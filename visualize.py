import cv2
import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from pylab import rcParams
from tensorflow.python.framework import ops

rcParams['figure.figsize'] = 12, 12
# acquire gpu memory
K.clear_session()
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list=str(0),
        allow_growth=True
    )
)
K.set_session(tf.Session(config=config))


def random_mask(array, mask_size=120):
    height = array.shape[0]
    width = array.shape[1]
    x = np.random.randint(height - mask_size)
    y = np.random.randint(width - mask_size)
    new_array = np.zeros_like(array)
    new_array[x:x + mask_size, y:y + mask_size, :] = array[x:x + mask_size, y:y + mask_size, :]
    return new_array


def modify_backprop(model, name, model_path):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        new_model = load_model(model_path)
    return new_model


def compile_saliency_function(model, activation_layer):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)  # taking maximum of output channel
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img], [saliency])


def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                   tf.cast(op.inputs[0] > 0., dtype)


def get_modified_model(model, model_path):
    K.set_learning_phase(0)
    register_gradient()
    guided_model = modify_backprop(model, "GuidedBackProp", model_path)
    saliency_fn = compile_saliency_function(guided_model, activation_layer="acitvation")
    return saliency_fn


def gen_patch_from_img(full_img, stride, patch_size):
    patches = []
    idx = []
    shape = full_img.shape
    for i in range(np.ceil((shape[0] - patch_size) / stride).astype(int) + 1):
        for j in range(np.ceil((shape[1] - patch_size) / stride).astype(int) + 1):
            if stride * i + patch_size >= shape[0]:
                x1 = shape[0] - patch_size
                x2 = shape[0]
            else:
                x1 = stride * i
                x2 = stride * i + patch_size
            if stride * j + patch_size >= shape[1]:
                y1 = shape[1] - patch_size
                y2 = shape[1]
            else:
                y1 = stride * j
                y2 = stride * j + patch_size
            patches.append(full_img[x1:x2, y1:y2])
            idx.append([x1, x2, y1, y2])
    patches = np.array(patches)
    idx = np.array(idx)
    return patches, idx


def multi_grad_cam(model, patches, desired_prediction, layer_name="acitvation", target_shape=(180, 180)):
    class_output = model.output[:, desired_prediction]
    conv_output = model.get_layer(layer_name).output
    grads = K.gradients(class_output, conv_output)[0]
    gradient_function = K.function([model.input], [conv_output, grads])
    batch_size = 30
    output = []
    grads_val = []
    idx = 0
    len_data = len(patches)
    while (1):
        end = idx + batch_size
        if end > len_data:
            end = len_data
        o, g = gradient_function([patches[idx:end]])
        output.extend(o)
        grads_val.extend(g)
        if end == len_data:
            break
        idx = end
    output, grads_val = np.array(output), np.array(grads_val)
    weights = np.mean(grads_val, axis=(1, 2))
    cams = output * weights[:, np.newaxis, np.newaxis, :]
    cams = np.sum(cams, axis=3)
    cams = [cv2.resize(cam, target_shape[::-1], cv2.INTER_LINEAR) for cam in cams]
    cams = np.maximum(cams, 0)
    cams = cams / np.max(cams, axis=(1, 2))[:, np.newaxis, np.newaxis]
    return cams


def calc_guided(model, model_path, patches, layer_name="acitvation"):
    K.set_learning_phase(0)
    register_gradient()
    guided_model = modify_backprop(model, "GuidedBackProp", model_path)
    saliency_fn = compile_saliency_function(guided_model, activation_layer=layer_name)
    batch_size = 30
    guided = []
    idx = 0
    len_data = len(patches)
    while (1):
        end = idx + batch_size
        if end > len_data:
            end = len_data
        g = saliency_fn([patches[idx:end]])[0]
        guided.extend(g)
        if end == len_data:
            break
        idx = end
    guided = np.array(guided)
    guided = guided / guided.max()
    return guided


def concat_data(idx, cams, guided, orig_shape):
    counter = np.zeros(orig_shape[0:2])
    concat_cam = np.zeros(orig_shape[0:2])
    concat_guided = np.zeros(orig_shape[0:3])
    for j, (x1, x2, y1, y2) in enumerate(idx):
        if not np.isnan(cams[j]).any() and not np.isnan(guided[j]).any():
            counter[x1:x2, y1:y2] += 1
            concat_cam[x1:x2, y1:y2] += cams[j]
            concat_guided[x1:x2, y1:y2] += guided[j]
        counter[counter == 0] += 1  # avoid zero division
    return concat_cam / counter, concat_guided / counter[..., np.newaxis]


def composit_img(orig, cam):
    jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    jetcam = cv2.addWeighted((orig * 255).astype("uint8"), 0.5, jetcam.astype("uint8"), 0.5, 0)
    return jetcam


if __name__ == "__main__":
    model_path = "model.h5"
    model = load_model(model_path)
    img = cv2.imread("body.jpg")[:, :, ::-1]
    img = cv2.GaussianBlur(img, (5, 5), 0)
    left = img[:, 33:590] / 255
    right = img[:, 602:602 + 590 - 33] / 255
    stride = 10
    patch_size = 150

    patches, idx = gen_patch_from_img(left, stride, patch_size)
    cams = multi_grad_cam(model, patches, 0, target_shape=(patch_size, patch_size))
    guided = calc_guided(model, model_path, patches)
    grad_cam, guided = concat_data(idx, cams, guided, left.shape)
    grad_cam = grad_cam / grad_cam.max()
    np.save("grad_cam_left.npy", grad_cam)
    np.save("guided_left.npy", guided)
    img1 = composit_img(left, grad_cam)
    cv2.imwrite("left_1.png", img1[:, :, ::-1])

    patches, idx = gen_patch_from_img(right, stride, patch_size)
    cams = multi_grad_cam(model, patches, 0, target_shape=(patch_size, patch_size))
    guided = calc_guided(model, model_path, patches)
    grad_cam, guided = concat_data(idx, cams, guided, right.shape)
    grad_cam = grad_cam / grad_cam.max()
    np.save("grad_cam_right.npy", grad_cam)
    np.save("guided_right.npy", guided)
    img1 = composit_img(right, grad_cam)
    cv2.imwrite("right_1.png", img1[:, :, ::-1])
