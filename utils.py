from keras import backend as K
from keras import layers
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg16
from keras import initializers

import numpy as np


def process_image(image_path, width, height, resize=True):
    if resize:
        img = load_img(image_path, target_size=(height, width))
    else:
        img = load_img(image_path)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img


def deprocess_image(x, width, height):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, height, width))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((height, width, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def expand_input(batch_size, input_o):
    expanded_input = input_o.copy()
    for x in range(batch_size - 1):
        expanded_input = np.append(expanded_input, input_o, axis=0)

    return expanded_input


def get_padding(image, axis_expanded=True):
    height = image.shape[1]
    width = image.shape[2]

    pad_height = (height//8 + 1) * 8 - height
    pad_width = (width//8 + 1) * 8 - width

    if axis_expanded:
        padding = (0, 0), (0, pad_height), (0, pad_width), (0, 0)
    else:
        padding = ((0, pad_height), (0, pad_width), (0, 0))

    new_image = np.pad(image, padding, 'reflect')
    return new_image


def remove_padding(image, ori_height, ori_width):
    new_image = image[0:ori_height, 0:ori_width, :]
    return new_image


def get_vgg_activation(layer_name, width, height):
    tensor = K.placeholder((1, height, width, 3))
    model = vgg16.VGG16(input_tensor=tensor, weights='imagenet', include_top=False)
    outputs_dict = {}
    for layer in model.layers:
        outputs_dict[layer.name] = layer.output
        layer.trainable = False
    return K.function([tensor], [outputs_dict[layer_name]])


def get_content_loss(args):
    new_activation, content_activation = args[0], args[1]
    shape = K.cast(K.shape(content_activation), dtype='float32')
    return K.sum(K.square(new_activation - content_activation)) / (shape[1] * shape[2] * shape[3])


def gram_matrix(activation):
    assert K.ndim(activation) == 3
    shape = K.cast(K.shape(activation), dtype='float32')
    shape = (shape[0] * shape[1], shape[2])

    activation = K.reshape(activation, shape)
    return K.dot(K.transpose(activation), activation) / (shape[0] * shape[1])


def get_style_loss(args, **kwargs):
    batch_size = kwargs['batch_size']
    new_activation, style_activation = args[0], args[1]

    loss_sum = K.variable(0.0)
    for i in range(batch_size):
        ori_gram_matrix = gram_matrix(style_activation[i])
        new_gram_matrix = gram_matrix(new_activation[i])
        loss_sum = loss_sum + K.sum(K.square(ori_gram_matrix - new_gram_matrix))
    return loss_sum


def get_tv_loss(args, **kwargs):
    image = args[0]
    width = kwargs['width']
    height = kwargs['height']
    x_diff = K.square(image[:, :height - 1, :, :] - image[:, 1:, :, :])
    y_diff = K.square(image[:, :, :width - 1, :] - image[:, :, 1:, :])
    x_diff = K.sum(x_diff) / height
    y_diff = K.sum(y_diff) / width
    return x_diff + y_diff


def residual_block(x, num):
    shortcut = x
    x = layers.Conv2D(128, (3, 3), strides=1, padding='same', name='resi_conv_%d_1' % num)(x)
    x = layers.BatchNormalization(name='resi_normal_%d_1' % num)(x)
    x = layers.Activation('relu', name='resi_relu_%d_1' % num)(x)
    x = layers.Conv2D(128, (3, 3), strides=1, padding='same', name='resi_conv_%d_2' % num)(x)
    x = layers.BatchNormalization(name='resi_normal_%d_2' % num)(x)
    m = layers.add([x, shortcut], name='resi_add_%d' % num)
    return m


def dummy_loss(y_true, y_pred):
    return y_pred


def zero_loss(y_true, y_pred):
    return K.variable(np.zeros(1,))


class OutputScale(layers.Layer):

    def __init__(self, **kwargs):
        super(OutputScale, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        return x * 150

    def compute_output_shape(self, input_shape):
        return input_shape


class AverageAddTwo(layers.Layer):

    def __init__(self, **kwargs):
        super(AverageAddTwo, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        return x[0] / 2 + x[1] / 2

    def compute_output_shape(self, input_shape):
        return input_shape


class InputReflect(layers.Layer):

    def __init__(self, width, height, **kwargs):
        super(InputReflect, self).__init__(**kwargs)
        self.width = width
        self.height = height

    def build(self, input_shape):
        init = initializers.RandomUniform(minval=-50, maxval=50, seed=None)
        self.kernel = self.add_weight(name='kernel', shape=(self.height, self.width, 3),
                                      initializer=init, trainable=True)

        super(InputReflect, self).build(input_shape)

    def call(self, x, mask=None):
        y = x * self.kernel
        return y

    def compute_output_shape(self, input_shape):
        return input_shape
