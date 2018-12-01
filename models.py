from keras import layers
from keras.applications import vgg16
from keras.models import Model

from utils import get_style_loss, get_content_loss, get_tv_loss, \
    residual_block, OutputScale, InputReflect, AverageAddTwo


def get_training_model(width, height, bs=1, bi_style=False):
    input_o = layers.Input(shape=(height, width, 3), dtype='float32', name='input_o')

    c1 = layers.Conv2D(32, (9, 9), strides=1, padding='same', name='conv_1')(input_o)
    c1 = layers.BatchNormalization(name='normal_1')(c1)
    c1 = layers.Activation('relu', name='relu_1')(c1)

    c2 = layers.Conv2D(64, (3, 3), strides=2, padding='same', name='conv_2')(c1)
    c2 = layers.BatchNormalization(name='normal_2')(c2)
    c2 = layers.Activation('relu', name='relu_2')(c2)

    c3 = layers.Conv2D(128, (3, 3), strides=2, padding='same', name='conv_3')(c2)
    c3 = layers.BatchNormalization(name='normal_3')(c3)
    c3 = layers.Activation('relu', name='relu_3')(c3)

    r1 = residual_block(c3, 1)
    r2 = residual_block(r1, 2)
    r3 = residual_block(r2, 3)
    r4 = residual_block(r3, 4)
    r5 = residual_block(r4, 5)

    d1 = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', name='conv_4')(r5)
    d1 = layers.BatchNormalization(name='normal_4')(d1)
    d1 = layers.Activation('relu', name='relu_4')(d1)

    d2 = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', name='conv_5')(d1)
    d2 = layers.BatchNormalization(name='normal_5')(d2)
    d2 = layers.Activation('relu', name='relu_5')(d2)

    c4 = layers.Conv2D(3, (9, 9), strides=1, padding='same', name='conv_6')(d2)
    c4 = layers.BatchNormalization(name='normal_6')(c4)
    c4 = layers.Activation('tanh', name='tanh_1')(c4)
    c4 = OutputScale(name='output')(c4)

    content_activation = layers.Input(shape=(height // 2, width // 2, 128), dtype='float32')
    style_activation1 = layers.Input(shape=(height, width, 64), dtype='float32')
    style_activation2 = layers.Input(shape=(height // 2, width // 2, 128), dtype='float32')
    style_activation3 = layers.Input(shape=(height // 4, width // 4, 256), dtype='float32')
    style_activation4 = layers.Input(shape=(height // 8, width // 8, 512), dtype='float32')

    if bi_style:
        style_activation1_2 = layers.Input(shape=(height, width, 64), dtype='float32')
        style_activation2_2 = layers.Input(shape=(height // 2, width // 2, 128), dtype='float32')
        style_activation3_2 = layers.Input(shape=(height // 4, width // 4, 256), dtype='float32')
        style_activation4_2 = layers.Input(shape=(height // 8, width // 8, 512), dtype='float32')

    total_variation_loss = layers.Lambda(get_tv_loss, output_shape=(1,), name='tv',
                                         arguments={'width': width, 'height': height})([c4])

    # Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(c4)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    style_loss1 = layers.Lambda(get_style_loss, output_shape=(1,),
                                name='style1', arguments={'batch_size': bs})([x, style_activation1])
    if bi_style:
        style_loss1_2 = layers.Lambda(get_style_loss, output_shape=(1,),
                                    name='style1_2', arguments={'batch_size': bs})([x, style_activation1_2])
        style_loss1 = AverageAddTwo(name='style1_out')([style_loss1, style_loss1_2])
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    content_loss = layers.Lambda(get_content_loss, output_shape=(1,), name='content')([x, content_activation])
    style_loss2 = layers.Lambda(get_style_loss, output_shape=(1,),
                                name='style2', arguments={'batch_size': bs})([x, style_activation2])
    if bi_style:
        style_loss2_2 = layers.Lambda(get_style_loss, output_shape=(1,),
                                    name='style2_2', arguments={'batch_size': bs})([x, style_activation2_2])
        style_loss2 = AverageAddTwo(name='style2_out')([style_loss2, style_loss2_2])
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    style_loss3 = layers.Lambda(get_style_loss, output_shape=(1,),
                                name='style3', arguments={'batch_size': bs})([x, style_activation3])
    if bi_style:
        style_loss3_2 = layers.Lambda(get_style_loss, output_shape=(1,),
                                    name='style3_2', arguments={'batch_size': bs})([x, style_activation3_2])
        style_loss3 = AverageAddTwo(name='style3_out')([style_loss3, style_loss3_2])
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    style_loss4 = layers.Lambda(get_style_loss, output_shape=(1,),
                                name='style4', arguments={'batch_size': bs})([x, style_activation4])
    if bi_style:
        style_loss4_2 = layers.Lambda(get_style_loss, output_shape=(1,),
                                    name='style4_2', arguments={'batch_size': bs})([x, style_activation4_2])
        style_loss4 = AverageAddTwo(name='style4_out')([style_loss4, style_loss4_2])
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if bi_style:
        model = Model(
            [input_o, content_activation, style_activation1, style_activation2, style_activation3, style_activation4,
             style_activation1_2, style_activation2_2, style_activation3_2, style_activation4_2],
            [content_loss, style_loss1, style_loss2, style_loss3, style_loss4, total_variation_loss, c4])
    else:
        model = Model(
            [input_o, content_activation, style_activation1, style_activation2, style_activation3, style_activation4],
            [content_loss, style_loss1, style_loss2, style_loss3, style_loss4, total_variation_loss, c4])
    model_layers = {layer.name: layer for layer in model.layers}
    original_vgg = vgg16.VGG16(weights='imagenet', include_top=False)
    original_vgg_layers = {layer.name: layer for layer in original_vgg.layers}

    # load image_net weight
    for layer in original_vgg.layers:
        if layer.name in model_layers:
            model_layers[layer.name].set_weights(original_vgg_layers[layer.name].get_weights())
            model_layers[layer.name].trainable = False

    print("training model built successfully!")
    return model


def get_evaluate_model(width, height):
    input_o = layers.Input(shape=(height, width, 3), dtype='float32', name='input_o')

    c1 = layers.Conv2D(32, (9, 9), strides=1, padding='same', name='conv_1')(input_o)
    c1 = layers.BatchNormalization(name='normal_1')(c1)
    c1 = layers.Activation('relu', name='relu_1')(c1)

    c2 = layers.Conv2D(64, (3, 3), strides=2, padding='same', name='conv_2')(c1)
    c2 = layers.BatchNormalization(name='normal_2')(c2)
    c2 = layers.Activation('relu', name='relu_2')(c2)

    c3 = layers.Conv2D(128, (3, 3), strides=2, padding='same', name='conv_3')(c2)
    c3 = layers.BatchNormalization(name='normal_3')(c3)
    c3 = layers.Activation('relu', name='relu_3')(c3)

    r1 = residual_block(c3, 1)
    r2 = residual_block(r1, 2)
    r3 = residual_block(r2, 3)
    r4 = residual_block(r3, 4)
    r5 = residual_block(r4, 5)

    d1 = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', name='conv_4')(r5)
    d1 = layers.BatchNormalization(name='normal_4')(d1)
    d1 = layers.Activation('relu', name='relu_4')(d1)

    d2 = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', name='conv_5')(d1)
    d2 = layers.BatchNormalization(name='normal_5')(d2)
    d2 = layers.Activation('relu', name='relu_5')(d2)

    c4 = layers.Conv2D(3, (9, 9), strides=1, padding='same', name='conv_6')(d2)
    c4 = layers.BatchNormalization(name='normal_6')(c4)
    c4 = layers.Activation('tanh', name='tanh_1')(c4)
    c4 = OutputScale(name='output')(c4)

    model = Model([input_o], c4)
    print("evaluate model built successfully!")
    return model


def get_temp_view_model(width, height, bs=1, bi_style=False):
    input_o = layers.Input(shape=(height, width, 3), dtype='float32')

    y = InputReflect(width, height, name='output')(input_o)
    total_variation_loss = layers.Lambda(get_tv_loss, output_shape=(1,), name='tv',
                                         arguments={'width': width, 'height': height})([y])

    content_activation = layers.Input(shape=(height//2, width//2, 128), dtype='float32')
    style_activation1 = layers.Input(shape=(height, width, 64), dtype='float32')
    style_activation2 = layers.Input(shape=(height//2, width//2, 128), dtype='float32')
    style_activation3 = layers.Input(shape=(height//4, width//4, 256), dtype='float32')
    style_activation4 = layers.Input(shape=(height//8, width//8, 512), dtype='float32')

    if bi_style:
        style_activation1_2 = layers.Input(shape=(height, width, 64), dtype='float32')
        style_activation2_2 = layers.Input(shape=(height // 2, width // 2, 128), dtype='float32')
        style_activation3_2 = layers.Input(shape=(height // 4, width // 4, 256), dtype='float32')
        style_activation4_2 = layers.Input(shape=(height // 8, width // 8, 512), dtype='float32')

    # Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(y)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    style_loss1 = layers.Lambda(get_style_loss, output_shape=(1,),
                                name='style1', arguments={'batch_size': bs})([x, style_activation1])
    if bi_style:
        style_loss1_2 = layers.Lambda(get_style_loss, output_shape=(1,),
                                      name='style1_2', arguments={'batch_size': bs})([x, style_activation1_2])
        style_loss1 = AverageAddTwo(name='style1_out')([style_loss1, style_loss1_2])
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    content_loss = layers.Lambda(get_content_loss, output_shape=(1,), name='content')([x, content_activation])
    style_loss2 = layers.Lambda(get_style_loss, output_shape=(1,),
                                name='style2', arguments={'batch_size': bs})([x, style_activation2])
    if bi_style:
        style_loss2_2 = layers.Lambda(get_style_loss, output_shape=(1,),
                                      name='style2_2', arguments={'batch_size': bs})([x, style_activation2_2])
        style_loss2 = AverageAddTwo(name='style2_out')([style_loss2, style_loss2_2])
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    style_loss3 = layers.Lambda(get_style_loss, output_shape=(1,),
                                name='style3', arguments={'batch_size': bs})([x, style_activation3])
    if bi_style:
        style_loss3_2 = layers.Lambda(get_style_loss, output_shape=(1,),
                                      name='style3_2', arguments={'batch_size': bs})([x, style_activation3_2])
        style_loss3 = AverageAddTwo(name='style3_out')([style_loss3, style_loss3_2])
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    style_loss4 = layers.Lambda(get_style_loss, output_shape=(1,),
                                name='style4', arguments={'batch_size': bs})([x, style_activation4])
    if bi_style:
        style_loss4_2 = layers.Lambda(get_style_loss, output_shape=(1,),
                                      name='style4_2', arguments={'batch_size': bs})([x, style_activation4_2])
        style_loss4 = AverageAddTwo(name='style4_out')([style_loss4, style_loss4_2])
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if bi_style:
        model = Model(
            [input_o, content_activation, style_activation1, style_activation2, style_activation3,
             style_activation4,
             style_activation1_2, style_activation2_2, style_activation3_2, style_activation4_2],
            [content_loss, style_loss1, style_loss2, style_loss3, style_loss4, total_variation_loss, y])
    else:
        model = Model(
            [input_o, content_activation, style_activation1, style_activation2, style_activation3,
             style_activation4],
            [content_loss, style_loss1, style_loss2, style_loss3, style_loss4, total_variation_loss, y])
    model_layers = {layer.name: layer for layer in model.layers}
    original_vgg = vgg16.VGG16(weights='imagenet', include_top=False)
    original_vgg_layers = {layer.name: layer for layer in original_vgg.layers}

    # load image_net weight
    for layer in original_vgg.layers:
        if layer.name in model_layers:
            model_layers[layer.name].set_weights(original_vgg_layers[layer.name].get_weights())
            model_layers[layer.name].trainable = False

    print("temp_view model built successfully!")
    return model
