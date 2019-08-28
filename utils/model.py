from densenet121 import *
from keras import applications

from keras.layers import Input, ZeroPadding2D, Conv2D, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam


def get_baseline_model(num_classes, img_width=32, img_height=32, channel=1, lr=0.001):
    model = Sequential()
    # input: 168X128 images with 1 channel -> (1, 168, 128) tensors.
    # this applies 64 convolution filters of size 3x3 each.
    model.add(Conv2D(64, (3, 3), padding='valid',
                     input_shape=(img_width, img_height, channel)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=lr, decay=1e-7, momentum=0.9, nesterov=True)
    # rms = RMSprop()
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['categorical_accuracy'])
    return model


def get_vgg_model(num_classes, img_width=48, img_height=48, channel=3, lr=1e-5):
    base_model = applications.vgg19.VGG19(include_top=False,
                                          weights=None,
                                          input_shape=(img_width, img_height, channel))

    # default shape is (224,224,3). However, minimum can be (48,48,3)

    x = base_model.output
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    predictions = Dense(num_classes, activation='softmax',
                        name='predictions')(x)

    model = Model(base_model.input, predictions)

    for layer in model.layers[:23]:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=lr, momentum=0.9, decay=1e-7),
                  metrics=['categorical_accuracy'])
    return model


def get_inception_model(num_classes, img_width=139, img_height=139, channel=3, lr=1e-5):
    # default shape is (224,224,3). However, minimum can be (139,139,3)

    base_model = applications.inception_v3.InceptionV3(weights=None,
                                                       include_top=False,
                                                       input_shape=(img_width, img_height, channel))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True

    model.compile(optimizer=SGD(lr=lr, decay=1e-7, momentum=0.9), loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model


def get_resnet_model(num_classes, img_width=200, img_height=200, channel=3, lr=1e-5):
    # default shape is (224,224,3). However, minimum can be (200,200,3)

    base_model = applications.resnet50.ResNet50(weights='imagenet',
                                                include_top=False,
                                                input_shape=(img_width, img_height, channel))

    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in model.layers[:172]:
        layer.trainable = False
    # for layer in model.layers[172:]:
    #     layer.trainable = True

    model.compile(optimizer=SGD(lr=lr, decay=1e-7, momentum=0.9), loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model


def get_dense_model(num_classes, img_width=200, img_height=200, channel=3, lr=1e-5, loss=None):
    nb_dense_block = 4
    growth_rate = 32
    reduction = 0.0
    dropout_rate = 0.0
    weight_decay = 1e-4
    classes = num_classes
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis
    concat_axis = 3
    img_input = Input(shape=(img_width, img_height, channel), name='data')

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 64
    nb_layers = [6, 12, 24, 16]  # For DenseNet-121

    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Convolution2D(nb_filter, 7, 7, subsample=(2, 2),
                      name='conv1', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx + 2
        x, nb_filter = dense_block(x, stage, nb_layers[
            block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression,
                             dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[
        -1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis,
                           name='conv' + str(final_stage) + '_blk_bn')(x)
    x = Scale(axis=concat_axis, name='conv' +
                                     str(final_stage) + '_blk_scale')(x)
    x = Activation('relu', name='relu' + str(final_stage) + '_blk')(x)
    x = GlobalAveragePooling2D(name='pool' + str(final_stage))(x)

    x = Dense(classes, name='fc6')(x)
    x = Activation('softmax', name='prob')(x)

    model = Model(img_input, x, name='densenet')

    optimizer = Adam(lr=1e-4)  # Using Adam instead of SGD to speed up training

    if loss is None:
        loss = 'categorical_crossentropy'
    model.compile(loss=loss,
                  optimizer=optimizer, metrics=["accuracy"])
    return model


def get_keras_app_densenet(num_classes, img_width=200, img_height=200, channel=3, lr=1e-5, loss=None):
    model = applications.densenet.DenseNet121(input_shape=(img_width,img_height,channel),weights=None, classes=num_classes)

    optimizer = Adam(lr=lr)  # Using Adam instead of SGD to speed up training

    if loss is None:
        loss = 'categorical_crossentropy'
    model.compile(loss=loss,
                  optimizer=optimizer, metrics=["accuracy"])
    return model
