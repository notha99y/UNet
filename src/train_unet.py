import os
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import keras
from keras.models import Model
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Dropout, Cropping2D, Input, merge
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as K  # want the Keras modules to be compatible
from keras.utils import to_categorical

from metrics import f1 as f1_score

from utils import resizing, make_dir


def UNet(filters_dims, activation='relu', kernel_initializer='glorot_uniform', padding='same'):
    inputs = Input((240, 320, 3))
    new_inputs = inputs
    conv_layers = []
    # Encoding Phase
    print("Encoding Phase")
    for i in range(len(filters_dims) - 1):
        print("Stage :", i+1)
        print("========================================")
        print(new_inputs.shape)
        conv = Conv2D(filters_dims[i], 3, activation=activation, padding=padding,
                      kernel_initializer=kernel_initializer)(new_inputs)
        conv = Conv2D(filters_dims[i], 3, activation=activation, padding=padding,
                      kernel_initializer=kernel_initializer)(conv)
        conv_layers.append(conv)
        new_inputs = MaxPooling2D(pool_size=(2, 2))(conv)
        print(new_inputs.shape)
        # op = BatchNormalization()(op)

    # middle phase
    print("middle phase")
    print("========================================")
    conv = Conv2D(filters_dims[-1], 3, activation=activation, padding=padding,
                  kernel_initializer=kernel_initializer)(new_inputs)
    conv = Conv2D(filters_dims[-1], 3, activation=activation, padding=padding,
                  kernel_initializer=kernel_initializer)(conv)
    new_inputs = Dropout(0.5)(conv)
    print(new_inputs.shape)

    filters_dims.reverse()
    conv_layers.reverse()

    # Decoding Phase
    print("Decoding Phase")
    for i in range(1, len(filters_dims)):
        print(i)
        print("========================================")

        print(new_inputs.shape)
        up = Conv2D(filters_dims[i], 3, activation=activation, padding=padding,
                    kernel_initializer=kernel_initializer)(UpSampling2D(size=(2, 2))(new_inputs))
        concat = merge([conv_layers[i-1], up], mode='concat', concat_axis=3)
        conv = Conv2D(filters_dims[i], 3, activation=activation, padding=padding,
                      kernel_initializer=kernel_initializer)(concat)
        new_inputs = Conv2D(filters_dims[i], 3, activation=activation, padding=padding,
                            kernel_initializer=kernel_initializer)(conv)
        print(new_inputs.shape)
    outputs = Conv2D(8, 1, activation='softmax', padding='same',
                     kernel_initializer='glorot_uniform')(new_inputs)
    print(outputs.shape)
    model = Model(input=inputs, output=outputs, name='UNet')
    model.compile(optimizer=Adam(lr=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'mse', f1_score])
    return model


def train(model, x, y, batch_size, epochs):
    # Running on multi GPU
    print('Tensorflow backend detected; Applying memory usage constraints')
    ss = K.tf.Session(config=K.tf.ConfigProto(gpu_options=K.tf.GPUOptions(allow_growth=True),
                                              log_device_placement=True))
    K.set_session(ss)
    ss.run(K.tf.global_variables_initializer())
    K.set_learning_phase(1)

    print("Getting data.. Image shape: {}. Masks shape : {}".format(x.shape,
                                                                    y.shape))
    print("The data will be split to Train Val: 80/20")

    # saving weights and logging
    weights_path = os.path.join(os.getcwd(), 'weights')
    make_dir(weights_path)
    filepath = 'weights/' + model.name + '.{epoch:02d}-{loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1,
                                 save_weights_only=True, save_best_only=True, mode='auto', period=1)
    tensor_board = TensorBoard(log_dir='logs/')

    history = model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs,
                        verbose=1, callbacks=[checkpoint, tensor_board], validation_split=0.2)

    return history


if __name__ == '__main__':
    unet_config = 'config/unet.json'
    print('unet json: {}'.format(os.path.abspath(unet_config)))
    with open(unet_config) as json_file:
        config = json.load(json_file)
    print("Initializing UNet model")
    model = UNet(filters_dims=config['filters_dims'],
                 activation=config['activation'],
                 kernel_initializer=config['kernel_initializer'],
                 padding=config['padding'])

    training_config = 'config/training.json'
    print('training json: {}'.format(os.path.abspath(training_config)))
    with open(training_config) as json_file:
        config = json.load(json_file)

    print("Loading data")
    data_path = os.path.join(os.getcwd(), 'data', 'processed')
    image_path = os.path.join(data_path, 'images')
    mask_path = os.path.join(data_path, 'labels', 'regions')

    transformed_images, transformed_masks = resizing(
        image_path, mask_path, verbose=False, plot=False)

    transformed_images = np.array(transformed_images)
    transformed_masks = np.array(transformed_masks)
    print("Performing one hot encoding")
    transformed_masks_oneHot = to_categorical(transformed_masks, 8)
    print("Initializing training instance")

    train(model=model,
          x=transformed_images,
          y=transformed_masks_oneHot,
          batch_size=config["batch_size"],
          epochs=config["epochs"])
