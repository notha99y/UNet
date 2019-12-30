import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Model
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Dropout, Cropping2D, Input, merge
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as K  # want the Keras modules to be compatible

from utils import resizing


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
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', help='weights file in hdf5 format', required=True)
    args = parser.parse_args()
    ss = K.tf.Session(config=K.tf.ConfigProto(gpu_options=K.tf.GPUOptions(allow_growth=True),
                                              log_device_placement=True))
    K.set_session(ss)
    ss.run(K.tf.global_variables_initializer())
    K.set_learning_phase(0)

    unet_config = 'config/unet.json'
    print('unet json: {}'.format(os.path.abspath(unet_config)))
    with open(unet_config) as json_file:
        config = json.load(json_file)
    print("Initializing UNet model")
    model = UNet(filters_dims=config['filters_dims'],
                 activation=config['activation'],
                 kernel_initializer=config['kernel_initializer'],
                 padding=config['padding'])

    # Loading weights
    model_name = os.path.basename(args.weights)
    print('Loading model: {}'.format(model_name))
    model.load_weights(args.weights)
    # Getting images
    data_path = os.path.join(os.getcwd(), 'data', 'processed')
    image_path = os.path.join(data_path, 'images')
    mask_path = os.path.join(data_path, 'labels', 'regions')

    transformed_images, transformed_masks = resizing(
        image_path, mask_path, verbose=False, plot=False)

    transformed_images = np.array(transformed_images)
    transformed_masks = np.array(transformed_masks)

    # validating user input
    while True:
        image_num = input("Choose image from 0 to {}".format(len(transformed_masks) - 1))
        try:
            image_num = int(image_num)
        except ValueError:
            print("please input a valid number from 0 to {} . {} is not a valid number".format(
                len(transformed_masks) - 1, image_num))
            continue
        if 0 <= image_num <= len(transformed_masks)-1:
            break
        else:
            print("Please input a number from 0 to {}. {} is out of range".format(
                len(transformed_masks) - 1, image_num))

    # plotting Segmentation
    while True:
        print('Showing image: {}'.format(image_num))
        test_image = transformed_images[image_num]
        test_mask = transformed_masks[image_num]
        prediction = model.predict(transformed_images[image_num:image_num+1])[0]

        # combining the prediction layers
        combined_prediction = np.zeros(prediction.shape[:-1])
        for lays in range(prediction.shape[-1]):
            combined_prediction += np.round(prediction[:, :, lays])*lays

        # Creating plots
        fig, ax = plt.subplots(2, 2)

        ax[0, 0].imshow((test_image))
        ax[0, 0].set_axis_off()
        ax[0, 0].set_title('Test_Image')

        ax[0, 1].matshow(test_mask)
        ax[0, 1].set_axis_off()
        ax[0, 1].set_title('Ground Truth Seg')

        ax[1, 0].matshow(combined_prediction)
        ax[1, 0].set_axis_off()
        ax[1, 0].set_title('Predicted Seg')

        ax[1, 1].set_axis_off()

        loss = model_name.split('-')[-1].split('.')[0]
        epoch = model_name.split('-')[0].split('.')[-1]
        text = '0: sky \n1: tree \n2: road\n3: grass\n4: water\n5: building\n6: mountain\n7: foreground'
        fig.text(0.65, 0.1, text)
        fig.text(0.7, 0.37, "Loss: {}".format(loss))
        fig.text(0.65, 0.4, "Epoch: {}".format(epoch), fontsize=20, fontweight='bold')

        plt.show()

        # validating user input
        while True:
            image_num = input("Choose image from 0 to {}".format(len(transformed_masks) - 1))
            try:
                image_num = int(image_num)
            except ValueError:
                print("please input a valid number from 0 to {} . {} is not a valid number".format(
                    len(transformed_masks) - 1, image_num))
                continue
            if 0 <= image_num <= len(transformed_masks)-1:
                break
            else:
                print("Please input a number from 0 to {}. {} is out of range".format(
                    len(transformed_masks) - 1, image_num))
