from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import sys
import threading
import copy
import inspect
import types

from keras import backend as K
from keras.utils.generic_utils import Progbar
import tensorflow as tf
import cv2


class ImageDataGenerator(object):
    '''Generate minibatches with
    real-time data augmentation.

    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        featurewise_standardize_axis: axis along which to perform feature-wise center and std normalization.
        samplewise_standardize_axis: axis along which to to perform sample-wise center and std normalization.
        zca_whitening: apply ZCA whitening.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided (before applying
            any other transformation).
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode it is at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".
        seed: random seed for reproducible pipeline processing. If not None, it will also be used by `flow` or
            `flow_from_directory` to generate the shuffle index in case of no seed is set.
    '''

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 featurewise_standardize_axis=None,
                 samplewise_standardize_axis=None,
                 zca_whitening=False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 dim_ordering=K.image_dim_ordering(),
                 seed=None,
                 verbose=1):
        self.config = copy.deepcopy(locals())
        self.config['config'] = self.config
        self.config['mean'] = None
        self.config['std'] = None
        self.config['principal_components'] = None
        self.config['rescale'] = rescale

        if dim_ordering not in {'tf', 'th'}:
            raise Exception('dim_ordering should be "tf" (channel after row and '
                            'column) or "th" (channel before row and column). '
                            'Received arg: ', dim_ordering)

        self.__sync_seed = self.config['seed'] or np.random.randint(0, 4294967295)

        self.default_pipeline = []
        self.default_pipeline.append(random_transform)
        self.default_pipeline.append(standardize)
        self.set_pipeline(self.default_pipeline)

        self.__fitting = False
        self.fit_lock = threading.Lock()

    @property
    def sync_seed(self):
        return self.__sync_seed

    @property
    def fitting(self):
        return self.__fitting

    @property
    def pipeline(self):
        return self.__pipeline

    def sync(self, image_data_generator):
        self.__sync_seed = image_data_generator.sync_seed
        return (self, image_data_generator)

    def set_pipeline(self, p):
        if p is None:
            self.__pipeline = self.default_pipeline
        elif type(p) is list:
            self.__pipeline = p
        else:
            raise Exception('invalid pipeline.')

    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_mode=None, save_format='jpeg'):
        return NumpyArrayIterator(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.config['dim_ordering'],
            save_to_dir=save_to_dir, save_prefix=save_prefix,
            save_mode=save_mode, save_format=save_format)

    def flow_from_list(self, X, y=None, batch_size=32, shuffle=True, seed=None,
                       save_to_dir=None, save_prefix='', save_mode=None, save_format='jpeg'):
        return ListArrayIterator(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.config['dim_ordering'],
            save_to_dir=save_to_dir, save_prefix=save_prefix,
            save_mode=save_mode, save_format=save_format)

    # def flow_with_mask(self, X, y=None, batch_size=32, shuffle=True, seed=None,
    #          save_to_dir=None, save_prefix='', save_mode=None, save_format='jpeg'):
    #     return ListArrayIteratorWithMask(
    #         X, y, self,
    #         batch_size=batch_size, shuffle=shuffle, seed=seed,
    #         dim_ordering=self.config['dim_ordering'],
    #         save_to_dir=save_to_dir, save_prefix=save_prefix,
    #         save_mode=save_mode, save_format=save_format)

    def flow_from_directory(self, directory,
                            color_mode=None, target_size=None,
                            image_reader='pil', reader_config=None,
                            read_formats=None,
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None, save_prefix='',
                            save_mode=None, save_format='jpeg'):
        if reader_config is None:
            reader_config = {'target_mode': 'RGB', 'target_size': (256, 256)}
        if read_formats is None:
            read_formats = {'png', 'jpg', 'jpeg', 'bmp'}
        return DirectoryIterator(
            directory, self,
            color_mode=color_mode, target_size=target_size,
            image_reader=image_reader, reader_config=reader_config,
            read_formats=read_formats,
            classes=classes, class_mode=class_mode,
            dim_ordering=self.config['dim_ordering'],
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix,
            save_mode=save_mode, save_format=save_format)

    def process(self, x):
        # get next sync_seed
        np.random.seed(self.__sync_seed)
        self.__sync_seed = np.random.randint(0, 4294967295)
        self.config['fitting'] = self.__fitting
        self.config['sync_seed'] = self.__sync_seed
        for p in self.__pipeline:
            x = p(x, **self.config)
        return x

    def fit_generator(self, generator, nb_iter):
        '''Fit a generator

        # Arguments
            generator: Iterator, generate data for fitting.
            nb_iter: Int, number of iteration to fit.
        '''
        with self.fit_lock:
            try:
                self.__fitting = nb_iter*generator.batch_size
                for i in range(nb_iter):
                    next(generator)
            finally:
                self.__fitting = False

    def fit(self, X, rounds=1):
        '''Fit the pipeline on a numpy array

        # Arguments
            X: Numpy array, the data to fit on.
            rounds: how many rounds of fit to do over the data
        '''
        # X = np.copy(X)
        with self.fit_lock:
            try:
                # self.__fitting = rounds*X.shape[0]
                self.__fitting = rounds * len(X)
                for r in range(rounds):
                    # for i in range(X.shape[0]):
                    for i in range(len(X)):
                        self.process(X[i])
            finally:
                self.__fitting = False


if __name__ == '__main__':
    pass
