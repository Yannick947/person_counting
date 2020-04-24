import pandas as pd
import numpy as np
import random
import os
from time import gmtime, strftime

import keras
import tensorflow as tf
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, BatchNormalization
from sklearn.model_selection import ParameterSampler

from person_counting.data_generators import data_generator_cnn as dgv_cnn
from person_counting.data_generators import data_generators as dgv
from person_counting.utils.visualization_utils import plot, visualize_predictions
from person_counting.utils.hyperparam_utils import create_callbacks, get_optimizer

LOGDIR_TOP = os.path.join('tensorboard\\')
n_runs = 100


def main():

    timestep_num, feature_num = dgv.get_filtered_lengths()
    datagen_train, datagen_test = dgv_cnn.create_datagen()
    # test_best(timestep_num, feature_num, datagen_train, datagen_test)

    hparams_samples = get_samples(n_runs)
    for sample in hparams_samples: 
        logdir = os.path.join(LOGDIR_TOP + strftime("%Y_%b_%d_%H_%M_%S", gmtime()))
        model = create_cnn(timestep_num, feature_num, sample)
        train(model, datagen_train, logdir, sample, datagen_test)


def get_samples(n_runs=10):
    '''
    '''

    #Put values multiple times into list to increase probability to be chosen
    param_grid = {
                  'dropout'             : [i / 100 for i in range(50)],
                  'kernel_size'         : [i for i in range (3, 10)],
                  'kernel_number'       : [i for i in range(3,10)],
                  'max_pool_size'       : [i for i in range(2, 5)],
                  'learning_rate'       : [i / 1e5 for i in range(100)],
                  'optimizer'           : ['Adam', 'Adam' 'RMSProp', 'SGD'], 
                  'layer_number'        : [2, 3], 
                  'batch_normalization' : [True, False], 
                  'y_stride'            : [1, 1, 1, 2]
                 }

    return list(ParameterSampler(param_grid, n_iter=n_runs))


def test_best(timestep_num,
              feature_num,
              datagen_train,
              datagen_test): 
    
    hparams = {
               'kernel_size'           : 3,
               'regularizer'           : 0.1,
               'kernel_number'         : 2
              }

    cnn_model = create_cnn(timestep_num, feature_num, hparams)
    history, model = train(cnn_model, datagen_train, hparams=hparams, datagen_test=datagen_test)
    plot(history)
    visualize_predictions(model, datagen_test)


def train(model, datagen_train, logdir=None, hparams=None, datagen_test=None):
    '''
    '''

    history = model.fit_generator(validation_steps=5,
                                  generator=datagen_train.datagen(),
                                  validation_data=datagen_test.datagen(),
                                  steps_per_epoch=30,
                                  epochs=15,
                                  verbose=1,
                                  shuffle=True,
                                  callbacks=create_callbacks(logdir, hparams)
                                  )

    return history, model


def create_cnn(timesteps, features, hparams):
    '''
    '''
    stride = (1, hparams['y_stride'])
    model = keras.Sequential()
    model.add(Conv2D(hparams['kernel_number'], hparams['kernel_size'], strides=stride, input_shape=(timesteps, features, 1)))    
    model.add(MaxPooling2D(pool_size=hparams['max_pool_size']))
 
    for _ in range(hparams['layer_number'] - 1):
        try:
            if hparams['batch_normalization']:
                model.add(BatchNormalization())

            model.add(Conv2D(hparams['kernel_number'], hparams['kernel_size'], strides=stride))
            model.add(MaxPooling2D(pool_size=(hparams['max_pool_size'], hparams['max_pool_size'])))

        except ValueError: 
            print('Tried to create a MaxPool Layer that is not possible to create,',
                  'because it would lead to negative dimensions. Creation was skipped')

    model.add(Flatten())
    model.add(Dense(1, activation='tanh'))
    
    optimizer = get_optimizer(hparams['optimizer'], hparams['learning_rate'])
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.summary()
    return model

if __name__ == '__main__':
    main() 

