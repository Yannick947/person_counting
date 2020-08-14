import sys
import os
from time import gmtime, strftime
import json

import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from keras import Model
from keras.layers import (Dense, MaxPooling2D, Conv2D, Flatten, BatchNormalization,
                          AveragePooling2D, Reshape, LSTM, Layer, Lambda, Input, GRU)
from keras import backend as K
from sklearn.model_selection import ParameterSampler
from scipy.stats import loguniform

from person_counting.data_generators import data_generator_cnn as dgv_cnn
from person_counting.data_generators import data_generators as dgv
from person_counting.utils.visualization_utils import plot_losses, visualize_predictions
from person_counting.utils.hyperparam_utils import create_callbacks, get_optimizer, get_static_hparams, hard_tanh
from person_counting.models.model_argparse import parse_args, check_args
from person_counting.bin.evaluate import evaluate_run, create_mae_rescaled, create_accuracy_rescaled, parse_model
from person_counting.utils.preprocessing import get_filtered_lengths

def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    check_args(args)
    
    hparams_samples = get_samples(args)

    for sample in hparams_samples: 
        try: 
            timestep_num, feature_num = get_filtered_lengths(args.top_path, sample)

            datagen_train, datagen_validation, datagen_test = dgv_cnn.create_datagen(args.top_path,
                                                                                    sample,
                                                                                    args.label_file, 
                                                                                    args.augmentation_factor, 
                                                                                    args.filter_hour_above, 
                                                                                    args.filter_category_noisy)

            rescale_factor = datagen_train.label_scaler.scale_
            print('Using rescale factor ', rescale_factor)

            logdir = os.path.join(args.topdir_log + '_cnn_' + strftime("%Y_%b_%d_%H_%M_%S", gmtime()))

            model = create_cnn(timestep_num, feature_num, sample, rescale_factor, snap_path=args.warm_start_path)
            log_load_params(logdir, sample, args.warm_start_path)

            history, model= train(model=model,
                                  datagen_train=datagen_train,
                                  logdir=logdir,
                                  hparams=sample,
                                  datagen_validation=datagen_validation,
                                  epochs=args.epochs)

            evaluate_run(model, history, datagen_validation,  mode='validation', logdir=logdir, visualize=True, top_path=args.top_path)
            evaluate_run(model, history, datagen_test, mode='test', logdir=logdir, visualize=True, top_path=args.top_path)
            
            K.clear_session()
            del model

        except OSError as ose:
            print(ose)
            print('OSError ocurred, continue with next set of parameters')
            continue

def get_samples(args):
    '''Get different samples of hyperparameters

    Arguments: 
        args: Arguments read from command line 

    returns list of samples for hyperparameters out of given hparam space
    '''

    #Put values multiple times into list to increase probability to be chosen
    param_grid = {
                  'pooling_type'        : ['avg', 'max'],
                  'kernel_size'         : [i for i in range(3, 7)],
                  'kernel_number'       : [i for i in range(5, 12)],
                  'pool_size_y'         : [2],
                  'pool_size_x'         : [2, 3],
                  'learning_rate'       : loguniform.rvs(a=5e-4, b=3e-3, size=100000),
                  'optimizer'           : ['Adam', 'Nadam', 'AdaGrad'], 
                  'layer_number'        : [3, 4, 5], 
                  'batch_normalization' : [False, True],
                  'regularization'      : [0, 0.05, 0, 0.01, 0.1], 
                  'filter_rows_lower'   : [0], 
                  'filter_cols_upper'   : [0], 
                  'filter_cols_lower'   : [0],
                  'batch_size'          : [16, 32], 
                  'loss'                : ['mae', 'mae', 'mae', 'mse'], 
                  'Recurrent_Celltype'  : ['GRU', 'LSTM'], 
                  'units'               : [i for i in range(4,20)],          
                  'squeeze_method'      : ['1x1_conv', 'squeeze'], 
                  '1x1_conv_filters'    : [i + 1 for i in range(20)],
                }

    randint = int(tf.random.uniform(shape=[], minval=0, maxval=10, dtype=tf.dtypes.int32, seed=None))
    dynamic_params = list(ParameterSampler(param_grid, n_iter=args.n_runs, random_state=randint))
    
    static_params = get_static_hparams(args)

    return [{**dp, **static_params} for dp in dynamic_params]


def train(model,
          datagen_train,
          logdir=None,
          hparams=None,
          datagen_validation=None,
          workers=32,
          use_multiprocessing=True, 
          epochs=70, 
          rescale_factor=1):
    '''Train a given model with given datagenerator

    Arguments: 
        model: Cnn keras model 
        datagen_train: Datagenerator for training
        logdir: Path to folder for logging
        hparams: Sample of hyperparameter
        datagen_validation: Datagenerator for evaluation during testing
        workers: Number of workers if multiprocessing is true
        use_multiprocessing: Flag if multiprocessing is enabled
        epochs: Number of epochs to train
    '''
    
    #Add num params parameter only for logging purposes
    hparams['number_params'] = model.count_params()
    max_metrics = {'epoch_mae_rescaled': 'min',
                   'epoch_acc_rescaled' : 'max'}

    history = model.fit_generator(validation_steps=int(len(datagen_validation)),
                                  generator=datagen_train,
                                  validation_data=datagen_validation,
                                  steps_per_epoch=int(len(datagen_train)),
                                  epochs=epochs,
                                  verbose=0,
                                  shuffle=True,
                                  callbacks=create_callbacks(logdir, hparams, save_best=True, max_metrics=max_metrics), 
                                  use_multiprocessing=use_multiprocessing, 
                                  workers=workers, 
                                  max_queue_size=workers*2 + 1)
    return history, model


def create_cnn(timesteps, features, hparams, rescale_factor, snap_path=None):
    '''Creates a convolutional-rnn nn with architecture defined in hparams

    Arguments: 
        timesteps: Amount of timesteps in input data
        features: Number of columns (features)
        hparams: Sample of hyperparameters

        returns keras model with cnn architecture
    '''
    model = None

    if (snap_path is not 'None') and (snap_path is not None): 
        model = parse_model(logdir=snap_path, compile_model=False)

    print('Actual model is using following hyper-parameters:')
    for key in hparams.keys():
        print(key, ': ', hparams[key])

    if model is None: 
        #define layer creations
        def create_conv_layer(layer):
            kernel_size = max(3, hparams['kernel_size'] - layer)
            return Conv2D(hparams['kernel_number'],
                        kernel_size,
                        use_bias=True,
                        activation='relu', 
                        kernel_initializer=keras.initializers.glorot_normal(), 
                        padding='same')

        if hparams['pooling_type'] == 'avg': 
            def create_pool_layer():
                return AveragePooling2D(pool_size=(hparams['pool_size_x'], hparams['pool_size_y']))
        else:
            def create_pool_layer():
                return MaxPooling2D(pool_size=(hparams['pool_size_x'], hparams['pool_size_y']))

        if hparams['Recurrent_Celltype'] == 'LSTM': 
            def create_rnn_layer(return_sequences=False): 
                return LSTM(units=hparams['units'],
                            activation=hard_tanh,
                            return_sequences=return_sequences,
                            kernel_regularizer=keras.regularizers.l2(hparams['regularization']))

        elif hparams['Recurrent_Celltype'] == 'GRU':
            def create_rnn_layer(return_sequences=False): 
                return GRU(units=hparams['units'],
                        activation=hard_tanh,
                        return_sequences=return_sequences,
                        kernel_regularizer=keras.regularizers.l2(hparams['regularization']))

        layers = list()
        layers.append(Input(shape=(timesteps, features, 2)))

        for layer_num in range(hparams['layer_number']):
            try:
                layers.append(create_conv_layer(layer=layer_num)(layers[-1]))
                layers.append(create_pool_layer()(layers[-1]))

            except ValueError:
                #Creation failed, due to negative resulting dimension. hparam must be adjusted for logging
                hparams['layer_number'] -= 1 
                print('Tried to create a Pool Layer that is not possible to create,',
                    'because it would lead to negative dimensions. Creation was skipped')
        
        #Use 1x1 Convolution to remove depth of model
        layers.append(Conv2D(hparams['1x1_conv_filters'], 1, use_bias=True, activation='relu', padding='same')(layers[-1]))
        
        #Squeeze 4th dimension and pass to time-series without loosing information
        layers.append(Lambda(squeeze_dim3, output_shape = squeeze_dim3_shape)(layers[-1]))

        layers.append(create_rnn_layer(return_sequences=True)(layers[-1]))
        layers.append(create_rnn_layer(return_sequences=False)(layers[-1]))
        layers.append(Dense(1, activation='relu', kernel_regularizer=keras.regularizers.l2(hparams['regularization']))(layers[-1]))
        
        model = Model(layers[0], layers[-1])

    #Create custom mae and acc for original unscaled data
    mae_rescaled = create_mae_rescaled(rescale_factor)
    acc_rescaled = create_accuracy_rescaled(rescale_factor)

    optimizer = get_optimizer(hparams['optimizer'], hparams['learning_rate'])
    model.compile(loss=hparams['loss'], metrics=['msle', mae_rescaled, acc_rescaled], optimizer=optimizer)
    model.summary()

    return model

def log_load_params(logdir, hparams, snap_path):
    """ Load and log model architecture for later loading of architecture"""
    if not os.path.exists(logdir): 
        os.mkdir(logdir)
        
    if (snap_path is not 'None') and (snap_path is not None): 
        update_keys = ['pool_size_y', 'pool_size_x', 'layer_number', 'pooling_type', 'kernel_size','squeeze_method',
                       'units', 'kernel_number',  'Recurrent_Celltype', 'batch_normalization', 'regularization', ]

        with open(os.path.join(snap_path, 'config.json'), 'r') as fp:
            config = json.load(fp)

        for key in update_keys: 
            hparams.update({key:config[key]}) 

    with open(os.path.join(logdir, 'config.json'), 'w') as fp:
        json.dump(hparams, fp) 

def squeeze_dim3(x3d):
    """ Squeeze 3rd dimension in CNN architecture """
    shape = tf.shape( x3d ) # get dynamic tensor shape
    x3d = tf.reshape( x3d, [shape[0], shape[1], shape[2] * shape[3]])
    return x3d

def squeeze_dim3_shape(x3d_shape):
    """ Squeeze shape of 3rd dimension in CNN architecture """
    in_batch, in_rows, in_cols, in_filters = x3d_shape
    if ( None in [ in_rows, in_cols] ) :
        output_shape = (in_batch, None, in_filters)
    else :
        output_shape = (in_batch, in_rows, in_cols * in_filters)
    return output_shape


if __name__ == '__main__':
    main() 
