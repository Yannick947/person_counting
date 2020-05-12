import sys
import pandas as pd
import os
from time import gmtime, strftime

import tensorflow as tf
import keras

from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, BatchNormalization, AveragePooling2D, Reshape, LSTM, Layer, Lambda, Input
from sklearn.model_selection import ParameterSampler
from scipy.stats import loguniform

from person_counting.data_generators import data_generator_cnn as dgv_cnn
from person_counting.data_generators import data_generators as dgv
from person_counting.utils.visualization_utils import plot_losses, visualize_predictions
from person_counting.utils.hyperparam_utils import create_callbacks, get_optimizer, get_static_hparams
from person_counting.models.model_argparse import parse_args
from person_counting.bin.evaluate import evaluate_model, create_mae_rescaled


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    hparams_samples = get_samples(args)

    for sample in hparams_samples: 
        timestep_num, feature_num = dgv.get_filtered_lengths(args.top_path, sample)

        datagen_train, datagen_test = dgv_cnn.create_datagen(args.top_path,
                                                             sample,
                                                             args.label_file, 
                                                             args.augmentation_factor, 
                                                             args.filter_hour_above, 
                                                             args.filter_category_noisy)

        rescale_factor = datagen_train.scaler.scaler_labels.scale_
        print('Using rescale factor ', rescale_factor)

        logdir = os.path.join(args.topdir_log + '_cnn_' + strftime("%Y_%b_%d_%H_%M_%S", gmtime()))
        model = create_cnn(timestep_num, feature_num, sample, rescale_factor)
        history, model= train(model=model,
                              datagen_train=datagen_train,
                              logdir=logdir,
                              hparams=sample,
                              datagen_test=datagen_test,
                              epochs=args.epochs)

        evaluate_model(model, history, datagen_test,  mode='test', logdir=logdir, visualize=False)
        evaluate_model(model, history, datagen_train, mode='train', logdir=logdir, visualize=False)


def get_samples(args):
    '''Get different samples of hyperparameters

    Arguments: 
        args: Arguments read from command line 

    returns list of samples for hyperparameters out of given hparam space
    '''

    #Put values multiple times into list to increase probability to be chosen
    param_grid = {
                  'pooling_type'        : ['avg', 'max'],
                  'kernel_size'         : [i for i in range(3, 5)],
                  'kernel_number'       : [i for i in range(3,7)],
                  'pool_size_y'         : [2],
                  'pool_size_x'         : [2],
                  'learning_rate'       : loguniform.rvs(a=1e-4, b=1e-2, size=100000),
                  'optimizer'           : ['Adam', 'Adam', 'SGD'], 
                  'layer_number'        : [1, 2, 3], 
                  'batch_normalization' : [False],
                  'regularization'      : [0], 
                  'filter_rows_lower'   : [150], 
                  'filter_cols_upper'   : [35], 
                  'filter_cols_lower'   : [25],
                  'filter_rows_factor'  : [1],
                  'batch_size'          : [32], 
                  'units'               : [i for i in range(3,7)],
                }

    dynamic_params = list(ParameterSampler(param_grid, n_iter=args.n_runs))
    static_params = get_static_hparams(args)

    return [{**dp, **static_params} for dp in dynamic_params]


def train(model,
          datagen_train,
          logdir=None,
          hparams=None,
          datagen_test=None,
          workers=16,
          use_multiprocessing=True, 
          epochs=50, 
          rescale_factor=1):
    '''Train a given model with given datagenerator

    Arguments: 
        model: Cnn keras model 
        datagen_train: Datagenerator for training
        logdir: Path to folder for logging
        hparams: Sample of hyperparameter
        datagen_test: Datagenerator for evaluation during testing
        workers: Number of workers if multiprocessing is true
        use_multiprocessing: Flag if multiprocessing is enabled
        epochs: Number of epochs to train
    '''

    print('Actual model is using following hyper-parameters:')
    for key in hparams.keys():
        print(key, ': ', hparams[key])
    
    #Add num params parameter only for logging purposes
    hparams['number_params'] = model.count_params()

    history = model.fit_generator(validation_steps=int(len(datagen_test)),
                                  generator=datagen_train.datagen(),
                                  validation_data=datagen_test.datagen(),
                                  steps_per_epoch=int(len(datagen_train)),
                                  epochs=epochs,
                                  verbose=1,
                                  shuffle=True,
                                  callbacks=create_callbacks(logdir, hparams, save_best=False), 
                                  use_multiprocessing=use_multiprocessing, 
                                  workers=workers, 
                                  max_queue_size=16,
                                  )

    return history, model


def create_cnn(timesteps, features, hparams, rescale_factor):
    '''Creates a convolutional nn with architecture defined in hparams

    Arguments: 
        timesteps: Amount of timesteps in input data
        features: Number of columns (features)
        hparams: Sample of hyperparameters

        returns keras model with cnn architecture
    '''

    conv_layer = Conv2D(hparams['kernel_number'],
                        hparams['kernel_size'],
                        use_bias=True,
                        activation='relu', 
                        kernel_initializer=keras.initializers.glorot_normal())
                          
    if hparams['pooling_type'] == 'avg': 
        pooling_layer =  AveragePooling2D(pool_size=(hparams['pool_size_x'], hparams['pool_size_y']))
    else: 
        pooling_layer = MaxPooling2D(pool_size=(hparams['pool_size_x'], hparams['pool_size_y']))


    #Create the model architecture
    model = keras.Sequential()
    model.add(AveragePooling2D(pool_size=(hparams['pool_size_x'], hparams['pool_size_y']),
                               input_shape=(timesteps, features, 1)))

    for _ in range(hparams['layer_number']):
        try:
            if hparams['batch_normalization']: 
                model.add(BatchNormalization())

            model.add(conv_layer) 
            model.add(pooling_layer)

        except ValueError:
            #Creation failed, hparam must be adjusted for logging
            hparams['layer_number'] -= 1 
            print('Tried to create a Pool Layer that is not possible to create,',
                  'because it would lead to negative dimensions. Creation was skipped')
    
    #Squeeze 4th dimension and pass to time-series module
    model.add(Lambda(squeeze_dim4, output_shape = squeeze_dim4_shape))
    model.add(LSTM(units=hparams['units'],activation='relu', return_sequences=True))
    model.add(LSTM(units=hparams['units'], activation='relu', return_sequences=False))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(hparams['regularization'])))
    
    mae_rescaled = create_mae_rescaled(rescale_factor)
    
    optimizer = get_optimizer(hparams['optimizer'], hparams['learning_rate'])
    model.compile(loss='mae', metrics=['mse', 'msle', mae_rescaled], optimizer=optimizer)
    model.summary()
    return model


def squeeze_dim4(x4d):
    shape = tf.shape( x4d ) # get dynamic tensor shape
    x3d = tf.reshape( x4d, [shape[0], shape[1], shape[2] * shape[3]])
    return x3d

def squeeze_dim4_shape(x4d_shape):
    in_batch, in_rows, in_cols, in_filters = x4d_shape
    if ( None in [ in_rows, in_cols] ) :
        output_shape = (in_batch, None, in_filters)
    else :
        output_shape = (in_batch, in_rows, in_cols * in_filters)
    return output_shape


if __name__ == '__main__':
    main() 

