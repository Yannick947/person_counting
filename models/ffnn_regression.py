import sys
import os
from time import gmtime, strftime

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from keras import backend as K
from keras.models import Model
from keras.layers import Dense, BatchNormalization, AveragePooling2D, LSTM, MaxPooling2D, Flatten, TimeDistributed, Input
from sklearn.model_selection import ParameterSampler
from scipy.stats import loguniform

from person_counting.data_generators import data_generator_cnn as dgv_lstm
from person_counting.data_generators import data_generators as dgv
from person_counting.utils.visualization_utils import plot_losses, visualize_predictions
from person_counting.utils.hyperparam_utils import create_callbacks, get_optimizer, get_static_hparams
from person_counting.models.model_argparse import parse_args
from person_counting.bin.evaluate import evaluate_model
from person_counting.models import lstm_regression as lsr


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    hparams_samples = get_samples(args)

    for sample in hparams_samples: 
        timestep_num, feature_num = dgv.get_filtered_lengths(args.top_path, sample)
        
        datagen_train, datagen_test = dgv_lstm.create_datagen(args.top_path,
                                                             sample,
                                                             args.label_file, 
                                                             args.augmentation_factor)

        logdir = os.path.join(args.topdir_log + '_ffnn' + strftime("%Y_%b_%d_%H_%M_%S", gmtime()))
        model = create_ffnn(timestep_num, feature_num, sample)
        history, model= lsr.train(model, datagen_train, logdir, sample, datagen_test)
        evaluate_model(model, history, datagen_test,  mode='test', logdir=logdir, visualize=False)

def create_ffnn(timesteps, features, hparams):
    ''' Creates a feed forward neural network with pooling at input 
    '''

    input_layer = Input(shape=((timesteps, features, 1)))
    if hparams['pooling_type'] == 'avg': 
        pooling1 = AveragePooling2D(pool_size=(hparams['pool_size_x'], int(features * hparams['pool_size_y_factor']))) (input_layer)
    else:
        pooling1 = MaxPooling2D(pool_size=(hparams['pool_size_x'], int(features * hparams['pool_size_y_factor']))) (input_layer)
    
    flatten = Flatten()(pooling1)
    dense1 = Dense(hparams['num_units_l1'])(flatten)
    # dense2 = Dense(hparams['num_units_l2'])(dense1)
    output_layer = Dense(1, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(hparams['regularizer']))(dense1)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = get_optimizer(hparams['optimizer'], hparams['learning_rate'])
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.summary()

    return model

def get_samples(args):
    '''Get different samples of hyperparameters

    Arguments: 
        args: Arguments read from command line 

    returns list of samples for hyperparameters out of given hparam space
    '''

    #Put values multiple times into list to increase probability to be chosen
    param_grid = {
                  'pooling_type'        : ['avg'],
                  'pool_size_y_factor'  : [i/30 for i in range(1,15)],
                  'pool_size_x'         : [i for i in range(8,22)],
                  'learning_rate'       : loguniform.rvs(a=1e-3, b=5e-1, size=10000),
                  'optimizer'           : ['Adam'], 
                  'regularizer'         : [i/100 for i in range(0,20)], 
                  'filter_rows_lower'   : [i for i in range(300)], 
                  'filter_cols_upper'   : [i for i in range(8, 20)], 
                  'filter_cols_lower'   : [i for i in range(10, 20)],
                  'batch_size'          : [32, 64, 128, 256], 
                  'num_units_l1'        : [i for i in range(3, 10)],
                  'num_units_l2'        : [i for i in range(1, 5)],
                  'filter_rows_factor'  : [1],
                  'activation'          : ['relu']
                }

    dynamic_params = list(ParameterSampler(param_grid, n_iter=args.n_runs))
    static_params = get_static_hparams(args)

    return [{**dp, **static_params} for dp in dynamic_params]


if __name__ == '__main__': 
    main()