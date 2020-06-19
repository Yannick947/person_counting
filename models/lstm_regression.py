import sys
import os
from time import gmtime, strftime

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from keras import backend as K
from keras.models import Model
from keras.legacy import interfaces
from keras.layers import Dense, BatchNormalization, AveragePooling2D, LSTM, MaxPooling2D, Flatten, TimeDistributed, Input
from sklearn.model_selection import ParameterSampler
from scipy.stats import loguniform

from person_counting.data_generators import data_generator_cnn as dgv_lstm
from person_counting.data_generators import data_generators as dgv
from person_counting.utils.visualization_utils import plot_losses, visualize_predictions
from person_counting.utils.hyperparam_utils import create_callbacks, get_optimizer, get_static_hparams
from person_counting.models.model_argparse import parse_args
from person_counting.bin.evaluate import evaluate_run
from person_counting.utils.preprocessing import get_filtered_lengths

def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    hparams_samples = get_samples(args)

    for sample in hparams_samples: 
        timestep_num, feature_num = get_filtered_lengths(args.top_path, sample)
        
        datagen_train, datagen_test = dgv_lstm.create_datagen(args.top_path,
                                                             sample,
                                                             args.label_file, 
                                                             args.augmentation_factor,
                                                             args.filter_hour_above, 
                                                             args.filter_category_noisy)

        logdir = os.path.join(args.topdir_log + '_lstm' + strftime("%Y_%b_%d_%H_%M_%S", gmtime()))
        model = create_lstm(timestep_num, feature_num, sample)
        history, model= train(model, datagen_train, logdir, sample, datagen_test)
        evaluate_run(model, history, datagen_test,  mode='test', logdir=logdir, visualize=True, top_path=args.top_path)


def train(model,
          datagen_train,
          logdir=None,
          hparams=None,
          datagen_test=None,
          workers=16,
          use_multiprocessing=True, 
          epochs=60):
    '''Trains a given lstm model

    Arguments: 
        model: Lstm keras model
        datagen_train: Generator for training data 
        logdir: Directory where shall be logged to
        hparams: Sample of hyperparameters for this run
        datagen_test: Data generator for test data
        worker: Number of workers used for training
        use_multiprocessing: Flag if multiprocessing shall be used, only use in gpu mode
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
                                  callbacks=create_callbacks(logdir, hparams, save_best=True), 
                                  use_multiprocessing=use_multiprocessing, 
                                  workers=workers, 
                                  max_queue_size=64
                                  )

    return history, model

def get_samples(args):
    '''Get different samples of hyperparameters

    Arguments: 
        args: Arguments read from command line 

    returns list of samples for hyperparameters out of given hparam space
    '''

    #Put values multiple times into list to increase probability to be chosen
    param_grid = {
                  'pooling_type'        : ['avg'],
                  'pool_size_y_factor'  : [i/100 for i in range(1,20)],
                  'pool_size_x'         : [i for i in range(15,25)],
                  'learning_rate'       : loguniform.rvs(a=1e-4, b=5e-1, size=100000),
                  'optimizer'           : ['Adam'], 
                  'batch_normalization' : [False],
                  'regularizer'         : [i/100 for i in range(0,35)], 
                  'filter_rows_lower'   : [150, 200, 250, 300], 
                  'filter_cols_upper'   : [i for i in range(12, 17)], 
                  'filter_cols_lower'   : [i for i in range(12, 17)],
                  'batch_size'          : [32, 64, 128, 256], 
                  'num_units'           : [i for i in range(30, 200)], 
                  'activation'          : ['sigmoid', 'relu']
                }

    dynamic_params = list(ParameterSampler(param_grid, n_iter=args.n_runs))
    static_params = get_static_hparams(args)

    return [{**dp, **static_params} for dp in dynamic_params]

def create_lstm(timesteps, features, hparams):
    '''
    '''

    input_layer = Input(shape=((timesteps, features, 1)))
    if hparams['pooling_type'] == 'avg': 
        try: 
            pooling1 = AveragePooling2D(pool_size=(hparams['pool_size_x'], int(features * hparams['pool_size_y_factor']))) (input_layer)
        except:
            pooling1 = MaxPooling2D(pool_size=(hparams['pool_size_x'], 2)) (input_layer)
    
    lstm1 = TimeDistributed(LSTM(
            units                       = hparams['num_units'],
            activation                  = hparams['activation'],
            recurrent_activation        = 'sigmoid',
            return_sequences            = False, 
            )) (pooling1)

    flatten = Flatten()(lstm1)
    output_layer = Dense(1, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(hparams['regularizer']))(flatten)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = get_optimizer(hparams['optimizer'], hparams['learning_rate'])
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.summary()

    return model

if __name__ == '__main__':
    main() 
