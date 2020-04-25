import sys
import pandas as pd
import os
from time import gmtime, strftime

import keras
import tensorflow as tf
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, BatchNormalization, AveragePooling2D
from sklearn.model_selection import ParameterSampler
from scipy.stats import loguniform

from person_counting.data_generators import data_generator_cnn as dgv_cnn
from person_counting.data_generators import data_generators as dgv
from person_counting.utils.visualization_utils import plot, visualize_predictions
from person_counting.utils.hyperparam_utils import create_callbacks, get_optimizer, get_static_hparams
from person_counting.models.model_argparse import parse_args
#Static params for local use, in colab set via arguments

def main(args=None):
    #comment out if not needed
    # test_best()

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    timestep_num, feature_num = dgv.get_filtered_lengths(args.top_path,
                                                         args.filter_rows_factor,
                                                         args.filter_cols)

    datagen_train, datagen_test = dgv_cnn.create_datagen(args.top_path, 
                                                         args.filter_rows_factor, 
                                                         args.filter_cols, 
                                                         args.batch_size, 
                                                         args.label_file)


    hparams_samples = get_samples(args)
    for sample in hparams_samples: 
        logdir = os.path.join(args.topdir_log + strftime("%Y_%b_%d_%H_%M_%S", gmtime()))
        model = create_cnn(timestep_num, feature_num, sample)
        train(model, datagen_train, logdir, sample, datagen_test)


def get_samples(args):
    '''
    '''

    print(loguniform.rvs(a=5e-6, b=5e-2, size=10))
    #Put values multiple times into list to increase probability to be chosen
    param_grid = {
                #   'dropout'             : [i / 100 for i in range(50)],
                  'kernel_size'         : [i for i in range(3, 10)],
                  'kernel_number'       : [i for i in range(3,10)],
                  'pool_size'           : [i for i in range(2, 5)],
                  'learning_rate'       : loguniform.rvs(a=1e-4, b=5e-2, size=100000),
                  'optimizer'           : ['Adam','RMSProp'], 
                  'layer_number'        : [1, 3], 
                  'batch_normalization' : [False],
                  'y_stride'            : [1, 1, 1, 2], 
                  'regularization'      : [i/100 for i in range(0,40)], 
                  'pooling_type'        : ['avg', 'max']
                 }

    dynamic_params = list(ParameterSampler(param_grid, n_iter=args.n_runs))
    static_params = get_static_hparams(args)

    return [{**dp, **static_params} for dp in dynamic_params]


def test_best(): 
    hparams = {
                'learning_rate'          : 0.0021052,
                'filter_rows_factor'     : 2,
                'filter_cols_factor'     : 1,
                'layer_number'           : 2,
                'pool_size'              : 4,
                'kernel_size'            : 4,
                'regularizer'            : 0.1,
                'kernel_number'          : 9,
                'y_stride'               : 1,
                'filter_cols'            : 5,
                'optimizer'              : 'Adam',
                'batch_normalization'    : False, 
                'batch_size'             : 8
              }
    top_path = 'C:/Users/Yannick/Google Drive/person_detection/bus_videos/pcds_dataset_detected/'
    label_file = 'pcds_dataset_labels_united.csv'
    timestep_num, feature_num = dgv.get_filtered_lengths(top_path=top_path,
                                                         filter_rows_factor=hparams['filter_rows_factor'],
                                                         filter_cols=hparams['filter_cols'],
                                                         )

    datagen_train, datagen_test = dgv_cnn.create_datagen(top_path=top_path, 
                                                         filter_rows_factor=hparams['filter_rows_factor'], 
                                                         filter_cols=hparams['filter_cols'], 
                                                         batch_size=hparams['batch_size'], 
                                                         label_file=label_file)


    cnn_model = create_cnn(timestep_num, feature_num, hparams)
    history, cnn_model = train(cnn_model,
                               datagen_train, './',
                               hparams,
                               datagen_test,
                               workers=1,
                               use_multiprocessing=False, 
                               epochs=12)

    plot(history)
    visualize_predictions(cnn_model, datagen_test)


def train(model,
          datagen_train,
          logdir=None,
          hparams=None,
          datagen_test=None,
          workers=16,
          use_multiprocessing=True, 
          epochs=20):
    '''
    '''
    print('Actual model is using following hyper-parameters:')
    for key in hparams.keys(): 
        print(key, ': ', hparams[key])

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
                                  max_queue_size=16, 
                                  )

    return history, model


def create_cnn(timesteps, features, hparams):
    '''
    '''

    stride = tuple([hparams['y_stride'], ['y_stride']])
    print(hparams['kernel_number'], hparams['kernel_size'], stride)

    conv_layer = Conv2D(hparams['kernel_number'],
                        hparams['kernel_size'],
                        input_shape=(timesteps, features, 1),
                        use_bias=True,
                        activation='relu', 
                        kernel_initializer=keras.initializers.glorot_normal(1)
                       )   

    if hparams['pooling_type'] == 'avg': 
        pooling_layer =  AveragePooling2D(pool_size=hparams['pool_size']) 
    else: 
        pooling_layer = MaxPooling2D(pool_size=hparams['pool_size'])

    model = keras.Sequential()
    model.add(conv_layer)    
    model.add(pooling_layer)
 
    for _ in range(hparams['layer_number'] - 1):
        try:
            if hparams['batch_normalization']: 
                model.add(BatchNormalization())
            model.add(Conv2D(hparams['kernel_number'],
                            hparams['kernel_size'],
                            use_bias=True,
                            activation='relu', 
                            kernel_initializer=keras.initializers.glorot_normal(1)
                       )
            )
            model.add(pooling_layer)

        except ValueError: 
            print('Tried to create a MaxPool Layer that is not possible to create,',
                  'because it would lead to negative dimensions. Creation was skipped')

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = get_optimizer(hparams['optimizer'], hparams['learning_rate'])
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.summary()
    return model

if __name__ == '__main__':
    main() 

