import pandas as pd
import numpy as np
import os
from time import gmtime, strftime

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense

from data_generators import data_generator_cnn as dgv_cnn
from data_generators import data_generators as dgv
from utils.visualization_utils import plot, visualize_predictions
from utils.hyperparam_utils import create_hyperparams_domains, create_callbacks

LOGDIR_TOP = os.path.join('tensorboard\\')


def main():

    timestep_num, feature_num = dgv.get_filtered_lengths()
    datagen_train, datagen_test = dgv.create_datagen()
    
    hp_domains, _ = create_hyperparams_domains()

    #TODO: implement random search
    for num_units in hp_domains['num_units'].domain.values:
        for regularizer in (hp_domains['regularizer'].domain.min_value, hp_domains['regularizer'].domain.max_value):
            for activation in hp_domains['activation'].domain.values:
                logdir = os.path.join(LOGDIR_TOP + strftime("%Y_%b_%d_%H_%M_%S", gmtime()))
                
                hparams = {
                                'num_units'             : num_units,
                                'regularizer'           : regularizer,
                                'activation'            : activation,
                            }
            
                lstm_model = create_lstm(timestep_num, feature_num, hparams)
                history, model = train(lstm_model, datagen_train, logdir, hparams, datagen_test)
        
                # plot(history)
                # visualize_predictions(model, datagen_test)


def train(model, datagen_train, logdir, hparams=None, datagen_test=None):
    '''
    '''
    history = model.fit_generator(validation_steps=5,
                                generator=datagen_train.datagen(),
                                validation_data=datagen_test.datagen(),
                                steps_per_epoch=30,
                                epochs=10,
                                verbose=1,
                                shuffle=True,
                                callbacks=create_callbacks(logdir, hparams)
                                )

    return history, model


def create_lstm(timesteps, features, hparams):
    '''
    '''
    
    model = tf.keras.Sequential()
    # input_shape is of dimension (timesteps, features)
    model.add(
        LSTM(
            kernel_initializer          = tf.keras.initializers.Zeros(),
            recurrent_initializer       = tf.keras.initializers.Zeros(),
            bias_initializer            = tf.keras.initializers.Zeros(),
            units                       = hparams['num_units'],
            activation                  = hparams['activation'],
            recurrent_activation        = 'sigmoid',
            kernel_regularizer          = tf.keras.regularizers.l2(hparams['regularizer']),
            input_shape                 = (timesteps, features),
            return_sequences            = False))

    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    return model


if __name__ == '__main__':
    main() 
