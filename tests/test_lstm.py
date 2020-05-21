import datetime
import sys
import os
import random
import time

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, BatchNormalization, AveragePooling2D, MaxPooling2D, Input

from person_counting.models import lstm_regression as lstm
from person_counting.data_generators import data_generators as dgv
from person_counting.data_generators import data_generator_cnn as dgv_lstm
from person_counting.bin.evaluate import evaluate_run
from person_counting.tests.test_cnn import test_input_csvs
from person_counting.utils.preprocessing import get_filtered_lengths

label_file = 'pcds_dataset_labels_united.csv'
LABEL_HEADER = ['file_name', 'entering', 'exiting', 'video_type']

def main():
    if sys.argv[1] == 'train_best_cpu': 
        top_path = 'C:/Users/Yannick/Google Drive/person_detection/bus_videos/pcds_dataset_detected/'
        workers = 1
        multi_processing = False
        train_best(workers, multi_processing, top_path)

    elif sys.argv[1] == 'train_best_gpu':
        top_path = '/content/drive/My Drive/person_detection/bus_videos/pcds_dataset_detected/'
        workers = 16
        multi_processing = True
        train_best(workers, multi_processing, top_path)

    elif sys.argv[1] == 'test_input_csvs':
        top_path = 'C:/Users/Yannick/Google Drive/person_detection/bus_videos/pcds_dataset_detected/'
        test_input_csvs(top_path)


def train_best(workers, multi_processing, top_path): 
    '''Train best lstm model with manually put hparams from prior tuning results
    
    Arguments: 
        workers: Number of workers
        multi_processing: Flag if multi-processing is enabled
        top_path: Path to parent directory where csv files are stored
    '''

    sample, timestep_num, feature_num = get_best_hparams(top_path)
    
    datagen_train, datagen_test = dgv_lstm.create_datagen(top_path=top_path, 
                                                         sample=sample,
                                                         label_file=label_file)

    model = lstm.create_lstm(timestep_num, feature_num, sample)
    history, lstm_model= lstm.train(model, datagen_train, logdir=None, hparams=sample, datagen_test=datagen_test)
    for gen, mode in zip([datagen_train, datagen_test], ['train', 'test']):
        evaluate_run(lstm_model, history, gen, mode=mode, logdir='./', visualize=True, top_path=top_path)

    lstm_model.save('test_best{}.h5'.format(min(history.history['val_loss'])))
    return lstm_model, history


def get_best_hparams(top_path):
    '''Set best hyperparameter set from prior tuning session
    Arguments: 
        top_path: Parent directory where shall be searched for csv files

    '''
    hparams = {
                'activation'             : 'relu',
                'batch_size'             : 32,
                'regularizer'            : 0.001,
                'filter_cols_upper'      : 15,
                'num_units'              : 99,
                'filter_cols_factor'     : 1,
                'pooling_type'           : 'avg',
                'filter_rows_factor'     : 1,
                'learning_rate'          : 0.018473,
                'y_stride'               : 1,
                'optimizer'              : 'Adam',
                'pool_size_x'            : 16, 
                'filter_cols_lower'      : 13,
                'augmentation_factor'    : 0,
                'filter_rows_lower'      : 150, 
                'pool_size_y_factor'     : 0.16, 
              }
              
    timestep_num, feature_num = get_filtered_lengths(top_path=top_path, sample=hparams)

    return hparams, timestep_num, feature_num

if __name__ == '__main__':
    main()