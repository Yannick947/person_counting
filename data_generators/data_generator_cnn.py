import os 
import pandas as pd
import numpy as np
from random import shuffle
import math

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from person_counting.data_generators.data_generators import Generator_CSVS
from person_counting.data_generators.data_generators import *


class Generator_CSVS_CNN(Generator_CSVS):
    '''
    Generators class to load csv files from 
    video folder structre like PCDS Dataset and
    train CNNs

    Arguments (**kwargs)
        length_t            : Length of the feature's DataFrame in time dimension
        length_y            : Length of the feature's DataFrame in y direction
        file_names          : File names to be processed
        filter_cols_upper,  : Amount of columns to be filtered at end and start of DataFrame
        filter_rows_factor  : Factor of rows to be filtered
        batch_size          : Batch size
        top_path            : Parent path where csv files are contained
        label_file          : Name of the label file

    '''

    def __init__(self,*args, **kwargs):
        super(Generator_CSVS_CNN, self).__init__(*args, **kwargs)

    def datagen(self):

        '''
        Datagenerator for bus video csv

        yields: Batch of samples in cnn shape
        '''

        batch_index = 0

        x_batch = np.zeros(shape=(self.batch_size,
                                self.length_t,
                                self.length_y, 
                                1))
        y_batch = np.zeros(shape=(self.batch_size, 1))

        while True:

            for file_name in self.file_names.sample(frac=1): 
                try: 
                    df_x, label = self.__getitem__(file_name)
                    self.file_names_processed.append(file_name)

                except FileNotFoundError as e: 
                    continue

                x_batch[batch_index,:,:,0] = df_x
                y_batch[batch_index] = label
                batch_index += 1

                # Shape for x must be 3D [samples, timesteps, features] and numpy arrays
                if batch_index == self.batch_size:
                    batch_index = 0
                    self.labels.extend(list(y_batch))
                    yield (x_batch, y_batch)

    def get_file_names(self):
        return self.file_names

    def reset_file_names_processed(self): 
        self.file_names_processed = list()
    
    def get_file_names_processed(self):
        return self.file_names_processed

def create_datagen(top_path, 
                   filter_rows_factor, 
                   filter_cols_upper,
                   filter_cols_lower, 
                   batch_size, 
                   label_file): 
    '''
    '''
    length_t, length_y = get_filtered_lengths(top_path,
                                              filter_rows_factor,
                                              filter_cols_upper, 
                                              filter_cols_lower)

    train_file_names, test_file_names = split_files(top_path, label_file)
    print_train_test_lengths(train_file_names, test_file_names, top_path)

    gen_train = Generator_CSVS_CNN(length_t,
                                   length_y,
                                   train_file_names,
                                   filter_cols_upper,
                                   filter_cols_lower,
                                   filter_rows_factor,
                                   batch_size=batch_size,
                                   top_path=top_path,
                                   label_file=label_file)
    gen_train.create_scaler()

    gen_test = Generator_CSVS_CNN(length_t,
                                  length_y,
                                  test_file_names,
                                  filter_cols_upper, 
                                  filter_cols_lower,
                                  filter_rows_factor,
                                  batch_size=batch_size,
                                  top_path=top_path,
                                  label_file=label_file)
    gen_test.create_scaler()

    return gen_train, gen_test


def main(args=None):

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)    
    length_t, length_y = get_filtered_lengths(args.top_path, args.filter_rows_factor,
                                              args.filter_cols_upper, args.filter_cols_lower)
                                              
    train_file_names, _ = split_files(args.top_path, args.label_file)

    gen = Generator_CSVS_CNN(length_t, length_y,
                         train_file_names, args.filter_cols_upper, args.filter_cols_lower,
                         args.filter_rows_factor, batch_size=16)

    for _ in range(5):
        print(next(gen.datagen))

if __name__ == '__main__':
    main()