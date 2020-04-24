import os 
import pandas as pd
import numpy as np
import math
from random import shuffle

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from data_generators.data_generators import *
from process_labels import correct_path_csv, load_labels
from scaler import CSVScaler



class Generator_CSVS_LSTM(Generator_CSVS):

    def __init__(self,*args, **kwargs):
        super(Generator_CSVS_LSTM, self).__init__(*args, **kwargs)


    def datagen(self):

        '''
        Datagenerator for bus video csv

        yields: Batch of samples
        '''

        batch_index = 0

        x_batch = np.zeros(shape=(self.batch_size,
                                self.length_t,
                                self.length_y))
        y_batch = np.zeros(shape=(self.batch_size, 1))

        while True:

            for file_name in self.file_names.sample(frac=1): 
                try: 
                    df_x, label = self.__getitem__(file_name)
                except FileNotFoundError as e: 
                    continue

                x_batch[batch_index,:,:] = df_x
                y_batch[batch_index] = label
                batch_index += 1

                # Shape for x must be 3D [samples, timesteps, features] and numpy arrays
                if batch_index == self.batch_size:
                    batch_index = 0
                    self.labels.extend(list(y_batch))
                    yield (x_batch, y_batch)


def create_datagen(top_path=TOP_PATH): 
    '''
    '''
    length_t, length_y = get_filtered_lengths(top_path)

    train_file_names, test_file_names = split_files()

    filter_cols, filter_rows_factor = get_filters(train_file_names)

    gen_train = Generator_CSVS_LSTM(length_t, length_y,
                                    train_file_names, filter_cols, 
                                    filter_rows_factor, batch_size=16)

    gen_test = Generator_CSVS_LSTM(length_t, length_y,
                                   test_file_names, filter_cols, 
                                   filter_rows_factor, batch_size=16)

    return gen_train, gen_test


def main():

    length_t, length_y = get_filtered_lengths(TOP_PATH)
    train_file_names, _ = split_files()

    filter_cols, filter_rows = get_filters(train_file_names)

    gen = Generator_CSVS(length_t, length_y,
                         train_file_names, filter_cols,
                         filter_rows, batch_size=16)

    for _ in range(5):
        print(next(gen.datagen))


if __name__ == '__main__':
    main()