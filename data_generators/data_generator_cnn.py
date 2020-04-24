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


def create_datagen(args): 
    '''
    '''
    length_t, length_y = get_filtered_lengths(args)

    train_file_names, test_file_names = split_files(args)

    gen_train = Generator_CSVS_CNN(length_t,
                                   length_y,
                                   train_file_names,
                                   args.filter_cols, 
                                   args.filter_rows_factor,
                                   batch_size=args.batch_size,
                                   top_path=args.top_path,
                                   label_file=args.label_file)

    gen_test = Generator_CSVS_CNN(length_t,
                                  length_y,
                                  test_file_names,
                                  args.filter_cols, 
                                  args.filter_rows_factor,
                                  batch_size=args.batch_size,
                                  top_path=args.top_path,
                                  label_file=args.label_file)

    return gen_train, gen_test

def main(args=None):

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)    
    length_t, length_y = get_filtered_lengths(args.top_path)
    train_file_names, _ = split_files(args)

    gen = Generator_CSVS_CNN(length_t, length_y,
                         train_file_names, args.filter_cols,
                         args.filter_rows_factor, batch_size=16)

    for _ in range(5):
        print(next(gen.datagen))

if __name__ == '__main__':
    main()