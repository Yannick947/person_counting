import os 
import pandas as pd
import numpy as np
import math
from random import shuffle

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

                #Error messages for debugging purposes
                except FileNotFoundError as e: 
                    continue

                except ValueError as e: 
                    continue

                x_batch[batch_index,:,:,0] = df_x
                y_batch[batch_index] = label
                batch_index += 1

                # Shape for x must be 4D [samples, timesteps, features, channels] and numpy array
                if batch_index == self.batch_size:
                    batch_index = 0
                    self.labels.extend(list(y_batch))
                    yield (x_batch, y_batch)


def create_datagen(top_path, 
                   sample, 
                   label_file, 
                   augmentation_factor=0): 
    '''
    Creates train and test data generators for cnn network. 

    Arguments: 
        top_path: Parent directory where shall be searched for training files
        sample: sample of hyperparameters used in this run
        label_file: Name of the label file containing all the labels
        augmentation_factor: Factor how much augmentation shall be done, 1 means
                             moving every pixel for one position
    '''

    length_t, length_y = get_filtered_lengths(top_path, sample)

    train_file_names, test_file_names = split_files(top_path, label_file)
    print_train_test_lengths(train_file_names, test_file_names, top_path, label_file)

    gen_train = Generator_CSVS_CNN(length_t,
                                   length_y,
                                   train_file_names,
                                   sample=sample,
                                   top_path=top_path,
                                   label_file=label_file, 
                                   augmentation_factor=augmentation_factor)
    gen_train.create_scaler()

    #Don't do augmentation here!
    gen_test = Generator_CSVS_CNN(length_t,
                                  length_y,
                                  test_file_names,
                                  sample=sample,
                                  top_path=top_path,
                                  label_file=label_file, 
                                  augmentation_factor=0)
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