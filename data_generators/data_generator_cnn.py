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
from person_counting.utils.scaler import CSVScaler

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
            for file_name in self.file_names:
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
                    yield (x_batch, y_batch)


def create_datagen(top_path, 
                   sample, 
                   label_file, 
                   augmentation_factor=0, 
                   filter_hour_above=0, 
                   filter_category_noisy=False): 
    '''
    Creates train and test data generators for lstm network. 

    Arguments: 
        top_path: Parent directory where shall be searched for training files
        sample: sample of hyperparameters used in this run
        label_file: Name of the label file containing all the labels
        augmentation_factor: Factor how much augmentation shall be done, 1 means
                             moving every pixel for one position
        filter_hour_above: Hour after which videos shall be filtered
        filter_category_noisy: Flag if noisy videos shall be filtered
    '''
    #Load filenames and lengths
    length_t, length_y = get_filtered_lengths(top_path, sample)
    train_file_names, test_file_names = split_files(top_path, label_file)

    #Apply filters
    train_file_names = apply_file_filters(train_file_names, filter_hour_above, filter_category_noisy)
    test_file_names = apply_file_filters(test_file_names, filter_hour_above, filter_category_noisy)

    print('Dataset contains: \n{} training csvs \n{} testing csvs'.format(len(train_file_names), len(test_file_names)))
    
    #TODO: Should be mix of train and test file names
    scaler = CSVScaler(top_path, label_file, train_file_names, sample, sample_size=100)

    gen_train = Generator_CSVS_CNN(length_t,
                                   length_y,
                                   train_file_names,
                                   scaler=scaler, 
                                   sample=sample,
                                   top_path=top_path,
                                   label_file=label_file, 
                                   augmentation_factor=augmentation_factor)
    
    #Don't do augmentation here!
    gen_test = Generator_CSVS_CNN(length_t,
                                  length_y,
                                  test_file_names,
                                  scaler=scaler, 
                                  sample=sample,
                                  top_path=top_path,
                                  label_file=label_file, 
                                  augmentation_factor=0)

    return gen_train, gen_test

