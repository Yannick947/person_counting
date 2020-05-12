import os 
import pandas as pd
import numpy as np
import math
from random import shuffle

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from person_counting.data_generators.data_generators import *
from person_counting.utils.scaler import CSVScaler

class Generator_CSVS_LSTM(Generator_CSVS):
    '''
    Generators class to load csv files from video folder structre
    like PCDS Dataset and train LSTMs
    '''

    def __init__(self,*args, **kwargs):
        """ Initialize Generator object.
        """
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

            for file_name in self.file_names: 
                try: 
                    df_x, label = self.__getitem__(file_name)

                except FileNotFoundError as e: 
                    continue
                except ValueError as e: 
                    continue

                x_batch[batch_index,:,:] = df_x
                y_batch[batch_index] = label
                batch_index += 1

                # Shape for x must be 3D [samples, timesteps, features] and numpy arrays
                if batch_index == self.batch_size:
                    batch_index = 0
                    self.labels.extend(list(y_batch))
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

    print_train_test_lengths(train_file_names, test_file_names, top_path, label_file)

    gen_train = Generator_CSVS_LSTM(length_t,
                                   length_y,
                                   train_file_names,
                                   sample=sample,
                                   top_path=top_path,
                                   label_file=label_file, 
                                   augmentation_factor=augmentation_factor)
    gen_train.create_scaler()

    gen_test = Generator_CSVS_LSTM(length_t,
                                  length_y,
                                  test_file_names,
                                  sample=sample,
                                  top_path=top_path,
                                  label_file=label_file, 
                                  augmentation_factor=0)
    gen_test.create_scaler()

    return gen_train, gen_test
