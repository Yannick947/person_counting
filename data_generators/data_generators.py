import sys
import os 
import pandas as pd
import numpy as np
import math
import abc
from random import shuffle

from tensorflow import keras
from sklearn.model_selection import train_test_split

from person_counting.models.model_argparse import parse_args
from person_counting.utils import preprocessing as pp
LABEL_HEADER = ['file_name', 'entering', 'exiting', 'video_type']

np.random.seed(7)
TEST_SIZE = 0.25

class Generator_CSVS(keras.utils.Sequence):
    '''Abstract class for Generators to load csv files from 
    video folder structre like PCDS Dataset
    '''
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 length_t,
                 length_y,
                 file_names,
                 sample,
                 top_path,
                 label_file, 
                 scaler,
                 augmentation_factor=0): 

        ''' Initialize Generator object.

            Arguments
                length_t            : Length of the feature's DataFrame in time dimension
                length_y            : Length of the feature's DataFrame in y direction
                file_names          : File names to be processed
                filter_cols_upper   : Amount of columns to be filtered at end and start of DataFrame
                sample              : Sample of hyperparameter
                top_path            : Parent path where csv files are contained
                label_file          : Name of the label file
                augmentation_factor : Factor how much augmentation shall be done, 1 means moving every
                                      pixel for 1 position 
        '''

        self.top_path               = top_path
        self.label_file             = label_file
        self.length_t               = length_t
        self.length_y               = length_y
        self.file_names             = file_names 
        self.filter_cols_upper      = sample['filter_cols_upper']
        self.filter_cols_lower      = sample['filter_cols_lower']
        self.filter_rows_factor     = sample['filter_rows_factor']
        self.batch_size             = sample['batch_size']
        self.labels                 = list()
        self.file_names_processed   = list()
        self.scaler                 = scaler
        self.df_y                   = pd.read_csv(self.top_path + self.label_file, header=None, names=LABEL_HEADER)
        self.augmentation_factor    = augmentation_factor
        self.sample                 = sample 
        self.unfiltered_length_t, self.unfiltered_length_y = pp.get_lengths(self.top_path)
        self.preprocessor           = pp.Preprocessor(length_t, length_y, top_path, sample, scaler, augmentation_factor)

    @abc.abstractmethod
    def datagen(self):
        '''Returns datagenerator
        '''
        return
    
    def __len__(self):
        '''Returns the amount of batches for the generator
        '''
        eval_batches = int(np.floor(len(self.file_names) / float(self.batch_size)))
        print('Generator contains ', len(self.file_names), ' files with ', eval_batches, ' batches of size ', self.batch_size)
        return eval_batches
    
    def __getitem__(self, file_name):
        '''Gets pair of features and labels for given filename
        Arguments: 
            file_name: The name of the file which shall be parsed
        '''
        df_x = self.__get_features(file_name)

        if df_x is not None: 
            if df_x.shape[0] != self.unfiltered_length_t or\
               df_x.shape[1] != self.unfiltered_length_y:

                raise ValueError ('File with wrong dimensions found')
        else:
                raise FileNotFoundError('Failed getting features for file {}'.format(file_name))

        df_x = self.preprocessor.preprocess_features(df_x)

        label = get_entering(file_name, self.df_y)
        label = self.preprocessor.preprocess_labels(label)

        self.file_names_processed.append(file_name)
        self.labels.append(label)
        return df_x, label

    def __get_features(self, file_name): 
        '''Get sample of features for given filename. 

        Arguments: 
            file_name: Name of given training sample

            returns: Features for given file_name
        '''

        full_path = file_name
        try:
            df_x = pd.read_csv(full_path, header=None)
            return df_x

        except Exception as e:
            return None

    def get_labels(self):
        '''Returns the labels which were yielded since calling reset_labels() 
        '''
        return np.array(self.labels)

    def reset_label_states(self): 
        '''Resets the labels which were processed
        '''
        self.labels = list()

    def get_file_names(self):
        return self.file_names

    def reset_file_names_processed(self): 
        self.file_names_processed = list()
    
    def get_file_names_processed(self):
        return self.file_names_processed

        
def get_feature_file_names(top_path): 
    '''
    Get names of all csv files for training

    Arguments: 
        top_path: Parent directory where to search for csv files
    '''
    csv_names = list()
    for root, _, files in os.walk(top_path):
        for file_name in files: 
            if (file_name[-4:] == '.csv') and not ('label' in file_name): 
                csv_names.append(os.path.join(root, file_name))
    return csv_names


def split_files(top_path, label_file):
    ''' Splits all files in the training set into train and test files
    and returns lists of names for train and test files
    #TODO: Split files according to the categories equally distributed
    '''

    df_names = pd.Series(get_feature_file_names(top_path))

    #replace .avi with .csv
    df_names = df_names.apply(lambda row: row[:-4] + '.csv')
    return train_test_split(df_names, test_size=TEST_SIZE, random_state=10)
            

def get_filters(file_names):
    '''Searches for the right columns amount of columns and rows to drop
    '''
    #TODO: Implement. Now dummy function
    raise NotImplementedError


def get_entering(file_name, df_y): 
    ''' Get number of entering persons to existing training sample. 

    Arguments: 
        file_name: Name of given training sample
        df_y: Dataframe with all labels for all samples

        returns: Label for given features
    '''

    try: 
        search_str = file_name.replace('\\', '/').replace('\\', '/')
        
        if 'front' in file_name:
            search_str = search_str[search_str.find('front'):]
        else: 
            search_str = search_str[search_str.find('back'):]

        entering = df_y.loc[df_y.file_name == search_str].entering
        return entering 

    except Exception as e:
        # print('No matching label found for existing csv file')
        return None


def get_exiting(file_name, df_y): 
    '''Get number of exiting persons to existing training sample. 

    Arguments: 
        file_name: Name of given training sample
        df_y: Dataframe with all labels for all samples

        returns: Exiting persons for given file
    '''
    try: 
        exiting = df_y.loc[df_y.file_name == file_name].exiting
        return exiting 

    except Exception as e:
        # print(e, ', no matching label found for existing csv file')
        return None
    


