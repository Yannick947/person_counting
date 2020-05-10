import sys
import os 
import pandas as pd
import numpy as np
import math
import abc
from random import shuffle

from tensorflow import keras
from sklearn.model_selection import train_test_split

from person_counting.utils import scaler
from person_counting.models.model_argparse import parse_args
from person_counting.data_generators import trajectory_augmentation as ta

LABEL_HEADER = ['file_name', 'entering', 'exiting', 'video_type']

np.random.seed(7)
TEST_SIZE = 0.3

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
        self.unfiltered_length_t, self.unfiltered_length_y = get_lengths(self.top_path)

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

        #Only remove when index is saved in csv
        if check_for_index_col(self.top_path):
            df_x.drop(df_x.columns[0], axis=1, inplace=True)

        df_x = clean_ends(df_x, del_leading=self.filter_cols_upper,
                                del_trailing=self.filter_cols_lower,
                                del_rows=self.sample['filter_rows_lower'])
        df_x = filter_rows(df_x, self.filter_rows_factor)

        if self.augmentation_factor > 0: 
            df_x = ta.augment_trajectory(df_x, self.augmentation_factor)

        if self.scaler is not None:
            df_x = self.scaler.transform_features(df_x)

        assert df_x is not None, 'Scaling or augmentation went wrong, check implementation'

        label = get_entering(file_name, self.df_y)

        if self.scaler is not None:
            label = self.scaler.transform_labels(label)
        else: 
            label = label.values

        assert label is not None, 'Scaling for label file went wrong'

        assert df_x.shape[0] == (self.length_t)\
           and df_x.shape[1] == (self.length_y),\
           'Shapes or not consistent for feature dframe'

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


def check_for_index_col(top_path): 
    '''Returns true if index column in sample csv file exists

    Arguments: 
        top_path: Path where the csv files are contained in 
    '''

    for root, _, files in os.walk(top_path): 
        for file_name in files:
            if file_name[-4:] == '.csv' and not ('label' in file_name):
                full_path = os.path.join(root, file_name)
                df = pd.read_csv(full_path, header=None)
                for i in range(df.shape[0]):
                    if df.iloc[i, 0] != i:
                        return False
                return True


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
    


def clean_ends(df, del_leading=5, del_trailing=5, del_rows=100):
    ''' Delete leading and trailing columns due to sparsity. 

    Arguments: 
        df: Dataframe to adjust
        del_leading: Number of leading columns to delete
        del_trailing: Number of trailing columns to delete
        
    returns: Dataframe with cleaned columns
    '''
    drop_indices = [i for i in range(del_leading)]
    df.drop(df.columns[drop_indices], axis=1, inplace=True)        

    col_length = df.shape[1]
    drop_indices = [col_length - i - 1 for i in range(del_trailing)]
    df.drop(df.columns[drop_indices], axis=1, inplace=True)
    
    row_length = df.shape[0]
    drop_indices = [row_length - i - 1 for i in range(del_rows)]
    df.drop(drop_indices, axis=0, inplace=True)
    return df


def filter_rows(df, filter_rows_factor): 
    '''Filters rows according to the filter_rows_factor in a given DataFrame
    
    Arguments: 
        df: Dataframe which shall be filtered 
        filter_rows_factor: The factor which shall be used for filtering rows, 
                            a factor of 4 will remove 3/4 rows and keeps every
                            forth row, starting with the first row
        
        returns: Filtered Dataframe
    '''

    return df.iloc[::filter_rows_factor, :]


def get_lengths(top_path):
    '''returns: Number of timesteps, number of features (columns) which csv files have
    '''

    for root, _, files in os.walk(top_path): 
        for file_name in files:
            if file_name[-4:] == '.csv' and not ('label' in file_name):
                full_path = os.path.join(root, file_name)
                df = pd.read_csv(full_path, header=None)
                if check_for_index_col(top_path):
                    print('Warning: Index column existing, make sure to drop it!')
                    return df.shape[0], df.shape[1] - 1
                else: 
                    return df.shape[0], df.shape[1]


def get_filtered_lengths(top_path, sample):
    '''Returns the length of the feature dataframes after filtering those with 
    the above given methods

    Arguments: 
        top_path: Path to parent directory of csv files
        sample: Sample of hyperparameters for this run
    '''

    timestep_num, feature_num = get_lengths(top_path)
    #TODO: Verify that the rounding is correct, maybe math.ceil() rounding in some cases has to be used

    if timestep_num % sample['filter_rows_factor'] != 0:
        filtered_length_t = int(timestep_num / sample['filter_rows_factor']) + 1

    else: 
        filtered_length_t = int(timestep_num / sample['filter_rows_factor'] - sample['filter_rows_lower'])

    filtered_length_y = feature_num - sample['filter_cols_upper'] - sample['filter_cols_lower']
    
    return filtered_length_t, filtered_length_y

def get_video_daytime(file_name): 
    ''' Extracts daytime when video was recorded from filename

    Arguments: 
        file_name: The name of the file
    returns hour, minutes of the video when it was recorded
    '''
    if file_name.find('FrontColor') != -1:
        time_stop_pos = file_name.find('FrontColor')
    elif file_name.find('BackColor') != -1:
        time_stop_pos = file_name.find('BackColor')
    
    else:
        raise FileNotFoundError('Not a valid file')
     
    hour = int(file_name[time_stop_pos - 8: time_stop_pos - 6])
    minutes = int(file_name[time_stop_pos - 5: time_stop_pos - 3])  
    
    return hour, minutes

def apply_file_filters(df, filter_hour_above=0, filter_category_noisy=False): 
    '''Apply filters to the df
    Arguments: 
        df: Dataframe with the file names
        filter_category_noisy: Flag if noisy videos shall be filtered
        filter_hour_above: Hour after which the videos shall be filter
    returns a dataframe with filtered filenames
    '''

    if filter_category_noisy: 
        df = df[~df.str.contains("noisy")]
    if filter_hour_above > 0: 
        df = df[df.apply(__filter_by_hour, args=(filter_hour_above,))]
    
    return df

def __filter_by_hour(row, filter_hour_above): 
    '''Function to filter a row of a df with hour after specified value
    Arguments: 
        row: Row of the dataframe
        filter_hour_above: Hour after which shall be filtered
    returns True if row shall be kept, False otherwise
    '''

    file_name = row
    hour, _ = get_video_daytime(file_name)
    if hour > filter_hour_above: 
        return False
    else: 
        return True
