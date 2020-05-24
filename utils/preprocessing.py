import os 

import pandas as pd
import numpy as np

from person_counting.data_generators import trajectory_augmentation as ta
from person_counting.utils import scaler

class Preprocessor(object):
    '''Preprocessor for video DataFrames
    '''

    def __init__(self, 
                 length_t,
                 length_y, 
                 top_path,
                 sample, 
                 feature_scaler,
                 label_scaler, 
                 augmentation_factor):
        self.length_t = length_t
        self.length_y = length_y
        self.top_path = top_path
        self.sample = sample
        self.augmentation_factor = augmentation_factor
        self.feature_scaler = feature_scaler
        self.label_scaler = label_scaler

    def preprocess_features(self, arr): 
        '''Preprocess features
        Arguments: 
            arr: Numpy array with features

        returns preprocessed numpy array
        '''

        arr = clean_ends(arr, del_leading=self.sample['filter_cols_upper'],
                              del_trailing=self.sample['filter_cols_lower'],
                              del_rows=self.sample['filter_rows_lower'])

        if self.augmentation_factor > 0: 
            arr = ta.augment_trajectory(arr, self.augmentation_factor)

        if self.feature_scaler is not None:
            arr = self.feature_scaler.transform(arr)

        assert arr is not None, 'Scaling or augmentation went wrong, check implementation'
        
        assert arr.shape[0] == (self.length_t)\
           and arr.shape[1] == (self.length_y)\
           and arr.shape[2] == 2,\
           'Shapes are not consistent for feature data frame'
        return arr


    def preprocess_labels(self, label): 
        '''Preprocess labels 
        '''
        if self.label_scaler is not None:
            label = self.label_scaler.transform_labels(label)
        else: 
            label = label.values

        assert label is not None, 'Scaling for label file went wrong'

        return label


def clean_ends(arr, del_leading=5, del_trailing=5, del_rows=0):
    ''' Delete leading and trailing columns due to sparsity. 

    Arguments: 
        arr: Dataframe to adjust
        del_leading: Number of leading columns to delete
        del_trailing: Number of trailing columns to delete
        
    returns: Dataframe with cleaned columns
    '''
    drop_indices = [i for i in range(del_leading)]
    arr = np.delete(arr, drop_indices, axis=1)

    col_length = arr.shape[1]
    drop_indices = [col_length - i - 1 for i in range(del_trailing)]
    arr = np.delete(arr, drop_indices, axis=1)

    row_length = arr.shape[0]
    drop_indices = [row_length - i - 1 for i in range(del_rows)]
    arr = np.delete(arr, drop_indices, axis=0)
    
    return arr


def get_lengths(top_path):
    '''returns: Number of timesteps, number of features (columns) which npy files have
    '''
    for root, _, files in os.walk(top_path): 
        for file_name in files:
            if file_name[-4:] == '.npy' and not ('label' in file_name):
                full_path = os.path.join(root, file_name)
                arr = np.load(full_path)
                return arr.shape[0], arr.shape[1]


def get_filtered_lengths(top_path, sample):
    '''Returns the length of the feature dataframes after filtering those with 
    the above given methods

    Arguments: 
        top_path: Path to parent directory of npy files
        sample: Sample of hyperparameters for this run
    '''

    timestep_num, feature_num = get_lengths(top_path)
    filtered_length_t = int(timestep_num  - sample['filter_rows_lower'])
    filtered_length_y = feature_num - sample['filter_cols_upper'] - sample['filter_cols_lower']
    
    return filtered_length_t, filtered_length_y


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
