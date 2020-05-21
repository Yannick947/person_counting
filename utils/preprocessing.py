import os 

import pandas as pd

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
                 scaler, 
                 augmentation_factor):
        self.length_t = length_t
        self.length_y = length_y
        self.top_path = top_path
        self.sample = sample
        self.augmentation_factor = augmentation_factor
        self.scaler = scaler

    def preprocess_features(self, df_x): 
        '''Preprocess features
        '''

        #Only remove when index is saved in csv
        if check_for_index_col(self.top_path):
            df_x.drop(df_x.columns[0], axis=1, inplace=True)

        df_x = clean_ends(df_x, del_leading=self.sample['filter_cols_upper'],
                                del_trailing=self.sample['filter_cols_lower'],
                                del_rows=self.sample['filter_rows_lower'])

        df_x = filter_rows(df_x, self.sample['filter_rows_factor'])

        if self.augmentation_factor > 0: 
            df_x = ta.augment_trajectory(df_x, self.augmentation_factor)

        if self.scaler is not None:
            df_x = self.scaler.transform_features(df_x)

        assert df_x is not None, 'Scaling or augmentation went wrong, check implementation'
        
        assert df_x.shape[0] == (self.length_t)\
           and df_x.shape[1] == (self.length_y),\
           'Shapes are not consistent for feature data frame'
        return df_x

    def preprocess_labels(self, label): 
        '''Preprocess labels 
        '''
        if self.scaler.scaler_labels is not None:
            label = self.scaler.transform_labels(label)
        else: 
            label = label.values

        assert label is not None, 'Scaling for label file went wrong'

        return label

    
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
