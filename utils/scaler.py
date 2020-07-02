import os 
import pandas as pd
import numpy as np
from random import shuffle

import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from person_counting.data_generators import data_generators as dgv
from person_counting.utils import preprocessing as pp

LABEL_HEADER = ['file_name', 'entering', 'exiting', 'video_type']

class FeatureScaler(MinMaxScaler):
    '''Scaler to scale feature frames and corresponding labels
    '''

    def __init__(self, top_path, file_names, sample, sample_size=50, **kwargs):
        '''Scaler to scale feature frames and corresponding labels
        Arguments: 
            top_path: Path to search csv files
            label_file: Name of the label file
            file_names: File names to be considered for scaling
            sample: Hyperparam sample
            sample_size: Sample size from which shall be drawn for fitting the scaler
        '''

        self.file_names         = file_names
        self.top_path           = top_path
        self.sample_size        = sample_size
        self.sample             = sample

        self.unfiltered_length_t, self.unfiltered_length_y = pp.get_lengths(self.top_path)
        self.length_t, self.length_y = pp.get_filtered_lengths(self.top_path, self.sample)
        self.fit_scaler()
        super(FeatureScaler, self).__init__(**kwargs)

    def fit_scaler(self):
        '''Set min max values for custom scaling
        '''      
        self.max_feature = 0  
        self.min_feature = 1
        if type(self.file_names) is not pd.Series:
            self.file_names = pd.Series(self.file_names, name=None)

        for i in range(self.file_names.shape[0]):           
            full_path = os.path.join(self.top_path, self.file_names.iloc[i])
            try: 
                arr = np.load(full_path)
            
                if self.min_feature > np.min(arr[np.nonzero(arr)]): 
                    self.min_feature = np.min(arr[np.nonzero(arr)])
            
                if self.max_feature < np.max(arr[np.nonzero(arr)]):
                    self.max_feature = np.max(arr[np.nonzero(arr)])
                
            except: 
                # TODO: Fix error "> np.min(arr[...])"
                continue
            
            if i == self.sample_size: 
                return

    def transform(self, arr, min_max_range=(0, 1)):
        mi, ma = min_max_range
        arr_std = np.where(arr > 0, (arr - self.min_feature) / (self.max_feature - self.min_feature), arr)
        arr_scaled = arr_std * (ma - mi) + mi
        return arr_scaled


class LabelScaler(MinMaxScaler):
    '''Scaler to scale feature frames and corresponding labels
    '''

    def __init__(self, top_path, label_file, file_names, sample):
        '''Scaler to scale feature frames and corresponding labels
        Arguments: 
            top_path: Path to search csv files
            label_file: Name of the label file
            file_names: File names to be considered for scaling
            sample: Hyperparam sample
        '''

        self.file_names         = file_names
        self.label_file         = label_file
        self.top_path           = top_path
        self.sample             = sample

        self.unfiltered_length_t, self.unfiltered_length_y = pp.get_lengths(self.top_path)
        self.length_t, self.length_y = pp.get_filtered_lengths(self.top_path, self.sample)
        
        super(LabelScaler, self).__init__()
        self.fit_scaler()


    def fit_scaler(self):
        '''Fit the label scaler
        '''
        df_y = pd.read_csv(self.top_path + self.label_file, header=None, names=LABEL_HEADER)
        df_y['file_name'] = df_y['file_name'].apply(lambda row: row[:-4] + '.npy')
        df_y = df_y[df_y['file_name'].apply(lambda row: any(row[-32:] in file_name[-32:] for file_name in self.file_names))]
        scale_frame = df_y.apply(lambda row: row.iloc[1] if row.iloc[1] > row.iloc[2] else row.iloc[2], axis=1)
        self.fit(scale_frame.values.reshape(-1, 1))

    def transform_labels(self, label):
        return pd.DataFrame(self.transform(label.values.reshape(-1, 1)))

    