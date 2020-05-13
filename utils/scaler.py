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

class CSVScaler(MinMaxScaler):
    '''Scaler to scale feature frames and corresponding labels
    '''

    def __init__(self, top_path, label_file, file_names, sample, sample_size):
        '''Scaler to scale feature frames and corresponding labels
        Arguments: 
            top_path: Path to search csv files
            label_file: Name of the label file
            file_names: File names to be considered for scaling
            sample: Hyperparam sample
            sample_size: Sample size from which shall be drawn for fitting the scaler
        
        '''
        self.file_names         = file_names
        self.label_file         = label_file
        self.scaler_features    = MinMaxScaler()
        self.scaler_labels      = MinMaxScaler()
        self.top_path           = top_path
        self.sample_size        = sample_size
        self.sample             = sample

        self.unfiltered_length_t, self.unfiltered_length_y = pp.get_lengths(self.top_path)

        self.fit_scalers()

    def fit_scalers(self, **kwargs): 
        self.__fit_features_scaler(**kwargs)
        self.__fit_labels_scaler()

    def __fit_labels_scaler(self):

        df_y = pd.read_csv(self.top_path + self.label_file, header=None, names=LABEL_HEADER)
        #TODO: make it work for entering and exiting, care at transformation, both must exist
        #don't consider the labels which were already filtered before.
        file_names = dgv.get_feature_file_names(self.top_path)
        df_y = df_y[df_y['file_name'].apply(lambda row: any(row[-32:] in csv_file_name[-32:] for csv_file_name in file_names))]
        df_y['entering'] = self.scaler_labels.fit_transform(df_y['entering'].values.reshape(-1, 1))
        self.df_y_scaled = df_y.copy()

    def __fit_features_scaler(self):
        df_fit_scale = pd.DataFrame()
        if type(self.file_names) is not pd.Series:
            self.file_names = pd.Series(self.file_names, name=None)

        for i in range(self.sample_size):           
            full_path = os.path.join(self.top_path, self.file_names.iloc[i])
            try: 
                df = pd.read_csv(full_path, header=None)
                if df.shape[0] != self.unfiltered_length_t or\
                   df.shape[1] != self.unfiltered_length_y:
                    raise ValueError('Wrong dimensions')
            except: 
                continue

            df = pp.clean_ends(df,
                                self.sample['filter_cols_upper'],
                                self.sample['filter_cols_lower'],
                                self.sample['filter_rows_lower'])    
            df_fit_scale = pd.concat([df_fit_scale, df], axis=0)

        self.scaler_features.fit(df_fit_scale.values)

    def transform_features(self, df): 
        return pd.DataFrame(self.scaler_features.transform(df.values))

    def transform_labels(self, label):
        return pd.DataFrame(self.scaler_labels.transform(label.values.reshape(-1, 1)))

