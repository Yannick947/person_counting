import os 
import pandas as pd
import numpy as np
from random import shuffle

import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

LABEL_HEADER = ['file_name', 'entering', 'exiting', 'video_type']

class CSVScaler(StandardScaler):
    
    def __init__(self, top_path, label_file, file_names):

        self.file_names         = file_names
        self.label_file         = label_file
        self.scaler_features    = StandardScaler()
        self.scaler_labels      = StandardScaler()
        self.top_path           = top_path
        self.fit_scalers()

    def fit_scalers(self, **kwargs): 
        self.__fit_train_scaler(**kwargs)
        self.__fit_test_scaler()

    def __fit_test_scaler(self):

        df_y = pd.read_csv(self.top_path + self.label_file, header=None, names=LABEL_HEADER)
        #TODO: make it work for entering and exiting, care at transformation, both must exist
        df_y['entering'] = self.scaler_labels.fit_transform(df_y['entering'].values.reshape(-1, 1))
        self.df_y_scaled = df_y.copy()

    def __fit_train_scaler(self, sample_size=30):
        df_fit_scale = pd.DataFrame()

        for i in range(sample_size):           
            full_path = os.path.join(self.top_path, self.file_names.iloc[i])
            try: 
                df = pd.read_csv(full_path, header=None)
                
            except: 
                continue
            df_fit_scale = pd.concat([df_fit_scale, df], axis=0)
        self.scaler_features.fit(df_fit_scale.values)

    def transform_features(self, df): 
        return pd.DataFrame(self.scaler_features.transform(df.values))

    def transform_labels(self, label):
        return pd.DataFrame(self.scaler_labels.transform(label.values.reshape(-1, 1)))

