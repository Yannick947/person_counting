import datetime
import sys
import os
import random
import time 

import pandas as pd
import numpy as np  
import seaborn as sns
import matplotlib.pyplot as plt

from person_counting.models import cnn_regression as cnn
from person_counting.data_generators import data_generators as dgv
from person_counting.data_generators import data_generator_cnn as dgv_cnn
from person_counting.utils import visualization_utils as vp

label_file = 'pcds_dataset_labels_united.csv'
LABEL_HEADER = ['file_name', 'entering', 'exiting', 'video_type']

def main():
    if sys.argv[1] == 'train_best_cpu': 
        top_path = 'C:/Users/Yannick/Google Drive/person_detection/bus_videos/pcds_dataset_detected/'
        workers = 1
        multi_processing = False
        ipython_mode = False
        train_best(workers, multi_processing, top_path, ipython_mode)

    elif sys.argv[1] == 'train_best_gpu':
        top_path = '/content/drive/My Drive/person_detection/bus_videos/pcds_dataset_detected/'
        workers = 16
        multi_processing = True
        ipython_mode = True
        train_best(workers, multi_processing, top_path, ipython_mode)

    elif sys.argv[1] == 'test_input_csvs':
        top_path = 'C:/Users/Yannick/Google Drive/person_detection/bus_videos/pcds_dataset_detected/'
        test_input_csvs(top_path)

    elif sys.argv[1] == 'show_feature_frames':
        top_path = 'C:/Users/Yannick/Google Drive/person_detection/bus_videos/pcds_dataset_detected/'
        show_feature_frames(top_path)


def show_feature_frames(top_path):
    #Put the params you want to visualize below in the get_best_params function
    hparams, _, _ = get_best_hparams(top_path)

    datagen_train, datagen_test = dgv_cnn.create_datagen(top_path=top_path, 
                                                         filter_rows_factor=hparams['filter_rows_factor'], 
                                                         filter_cols_upper=hparams['filter_cols_upper'],
                                                         filter_cols_lower=hparams['filter_cols_lower'], 
                                                         batch_size=1, 
                                                         label_file=label_file, 
                                                         augmentation_factor=0.1)
    for datagen in [datagen_test, datagen_train]:                                                   
        gen = datagen.datagen()
        sns.set()
        for _ in range(10):
            with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
                feature_frame, label = next(gen)
                print('Label: ', datagen.scaler.scaler_labels.inverse_transform(label))
                ax = sns.heatmap(data=feature_frame[0, :, :, 0], vmin=0, vmax=1)
                plt.show()


def test_input_csvs(top_path):
    #Put random list of ints here, files have to be verified by hand afterwards
    test_indices = [1, 13, 15, 18, 25, 39, 50, 77, 88, 99]
    batch_size = 2
    testing_csv_names = list()

    for root, _, files in os.walk(top_path): 
        for i, file_name in enumerate(files): 
            if (file_name[-4:] == '.csv') and not ('label' in file_name) and (i in test_indices): 
                full_path = os.path.join(root, file_name)
                path = full_path[full_path.find('front_in'):].replace('\\', '/')
                testing_csv_names.append(path)

    df_testing_csv_names = pd.Series(testing_csv_names, name='file_names')

    df_verify = get_verification_data(top_path, testing_csv_names)
    hparams, timestep_num, feature_num = get_best_hparams(top_path)

    datagen = dgv_cnn.Generator_CSVS_CNN(length_t=timestep_num,
                                         length_y=feature_num,
                                         file_names=df_testing_csv_names,
                                         filter_cols_upper=hparams['filter_cols_upper'],
                                         filter_cols_lower=hparams['filter_cols_lower'],
                                         filter_rows_factor=hparams['filter_rows_factor'],
                                         batch_size=batch_size, 
                                         top_path=top_path,
                                         label_file=label_file)
    generator = datagen.datagen()

    for i in range(len(datagen)):
        _, labels_generated = next(generator)
        file_names_verify = datagen.get_file_names_processed()
        verify_labels = get_verification_labels(file_names_verify, df_verify, batch_size).reshape(batch_size, 1)
        labels_saved = datagen.get_labels().reshape(batch_size, 1)
        assert (labels_saved == verify_labels).all() and (verify_labels == labels_generated.reshape(batch_size, 1)).all(),\
               'Input test for generator fails for files {}'.format(file_names_verify)

        datagen.reset_label_states()
        datagen.reset_file_names_processed()
    print('Input test for generator passed for all verification files') 


def get_verification_labels(file_names_verify, df_verify, batch_size):
    entering = np.zeros(shape=batch_size)
    for i, file_name in enumerate(file_names_verify): 
        entering[i] = df_verify.loc[df_verify.file_name == file_name].iloc[:, 1].values
    return entering


def get_verification_data(top_path, testing_csv_names):
    df_y = pd.read_csv(top_path + label_file, header=None, names=LABEL_HEADER)
    df_verify = pd.DataFrame()
    df_verify = df_y[df_y['file_name'].apply(lambda row: any(row[-32:] in csv_file_name[-32:] for csv_file_name in testing_csv_names))]
    
    return df_verify 


def train_best(workers, multi_processing, top_path, ipython_mode): 
    hparams, timestep_num, feature_num = get_best_hparams(top_path)

    datagen_train, datagen_test = dgv_cnn.create_datagen(top_path=top_path, 
                                                         filter_rows_factor=hparams['filter_rows_factor'], 
                                                         filter_cols_upper=hparams['filter_cols_upper'],
                                                         filter_cols_lower=hparams['filter_cols_lower'], 
                                                         batch_size=hparams['batch_size'], 
                                                         label_file=label_file)

    cnn_model = cnn.create_cnn(timestep_num, feature_num, hparams)
    history, cnn_model = cnn.train(cnn_model,
                                   datagen_train,
                                   './',
                                   hparams,
                                   datagen_test,
                                   workers=workers,
                                   use_multiprocessing=multi_processing, 
                                   epochs=15)

    vp.plot(history, ipython_mode=ipython_mode)
    for gen in [datagen_train, datagen_test]:
        vp.visualize_predictions(cnn_model, gen, ipython_mode=ipython_mode)

    save_path = os.path.join(top_path, '/person_counting/model_snapshots/')
    cnn_model.save('test_best{}.h5'.format(min(history.history['val_loss'])))

def get_best_hparams(top_path):
    hparams = {
                'kernel_number'          : 5,
                'batch_size'             : 32,
                'regularization'         : 0.1,
                'filter_cols_upper'      : 70,
                'layer_number'           : 4,
                'kernel_size'            : 3,
                'filter_cols_factor'     : 1,
                'pooling_type'           : 'avg',
                'filter_rows_factor'     : 1,
                'learning_rate'          : 0.00017278,
                'y_stride'               : 1,
                'optimizer'              : 'Adam',
                'pool_size_x'            : 3,
                'batch_normalization'    : False, 
                'pool_size_y'            : 3,
                'filter_cols_lower'      : 35,   

              }
              
    timestep_num, feature_num = dgv.get_filtered_lengths(top_path=top_path,
                                            filter_rows_factor=hparams['filter_rows_factor'],
                                            filter_cols_upper=hparams['filter_cols_upper'], 
                                            filter_cols_lower=hparams['filter_cols_lower'],
                                            )

    return hparams, timestep_num, feature_num

if __name__ == '__main__': 
    main()
