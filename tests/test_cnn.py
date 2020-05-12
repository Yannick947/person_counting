import datetime
import sys
import os
import random
import time 

import pandas as pd
import numpy as np  
import seaborn as sns
import matplotlib.pyplot as plt
import cv2 as cv
from keras.models import Model
from keras.layers import Dense, BatchNormalization, AveragePooling2D, MaxPooling2D, Input

from person_counting.models import cnn_regression as cnn
from person_counting.data_generators import data_generators as dgv
from person_counting.data_generators import data_generator_cnn as dgv_cnn
from person_counting.bin.evaluate import evaluate_model

label_file = 'pcds_dataset_labels_united.csv'
LABEL_HEADER = ['file_name', 'entering', 'exiting', 'video_type']

def main():
    if sys.argv[1] == 'train_best_cpu': 
        top_path = 'C:/Users/Yannick/Google Drive/person_detection/pcds_dataset_detections/pcds_dataset_detected/'
        workers = 1
        multi_processing = False
        ipython_mode = False
        train_best(workers, multi_processing, top_path, ipython_mode, epochs=2)

    elif sys.argv[1] == 'train_best_gpu':
        top_path = '/content/drive/My Drive/person_detection/pcds_dataset_detections/pcds_dataset_detected/'
        workers = 16
        multi_processing = True
        ipython_mode = True
        train_best(workers, multi_processing, top_path, ipython_mode, epochs=25)

    elif sys.argv[1] == 'test_input_csvs':
        top_path = 'C:/Users/Yannick/Google Drive/person_detection/pcds_dataset_detections/pcds_dataset_detected/'
        test_input_csvs(top_path)

    elif sys.argv[1] == 'show_feature_frames':
        top_path = 'C:/Users/Yannick/Google Drive/person_detection/pcds_dataset_detections/pcds_dataset_detected/'
        show_feature_frames(top_path)


def show_feature_frames(top_path):
    #Put the params you want to visualize below in the get_best_params function
    hparams, timestep_num, feature_num = get_best_hparams(top_path)
    datagen_train, datagen_test = dgv_cnn.create_datagen(top_path=top_path, 
                                                         sample=hparams,
                                                         label_file=label_file, 
                                                         augmentation_factor=0.1, 
                                                         filter_hour_above=16, 
                                                         filter_category_noisy=True)
    for datagen in [datagen_test, datagen_train]:                                                   
        gen = datagen.datagen()
        sns.set()
        for _ in range(8):
            with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
                datagen.reset_file_names_processed() 
                feature_frame, label = next(gen)
                file_names = datagen.get_file_names_processed()
                daytime = dgv_cnn.get_video_daytime(file_names[0])

                print('Video name: ', file_names[0])
                print('Label: ', datagen.scaler.scaler_labels.inverse_transform(label)[0])
                print('Daytime of video: ', daytime[0], ':', daytime[1], '\n')

                fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))
                sns.heatmap(data=feature_frame[0, :, :, 0], vmin=0, vmax=1, ax=ax1)
                
                kernel = np.ones((2,3), np.uint8)
                opening = cv.dilate(feature_frame[0, :, :, 0], kernel)
                kernel = np.ones((2,2), np.uint8)
                opening = cv.erode(opening, kernel)
                sns.heatmap(data=opening, ax=ax2)

                pool_model = create_pooling_model(hparams, timestep_num, feature_num)
                pooled_frame = pool_model.predict(feature_frame)
                sns.heatmap(data=pooled_frame[0, :, :, 0], vmin=0, ax=ax3)

                plt.show()



def create_pooling_model(hparams, timesteps, features): 
    input_layer = Input(shape=((timesteps, features, 1)))

    if hparams['pooling_type'] == 'avg': 
        pooling = AveragePooling2D(pool_size=(hparams['pool_size_x'], hparams['pool_size_y'])) (input_layer)
    else:
        pooling = MaxPooling2D(pool_size=(hparams['pool_size_x'], hparams['pool_size_y'])) (input_layer)
    
    model = Model(inputs=input_layer, outputs=pooling)
    model.compile(loss='mean_squared_error', optimizer='Adam')

    return model



def test_input_csvs(top_path):
    ''' Test if the csv files have their proper label at the start 
    of the training session after the data generator did the "preprocessing"

    Arguments: 
        top_path: Parent directory where shall be searched for csv files
    '''

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
                                         sample=hparams,
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
    ''' Get verification labels from df and filters the ones needid for testing
    Arguments: 
        file_names_verify: File names for verfication used
        df_verify: Dataframe with labels
        batch_size: Batch_size for testing
    '''

    #TODO: Needs to be fixed
    entering = np.zeros(shape=batch_size)
    for i, file_name in enumerate(file_names_verify): 
        entering[i] = df_verify.loc[df_verify.file_name == file_name].iloc[:, 1].values
    return entering


def get_verification_data(top_path, testing_csv_names):
    '''Get verification labels directly from storage
    Arguments: 
        top_path: Parent directory where shall be searched for csvs
        testing_csv_names: Names of of which shall be tested
    '''

    df_y = pd.read_csv(top_path + label_file, header=None, names=LABEL_HEADER)
    df_verify = pd.DataFrame()
    df_verify = df_y[df_y['file_name'].apply(lambda row: any(row[-32:] in csv_file_name[-32:] for csv_file_name in testing_csv_names))]
    
    return df_verify 


def train_best(workers, multi_processing, top_path, ipython_mode, epochs=25): 
    '''Train best cnn model with manually put hparams from prior tuning results
    
    Arguments: 
        workers: Number of workers
        multi_processing: Flag if multi-processing is enabled
        top_path: Path to parent directory where csv files are stored
        ipython_mode: If running in a jupyter notebook or colab
    '''
    hparams, timestep_num, feature_num = get_best_hparams(top_path)

    datagen_train, datagen_test = dgv_cnn.create_datagen(top_path=top_path, 
                                                         sample=hparams,
                                                         label_file=label_file)

    cnn_model = cnn.create_cnn(timestep_num, feature_num, hparams, datagen_train.scaler.scaler_labels.scale_)
    history, cnn_model = cnn.train(cnn_model,
                                   datagen_train,
                                   './',
                                   hparams,
                                   datagen_test,
                                   workers=workers,
                                   use_multiprocessing=multi_processing, 
                                   epochs=epochs)

    for gen, mode in zip([datagen_train, datagen_test], ['train', 'test']):
        evaluate_model(cnn_model, history, gen, mode=mode, logdir='./', visualize=True)

    save_path = os.path.join(top_path, '/person_counting/model_snapshots/')
    cnn_model.save('test_best{}.h5'.format(min(history.history['val_loss'])))
    return cnn_model, history

def get_best_hparams(top_path):
    '''Set best hyperparameter set from prior tuning session
    Arguments: 
        top_path: Parent directory where shall be searched for csv files

    '''
    hparams = {
                'kernel_number'          : 5,
                'batch_size'             : 32,
                'regularization'         : 0.1,
                'filter_cols_upper'      : 35,
                'layer_number'           : 1,
                'kernel_size'            : 4,
                'filter_cols_factor'     : 1,
                'pooling_type'           : 'max',
                'filter_rows_factor'     : 1,
                'learning_rate'          : 0.0029459,
                'y_stride'               : 1,
                'optimizer'              : 'Adam',
                'pool_size_x'            : 2,
                'pool_size_y'            : 2,
                'batch_normalization'    : False, 
                'filter_cols_lower'      : 25,
                'augmentation_factor'    : 0,
                'filter_rows_lower'      : 150, 
                'pool_size_y_factor'     : 0.01, 
                'units'                  : 5,
              }
              
    timestep_num, feature_num = dgv.get_filtered_lengths(top_path=top_path, sample=hparams)

    return hparams, timestep_num, feature_num

if __name__ == '__main__': 
    main()
