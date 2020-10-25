import sys
import os
import time 
import io
from contextlib import redirect_stdout

from person_counting.models import cnn_regression as cnn
from person_counting.data_generators import data_generators as dgv
from person_counting.data_generators import data_generator_cnn as dgv_cnn
from person_counting.utils.preprocessing import get_filtered_lengths, get_video_daytime

label_file = 'pcds_dataset_labels_united.csv'
LABEL_HEADER = ['file_name', 'entering', 'exiting', 'video_type']
SNAP_PATH = 'C:/Users/Yannick/Google Drive/person_counting/tensorboard/cnn_regression/warm_start/t2_2020-09-11-20-22-27_cnn_2020_Sep_12_04_14_37'
top_path = 'C:/Users/Yannick/Google Drive/person_detection/pcds_dataset_detections/pcds_dataset_detected/'

TEST_BATCH_SIZES = [1, 2, 4, 8, 16, 32]

def main():
    for batch_size in TEST_BATCH_SIZES:

        # Supress printing
        trap = io.StringIO()
        with redirect_stdout(trap):
            hparams, timestep_num, feature_num = get_hparams(top_path, batch_size)

            _, _, datagen_test = dgv_cnn.create_datagen(top_path=top_path, 
                                                        sample=hparams,
                                                        label_file=label_file,
                                                        filter_hour_above=24)

            cnn_model = cnn.create_cnn(timestep_num, feature_num, hparams, datagen_test.label_scaler.scale_, snap_path=SNAP_PATH)

        # Inference for whole data generator 
        time = dataset_inference(cnn_model, datagen_test)
        print(f'Average inference time for batch size {batch_size} is ', time / len(datagen_test) / batch_size)

def time_measure(method):
    def timed(*args, **kw):
        ts = time.time()
        _ = method(*args, **kw)
        te = time.time()

        return te - ts
    return timed

@time_measure
def dataset_inference(cnn_model, datagen):
    """ Do inference for the whole generator without returning anything
    """
    for i in range(len(datagen)):
        img, label = datagen[i]

        dummy = cnn_model.predict(img) 
        dummy = datagen.label_scaler.inverse_transform(label)[0]

    return 

def get_hparams(top_path, batch_size):
    '''Set best hyperparameter set from prior tuning session
    
    Arguments: 
        top_path: Parent directory where shall be searched for csv files
    '''

    hparams = {
                'kernel_number'          : 5,
                'batch_size'             : batch_size,
                'regularization'         : 0.1,
                'filter_cols_upper'      : 0,
                'layer_number'           : 5,
                'kernel_size'            : 4,
                'pooling_type'           : 'max',
                'learning_rate'          : 0.0029459,
                'y_stride'               : 1,
                'optimizer'              : 'Adam',
                'pool_size_x'            : 2,
                'pool_size_y'            : 2,
                'batch_normalization'    : False, 
                'filter_cols_lower'      : 0,
                'augmentation_factor'    : 0,
                'filter_rows_lower'      : 0, 
                'pool_size_y_factor'     : 0, 
                'units'                  : 5,
                'loss'                   : 'msle',
                'Recurrent_Celltype'     : 'LSTM',
                'squeeze_method'         : '1x1_conv',
                'schedule_step'          : 5,
                'warm_start_path'        : 'None',
              }
              
    timestep_num, feature_num = get_filtered_lengths(top_path=top_path, sample=hparams)

    return hparams, timestep_num, feature_num

if __name__ == '__main__': 
    main()
