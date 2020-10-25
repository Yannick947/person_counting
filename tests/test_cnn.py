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

from person_counting.models import cnn_classification as cnn_cls
from person_counting.models import cnn_regression as cnn
from person_counting.data_generators import data_generators as dgv
from person_counting.data_generators import data_generator_cnn as dgv_cnn
from person_counting.data_generators import data_generator_cnn_classification as dg_cls
from person_counting.bin.evaluate import evaluate_run
from person_counting.bin.evaluate_cls import evaluate_run_cls
from person_counting.utils.preprocessing import get_filtered_lengths, get_video_daytime
from person_counting.utils.visualization_utils import visualize_input_2d, visualize_input_3d
from person_counting.data_generators.trajectory_augmentation import augment_trajectory

label_file = "pcds_dataset_labels_united.csv"
LABEL_HEADER = ["file_name", "entering", "exiting", "video_type"]

# The of the model you want to use for evaluation
SNAP_PATH = "C:/Users/Yannick/Google Drive/person_counting/tensorboard/cnn_regression/warm_start/t2_2020-09-11-20-22-27_cnn_2020_Sep_12_04_14_37"

# TODO: Seperate all test cases in different folder


def main():
    if sys.argv[1] == "train_best_cpu":
        top_path = "C:/Users/Yannick/Google Drive/person_detection/pcds_dataset_detections/pcds_dataset_detected/"
        workers = 0
        multi_processing = False
        train_best(workers, multi_processing, top_path, epochs=0, snap_path=SNAP_PATH)

    elif sys.argv[1] == "train_best_gpu":
        top_path = "/content/drive/My Drive/person_detection/pcds_dataset_detections/pcds_dataset_detected/"
        workers = 16
        multi_processing = True
        train_best(workers, multi_processing, top_path, epochs=12)

    elif sys.argv[1] == "test_input_csvs":
        top_path = "C:/Users/Yannick/Google Drive/person_detection/pcds_dataset_detections/pcds_dataset_detected/"
        test_input_csvs(top_path)

    elif sys.argv[1] == "train_best_cpu_cls":
        top_path = "C:/Users/Yannick/Google Drive/person_detection/pcds_dataset_detections/pcds_dataset_detected/"
        workers = 1
        multi_processing = False
        train_best_cls(workers, multi_processing, top_path, epochs=1)

    elif sys.argv[1] == "show_feature_frames":
        top_path = "C:/Users/Yannick/Google Drive/person_detection/pcds_dataset_detections/pcds_dataset_detected/"
        show_feature_frames(top_path, show_augmentation=True)


def show_feature_frames(top_path, show_augmentation=False):
    # Put the params you want to visualize below in the get_best_params function
    hparams, timestep_num, feature_num = get_best_hparams(top_path)
    datagen_train, datagen_validation, datagen_test = dgv_cnn.create_datagen(
        top_path=top_path,
        sample=hparams,
        label_file=label_file,
        augmentation_factor=0.0,
        filter_hour_below=16,
        filter_hour_above=20,
        filter_category_noisy=True,
    )

    pool_model = create_pooling_model(hparams, timestep_num, feature_num)
    for datagen in [datagen_test, datagen_train]:
        sns.set()

        for i in range(len(datagen)):
            with pd.option_context("display.max_rows", None, "display.max_columns", None):
                datagen.reset_file_names_processed()
                feature_frame, label = datagen[i]

                if datagen.label_scaler.inverse_transform(label)[0] < 4:
                    continue

                file_names = datagen.get_file_names_processed()
                daytime = get_video_daytime(file_names[0])

                print("Video name: ", file_names[0])
                print("Label: ", datagen.label_scaler.inverse_transform(label)[0])
                print("Daytime of video: ", daytime[0], ":", daytime[1], "\n")

                # visualize_input_3d(feature_frame, pool_model, save_plots=False)
                visualize_input_2d(feature_frame, feature_num, timestep_num, pool_model, save_plots=True)

                if show_augmentation:
                    for augmentation_factor in [0.1, 0.5, 1.0]:
                        feature_frame = augment_trajectory(feature_frame[0, :, :, :], aug_factor=augmentation_factor)
                        feature_frame = np.expand_dims(feature_frame, axis=0)
                        visualize_input_2d(feature_frame, feature_num, timestep_num, pool_model, save_plots=True)


def create_pooling_model(hparams, timesteps, features):
    input_layer = Input(shape=((timesteps, features, 2)))

    if hparams["pooling_type"] == "avg":
        pooling = AveragePooling2D(pool_size=(hparams["pool_size_x"], hparams["pool_size_y"]))(input_layer)
    else:
        pooling = MaxPooling2D(pool_size=(hparams["pool_size_x"], hparams["pool_size_y"]))(input_layer)

    model = Model(inputs=input_layer, outputs=pooling)
    model.compile(loss="mean_squared_error", optimizer="Adam")

    return model


def test_input_csvs(top_path):
    """Test if the csv files have their proper label at the start
    of the training session after the data generator did the "preprocessing"

    Arguments:
        top_path: Parent directory where shall be searched for csv files
    """

    # Put random list of ints here, files have to be verified by hand afterwards
    test_indices = [1, 13, 15, 18, 25, 39, 50, 77, 88, 99]
    batch_size = 2
    testing_csv_names = list()

    for root, _, files in os.walk(top_path):
        for i, file_name in enumerate(files):
            if (file_name[-4:] == ".csv") and not ("label" in file_name) and (i in test_indices):
                full_path = os.path.join(root, file_name)
                path = full_path[full_path.find("front_in") :].replace("\\", "/")
                testing_csv_names.append(path)

    df_testing_csv_names = pd.Series(testing_csv_names, name="file_names")

    df_verify = get_verification_data(top_path, testing_csv_names)
    hparams, timestep_num, feature_num = get_best_hparams(top_path)

    datagen = dgv_cnn.Generator_CSVS_CNN(
        length_t=timestep_num,
        length_y=feature_num,
        file_names=df_testing_csv_names,
        sample=hparams,
        top_path=top_path,
        label_file=label_file,
    )
    generator = datagen.datagen()

    for i in range(len(datagen)):
        _, labels_generated = next(generator)
        file_names_verify = datagen.get_file_names_processed()
        verify_labels = get_verification_labels(file_names_verify, df_verify, batch_size).reshape(batch_size, 1)
        labels_saved = datagen.get_labels().reshape(batch_size, 1)
        assert (labels_saved == verify_labels).all() and (
            verify_labels == labels_generated.reshape(batch_size, 1)
        ).all(), "Input test for generator fails for files {}".format(file_names_verify)

        datagen.reset_label_states()
        datagen.reset_file_names_processed()
    print("Input test for generator passed for all verification files")


def get_verification_labels(file_names_verify, df_verify, batch_size):
    """Get verification labels from df and filters the ones needid for testing
    Arguments:
        file_names_verify: File names for verfication used
        df_verify: Dataframe with labels
        batch_size: Batch_size for testing
    """

    entering = np.zeros(shape=batch_size)
    for i, file_name in enumerate(file_names_verify):
        entering[i] = df_verify.loc[df_verify.file_name == file_name].iloc[:, 1].values
    return entering


def get_verification_data(top_path, testing_csv_names):
    """Get verification labels directly from storage
    Arguments:
        top_path: Parent directory where shall be searched for csvs
        testing_csv_names: Names of of which shall be tested
    """

    df_y = pd.read_csv(top_path + label_file, header=None, names=LABEL_HEADER)
    df_verify = pd.DataFrame()
    df_verify = df_y[
        df_y["file_name"].apply(
            lambda row: any(row[-32:] in csv_file_name[-32:] for csv_file_name in testing_csv_names)
        )
    ]

    return df_verify


def train_best(workers, multi_processing, top_path, epochs=25, snap_path=None):
    """Train best cnn model with manually put hparams from prior tuning results

    Arguments:
        workers: Number of workers
        multi_processing: Flag if multi-processing is enabled
        top_path: Path to parent directory where csv files are stored
    """
    hparams, timestep_num, feature_num = get_best_hparams(top_path)

    datagen_train, datagen_validation, datagen_test = dgv_cnn.create_datagen(
        top_path=top_path, sample=hparams, label_file=label_file, filter_hour_above=20
    )
    cnn_model = cnn.create_cnn(
        timestep_num, feature_num, hparams, datagen_train.label_scaler.scale_, snap_path=snap_path
    )

    history, cnn_model = cnn.train(
        cnn_model,
        datagen_train,
        "./",
        hparams,
        datagen_test,
        workers=workers,
        use_multiprocessing=multi_processing,
        epochs=epochs,
    )
    cnn_model.summary()

    for gen, mode in zip([datagen_validation, datagen_test], ["validation", "test"]):
        evaluate_run(cnn_model, history, gen, mode=mode, logdir="./", visualize=True, top_path=top_path)

    return cnn_model, history


def train_best_cls(workers, multi_processing, top_path, epochs=25):
    """Train best cnn model with manually put hparams from prior tuning results

    Arguments:
        workers: Number of workers
        multi_processing: Flag if multi-processing is enabled
        top_path: Path to parent directory where csv files are stored
    """
    hparams, timestep_num, feature_num = get_best_hparams(top_path)

    datagen_train, datagen_test = dg_cls.create_datagen(
        top_path=top_path, sample=hparams, label_file=label_file, filter_hour_above=12
    )

    cnn_model = cnn_cls.create_cnn(timestep_num, feature_num, hparams, datagen_train.num_classes)
    history, cnn_model = cnn_cls.train(
        cnn_model,
        datagen_train,
        "./",
        hparams,
        datagen_test,
        workers=workers,
        use_multiprocessing=multi_processing,
        epochs=epochs,
    )

    for gen, mode in zip([datagen_train, datagen_test], ["train", "test"]):
        evaluate_run_cls(cnn_model, history, gen, mode=mode, logdir="./", visualize=True, top_path=top_path)

    return cnn_model, history


def get_best_hparams(top_path):
    """Set best hyperparameter set from prior tuning session

    Arguments:
        top_path: Parent directory where shall be searched for csv files
    """

    hparams = {
        "kernel_number": 5,
        "batch_size": 16,
        "regularization": 0.1,
        "filter_cols_upper": 0,
        "layer_number": 5,
        "kernel_size": 4,
        "pooling_type": "max",
        "learning_rate": 0.0029459,
        "y_stride": 1,
        "optimizer": "Adam",
        "pool_size_x": 2,
        "pool_size_y": 2,
        "batch_normalization": False,
        "filter_cols_lower": 0,
        "augmentation_factor": 0,
        "filter_rows_lower": 0,
        "pool_size_y_factor": 0,
        "units": 5,
        "loss": "msle",
        "Recurrent_Celltype": "LSTM",
        "squeeze_method": "1x1_conv",
        "schedule_step": 5,
        "warm_start_path": "None",
    }

    timestep_num, feature_num = get_filtered_lengths(top_path=top_path, sample=hparams)

    return hparams, timestep_num, feature_num


if __name__ == "__main__":
    main()
