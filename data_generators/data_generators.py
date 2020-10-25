import sys
import os
import pandas as pd
import numpy as np
import math
import abc
from random import shuffle
import random

from tensorflow import keras
from sklearn.model_selection import train_test_split

from person_counting.models.model_argparse import parse_args
from person_counting.utils import preprocessing as pp

LABEL_HEADER = ["file_name", "entering", "exiting", "video_type"]

VAL_SIZE = 0.2
TEST_SIZE = 0.1


class Generator_CSVS(keras.utils.Sequence):
    """Abstract class for Generators to load npy files from
    video folder structre like PCDS Dataset
    """

    __metaclass__ = abc.ABCMeta

    def __init__(
        self,
        length_t,
        length_y,
        file_names,
        sample,
        top_path,
        label_file,
        feature_scaler,
        label_scaler,
        augmentation_factor=0,
        inverse_probability=0.5,
    ):

        """Initialize Generator object.

        Arguments
            length_t            : Length of the feature's DataFrame in time dimension
            length_y            : Length of the feature's DataFrame in y direction
            file_names          : File names to be processed
            filter_cols_upper   : Amount of columns to be filtered at end and start of DataFrame
            sample              : Sample of hyperparameter
            top_path            : Parent path where npy files are contained
            label_file          : Name of the label file
            augmentation_factor : Factor how much augmentation shall be done, 1 means moving every
                                  pixel for 1 position
        """
        self.top_path = top_path
        self.label_file = label_file
        self.length_t = length_t
        self.length_y = length_y
        self.file_names = file_names
        self.filter_cols_upper = sample["filter_cols_upper"]
        self.filter_cols_lower = sample["filter_cols_lower"]
        self.batch_size = sample["batch_size"]
        self.labels = list()
        self.file_names_processed = list()
        self.feature_scaler = feature_scaler
        self.label_scaler = label_scaler
        self.augmentation_factor = augmentation_factor
        self.sample = sample
        self.unfiltered_length_t, self.unfiltered_length_y = pp.get_lengths(self.top_path)
        self.preprocessor = pp.Preprocessor(
            length_t, length_y, top_path, sample, feature_scaler, label_scaler, augmentation_factor
        )
        self.df_y = self.load_label_file()
        self.index_to_file_name = self.load_index_to_filename()
        self.shuffle = True

        self.eval_batches = int(np.floor(len(self.file_names) / float(self.batch_size)))
        print(
            "Generator contains ",
            len(self.file_names),
            " files with ",
            self.eval_batches,
            " batches of size ",
            self.batch_size,
        )
        self.on_epoch_end()

    @abc.abstractmethod
    def datagen(self):
        """Returns datagenerator"""
        raise NotImplementedError

    def __len__(self):
        """Returns the amount of batches for the generator to be used when flipping the files is applied (*2)"""
        return self.eval_batches * 2

    def __getitem__(self, index):
        """Gets an item by its index"""
        x_batch = np.zeros(shape=(self.batch_size, self.length_t, self.length_y, 2))

        y_batch = np.zeros(shape=(self.batch_size, 1))
        batch_index = 0

        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            try:
                if i >= len(self.index_to_file_name):
                    # This is the inverse case when array shall be flipped
                    i = i - len(self.index_to_file_name)
                    i = self.indexes[i]
                    arr_x, label = self.get_sample(self.index_to_file_name[i], flip_arr=True)
                else:
                    i = self.indexes[i]
                    arr_x, label = self.get_sample(self.index_to_file_name[i], flip_arr=False)

            # Error messages for debugging purposes
            except FileNotFoundError as e:
                print(e)
                continue

            except ValueError as e:
                print(e)
                print(self.index_to_file_name[i])
                # os.remove(self.index_to_file_name[i])
                continue

            x_batch[batch_index, :, :, :] = arr_x
            y_batch[batch_index] = label
            batch_index += 1

        return (x_batch, y_batch)

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.index_to_file_name))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def load_index_to_filename(self):
        """Get a dict mapping indices to file names"""
        index_to_file_name = dict()
        for index, file_name in enumerate(self.file_names):
            index_to_file_name[index] = file_name

        return index_to_file_name

    def get_sample(self, file_name, flip_arr=False):
        """Gets pair of features and labels for given filename
        Arguments:
            file_name: The name of the file which shall be parsed
        """
        arr_x = self.__get_features(file_name)

        if arr_x is not None:
            if (
                arr_x.shape[0] != self.unfiltered_length_t
                or arr_x.shape[1] != self.unfiltered_length_y
                or arr_x.shape[2] != 2
            ):

                raise ValueError("File with wrong dimensions found")
        else:
            raise FileNotFoundError("Failed getting features for file {}".format(file_name))

        arr_x = self.preprocessor.preprocess_features(arr_x, file_name)
        # inverse case is getting the label entering for back out with flipped input and exiting for front in
        # labels have to be inverted according to both parameters (4 cases)
        if not flip_arr:
            if "back_out" in file_name:
                label = get_label(file_name=file_name, df_y=self.df_y, inverse=True)
            else:
                arr_x = np.flip(arr_x, axis=1)
                label = get_label(file_name=file_name, df_y=self.df_y, inverse=True)

        else:
            if "front_in" in file_name:
                label = get_label(file_name=file_name, df_y=self.df_y, inverse=False)
            else:
                arr_x = np.flip(arr_x, axis=1)
                label = get_label(file_name=file_name, df_y=self.df_y, inverse=False)

        label = self.preprocessor.preprocess_labels(label)

        self.file_names_processed.append(file_name)
        self.labels.append(label)
        return arr_x, label

    def __get_features(self, file_name):
        """Get sample of features for given filename.

        Arguments:
            file_name: Name of given training sample

            returns: Features for given file_name
        """

        try:
            arr = np.load(file_name)
            return arr

        except Exception as e:
            return None

    def load_label_file(self):
        """Loads the label file as DataFrame"""
        df_y = pd.read_csv(self.top_path + self.label_file, header=None, names=LABEL_HEADER)
        df_y["file_name"] = df_y["file_name"].apply(lambda row: row[:-4] + ".npy")
        return df_y

    def get_labels(self):
        """Returns the labels which were yielded since calling reset_labels()"""
        return np.array(self.labels)

    def reset_label_states(self):
        """Resets the labels which were processed"""
        self.labels = list()

    def get_file_names(self):
        """Get all file names used of this generator"""
        return self.file_names

    def reset_file_names_processed(self):
        """Reset the filenames which were processed since last execution  of this function"""
        self.file_names_processed = list()

    def get_file_names_processed(self):
        """Gets the file names which were processed since last execution of reset_file_names_processed"""
        return self.file_names_processed


def get_label(file_name, df_y, inverse=False):
    """Get the label for a given file
    Arguments:
        file_name: Name of the file for training
        df_y: Dataframe with labels for corresponding file names
        inverse: If the input array was flipped before so the opposite
                 label shall be loaded

    returns an int indicating the label
    """
    if (("back_out" in file_name) and (inverse is False)) or (("front_in" in file_name) and (inverse is True)):

        return get_exiting(file_name, df_y)

    elif (("front_in" in file_name) and (inverse is False)) or (("back_out" in file_name) and (inverse is True)):

        return get_entering(file_name, df_y)

    else:
        raise FileNotFoundError


def get_feature_file_names(top_path):
    """
    Get names of all npy files for training

    Arguments:
        top_path: Parent directory where to search for npy files
    """
    names = list()
    for root, _, files in os.walk(top_path):
        for file_name in files:
            if (file_name[-4:] == ".npy") and not ("label" in file_name):
                names.append(os.path.join(root, file_name))
    return names


def split_files(top_path, label_file):
    """Splits all files in the training set into train and test files
    and returns lists of names for train and test files
    """

    df_names = pd.Series(get_feature_file_names(top_path))

    # replace .avi with .npy
    df_names = df_names.apply(lambda row: row[:-4] + ".npy")
    train, test = train_test_split(df_names, train_size=1 - TEST_SIZE - VAL_SIZE, random_state=42)

    validation, test = train_test_split(test, train_size=VAL_SIZE / (VAL_SIZE + TEST_SIZE), random_state=42)
    return train, validation, test


def get_entering(file_name, df_y):
    """Get number of entering persons to existing training sample.

    Arguments:
        file_name: Name of given training sample
        df_y: Dataframe with all labels for all samples

        returns: Label for given features
    """
    search_str = file_name.replace("\\", "/").replace("\\", "/")

    if "front_in" in file_name:
        search_str = search_str[search_str.find("front_in") :]
    else:
        search_str = search_str[search_str.find("back_out") :]

    entering = df_y.loc[df_y.file_name == search_str].entering
    return entering


def get_exiting(file_name, df_y):
    """Get number of exiting persons to existing training sample.

    Arguments:
        file_name: Name of given training sample
        df_y: Dataframe with all labels for all samples

        returns: DF with exiting persons for given file
    """
    search_str = file_name.replace("\\", "/").replace("\\", "/")

    if "front_in" in file_name:
        search_str = search_str[search_str.find("front_in") :]
    else:
        search_str = search_str[search_str.find("back_out") :]

    entering = df_y.loc[df_y.file_name == search_str].exiting
    return entering


def get_video_class(file_name, df_y):
    """Get video type

    Arguments:
        file_name: Name of given training sample
        df_y: Dataframe with all labels for all samples

        returns: DataFrame with video_type for given file
    """
    try:
        search_str = file_name.replace("\\", "/").replace("\\", "/")

        if "front" in file_name:
            search_str = search_str[search_str.find("front_in") :]
        else:
            search_str = search_str[search_str.find("back_out") :]

        video_type = df_y.loc[df_y.file_name == search_str].video_type
        return video_type

    except Exception as e:
        # print('No matching label found for existing npy file')
        return None
