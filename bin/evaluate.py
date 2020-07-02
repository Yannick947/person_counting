import math
import statistics
import os
import csv

import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
from tensorflow.core.util import event_pb2
import keras
from keras.models import load_model
import pandas as pd
import numpy as np 

from person_counting.utils.visualization_utils import plot_losses, visualize_predictions, visualize_filters, save_confusion_matrix
from person_counting.data_generators.data_generators import get_label, get_video_class, get_exiting

CATEGORY_MAPPING = {0: 'normal_uncrowded', 1: 'normal_crowded', 2: 'noisy_uncrowded', 3: 'noisy_crowded'}


class Evaluate(keras.callbacks.Callback):
    """ Evaluation callback for arbitrary datasets.
    """

    def __init__(self, metric_name, mode, logdir):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.
        """
        self.metric_name = metric_name
        self.mode = mode 
        self.logdir = logdir
        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        max_metric = get_max_val(self.metric_name, self.mode, self.logdir)

        if max_metric is not None: 
            logs[self.metric_name + '_max'] = max_metric
        else: 
            print('Care to specify correct metric names')


def get_max_val(tag, mode, logdir): 
    '''Gets the maximum value from the logged validation file for a certain metric
    '''
    for file_name in os.listdir(os.path.join(logdir, 'validation')):
        if file_name[-3:] == '.v2':
            serialized_examples = tf.data.TFRecordDataset(os.path.join(logdir, 'validation', file_name))

    maximum_val = None
    for serialized_example in serialized_examples:
        event = event_pb2.Event.FromString(serialized_example.numpy())
        for value in event.summary.value:
            if value.tag == tag:
                if maximum_val == None: 
                    maximum_val = value.simple_value
                elif maximum_val < value.simple_value and mode == 'max':
                    maximum_val = value.simple_value
                elif maximum_val > value.simple_value and mode == 'min': 
                    maximum_val = value.simple_value

    return maximum_val


def evaluate_run(model, history, gen, mode, logdir, top_path, visualize=True):
    ''' Evaluate a run of certain hyperparameters
    Arguments: 
        model: Last model which was created during training
        history: Keras history object which was created during training
        gen: Generator for data
        mode: Mode (training or testing)
        logdir: Directory where logging is done
        top_path: Parent directory where is logged to 
        visualize: Flag indicating if plots shall be created and saved
    '''

    #Search for best model in logdir if existing
    model = parse_model(model, logdir, scale=gen.label_scaler.scale_)
    mae_rescaled = create_mae_rescaled(gen.label_scaler.scale_)
    accuracy_rescaled = create_accuracy_rescaled(gen.label_scaler.scale_)
    model.compile(optimizer='adam', loss=mae_rescaled, metrics=['msle', 'mae', mae_rescaled, accuracy_rescaled])
    
    y_pred, y_pred_orig, y_true, y_true_orig, video_cats, feature_frames = get_predictions(model, gen, top_path)
    if mode == 'test': 
        save_test_evaluation(model, feature_frames, y_true, logdir)

    evaluate_predictions(history, y_pred, y_pred_orig,
                         y_true, y_true_orig, model=model,
                         visualize=visualize, mode=mode, 
                         logdir=logdir, video_categories=video_cats)

def save_test_evaluation(model, feature_frames, y_true, logdir): 
    """ Save the metrics for the test run as csv file
    """
    print('Evaluation for test files: ')
    metrics = model.evaluate(x=feature_frames, y=np.array(y_true))
    with open(os.path.join(logdir, 'test_metrics.csv'), 'w') as csv_file:
        writer = csv.writer(csv_file)
        metric_keys = ['loss', 'msle', 'mae', 'mae_rescaled', 'accuracy_rescaled']
        for key, value in zip(metric_keys, metrics):
            writer.writerow([key, value])
            print(key, value)

def parse_model(model=None, logdir=None, compile_model=True, scale=0.066):
    ''' Parse logdir for best model
    Arguments: 
        model: Last model during training
        logdir: Path to directory where best model might be stored
        compile_model: Flag whether model shall be compiled on loading
        scale: Scale factor that was used for rescaling the data

    returns best Keras model during training, if not existent, returns 
            last model during training
    '''
    saved_models = list()

    for file_name in os.listdir(logdir):
        if file_name[-5:] == '.hdf5':
            saved_models.append(file_name)
    
    if len(saved_models) == 0:
        print("No model to load")
        return model

    epoch = 0
    best_model = None
    for i, file_name in enumerate(saved_models):
        cur_epoch = int(file_name[:3].replace('_', '')) 
        if  cur_epoch > epoch:
            epoch = cur_epoch
            if best_model is not None:
                os.remove(os.path.join(logdir, best_model))
            best_model = os.path.join(logdir, file_name)

        if i + 1 == len(saved_models):
            print('\nLoading from {}, which is the best within this training ..'.format(best_model))
            return load_model(best_model, custom_objects={'tf': tf,
                                                          'hard_tanh':hard_tanh,
                                                          'acc_rescaled':create_accuracy_rescaled(scale),
                                                          'mae_rescaled':create_mae_rescaled(scale)},
                                                          compile=compile_model) 
    print("No model loaded")
    return model


def evaluate_predictions(history,
                         y_pred,
                         y_pred_orig,
                         y_true,
                         y_true_orig,
                         visualize,
                         model,
                         mode='test',
                         logdir=None,
                         video_categories=None):
    '''Evaluate predictions of the best model
    Arguments: 
        history: Keras history object created during training 
        y_pred: Predictions 
        y_pred_orig: Predictions retransformed
        y_true: Ground truth 
        y_true_orig: Ground truth retransformed
        visualize: Flag indicating if visualization shall be done
        mode: Mode ('validation', or 'test')
        model: Last model created during training
        logdir: Directory where logging is done
    '''
    print_stats(y_pred_orig, y_true_orig, mode)

    if visualize == True:
        visualize_predictions(y_true=y_true_orig, y_pred=y_pred_orig, mode=mode, logdir=logdir, video_categories=video_categories)
        visualize_filters(model, logdir=logdir)
        plot_losses(history, logdir=logdir)
        save_confusion_matrix(y_true_orig, y_pred_orig, logdir)


def get_stats(y_true, predictions):
    '''Gets stats for GT and predictions
    '''

    difference = 0
    difference_dummy = 0
    mean_ground_truth = sum(y_true) / len(y_true)

    for prediction, y in zip(predictions, y_true):
        difference += abs(prediction - y)
        difference_dummy += abs(mean_ground_truth - y)
    
    mean_difference_pred = difference / len(predictions)
    mean_difference_dummy = difference_dummy / len(y_true)

    return mean_difference_pred, mean_difference_dummy, mean_ground_truth


def create_mae_rescaled(scale_factor):
    '''Create a callback function which tracks mae rescaled
    Arguments: 
        scale_factor: Scaling factor with which the labels were scaled initially
    '''
    def mae_rescaled(y_true, y_pred):    
        difference = abs(y_pred - y_true)
        return difference / scale_factor

    return mae_rescaled

def create_accuracy_rescaled(scale_factor):
    '''Create a callback function which calculates accuracy rescaled metric
    Arguments: 
        scale_factor: Scaling factor with which the labels were scaled initially
    '''
    def acc_rescaled(y_true, y_pred):    
        """ Calculates the accuracy for the rescaled values
        """
        y_true_rescaled = tf.round(tf.math.divide(y_true, scale_factor))
        y_pred_rescaled =  tf.round(tf.math.divide(y_pred, scale_factor))
        return math_ops.cast(math_ops.equal(y_true_rescaled, y_pred_rescaled), K.floatx())

    return acc_rescaled

def get_predictions(model, gen, top_path): 
    '''Generate predictions from generator and model

    Arguments: 
        model: Model which shall predict
        gen: Generator to load data
    returns predictions and corresponding ground_truth
    '''
    gen.reset_label_states()
    gen.reset_file_names_processed()
    gen.batch_size = 1

    y_true = list()
    feature_frames = list()
    y_true_orig = list()
    video_type = list()

    df_y = gen.load_label_file()
    for inverse in [True, False]:
        for file_name in gen.file_names: 
            try: 
                arr_x = np.load(file_name)
                arr_x = gen.preprocessor.preprocess_features(arr_x, file_name)
                if inverse: 
                    if 'back_out' in file_name: 
                        y = get_label(file_name=file_name, df_y=df_y, inverse=True)
                    else: 
                        arr_x = np.flip(arr_x, axis=1)
                        y = get_label(file_name=file_name, df_y=df_y, inverse=True)

                else: 
                    if 'front_in' in file_name: 
                        y = get_label(file_name=file_name, df_y=df_y, inverse=False)
                    else: 
                        arr_x = np.flip(arr_x, axis=1)
                        y = get_label(file_name=file_name, df_y=df_y, inverse=False)
                        
                y_true_orig.append(int(y.values[0]))
                y_processed = np.copy(gen.preprocessor.preprocess_labels(y))

                y_true.append(y_processed[0])
                feature_frames.append(arr_x)

                video_category = get_video_class(file_name, df_y).values[0]
                video_type.append(CATEGORY_MAPPING[video_category])

            except: 
                print('Failed reading feature file for ', file_name)
                continue

    feature_frames = np.stack(feature_frames, axis=0)
    y_pred = model.predict(feature_frames)
    y_pred_orig = gen.label_scaler.inverse_transform(y_pred)
    
    return np.squeeze(y_pred), np.squeeze(y_pred_orig), np.squeeze(np.array(y_true)), np.array(y_true_orig), np.array(video_type), feature_frames
    

def print_stats(predictions, y_true, mode):
    '''Print stats of predictions and ground_truth
    
    Arguments: 
        predictions: Predictions of estimator as numpy array
        y_true: Ground truth as numpy array
        mode: Mode (train or test) as string
    '''

    mean_difference, mean_difference_dummy, mean_ground_truth = get_stats(y_true, predictions)
    y_true = y_true.astype(float)   

    print('\nFor mode: ', mode)
    print('Mean of ground truth: ', mean_ground_truth)
    print('Mean of predictions: ', sum(predictions) / len(predictions))
    print('\nStd of ground truth: ', np.std(y_true))
    print('Std of predicitons: ', np.std(predictions))
    print('\nMean difference between ground truth and predictions is: ', mean_difference)
    print('Mean difference between dummy estimator (voting always for mean of ground truth) and ground truth: ', mean_difference_dummy)

def hard_tanh(x): 
    '''Hard tanh function
    Arguments: 
        x: Input value
    
    hard_tanh(x) = {-1,      for x < -2, 
                    tanh(x), for x > -2 and x < 2
                    1,       for x > 2              }

    returns value according to hard tanh function
    '''
    return tf.maximum(tf.cast(-1, tf.float32), tf.minimum(tf.cast(1, tf.float32), tf.cast(keras.backend.tanh(x) * 1.05, tf.float32)))
