import math
import statistics
import os

import numpy as np 
from keras.models import load_model
import tensorflow as tf
import pandas as pd

from person_counting.utils.visualization_utils import plot_losses, visualize_predictions, visualize_filters
from person_counting.data_generators.data_generators import get_entering

LABEL_HEADER = ['file_name', 'entering', 'exiting', 'video_type']

def evaluate_model(model, history, gen, mode, logdir, top_path, visualize=True):

    if type(model) == list:
        model = load_model(model) 
    
    y_pred, y_pred_orig, y_true, y_true_orig = get_predictions(model, gen, top_path)
    evaluate_predictions(history, y_pred, y_pred_orig,
                         y_true, y_true_orig, model=model,
                         visualize=visualize, mode=mode, 
                         logdir=logdir)


def evaluate_predictions(history, y_pred, y_pred_orig,
                         y_true, y_true_orig, visualize,
                         mode, model, logdir=None):
    '''Evaluate predictions
    '''

    print_stats(y_pred_orig, y_true_orig, mode)

    if visualize == True:
        plot_losses(history, logdir=logdir)
        visualize_predictions(y_pred_orig, y_true_orig, logdir=logdir)
        visualize_filters(model, logdir=logdir)


def get_stats(y_true, predictions):
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
    
    df_y = pd.read_csv(os.path.join(top_path, gen.label_file), header=None, names=LABEL_HEADER)

    for file_name in gen.file_names: 
        try: 
            x = pd.read_csv(file_name, header=None)
            x = gen.preprocessor.preprocess_features(x)

            y = get_entering(file_name, df_y)
            y_true_orig.append(y.values[0])
            y_processed = np.copy(gen.preprocessor.preprocess_labels(y))

            y_true.append(y_processed[0])
            feature_frames.append(x)
        except: 
            print('Failed reading feature file for ', file_name)
            continue

    #Reshape features
    feature_frames = np.dstack(feature_frames)
    feature_frames = np.moveaxis(feature_frames, 2, 0)[..., np.newaxis]

    print(model.evaluate(x=feature_frames, y=np.array(y_true)))
    y_pred = model.predict(feature_frames)

    y_pred_orig = gen.scaler.scaler_labels.inverse_transform(y_pred)
    
    return np.squeeze(y_pred), np.squeeze(y_pred_orig), np.squeeze(np.array(y_true)), np.array(y_true_orig)
    

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
