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
    model = parse_model(model, logdir)
    model.compile(optimizer='adam', loss=create_mae_rescaled(gen.scaler.scaler_labels.scale_))
    
    y_pred, y_pred_orig, y_true, y_true_orig = get_predictions(model, gen, top_path)

    evaluate_predictions(history, y_pred, y_pred_orig,
                         y_true, y_true_orig, model=model,
                         visualize=visualize, mode=mode, 
                         logdir=logdir)

def parse_model(model, logdir):
    ''' Parse logdir for best model
    Arguments: 
        model: Last model during training
        logdir: Path to directory where best model might be stored
    returns best Keras model during training, if not existent, returns 
            last model during training
    '''
    saved_models = list()
    print('\nLoading best model within this training ..')

    for files in os.listdir(logdir):
        for file_name in files: 
            if file_name[-5:] == '.hdf5':
                saved_models.append(file_name)
    
    if len(saved_models) == 0:
        return model

    epoch = 0
    best_model = None
    for i, file_name in enumerate(saved_models):
        cur_epoch = int(file_name[:3].replace('_', '')) 
        if  cur_epoch > epoch:
            epoch = cur_epoch
            os.remove(os.path.join(logdir, best_model))
            best_model = os.path.join(logdir, file_name)

        elif (i + 1) == len(saved_models):
            return load_model(best_model, custom_objects={'tf': tf}, compile=False) 

    return model


def evaluate_predictions(history, y_pred, y_pred_orig,
                         y_true, y_true_orig, visualize,
                         model, mode='Test', logdir=None):
    '''Evaluate predictions of the best model
    Arguments: 
        history: Keras history object created during training 
        y_pred: Predictions 
        y_pred_orig: Predictions retransformed
        y_true: Ground truth 
        y_true_orig: Ground truth retransformed
        visualize: Flag indicating if visualization shall be done
        mode: Mode ('Train', or 'Test')
        model: Last model created during training
        logdir: Directory where logging is done

    '''

    print_stats(y_pred_orig, y_true_orig, mode)

    if visualize == True:
        visualize_predictions(y_true=y_true_orig, y_pred=y_pred_orig, mode=mode, logdir=logdir)
        visualize_filters(model, logdir=logdir)
        plot_losses(history, logdir=logdir)


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
    #TODO: Get Attributes for eval
    attributes = pd.DataFrame(columns=['category', 'entering', 'exiting'])

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
