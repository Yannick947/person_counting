import math
import statistics

import numpy as np 
from keras.models import load_model
import tensorflow as tf

from person_counting.utils.visualization_utils import plot_losses, visualize_predictions, visualize_filters


def evaluate_model(model, history, gen, mode, logdir, visualize=True):

    if type(model) == list:
        model = load_model(model) 
    
    pred_test, y_true_test = get_predictions(model, gen)
    evaluate_predictions(history, pred_test, y_true_test, model=model, visualize=visualize, mode=mode)
    mean_difference, mean_difference_dummy, _ = get_stats(y_true_test, pred_test)
    write_custom_metrics(mean_difference, mean_difference_dummy, logdir)

def write_custom_metrics(mean_difference, mean_difference_dummy, logdir):
    try:
        file_writer = tf.summary.create_file_writer(logdir)
        file_writer.set_as_default()
        tf.summary.scalar('mean_difference', tf.squeeze(float(mean_difference)))
        tf.summary.scalar('mean_difference_dummy', tf.squeeze(float(mean_difference_dummy)))
    except: 
        print('failed writing summary for evaluation')
        
def evaluate_predictions(history, predictions, y_true, visualize, mode, model, ipython_mode=True):
    print_stats(predictions, y_true, mode)
    if visualize == True:
        plot_losses(history, ipython_mode)
        visualize_predictions(predictions, y_true, ipython_mode=ipython_mode)
        visualize_filters(model)

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
    print('\nMean difference between Ground truth and predictions is: ', mean_difference)
    print('Mean of dummy estimator, voting for mean of ground truth: ', mean_difference_dummy)

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

    def lossFunction(y_true, y_pred):    
        difference = 0

        for prediction, y in zip(y_pred, y_true):
            difference += abs(prediction - y)
    
        mean_difference_pred = difference / len(y_pred)
        return mean_difference_pred * scale_factor

    return lossFunction

def get_predictions(model, gen): 
    '''Generate predictions from generator and model

    Arguments: 
        model: Model which shall predict
        gen: Generator to load data
    returns predictions and corresponding ground_truth
    '''
    gen.reset_label_states()
    gen.reset_file_names_processed()
    gen.batch_size = 1

    predictions = model.predict_generator(generator=gen.datagen(), steps=len(gen))
    if gen.scaler is not None: 
        predictions_inverse = gen.scaler.scaler_labels.inverse_transform(predictions)
        y_true = gen.scaler.scaler_labels.inverse_transform(gen.get_labels())
        if len(y_true) != len(predictions): 
            print('Length of predictions and ground truth doesnt match!')
            y_true = y_true[:len(predictions_inverse)]
        print('Number of test samples for evaluation', len(predictions))
        print('Number of unique files used for evaluation, ', len(set(gen.get_file_names_processed())))
        return predictions_inverse, y_true

    else: 
        y_true = gen.get_labels()
        if len(y_true) != len(predictions): 
            y_true = y_true[:len(predictions_inverse)]
        print('No scaler found, calculations done on data given')
        return predictions, y_true

