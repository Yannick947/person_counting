import os

import numpy as np
import pandas as pd
from tensorflow.keras import backend as K

from src.evaluation.evaluate import parse_model
from src.data_generators.data_generators import get_entering, get_video_class
from src.utils.visualization_utils import plot_losses, visualize_predictions, visualize_filters

LABEL_HEADER = ["file_name", "entering", "exiting", "video_type"]
CATEGORY_MAPPING = {0: "normal_uncrowded", 1: "normal_crowded", 2: "noisy_uncrowded", 3: "noisy_crowded"}


# TODO: Update to new input format
def evaluate_run_cls(model, history, gen, mode, logdir, top_path, visualize=True):
    """Evaluate a run of certain hyperparameters
    Arguments:
        model: Last model which was created during training
        history: Keras history object which was created during training
        gen: Generator for data
        mode: Mode (training or testing)
        logdir: Directory where logging is done
        top_path: Parent directory where is logged to
        visualize: Flag indicating if plots shall be created and saved
    """

    # Search for best model in logdir if existing
    model = parse_model(model, logdir)
    model.compile(optimizer="adam", loss=f1)

    y_pred_orig, y_true_orig, video_categories = get_predictions(model, gen, top_path)

    evaluate_predictions(
        history,
        y_pred_orig,
        y_true_orig,
        model=model,
        visualize=visualize,
        mode=mode,
        logdir=logdir,
        video_categories=video_categories,
    )


def evaluate_predictions(history, y_pred, y_true, visualize, model, mode="Test", logdir=None, video_categories=None):
    """Evaluate predictions of the best model
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

    """

    print_stats(y_pred, y_true, mode)

    if visualize == True:
        visualize_predictions(y_true=y_true, y_pred=y_pred, mode=mode, logdir=logdir, video_categories=video_categories)
        visualize_filters(model, logdir=logdir)
        plot_losses(history, logdir=logdir)


def get_stats(y_true, predictions):
    """Gets stats for GT and predictions"""

    difference = 0
    difference_dummy = 0
    mean_ground_truth = sum(y_true) / len(y_true)

    for prediction, y in zip(predictions, y_true):
        difference += abs(prediction - y)
        difference_dummy += abs(mean_ground_truth - y)

    mean_difference_pred = difference / len(predictions)
    mean_difference_dummy = difference_dummy / len(y_true)

    return mean_difference_pred, mean_difference_dummy, mean_ground_truth


def get_predictions(model, gen, top_path):
    """Generate predictions from generator and model

    Arguments:
        model: Model which shall predict
        gen: Generator to load data
    returns predictions and corresponding ground_truth
    """
    gen.reset_label_states()
    gen.reset_file_names_processed()
    gen.batch_size = 1

    feature_frames = list()
    y_true_orig = list()
    video_type = list()
    # TODO: Get Attributes for eval

    df_y = pd.read_csv(os.path.join(top_path, gen.label_file), header=None, names=LABEL_HEADER)

    for file_name in gen.file_names:
        try:
            x = pd.read_csv(file_name, header=None)
            x = gen.preprocessor.preprocess_features(x)
            feature_frames.append(x)

            y = get_entering(file_name, df_y)
            y_true_orig.append(int(y.values[0]))
            video_category = get_video_class(file_name, df_y).values[0]
            video_type.append(CATEGORY_MAPPING[video_category])

        except:
            print("Failed reading feature file for ", file_name)
            continue

    # Reshape features
    feature_frames = np.dstack(feature_frames)
    feature_frames = np.moveaxis(feature_frames, 2, 0)[..., np.newaxis]

    y_pred = model.predict(feature_frames)
    y_pred_squeezed = np.zeros(shape=y_pred.shape[0])

    for i in range(y_pred.shape[0]):
        val_pred = int(np.argmax(y_pred[i, :], axis=0))
        y_pred_squeezed[i] = val_pred

    return np.squeeze(y_pred_squeezed), np.squeeze(np.array(y_true_orig)), np.array(video_type)


def print_stats(predictions, y_true, mode):
    """Print stats of predictions and ground_truth

    Arguments:
        predictions: Predictions of estimator as numpy array
        y_true: Ground truth as numpy array
        mode: Mode (train or test) as string
    """

    mean_difference, mean_difference_dummy, mean_ground_truth = get_stats(y_true, predictions)
    y_true = y_true.astype(float)

    print("\nFor mode: ", mode)
    print("Mean of ground truth: ", mean_ground_truth)
    print("Mean of predictions: ", sum(predictions) / len(predictions))
    print("\nStd of ground truth: ", np.std(y_true))
    print("Std of predicitons: ", np.std(predictions))
    print("\nMean difference between ground truth and predictions is: ", mean_difference)
    print(
        "Mean difference between dummy estimator (voting always for mean of ground truth) and ground truth: ",
        mean_difference_dummy,
    )


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
