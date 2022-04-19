import os
import sys
from time import gmtime, strftime

import tensorflow as tf
from sklearn.model_selection import ParameterSampler
from tensorflow import keras
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import (
    Dense,
    MaxPooling2D,
    Conv2D,
    AveragePooling2D,
    LSTM,
    Lambda,
    Input,
)

from src.data_generators import (
    data_generator_cnn_classification as dg_cnn_cls,
)
from src.evaluation.evaluate_cls import evaluate_run_cls, f1
from src.models.cnn_regression import squeeze_dim3, squeeze_dim3_shape
from src.models.model_argparse import parse_args
from src.utils.hyperparam_utils import (
    create_callbacks,
    get_optimizer,
    get_static_hparams,
)
from src.utils.preprocessing import get_filtered_lengths


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    hparams_samples = get_samples(args)

    for sample in hparams_samples:
        timestep_num, feature_num = get_filtered_lengths(args.top_path, sample)

        datagen_train, datagen_test = dg_cnn_cls.create_datagen(
            args.top_path,
            sample,
            args.label_file,
            args.augmentation_factor,
            args.filter_hour_above,
            args.filter_category_noisy,
        )

        logdir = os.path.join(args.topdir_log + "_cnn_" + strftime("%Y_%b_%d_%H_%M_%S", gmtime()))
        model = create_cnn(timestep_num, feature_num, sample, datagen_test.num_classes)
        history, model = train(
            model=model,
            datagen_train=datagen_train,
            logdir=logdir,
            hparams=sample,
            datagen_test=datagen_test,
            epochs=args.epochs,
        )

        evaluate_run_cls(
            model,
            history,
            datagen_test,
            mode="test",
            logdir=logdir,
            visualize=True,
            top_path=args.top_path,
        )
        evaluate_run_cls(
            model,
            history,
            datagen_train,
            mode="train",
            logdir=logdir,
            visualize=True,
            top_path=args.top_path,
        )


def get_samples(args):
    """Get different samples of hyperparameters

    Arguments:
        args: Arguments read from command line

    returns list of samples for hyperparameters out of given hparam space
    """
    from scipy.stats import loguniform

    # Put values multiple times into list to increase probability to be chosen
    param_grid = {
        "pooling_type": ["avg", "max"],
        "kernel_size": [i for i in range(3, 5)],
        "kernel_number": [i for i in range(2, 5)],
        "pool_size_y": [2],
        "pool_size_x": [2],
        "learning_rate": loguniform.rvs(a=1e-6, b=1e-4, size=100000),
        "optimizer": ["Adam"],
        "layer_number": [1, 2, 3],
        "batch_normalization": [False, True],
        "regularization": [0],
        "filter_cols_upper": [i for i in range(15, 35)],
        "filter_cols_lower": [i for i in range(15, 30)],
        "batch_size": [32, 64, 128],
        "units": [i for i in range(2, 14)],
        "loss": ["categorical_crossentropy"],
    }

    randint = int(tf.random.uniform(shape=[], minval=0, maxval=10, dtype=tf.dtypes.int32, seed=None))
    dynamic_params = list(ParameterSampler(param_grid, n_iter=args.n_runs, random_state=randint))

    static_params = get_static_hparams(args)

    return [{**dp, **static_params} for dp in dynamic_params]


def train(
        model,
        datagen_train,
        logdir=None,
        hparams=None,
        datagen_test=None,
        workers=16,
        use_multiprocessing=True,
        epochs=50,
):
    """Train a given model with given datagenerator

    Arguments:
        model: Cnn keras model
        datagen_train: Datagenerator for training
        logdir: Path to folder for logging
        hparams: Sample of hyperparameter
        datagen_test: Datagenerator for evaluation during testing
        workers: Number of workers if multiprocessing is true
        use_multiprocessing: Flag if multiprocessing is enabled
        epochs: Number of epochs to train
    """

    print("Actual model is using following hyper-parameters:")
    for key in hparams.keys():
        print(key, ": ", hparams[key])

    # Add num params parameter only for logging purposes
    hparams["number_params"] = model.count_params()

    history = model.fit_generator(
        validation_steps=int(len(datagen_test)),
        generator=datagen_train.datagen(),
        validation_data=datagen_test.datagen(),
        steps_per_epoch=int(len(datagen_train)),
        epochs=epochs,
        verbose=1,
        shuffle=True,
        callbacks=create_callbacks(logdir, hparams, save_best=True),
        use_multiprocessing=use_multiprocessing,
        workers=workers,
        max_queue_size=16,
    )
    return history, model


def create_cnn(timesteps, features, hparams, num_classes):
    """Creates a convolutional nn with architecture defined in hparams

    Arguments:
        timesteps: Amount of timesteps in input data
        features: Number of columns (features)
        hparams: Sample of hyperparameters
        num_classes: Number of classes for entering persons

        returns keras model with cnn architecture
    """

    def create_conv_layer():
        return Conv2D(
            hparams["kernel_number"],
            hparams["kernel_size"],
            use_bias=True,
            activation="relu",
            kernel_initializer=keras.initializers.glorot_normal(),
            padding="same",
        )

    if hparams["pooling_type"] == "avg":

        def create_pool_layer():
            return AveragePooling2D(pool_size=(hparams["pool_size_x"], hparams["pool_size_y"]))

    else:

        def create_pool_layer():
            return MaxPooling2D(pool_size=(hparams["pool_size_x"], hparams["pool_size_y"]))

    layers = list()
    layers.append(Input(shape=(timesteps, features, 1)))

    for _ in range(hparams["layer_number"]):
        try:
            layers.append(create_conv_layer()(layers[-1]))
            layers.append(create_conv_layer()(layers[-1]))
            layers.append(create_pool_layer()(layers[-1]))

        except ValueError:
            # Creation failed, hparam must be adjusted for logging
            hparams["layer_number"] -= 1
            print(
                "Tried to create a Pool Layer that is not possible to create,",
                "because it would lead to negative dimensions. Creation was skipped",
            )

    # Squeeze 4th dimension and pass to time-series module
    layers.append(Lambda(squeeze_dim3, output_shape=squeeze_dim3_shape)(layers[-1]))
    layers.append(
        LSTM(
            units=num_classes,
            activation="relu",
            return_sequences=False,
            kernel_regularizer=keras.regularizers.l2(hparams["regularization"]),
        )(layers[-1])
    )
    layers.append(
        Dense(
            num_classes,
            activation="softmax",
            kernel_regularizer=keras.regularizers.l2(hparams["regularization"]),
        )(layers[-1])
    )

    model = Model(layers[0], layers[-1])

    optimizer = get_optimizer(hparams["optimizer"], hparams["learning_rate"])
    model.compile(
        loss=hparams["loss"],
        metrics=[f1, keras.metrics.categorical_accuracy],
        optimizer=optimizer,
    )
    model.summary()

    return model


if __name__ == "__main__":
    main()
