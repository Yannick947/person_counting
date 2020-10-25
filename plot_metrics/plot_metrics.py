import os

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib as mpl
import matplotlib.pyplot as plt

LOGGING_METRIC = "epoch_acc_rescaled"
Y_LABEL = "Accuracy"
LOG_DIR_SUP = "C:/Users/Yannick/Google Drive/person_counting/tensorboard/cnn_regression/cold_start"
LOG_DIRS = {"best model": "best", "second best model": "second_best", "third best model": "third_best"}
SMOOTH_NUM = 1

plt.style.use("ggplot")


def plot_tensorflow_log(log_paths):
    figure = plt.figure(figsize=(8, 8))
    for net, path in log_paths.items():

        event_acc = EventAccumulator(path)
        event_acc.Reload()
        print("Available metrics: ", event_acc.Tags())
        parsed_metrics = list()
        for metric in event_acc.Tags()["scalars"]:

            if (metric in LOGGING_METRIC) and (metric not in parsed_metrics):
                metric_values = event_acc.Scalars(metric)

                steps = len(metric_values)
                x = np.arange(steps)
                y = np.zeros([steps, 2])
                y_smooth = np.zeros([steps, 2])

                for i in range(steps):
                    y[i] = metric_values[i][2]  # value

                    if i > SMOOTH_NUM:
                        y_smooth[i] = np.average(y[i - SMOOTH_NUM : i])
                    else:
                        y_smooth[i] = np.average(y[0:i])

                if ("validation" in path) or ("mAP" in metric):
                    # validation folder is the last which is parsed so show the plot and save it
                    plt.plot(x, y_smooth[:], label=f'validation Accuracy for {net[:net.find("validation")]}')
                    parsed_metrics.append(metric)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.ylabel(Y_LABEL)
    plt.xlabel("epoch")
    plt.show()
    # figure.savefig(f'./plot_metrics/pcds_bestvalidations_{LOGGING_METRIC}.png', type='png', dpi=2400)
    mpl.use("pgf")
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )
    figure.savefig(f"./plot_metrics/pcds_bestvalidations_{LOGGING_METRIC}.pgf", type="pgf")


def get_log_paths():
    log_paths = dict()

    for net, log_dir in LOG_DIRS.items():
        for root, _, files in os.walk(os.path.join(LOG_DIR_SUP, log_dir)):
            for file_name in files:
                if file_name[-3:] == ".v2":

                    if "validation" in root:
                        log_paths[net + " validation"] = os.path.join(root, file_name)
                    elif not "train" in root:
                        log_paths[net + " train_val"] = os.path.join(root, file_name)
                    else:
                        log_paths[net + " train"] = os.path.join(root, file_name)
    return log_paths


if __name__ == "__main__":
    log_paths = get_log_paths()
    plot_tensorflow_log(log_paths)
