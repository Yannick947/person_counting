import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from vis.visualization import visualize_cam
from vis.utils import utils
from keras import activations
from vis.visualization import visualize_saliency, overlay

from person_counting.models import cnn_regression as cnn
from person_counting.data_generators import data_generators as dgv
from person_counting.data_generators import data_generator_cnn as dgv_cnn
from person_counting.utils.preprocessing import get_filtered_lengths

"""
CBAM activation filter visualization is applied for further understanding 
of the attention of the cnn
"""
label_file = "pcds_dataset_labels_united.csv"
LABEL_HEADER = ["file_name", "entering", "exiting", "video_type"]
SNAP_PATH = "C:/Users/Yannick/Google Drive/person_counting/tensorboard/cnn_regression/warm_start/best"
top_path = "C:/Users/Yannick/Google Drive/person_detection/pcds_dataset_detections/pcds_dataset_detected/"
workers = 0
multi_processing = False

# Set the layer index which you want to visualize
LAYER_IDX = 7  # 11 -> 1x1 conv, 9 -> last convulitional layer



def main():
    hparams, timestep_num, feature_num = get_best_hparams(top_path)

    datagen_train, _, datagen_test = dgv_cnn.create_datagen(
        top_path=top_path, sample=hparams, label_file=label_file, filter_hour_above=10
    )
    cnn_model = cnn.create_cnn(
        timestep_num, feature_num, hparams, datagen_train.label_scaler.scale_, snap_path=SNAP_PATH
    )


    cnn_model.layers[LAYER_IDX].activation = activations.linear
    # cnn_model = utils.apply_modifications(cnn_model)

    for i in range(len(datagen_test)):
        img, label = datagen_test[i]

        if datagen_test.label_scaler.inverse_transform(label)[0] < 8:
            continue

        print("Predicted number of persons: ", cnn_model.predict(img) / datagen_test.label_scaler.scale_)
        print("Correct number of persons: ", datagen_test.label_scaler.inverse_transform(label)[0])

        f, ax = plt.subplots(1, 3, dpi=1200)

        grads = visualize_cam(
            model=cnn_model, LAYER_IDX=LAYER_IDX, filter_indices=None, seed_input=img[0], backprop_modifier=None
        )

        jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
        ax[0].imshow(jet_heatmap[:, 0:450, :, 0])
        ax[1].imshow(img[0, 0:450, :, 0])
        ax[2].imshow(img[0, 0:450, :, 1])
        for i in range(len(ax)):
            ax[i].get_xaxis().set_visible(False)
            ax[i].get_yaxis().set_visible(False)

        plt.xticks(ticks=None)
        plt.yticks(ticks=None)
        plt.show()


def get_best_hparams(top_path):
    """Set best hyperparameter set from prior tuning session

    Arguments:
        top_path: Parent directory where csv files shall be searched
    """

    hparams = {
        "kernel_number": 5,
        "batch_size": 1,
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
