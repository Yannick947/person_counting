import os
import urllib.request

import tensorflow as tf
from tensorflow.python.keras.models import load_model

from evaluation.evaluate import create_accuracy_rescaled, hard_tanh, create_mae_rescaled
from inference.detection_inference import detect_video
from src import PROJECT_ROOT

COUNTER_MODEL_PATH = os.path.join(PROJECT_ROOT)
SCALE = 0.066  # Calculated by 1 / (Maximum number of persons per video) for the trained model
sample_bus_video_link = "https://drive.google.com/file/d/1JfRU1FasPynvUMk-LAqQXubAWL2gV1D9/view?usp=sharing"

BUS_VIDEO_NAME = 'bus_video.avi'
urllib.request.urlretrieve(sample_bus_video_link, )
local_bus_video_path = os.path.join(os.getcwd(), BUS_VIDEO_NAME)


def main():
    detected_video_frame = detect_video(video_path=local_bus_video_path)
    person_counter = load_model(
        filepath=COUNTER_MODEL_PATH,
        custom_objects={
            "tf": tf,
            "hard_tanh": hard_tanh,
            "acc_rescaled": create_accuracy_rescaled(SCALE),
            "mae_rescaled": create_mae_rescaled(SCALE),
        })




if __name__ == '__main__':
    main()
