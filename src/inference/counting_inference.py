import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model

from evaluation.evaluate import create_accuracy_rescaled, hard_tanh, create_mae_rescaled
#from inference.detection_inference import detect_video
from src import PROJECT_ROOT

COUNTER_MODEL_PATH = os.path.join(os.path.dirname(PROJECT_ROOT), "models/person_counter.hdf5")
SCALE = 0.066  # Calculated by 1 / (Maximum number of persons per video) for the trained model
sample_bus_video_link = "https://drive.google.com/file/d/1JfRU1FasPynvUMk-LAqQXubAWL2gV1D9/view?usp=sharing"

BUS_VIDEO_NAME = 'bus_video.avi'
local_bus_video_path = os.path.join(os.getcwd(), BUS_VIDEO_NAME)

logger = logging.getLogger(__name__)
# urllib.request.urlretrieve(sample_bus_video_link, local_bus_video_path)


def main():
    detected_video_frame = get_detected_video_frame(video_path=local_bus_video_path)
    detected_video_frame = np.expand_dims(detected_video_frame, axis=0)
    person_counter = load_model(
        filepath=COUNTER_MODEL_PATH,
        custom_objects={
            "tf": tf,
            "hard_tanh": hard_tanh,
            "acc_rescaled": create_accuracy_rescaled(SCALE),
            "mae_rescaled": create_mae_rescaled(SCALE),
        })
    persons_entering = np.squeeze(person_counter.predict(detected_video_frame)).item() / SCALE

    detected_video_frame_flipped = np.flip(detected_video_frame, axis=1)
    exiting_persons = np.squeeze(person_counter.predict(detected_video_frame_flipped)).item() / SCALE
    logger.info(f"Number of persons that entered into the bus: {persons_entering}")
    logger.info(f"Number of persons that exited the bus: {exiting_persons}.")


def get_detected_video_frame(video_path=local_bus_video_path) -> np.array:
    if os.path.exists(local_bus_video_path):
        return np.load("./detected_video.npy")
        #return detect_video(video_path=video_path)
    else:
        logger.info("Video was not downloaded, using previously detected sample video detections. ")
        return np.load("./detected_video.npy")


if __name__ == '__main__':
    main()
