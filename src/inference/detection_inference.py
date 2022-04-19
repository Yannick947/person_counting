import random
from logging import getLogger
from typing import Optional, List, Tuple

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from matplotlib import pyplot as plt

# Use the inception resnet for more accurate but significantly slower detections
# module_handle = r"C:\Users\Yannick\Downloads\faster_rcnn_openimages_v4_inception_resnet_v2_1"
module_handle = r"C:\Users\Yannick\Downloads\openimages_v4_ssd_mobilenet_v2_1"
CLASSES_TO_FILTER = {"Person", "Man", "Woman"}

detector = hub.load(module_handle).signatures['default']

UPPER_VIDEO_LENGTH = 700  # TODO: Parse this parameter from the counting model
PREDICTION_THRESHOLD = 0.5
PREDICTION_THRESHOLD_INCEPTION_RESNET = 0.15
DOWNSCALE_FACTOR_Y = 3  # Was empirically set to 3 since there is quite no information loss and sparsity is reduced
logger = getLogger(__name__)

out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (320, 240))


def generate_detection_frame(video_path: str) -> np.array:
    """ Generate the csv files by passing videos to a model
    """

    vcapture = cv2.VideoCapture(video_path)
    num_frames = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))

    # create empty arrs for predictions.
    detections_y = create_zeroed_array(vcapture=vcapture)
    detections_x = create_zeroed_array(vcapture=vcapture)

    # Fill arrs with predictions
    detections_x, detections_y = fill_pred_image(detections_x=detections_x,
                                                 detections_y=detections_y,
                                                 vcapture=vcapture)

    stacked_arr = np.stack([detections_x, detections_y], axis=-1, out=None)

    print('Finished video detection')
    vcapture.release()
    return stacked_arr


def create_zeroed_array(vcapture, filter_upper_frames: int = UPPER_VIDEO_LENGTH, fps: Optional[int] = None):
    frame_rate = int(vcapture.get(cv2.CAP_PROP_FPS))
    height = int((vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT)) / DOWNSCALE_FACTOR_Y)

    if fps == None:
        detections_y_length = int(filter_upper_frames)

    else:
        print('Care about using fps argument, not tested yet -> trial mode')
        detections_y_length = int(filter_upper_frames * fps / frame_rate)

    return np.zeros(shape=(detections_y_length, height))


def fill_pred_image(detections_x, detections_y, vcapture):
    """ Fill an array with predictions for time and place of a model
    """
    success = True
    frame_index = 0
    frame_rate = int(vcapture.get(cv2.CAP_PROP_FPS))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    logger.info(f"Video properties:\nImage height: {height}, Image width: {width}, frame rate: {frame_rate}.")

    while success:
        frame_index += 1
        success, image = vcapture.read()

        if success:
            centers, scores = get_person_centers_resnet(image=image)
            for center, score in zip(centers, scores):

                # fill arr with probability at the center of y axis, consider rezizing with args_downscale_factor_y
                try:
                    t = int(frame_index)
                    x = int(center[0] * (height / width) / DOWNSCALE_FACTOR_Y)
                    y = int(center[1] / DOWNSCALE_FACTOR_Y)

                    # If position already contains a pixel due to downscaling, search new position along movement axis
                    if detections_y[t, y] == 0:
                        detections_y[t, y] = score
                        detections_x[t, x] = score

                    else:
                        t, y = get_destination(detections_y, (t, y), axis='y')
                        t, x = get_destination(detections_x, (t, x), axis='x')
                        detections_y[t, y] = score
                        detections_x[t, x] = score
                except ValueError as exc:
                    print(exc)
                    print(f'Detection out of bounds, t: {t}, y: {y}, x: {x}')

    return detections_y, detections_x


def get_person_centers_resnet(image: np.array) -> Tuple[List[Tuple[int, int]], List[float]]:
    converted_img = tf.image.convert_image_dtype(image, tf.float32)[tf.newaxis, ...]
    result = detector(converted_img)
    result = {key: value.numpy() for key, value in result.items()}
    boxes, class_entities, scores = [], [], []
    for i in range(len(result["detection_boxes"])):
        if is_valid_detection(result=result, i=i):
            print(result["detection_boxes"][i], result["detection_class_entities"][i], result["detection_scores"][i])
            boxes.append(to_absolute_centers(result["detection_boxes"][i], image=image))
            class_entities.append(result["detection_class_entities"][i])
            scores.append(min(1.0, result["detection_scores"][i] + 0.7))

    return boxes, scores


def to_absolute_centers(detection_box: List[float], image: np.array) -> Tuple[int, int]:
    ymin, xmin, ymax, xmax = detection_box[0], detection_box[1], detection_box[2], detection_box[3]
    im_height = image.shape[0]
    im_width = image.shape[1]
    absolute_box = (xmin * im_width, ymin * im_height, xmax * im_width, ymax * im_height)
    center = get_center_from_absolute_box(box=absolute_box)
    return center


def is_valid_detection(result: dict, i: int) -> bool:
    return result["detection_class_entities"][i].decode('UTF-8') in CLASSES_TO_FILTER and result["detection_scores"][
        i] > PREDICTION_THRESHOLD_INCEPTION_RESNET


def get_center_from_absolute_box(box: tuple) -> Tuple:
    """ Expecting the box to be absolute of format xA, yA, xB, yB.
    """
    (xA, yA, xB, yB) = box
    width = abs(xB - xA)
    height = abs(yB - yA)

    center_x = min(xA, xB) + width / 2
    center_y = min(yA, yB) + height / 2
    return center_x, center_y


def get_destination(arr, old_position, axis):
    '''Get new destination for pixel around the old position

    Arguments:
        arr: Dataframe with current pixels
        old_position: position where pixel would be placed initially

    returns t and y coordinate for new position of pixel
    '''
    # Directions in which pixel can be moved
    if axis == 'y':
        directions = [(-1, -1), (1, 1)]
    else:
        directions = [(1, 0), (-1, 0)]

    for _ in range(len(directions)):
        direction = random.sample(directions, 1)[0]

        x = old_position[0] + direction[0]
        y = old_position[1] + direction[1]
        new_pos = (x, y)

        if (new_pos[0] < 0) or (new_pos[0] >= arr.shape[0]) or (new_pos[1] < 0) or (new_pos[1] >= arr.shape[1]):
            continue

        if arr[new_pos[0], new_pos[1]] == 0:
            return new_pos[0], new_pos[1]

    # if all position are already a detection pixel, return the old value
    return old_position


def detect_video(video_path: str) -> np.array:
    detection_frame = generate_detection_frame(video_path=video_path)
    return detection_frame


if __name__ == '__main__':
    detected_frame = detect_video(video_path="./bus_video.avi")

    plt.imshow(detected_frame[:, :, 0])
    plt.show()

    plt.imshow(detected_frame[:, :, 1])
    plt.show()

    np.save("./detected_video.npy", arr=detected_frame)
