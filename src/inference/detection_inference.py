import os
import random

import numpy as np
from tensorflow.keras.models import load_model

from inference import custom_object_person_detector
from src import PROJECT_ROOT


def generate_csv(video_path, model, args, filter_upper_frames=1000):
    """ Generate the csv files by passing videos to a model
    """

    vcapture = cv2.VideoCapture(video_path)
    num_frames = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))

    # create empty arrs for predictions.
    detections_y = create_zeroed_array(args, vcapture, filter_upper_frames)
    detections_x = create_zeroed_array(args, vcapture, filter_upper_frames)

    # Fill arrs with predictions
    detections_x, detections_y = fill_pred_image(model, detections_x, detections_y, vcapture, args)

    stacked_arr = np.stack([detections_x, detections_y], axis=-1, out=None)

    print('Finished video detection')
    vcapture.release()
    return stacked_arr


def create_zeroed_array(fps: int, vcapture, filter_upper_frames: int, downscale_factor_y: float):
    frame_rate = int(vcapture.get(cv2.CAP_PROP_FPS))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps == None:
        detections_y_length = int(filter_upper_frames)

    else:
        print('Care about using fps argument, not tested yet -> trial mode')
        detections_y_length = int(filter_upper_frames * fps / frame_rate)

    return np.zeros(shape=(detections_y_length, int(height / downscale_factor_y)))


def fill_pred_image(model, detections_x, detections_y, vcapture, args):
    """ Fill an array with predictions for time and place of a model
    """
    success = True
    frame_index = 0
    frame_rate = int(vcapture.get(cv2.CAP_PROP_FPS))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))

    if args.fps:
        time_scale_factor = int(frame_rate / args.fps)
    else:
        time_scale_factor = 1

    while success:
        frame_index += 1
        success, image = vcapture.read()

        if success and ((frame_index % time_scale_factor) == 0):
            image = preprocess_image(image)

            # 800 and 1300 are values usally used during training, adjust if used differently
            image, scale = resize_image(image, min_side=800, max_side=1333)
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
            boxes /= scale

            for box, score, label in zip(boxes[0], scores[0], labels[0]):

                # scores are sorted so we can break
                if score < args.predict_threshold:
                    break

                # If model predicts more classes than we want to have, filter here
                if (label != 0):
                    continue

                b = box.astype(int)

                # fill arr with probability at the center of y axis, consider rezizing with args_downscale_factor_y
                try:
                    t = int(frame_index / time_scale_factor)
                    x = int((b[0] + b[2]) * (height / width) / args.downscale_factor_y / 2)
                    y = int((b[3] + b[1]) / args.downscale_factor_y / 2)

                    # If position already contains a pixel due to downscaling, search new position along movement axis
                    if detections_y[t, y] == 0:
                        detections_y[t, y] = score
                        detections_x[t, x] = score

                    else:
                        t, y = get_destination(detections_y, (t, y), axis='y')
                        t, x = get_destination(detections_x, (t, x), axis='x')
                        detections_y[t, y] = score
                        detections_x[t, x] = score
                except:
                    print('Detection out of bounds, t: {}, y: {}, x: {}'.format(t, y, x))

    return detections_y, detections_x


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


def create_zeroed_array(args, vcapture, filter_upper_frames):
    frame_rate = int(vcapture.get(cv2.CAP_PROP_FPS))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.fps == None:
        detections_y_length = int(filter_upper_frames)

    else:
        print('Care about using fps argument, not tested yet -> trial mode')
        detections_y_length = int(filter_upper_frames * args.fps / frame_rate)

    return np.zeros(shape=(detections_y_length, int(height / args.downscale_factor_y)))


def detect_video(video_path: str) -> np.array:
    model_path = os.path.join(os.path.dirname(PROJECT_ROOT), "models", "person_detector_retinanet_resnet50.h5")
    person_detector = load_model(model_path,
                                 custom_objects=custom_object_person_detector)
    detection_frame = generate_csv(video_path=video_path, model=person_detector)
    return detection_frame


if __name__ == '__main__':
    detect_video(video_path="./bus_video.avi")
