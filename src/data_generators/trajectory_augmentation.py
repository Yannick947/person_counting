import math
import random
import time

import numpy as np


def time_measure(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000))
        return result

    return timed


# For testing the performance, comment in the decorator to measure time for execution
# @time_measure
def augment_trajectory(arr, aug_factor=0.1):
    """Augment a trajectory by moving certain pixels which are detections
    into random directions
    """
    # Get list of indices and sample subset
    indices = get_indices_detections(arr)
    indices_sampled = random.sample(indices, math.ceil(len(indices) * aug_factor))
    random.shuffle(indices_sampled)

    for index_tuple in indices_sampled:
        new_dest, indices = get_destination(arr, index_tuple, indices)
        arr = move_pixel(arr, index_tuple, new_dest)

    return arr


def get_destination(arr, old_position, indices):
    """Get destinations where pixel can be moved to"""
    directions = [(1, 1), (-1, -1), (1, 1), (-1, -1), (0, -1), (0, 1), (1, 0), (-1, 0)]

    for _ in range(len(directions)):
        direction = random.sample(directions, 1)[0]

        x = old_position[0] + direction[0]
        y = old_position[1] + direction[1]
        z = old_position[2]
        new_pos = (x, y, z)

        if (new_pos[0] < 0) or (new_pos[0] >= arr.shape[0]) or (new_pos[1] < 0) or (new_pos[1] >= arr.shape[1]):
            continue

        if new_pos not in indices:
            index_old_pos = indices.index(old_position)
            indices[index_old_pos] = new_pos
            return new_pos, indices

    # if all position are already a detection pixel, return the old value
    return old_position, indices


def get_indices_detections(arr):
    """Returns list of tuples of indices of detections"""
    nonzeros = np.nonzero(arr)
    detections = list()
    for i in range(len(nonzeros[0])):
        detections.append((nonzeros[0][i], nonzeros[1][i], nonzeros[2][i]))
    return detections


def move_pixel(arr, old_dest, new_dest):
    """ Move pixel to new location """
    save_val = np.copy(arr[old_dest[0], old_dest[1], old_dest[2]])
    arr[old_dest[0], old_dest[1], old_dest[2]] = 0
    arr[new_dest] = save_val

    return arr


def main():
    test_arr = np.random.rand(4, 4, 2)
    test_arr[:, 2, 0] = 0
    print(test_arr, test_arr.shape)
    arr_augmented = augment_trajectory(test_arr, 0.5)
    print(arr_augmented, arr_augmented.shape)


if __name__ == "__main__":
    main()
