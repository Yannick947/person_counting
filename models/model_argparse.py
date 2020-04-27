import argparse

def parse_args(args):
    """ Parse the arguments.
    """

    parser = argparse.ArgumentParser(description='Parsing Arguments for training model on video detected data')
    parser.add_argument('--n-runs',             help='Number of samples which shall be chosen randomly from hyperparameter serach space', default=50, type=int)
    parser.add_argument('--topdir-log',         help='The parent directory for logging tensorboard data', default='tensorboard/')
    parser.add_argument('--top-path',           help='The parent directory where csv feature and label files are stored', default='C:/Users/Yannick/Google Drive/person_detection/bus_videos/pcds_dataset_detected/')
    parser.add_argument('--label-file',         help='The name of the label file', default='pcds_dataset_labels_united.csv')
    parser.add_argument('--filter-rows-factor', help="The factor for filtering rows (3 indicates every third row will be left after filtering", default=2, type=int)
    parser.add_argument('--filter-cols-upper',  help='Removes this total amount of columns from the upper boundary of the images', default=10, type=int)
    parser.add_argument('--filter-cols-lower',  help='Removes this total amount of columns from the lower boundary of the images', default=25, type=int)
    parser.add_argument('--filter-cols-factor', help='After removing cols from the ends, filter the csv files with this factor (same as for rows)', default=1, type=int)
    parser.add_argument('--batch-size',         help='The batch size for training', default=16, type=int)
    return parser.parse_args(args)