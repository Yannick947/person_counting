import argparse

def parse_args(args):
    """ Parse the arguments.
    """

    parser = argparse.ArgumentParser(description='Parsing Arguments for training model on video detected data')
    parser.add_argument('--n-runs',                 help='Number of samples which shall be chosen randomly from hyperparameter serach space', default=50, type=int)
    parser.add_argument('--topdir-log',             help='The parent directory for logging tensorboard data', default='tensorboard/')
    parser.add_argument('--top-path',               help='The parent directory where csv feature and label files are stored', default='C:/Users/Yannick/Google Drive/person_detection/pcds_dataset_detections/pcds_dataset_detected_100pcdsimgs_frontinonly/')
    parser.add_argument('--label-file',             help='The name of the label file', default='pcds_dataset_labels_united.csv')
    parser.add_argument('--filter-rows-factor',     help="The factor for filtering rows (3 indicates every third row will be left after filtering", type=int)
    parser.add_argument('--filter-cols-upper',      help='Removes this total amount of columns from the upper boundary of the images',  type=int)
    parser.add_argument('--filter-cols-lower',      help='Removes this total amount of columns from the lower boundary of the images',  type=int)
    parser.add_argument('--filter-cols-factor',     help='After removing cols from the ends, filter the csv files with this factor (same as for rows)', type=int)
    parser.add_argument('--batch-size',             help='The batch size for training', type=int)
    parser.add_argument('--augmentation-factor',    help='Factor determines how much augmentation will be done [0,1]. If chosen to 1, every detection pixel will be moved randomly 1 step', default=0, type=float)
    parser.add_argument('--filter-category-noisy',  help='If set to true, category noisy will not be loaded to the dataset', action='store_true', default=False)
    parser.add_argument('--filter-hour-above',      help='Filter videos which are after this hour during the day due to darkness', type=int, default=0)
    parser.add_argument('--epochs',                 help='Number of epochs to train', type=int, default=50)
    parser.add_argument('--warm-start-path',        help='Starting from the best current available snapshot in the provided folder', default='None')
    parser.add_argument('--schedule-step',          help='The epoch when the learning rate shall be scheduled', type=int, default=0)
    return parser.parse_args(args)

def check_args(args): 
    if args.augmentation_factor > 0: 
        print('Care using Augmentation, significant performance deacreases might be possible depneding on the size of your data')
