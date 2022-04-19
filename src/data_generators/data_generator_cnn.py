from src.data_generators.data_generators import *
from src.utils.preprocessing import apply_file_filters
from src.utils.preprocessing import get_filtered_lengths
from src.utils.scaler import FeatureScaler, LabelScaler


class Generator_CSVS_CNN(Generator_CSVS):
    """
    Generators class to load npy files from
    video folder structre like PCDS Dataset and
    train CNNs

    Arguments (**kwargs)
        length_t            : Length of the feature's DataFrame in time dimension
        length_y            : Length of the feature's DataFrame in y direction
        file_names          : File names to be processed
        filter_cols_upper,  : Amount of columns to be filtered at end and start of DataFrame
        batch_size          : Batch size
        top_path            : Parent path where csv files are contained
        label_file          : Name of the label file

    """

    def __init__(self, *args, **kwargs):
        super(Generator_CSVS_CNN, self).__init__(*args, **kwargs)


def create_datagen(
        top_path,
        sample,
        label_file,
        augmentation_factor=0,
        filter_hour_below=7,
        filter_hour_above=24,
        filter_category_noisy=False,
        supercharge_crowdeds=False,
):
    """
    Creates train and test data generators for lstm network.

    Arguments:
        top_path: Parent directory where shall be searched for training files
        sample: sample of hyperparameters used in this run
        label_file: Name of the label file containing all the labels
        augmentation_factor: Factor how much augmentation shall be done, 1 means
                             moving every pixel for one position
        filter_hour_above: Hour after which videos shall be filtered
        filter_category_noisy: Flag if noisy videos shall be filtered
    """
    # Load filenames and lengths
    length_t, length_y = get_filtered_lengths(top_path, sample)
    train_file_names, validation_file_names, test_file_names = get_file_split(
        top_path, supercharge_crowdeds=supercharge_crowdeds
    )

    # Apply filters
    train_file_names = apply_file_filters(
        df=train_file_names,
        filter_hour_above=filter_hour_above,
        filter_category_noisy=filter_category_noisy,
        filter_hour_below=filter_hour_below,
    )

    validation_file_names = apply_file_filters(
        df=validation_file_names,
        filter_hour_above=filter_hour_above,
        filter_category_noisy=filter_category_noisy,
        filter_hour_below=filter_hour_below,
    )

    test_file_names = apply_file_filters(
        df=test_file_names,
        filter_hour_above=filter_hour_above,
        filter_category_noisy=filter_category_noisy,
        filter_hour_below=filter_hour_below,
    )

    scale_files = pd.concat([train_file_names, validation_file_names, test_file_names])

    print(
        "Dataset contains: \n{} training files \n{} validation files \n{} testing files".format(
            len(train_file_names), len(validation_file_names), len(test_file_names)
        )
    )

    feature_scaler = FeatureScaler(top_path, scale_files, sample)
    label_scaler = LabelScaler(top_path, label_file, scale_files, sample)

    gen_train = Generator_CSVS_CNN(
        length_t=length_t,
        length_y=length_y,
        file_names=train_file_names,
        feature_scaler=feature_scaler,
        label_scaler=label_scaler,
        sample=sample,
        top_path=top_path,
        label_file=label_file,
        augmentation_factor=augmentation_factor,
    )

    # Don't do augmentation here!
    gen_validation = Generator_CSVS_CNN(
        length_t=length_t,
        length_y=length_y,
        file_names=validation_file_names,
        feature_scaler=feature_scaler,
        label_scaler=label_scaler,
        sample=sample,
        top_path=top_path,
        label_file=label_file,
        augmentation_factor=0,
    )

    gen_test = Generator_CSVS_CNN(
        length_t=length_t,
        length_y=length_y,
        file_names=test_file_names,
        feature_scaler=feature_scaler,
        label_scaler=label_scaler,
        sample=sample,
        top_path=top_path,
        label_file=label_file,
        augmentation_factor=0,
    )

    return gen_train, gen_validation, gen_test


def get_file_split(top_path, supercharge_crowdeds=False):
    """Get filenames previously splitted"""

    if top_path[-2:] != "\\\\" and top_path[-1] != "/":
        top_path += "/"

    if supercharge_crowdeds:
        train = top_path + pd.read_csv(
            os.path.join(top_path, "supercharged_crowdeds_train_split.csv"), header=None, squeeze=True
        )
    else:
        train = top_path + pd.read_csv(os.path.join(top_path, "train_split.csv"), header=None, squeeze=True)

    val = top_path + pd.read_csv(os.path.join(top_path, "validation_split.csv"), header=None, squeeze=True)
    test = top_path + pd.read_csv(os.path.join(top_path, "test_split.csv"), header=None, squeeze=True)

    train = train.apply(lambda row: row.replace("\\", "/"))
    val = val.apply(lambda row: row.replace("\\", "/"))
    test = test.apply(lambda row: row.replace("\\", "/"))

    return train, val, test
