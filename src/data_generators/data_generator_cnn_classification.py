from src.data_generators.data_generators import *
from src.data_generators.data_generators import Generator_CSVS
from src.utils.preprocessing import apply_file_filters
from src.utils.preprocessing import get_filtered_lengths


class Generator_CSVS_CNN_CLS(Generator_CSVS):
    """
    Generators class to load csv files from
    video folder structre like PCDS Dataset and
    train CNNs in Classification mode

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
        super(Generator_CSVS_CNN_CLS, self).__init__(*args, **kwargs)
        self.num_classes = self.get_num_classes(self.top_path, self.label_file)

    def get_num_classes(self, top_path, label_file):
        df_y = pd.read_csv(top_path + label_file, header=None, names=LABEL_HEADER)
        # TODO: make it work for entering and exiting, care at transformation, both must exist
        all_file_names = get_feature_file_names(top_path)
        df_y = df_y[
            df_y["file_name"].apply(
                lambda row: any(row[-32:] in csv_file_name[-32:] for csv_file_name in all_file_names)
            )
        ]
        return int(df_y["entering"].max()) + 1

    def datagen(self):
        """Datagenerator for bus video csv for cnn classification

        yields: Batch of samples in cnn shape for classification
        """

        batch_index = 0

        x_batch = np.zeros(shape=(self.batch_size, self.length_t, self.length_y, 1))

        y_batch = np.zeros(shape=(self.batch_size, self.num_classes))

        while True:
            for file_name in self.file_names:
                try:
                    df_x, label = self.get_sample(file_name)

                # Error messages for debugging purposes
                except FileNotFoundError as e:
                    continue

                except ValueError as e:
                    continue

                x_batch[batch_index, :, :, 0] = df_x
                y_batch[batch_index, int(label[0])] = 1
                batch_index += 1

                # Shape for x must be 4D [samples, timesteps, features, channels] and numpy array
                if batch_index == self.batch_size:
                    batch_index = 0
                    yield (x_batch, y_batch)


def create_datagen(
        top_path, sample, label_file, augmentation_factor=0, filter_hour_above=24, filter_category_noisy=False
):
    """
    Creates train and test data generators for cnn network.

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
    train_file_names, test_file_names = split_files(top_path, label_file)

    # Apply filters
    train_file_names = apply_file_filters(train_file_names, filter_hour_above, filter_category_noisy)
    test_file_names = apply_file_filters(test_file_names, filter_hour_above, filter_category_noisy)

    print("Dataset contains: \n{} training csvs \n{} testing csvs".format(len(train_file_names), len(test_file_names)))

    # TODO: Should be mix of train and test file names, feature and label scaler refactoring
    scaler = CSVScaler_CLS(top_path, label_file, train_file_names, sample, sample_size=100)

    gen_train = Generator_CSVS_CNN_CLS(
        length_t=length_t,
        length_y=length_y,
        file_names=train_file_names,
        scaler=scaler,
        sample=sample,
        top_path=top_path,
        label_file=label_file,
        augmentation_factor=augmentation_factor,
    )

    # Don't do augmentation here!
    gen_test = Generator_CSVS_CNN_CLS(
        length_t=length_t,
        length_y=length_y,
        file_names=test_file_names,
        scaler=scaler,
        sample=sample,
        top_path=top_path,
        label_file=label_file,
        augmentation_factor=0,
    )

    return gen_train, gen_test
