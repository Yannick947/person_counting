
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
import os


def visualize_predictions(**kwargs):
    '''Visualize predictions over ground truth
    '''
    scatterplot(**kwargs)
    violinplot(**kwargs)


def add_plot_attributes(method):
    def plot_method(y_pred, y_true, mode, logdir):
        figure = plt.figure()
        plt.subplot(1,1,1)
        method(y_pred, y_true, mode, logdir)
        plt.title('Predictions over ground truth')
        plt.xlabel('Ground truth')
        plt.ylabel('Predictions')

        plt.xticks(np.arange(min(np.append(y_pred, y_true)),
                            max(np.append(y_pred , y_true))))

        plt.yticks(np.arange(min(np.append(y_pred, y_true)),
                            max(np.append(y_pred , y_true))))

        if logdir is not None:
            save_name = os.path.join(logdir, '{plottype}_{mode}_Pred_GT.png'.format(mode=mode,
                                                                                    plottype=method.__name__))
            figure.savefig(save_name)
            print(method.__name__, ' created for mode {mode} and saved in {save_name}'.format(mode=mode,
                                                                                              save_name=save_name))
        plt.show()

    return plot_method


@add_plot_attributes
def scatterplot(y_true, y_pred, mode, logdir):
    '''Create a scatterplot for predictions
    '''
    #TODO: Implement colour for different categories (noisy, crowd/uncrowd)
    plt.scatter(y_pred, y_true)


@add_plot_attributes
def violinplot(y_true, y_pred, mode, logdir):
    ''' Create a violinplot for predictions
    '''
    sns.violinplot(x=y_pred, y=y_true) 


def plot_losses(history, logdir=None):
    '''Plot the history of a model
    '''
    # plot history
    figure = plt.figure()
    plt.subplot(1, 1, 1)
    for key in history.history.keys():
        plt.plot(history.history[key], label=key)

    plt.legend()
    plt.show()
    if logdir is not None:
        save_name = os.path.join(logdir, 'losses.png')
        figure.savefig(save_name)
        print('Figure saved as losses.png')


def visualize_filters(model, logdir=None):
    if logdir is not None:
        print('\nSaving convolutional filters for vizualization ..')

        for il, layer in enumerate(model.layers):
            if 'conv' not in layer.name:
                continue

            # get filter weights
            filters, _ = layer.get_weights()
            print('layer_name: ', layer.name,', shape: ', filters.shape)
            f_min, f_max = filters.min(), filters.max()
            filters = (filters - f_min) / (f_max - f_min)
            figure = plt.figure()

            for i in range(int(filters.shape[3])):
                f = filters[:, :, :, i]
                ax = plt.subplot(int(filters.shape[3]), 1, i + 1)
                plt.imshow(f[:, :, 0], cmap='gray')
                figure.add_axes(ax)
            
            save_name = os.path.join(logdir, 'filters_layer{}.png'.format(il))
            figure.savefig(save_name)
            plt.show()
