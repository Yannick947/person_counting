
import matplotlib.pyplot as plt
import numpy as np
import time
import os

axs = list()
vis_path = '/content/drive/My Drive/person_counting/visualizations/'

def visualize_predictions(y_true, predictions, mode='val', logdir=None):
    '''Visualize predictions over ground truth
    '''
    #TODO: Implement colour for different categories (noisy, crowd/uncrowd)
    figure = plt.figure()
    plt.subplot(1,1,1)
    plt.scatter(predictions, y_true)
    plt.title('Predictions over ground truth')
    plt.xlabel('Predictions')
    plt.ylabel('Ground truth')

    plt.xticks(np.arange(min(np.append(predictions, y_true)),
                         max(np.append(predictions , y_true))))

    plt.yticks(np.arange(min(np.append(predictions, y_true)),
                         max(np.append(predictions , y_true))))

    if logdir is not None:
        save_name = os.path.join(logdir, '{mode}_Pred_GT.png'.format(mode=mode))
        figure.savefig(save_name)
        print('Figure for mode {mode} saved in {save_name}'.format(mode=mode, save_name=save_name))
    
    plt.show()

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
        for il, layer in enumerate(model.layers):
            if 'conv' not in layer.name:
                continue

            # get filter weights
            filters, _ = layer.get_weights()
            print('layer_name: ', layer.name,', shape: ', filters.shape)
            f_min, f_max = filters.min(), filters.max()
            filters = (filters - f_min) / (f_max - f_min)
            n_filters = len(filters)
            ix = 1

            figure = plt.figure()
            for i in range(n_filters):
                f = filters[:, :, :, i]
                ax = plt.subplot(n_filters, 1, ix)
                plt.imshow(f[:, :, 0], cmap='gray')
                figure.add_axes(ax)
                ix += 1
            
            save_name = os.path.join(logdir, 'filters_layer{}.png'.format(il))
            figure.savefig(save_name)
            plt.show()
