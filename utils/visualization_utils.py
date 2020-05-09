
import matplotlib.pyplot as plt
import numpy as np
import time
import os

axs = list()
vis_path = '/content/drive/My Drive/person_counting/visualizations/'

def visualize_predictions(y_true, predictions, mode='val', ipython_mode=False):
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

    if ipython_mode == False: 
        plt.show()
    else: 
        plt.show()

    save_name = vis_path + '{mode}_Pred_GT.png'.format(mode=mode)
    figure.savefig(save_name)
    print('Figure for mode {mode} saved in {save_name}'.format(mode=mode, save_name=save_name))


def plot_losses(history, ipython_mode=False):
    '''Plot the history of a model
    '''
    # plot history
    figure = plt.figure()
    plt.subplot(1, 1, 1)
    for key in history.history.keys():
        plt.plot(history.history[key], label=key)

    plt.legend()
    plt.show()
    save_name = vis_path + 'losses.png'
    print(save_name)
    figure.savefig(save_name)
    print('Figure saved as losses.png')

def visualize_filters(model):
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
            # get the filter
            f = filters[:, :, :, i]
            # specify subplot and turn of axis
            ax = plt.subplot(n_filters, 1, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(f[:, :, 0], cmap='gray')
            figure.add_axes(ax)
            ix += 1
        
        save_name = vis_path + 'filters_layer{}.png'.format(il)
        figure.savefig(save_name)
        plt.show()
