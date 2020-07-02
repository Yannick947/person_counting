import time
import os

import seaborn as sns
import matplotlib.pyplot as plt
import cv2 as cv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib.backends.backend_pgf import FigureCanvasPgf
import numpy as np
import pathlib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

plt.style.use('ggplot')

def visualize_predictions(**kwargs):
    '''Visualize predictions over ground truth
    '''
    plt.clf()
    scatterplot(**kwargs)
    plt.clf()
    violinplot(**kwargs)
    plt.clf()
    boxplot(**kwargs)

def add_plot_attributes(method):
    def plot_method(y_pred, y_true, mode, logdir, video_categories):
        method_name = method.__name__.capitalize()
        figure = plt.figure()
        plt.subplot(1,1,1)
        method(y_pred, y_true, mode, logdir, video_categories)
        plt.xlabel('Ground truth')
        plt.ylabel('Predictions')

        plt.xticks(np.arange(min(np.append(y_pred, y_true)),
                            max(np.append(y_pred , y_true))))

        plt.yticks(np.arange(min(np.append(y_pred, y_true)),
                            max(np.append(y_pred , y_true))))

        if logdir is not None:
            save_name = os.path.join(logdir, '{}_{}_Pred_GT.png'.format(mode, method_name))
            figure.savefig(save_name)
            print(method.__name__, ' created for mode {} and saved in {}'.format(mode, save_name))
        plt.show()

    return plot_method

@add_plot_attributes
def boxplot(y_true, y_pred, mode, logdir, video_categories):
    '''Create a boxplot for predictions
    '''
    sns.boxplot(x=y_pred, y=y_true)

@add_plot_attributes
def scatterplot(y_true, y_pred, mode, logdir, video_categories):
    '''Create a scatterplot for predictions
    '''
    sns.scatterplot(y_pred, y_true, hue=video_categories)

@add_plot_attributes
def violinplot(y_true, y_pred, mode, logdir, video_categories):
    ''' Create a violinplot for predictions
    '''
    sns.violinplot(x=y_pred, y=y_true, scale='width') 


def plot_losses(history, logdir=None):
    '''Plot the history of a model
    '''
    # plot history
    figure = plt.figure()
    plt.subplot(1, 1, 1)
    for key in history.history.keys():
        if (not '_max' in key) and (not '_rescaled' in key):
            plt.plot(history.history[key], label=key)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=2)
    plt.yscale('log')
    plt.show()
    if logdir is not None:
        figure.savefig(os.path.join(logdir, 'losses'))
        print('Figure saved as losses.png')


def visualize_filters(model, logdir=None):
    if logdir is not None:
        print('\nSaving convolutional filters for vizualization ..')
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
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
                ax = plt.subplot(1, int(filters.shape[3]), i + 1)
                plt.imshow(f[:, :, 0], cmap='gray')
                figure.add_axes(ax)

            plt.subplots_adjust(hspace=0.9)
            save_name = os.path.join(logdir, 'filters_layer{}.png'.format(il))
            figure.savefig(save_name, dpi=1200)
            plt.show()
    plt.style.use('ggplot')

def visualize_input_2d(feature_frame, feature_num, timestep_num, pool_model, save_plots=False):
    dimensions = feature_frame.shape[3]
    fig, axs = plt.subplots(nrows=1,
                            ncols=dimensions * 2, 
                            figsize=(3 * dimensions * 2 ,
                                     3 * (timestep_num / feature_num) * 0.3))

    for dim in range(dimensions):
        sns.heatmap(data=feature_frame[0, :, :, dim], vmin=0, vmax=1, ax=axs[dim * 2], cbar=False)

        pooled_frame = pool_model.predict(feature_frame)
        sns.heatmap(data=pooled_frame[0, :, :, dim], vmin=0, ax=axs[dim * 2 + 1], cbar=False)

        if save_plots == True:
            for i, ax in enumerate(axs):
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig('ax{}_figure.png'.format(i), bbox_inches=extent, dpi=800)

    set_titles(axs, fig, feature_num, timestep_num)
    plt.show()

    if save_plots == True:
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        fig.savefig('last_run.png', format='png', dpi=fig.dpi)
        

def visualize_input_3d(feature_frame, pool_model, save_plots=False):
    """ Visualize input in 3D (time, x, y)
    """
    indices = np.argwhere(feature_frame[0, :,:,:] > 0)
    t = [i[0] for i in indices]
    dim1 = [i[1] for i in indices]
    dim2 = [i[2] for i in indices]
    prob = [feature_frame[0, it, idim1, idim2] for  it, idim1, idim2 in zip(t, dim1, dim2)]
    points_x, points_y, points_z = match_indices(t, dim1, dim2, prob)

    # points = pd.DataFrame([points_x, points_y, points_z], axis columns=['t', 'x', 'y'])
    # points.to_csv('points_3d_plot.csv', index=None)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  
    ax.scatter(xs=points_x, ys=points_y, zs=points_z, c=points_z)
    ax.set_xlabel('Frame number')
    ax.set_ylabel('x_coordinate')
    ax.set_zlabel('y_coordinate')
    plt.show()
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False
    })
    fig.savefig('3d_plot.pgf')
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def match_indices(t, dim1, dim2, prob):
    """Return list of coordinates of corresponding points of original image
    """

    t_points = list()
    x_points = list()
    y_points = list()
    for i, it in enumerate(t):
        matching_index_candidates = np.argwhere(t == it)
        for candidate in matching_index_candidates: 
            if (prob[int(candidate[0])] == prob[i]) and (i != candidate[0]):
                t_points.append(t[i]) 
                if dim2[i] == 1:
                    x_points.append(dim1[i])
                    y_points.append(dim1[int(candidate[0])])
                else: 
                    x_points.append(dim1[int(candidate[0])])
                    y_points.append(dim1[i])

    assert len(t_points) == len(x_points) == len(y_points)
    return t_points, x_points, y_points
        
def set_titles(axs, fig, feature_num, timestep_num):

    axs[0].set_title('Dimension x raw')
    axs[1].set_title('Dimension x pooled')
    axs[2].set_title('Dimension y raw')
    axs[3].set_title('Dimension y pooled')
    plt.setp(axs.flat, xlabel='X-label', ylabel='Y-label')
    for i in range(axs.shape[0]):
        axs[i].set_xticks(range(0, feature_num, 10))
        axs[i].set_yticks(range(0, timestep_num, 30))
        axs[i].set_ylabel('Frame number t')
        if i >= int(axs.shape[0] / 2):
            axs[i].set_xlabel('Coordinate y')
        else: 
            axs[i].set_xlabel('Coordinate x')

    fig.tight_layout()

def save_confusion_matrix(y_true, y_pred, logdir):
    plt.clf()
    fig = plt.figure(figsize=(10,10))
    y_pred = np.rint(y_pred)
    mat = confusion_matrix(y_true, y_pred, labels=np.sort(np.unique(y_true)))
    sns.heatmap(data=mat, annot=True, linewidths=.8)
    plt.xlabel('PredictionsGround truth')
    plt.ylabel('Ground truth')
    fig.savefig(os.path.join(logdir, 'confusion_matrix.png'), format='png', dpi=fig.dpi)
