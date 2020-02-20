import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import utils
import config_bayesian as cfg


def draw_distributions(filename, type='mean', node_no=0):
    file_desc = utils.get_file_info(filename)
    layer = file_desc['layer_name']
    means, std = utils.load_mean_std_from_file(filename)
    data = means if type=='mean' else stds

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(len(data)):
        sample = data[i].reshape((file_desc['batch_size'], -1))
        sample = sample[:, node_no]
        sns.distplot(sample, norm_hist=True, ax=ax)
        ax.axvline(np.mean(sample), color='r', linestyle='-')
        iteration = i % file_desc['recording_frequency_per_epoch']
        epoch = i // file_desc['recording_frequency_per_epoch']
        plt.title(f'Distribution for {layer} node {node_no}: Epoch-{epoch} Iteration-{iteration}')
        plt.xlabel(f'Value of {type}')
        plt.ylabel('Density')
        plt.show(block=False)
        plt.pause(0.5)
        ax.clear()
    plt.close()


def draw_lineplot(filename, type='mean', node_no=0):
    file_desc = utils.get_file_info(filename)
    layer = file_desc['layer_name']
    means, stds = utils.load_mean_std_from_file(filename)
    data = means if type=='mean' else stds

    means = []
    for i in range(len(data)):
        sample = data[i].reshape((file_desc['batch_size'], -1))
        means.append(np.mean(sample[:, node_no]))

    x = np.hstack([np.arange(0, file_desc['number_of_epochs'], 1 / file_desc['recording_frequency_per_epoch'])])
    sns.lineplot(x, means)
    plt.title(f'Mean value of {type} for node {node_no} of {layer}')
    plt.xlabel('Epoch Number')
    plt.ylabel(f'Mean of {type}s')
    plt.show()

# draw_lineplot("checkpoints/MNIST/bayesian/lenet/fc3.txt", 'mean', 3)