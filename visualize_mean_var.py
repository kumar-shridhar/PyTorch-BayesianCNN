import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import utils


def draw_distributions(filename, type='mean', node_no=0, save_plots=False, plot_time=0.5):
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
        plt.pause(plot_time)
        ax.clear()
    plt.close()

    if save_plots:
        raise NotImplementedError


def draw_lineplot(filename, type='mean', node_no=0, save_plots=False, plot_time=5):
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
    plt.show(block=False)
    plt.pause(plot_time)
    if save_plots:
        plt.savefig('temp.jpg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Visualize Mean and Variance")
    parser.add_argument('--filename', type=str, help='path to log file', required=True)
    parser.add_argument('--data_type', default='mean', type=str, help='Draw plots for what? mean or std?')
    parser.add_argument('--node_no', default=0, type=int, help='Draw plots for which node?')
    parser.add_argument('--plot_type', default='lineplot', type=str, help='Which plot to draw? lineplot or distplot?')
    parser.add_argument('--plot_time', default=1, type=int, help='Pause the plot for how much time?')
    parser.add_argument('--save_plots', default=0, type=int, help='Save plots? 0 (No) or 1 (Yes)')
    args = parser.parse_args()

    if args.plot_type=='lineplot':
        draw_lineplot(args.filename, args.data_type, args.node_no, bool(args.save_plots), args.plot_time)
    elif args.plot_type=='distplot':
        draw_distributions(args.filename, args.data_type, args.node_no, bool(args.save_plots), args.plot_time)
    else:
        raise NotImplementedError