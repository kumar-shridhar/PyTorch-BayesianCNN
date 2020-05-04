import os
import imageio
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import utils


def draw_distributions(filename, save_dir, type='mean', node_no=0, save_plots=False, plot_time=0.5):
    file_desc = utils.get_file_info(filename)
    layer = file_desc['layer_name']
    batch_size = file_desc['batch_size']
    freq = file_desc['recording_frequency_per_epoch']
    means, stds = utils.load_mean_std_from_file(filename)
    frames = []
    if type=='both':
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for i in range(len(means)):
            mean = np.mean(means[i].reshape((batch_size, -1))[:, node_no])
            std = np.sum(np.square(stds[i].reshape((batch_size, -1))[:, node_no])) / batch_size
            sns.distplot(np.random.normal(loc=mean, scale=std, size=1000), ax=ax, hist=False)
            ax.axvline(mean, color='r', linestyle='-')
            iteration = i % freq
            epoch = i // freq
            plt.title(f'Distribution for {layer} node {node_no}: Epoch-{epoch} Iteration-{iteration}')
            plt.xlabel(f'Value')
            plt.ylabel('Density')
            fig.canvas.draw()
            if save_plots:
                frame = np.array(fig.canvas.renderer.buffer_rgba())
                frames.append(frame)
            plt.pause(0.1)
            ax.clear()
        plt.close()
    else:
        data = means if type=='mean' else stds
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for i in range(len(data)):
            sample = data[i].reshape((batch_size, -1))
            sample = sample[:, node_no]
            sns.distplot(sample, norm_hist=True, ax=ax)
            ax.axvline(np.mean(sample), color='r', linestyle='-')
            iteration = i % freq
            epoch = i // freq
            plt.title(f'Distribution for {layer} node {node_no}: Epoch-{epoch} Iteration-{iteration}')
            plt.xlabel(f'Value of {type}')
            plt.ylabel('Density')
            fig.canvas.draw()
            if save_plots:
                frame = np.array(fig.canvas.renderer.buffer_rgba())
                frames.append(frame)
            plt.pause(0.1)
            ax.clear()
        plt.close()

    if save_plots:
        imageio.mimsave(save_dir + f'{layer}-node_{node_no}-{type}-distplot.gif', frames, fps=1/plot_time)


def draw_lineplot(filename, save_dir, type='mean', node_no=0, save_plots=False, plot_time=5):
    file_desc = utils.get_file_info(filename)
    layer = file_desc['layer_name']
    means, stds = utils.load_mean_std_from_file(filename)
    data = means if type=='mean' else stds

    means = []
    for i in range(len(data)):
        sample = data[i].reshape((file_desc['batch_size'], -1))
        means.append(np.mean(sample[:, node_no]))

    x = np.hstack([np.arange(0, file_desc['number_of_epochs'], 1 / freq)])
    sns.lineplot(x, means)
    plt.title(f'Average value of {type} for node {node_no} of {layer}')
    plt.xlabel('Epoch Number')
    plt.ylabel(f'Average {type}s')
    plt.show(block=False)
    plt.pause(plot_time)
    if save_plots:
        plt.savefig(save_dir + f'{layer}-node_{node_no}-{type}-lineplot.jpg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Visualize Mean and Variance")
    parser.add_argument('--filename', type=str, help='path to log file', required=True)
    parser.add_argument('--data_type', default='mean', type=str, help='Draw plots for what? mean or std?')
    parser.add_argument('--node_no', default=0, type=int, help='Draw plots for which node?')
    parser.add_argument('--plot_type', default='both', type=str, help='Which plot to draw? lineplot or distplot?')
    parser.add_argument('--plot_time', default=1, type=float, help='Pause the plot for how much time?')
    parser.add_argument('--save_plots', default=0, type=int, help='Save plots? 0 (No) or 1 (Yes)')
    parser.add_argument('--save_dir', default='', type=str, help='Save plots to which directory?(End with a /)')
    args = parser.parse_args()

    save_dir = None
    if args.save_plots:
        save_dir = None if args.save_dir=='' else args.save_dir
        if not save_dir:
            save_dir = "/".join(args.filename.split("/")[:-1]) + '/plots/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    if args.plot_type=='lineplot':
        draw_lineplot(args.filename, save_dir, args.data_type, args.node_no, bool(args.save_plots), args.plot_time)
    elif args.plot_type=='distplot':
        draw_distributions(args.filename, save_dir, args.data_type, args.node_no, bool(args.save_plots), args.plot_time)
    else:
        raise NotImplementedError