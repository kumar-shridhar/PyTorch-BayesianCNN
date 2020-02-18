import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import utils
import config_bayesian as cfg


def plot_dist(file, type, node_no):
    means, stds, freq_per_epoch = utils.load_mean_std_from_file(file)

    if type=='mean':
        data = means
    else:
        data = stds

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(data)):
        sample = data[i].reshape((cfg.batch_size, -1))
        sns.kdeplot(sample[:, node_no], ax=ax)
        plt.show(block=False)
        plt.pause(0.5)
        ax.clear()


def plot_line(file, type, node_no):
    means, stds, freq_per_epoch = utils.load_mean_std_from_file(file)

    if type=='mean':
        data = means
    else:
        data = stds

    means = []
    for i in range(len(data)):
        sample = data[i].reshape((cfg.batch_size, -1))
        means.append(np.mean(sample[:, node_no]))
    
    sns.lineplot(data=np.array(means))
    plt.show(block=False)
    plt.pause(5)
    plt.close()

plot_dist("checkpoints/MNIST/bayesian/lenet/fc3.txt", 'mean', 9)