import sys
sys.path.append('..')

import os
import torch
import numpy as np
from collections import OrderedDict

import utils
import utils_mixture
import config_bayesian as cfg


def feedforward_and_save_mean_var(net, dataloader, task_no, num_ens=1):
    cfg.mean_var_dir = "Mixtures/mean_vars/task-{}/".format(task_no)
    if not os.path.exists(cfg.mean_var_dir):
        os.makedirs(cfg.mean_var_dir, exist_ok=True)
        cfg.record_mean_var = True
        cfg.record_layers = None  # All layers
        cfg.record_now = True
        cfg.curr_batch_no = 0  # Not required
        cfg.curr_epoch_no = 0  # Not required

    net.train()  # To get distribution of mean and var
    accs = []

    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)
        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1).data

        log_outputs = utils.logmeanexp(outputs, dim=2)
        accs.append(metrics.acc(log_outputs, labels))
    return np.mean(accs)


def _get_layer_wise_mean_var_per_task(mean_var_path):
    # Order files according to creation time
    files = os.listdir(mean_var_path)
    files = [os.path.join(mean_var_path, f) for f in files]
    files.sort(key=os.path.getctime)
    layer_names = [f.split('/')[-1].split('.')[0] for f in files]

    # To get the correct model architecture
    mean_var = OrderedDict()
    for i in range(len(files)):
        mean, var = utils.load_mean_std_from_file(files[i])
        mean, var = np.vstack(mean), np.vstack(var)  # shape is (len(trainset), output shape)
        mean_var[layer_names[i]] = [mean, var]
    return mean_var


def get_mean_vars_for_all_tasks(mean_var_dir):
    first = True
    for task in os.listdir(mean_var_dir):
        path_to_task = os.path.join(mean_var_dir, task)
        mean_var_per_task = _get_layer_wise_mean_var_per_task(path_to_task)
        if first:
            first = False
            all_tasks = mean_var_per_task.copy()
        else:
            for layer, value in mean_var_per_task.items():
                all_tasks[layer][0] = np.vstack([all_tasks[layer][0], value[0]])  # mean
                all_tasks[layer][0] = np.vstack([all_tasks[layer][1], value[1]])  # var
    return all_tasks

def main():
    num_tasks = 2
    weights_dir = "checkpoints/MNIST/bayesian/splitted/2-tasks/"

    loader_task1, loader_task2 = utils_mixture.get_splitmnist_dataloaders(num_tasks)
    train_loader_task1 = loader_task1[0]
    train_loader_task2 = loader_task2[0]

    net_task1, net_task2 = utils_mixture.get_splitmnist_models(num_tasks, True, weights_dir)
    net_task1.cuda()
    net_task2.cuda()

    print("Task-1 Accuracy:", feedforward_and_save_mean_var(net_task1, train_loader_task1, task_no=1))
    print("Task-2 Accuracy:", feedforward_and_save_mean_var(net_task2, train_loader_task2, task_no=2))

    mean_vars_all_tasks = get_mean_vars_for_all_tasks("Mixtures/mean_vars/")
    


if __name__ == '__main__':
    data = get_mean_vars_for_all_tasks("Mixtures/mean_vars/")