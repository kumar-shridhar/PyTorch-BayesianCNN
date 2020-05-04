import sys
sys.path.append('..')

import os
import torch
import numpy as np
from collections import OrderedDict

import gmm
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


def _get_ordered_layer_name(mean_var_path):
    # Order files according to creation time
    files = os.listdir(mean_var_path)
    files = [os.path.join(mean_var_path, f) for f in files]
    files.sort(key=os.path.getctime)
    layer_names = [f.split('/')[-1].split('.')[0] for f in files]
    return layer_names


def _get_layer_wise_mean_var_per_task(mean_var_path):
    # Order files according to creation time
    # To get the correct model architecture
    files = os.listdir(mean_var_path)
    files = [os.path.join(mean_var_path, f) for f in files]
    files.sort(key=os.path.getctime)
    layer_names = [f.split('/')[-1].split('.')[0] for f in files]

    mean_var = OrderedDict()
    for i in range(len(files)):
        data = {}
        mean, var = utils.load_mean_std_from_file(files[i])
        mean, var = np.vstack(mean), np.vstack(var)  # shape is (len(trainset), output shape)
        data['mean'] = mean
        data['var'] = var
        data['mean.mu'] = np.mean(mean, axis=0)
        data['var.mu'] = np.mean(var, axis=0)
        data['mean.var'] = np.var(mean, axis=0)
        data['var.var'] = np.var(var, axis=0)
        mean_var[layer_names[i]] = data
    return mean_var


def get_mean_vars_for_all_tasks(mean_var_dir):
    all_tasks = {}
    for task in os.listdir(mean_var_dir):
        path_to_task = os.path.join(mean_var_dir, task)
        mean_var_per_task = _get_layer_wise_mean_var_per_task(path_to_task)
        all_tasks[task] = mean_var_per_task
    return all_tasks


def fit_to_gmm(num_tasks, layer_name, data_type, all_tasks):
    data = np.vstack([all_tasks[f'task-{i}'][layer_name][data_type] for i in range(1, num_tasks+1)])
    data = torch.tensor(data).float()
    #data_mu = torch.cat([torch.tensor(all_tasks[f'task-{i}'][layer_name][data_type+'.mu']).unsqueeze(0) for i in range(1, num_tasks+1)], dim=0).float().unsqueeze(0)
    #data_var = torch.cat([torch.tensor(all_tasks[f'task-{i}'][layer_name][data_type+'.var']).unsqueeze(0) for i in range(1, num_tasks+1)], dim=0).float().unsqueeze(0)
    model = gmm.GaussianMixture(n_components=num_tasks, n_features=np.prod(data.shape[1:]))#, mu_init=data_mu, var_init=data_var)
    data = data[torch.randperm(data.size()[0])]  # Shuffling of data
    model.fit(data)
    return model.predict(data)


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
    all_tasks = get_mean_vars_for_all_tasks("Mixtures/mean_vars/")
    y = fit_to_gmm(2, 'fc3', 'mean', all_tasks)
    print("Cluster0", (1-y).sum())
    print("Cluster1", y.sum())
