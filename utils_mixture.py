import os
import torch
import numpy as np

import data
from main_bayesian import getModel
import config_mixtures as cfg


def _get_splitmnist_datasets(num_tasks):
    datasets = []
    for i in range(1, num_tasks + 1):
        name = 'SplitMNIST-{}.{}'.format(num_tasks, i)
        datasets.append(data.getDataset(name))
    return datasets


def get_splitmnist_dataloaders(num_tasks, return_datasets=False):
    loaders = []
    datasets = _get_splitmnist_datasets(num_tasks)
    for i in range(1, num_tasks + 1):
        trainset, testset, _, _ = datasets[i-1]
        curr_loaders = data.getDataloader(
            trainset, testset, cfg.valid_size, cfg.batch_size, cfg.num_workers)
        loaders.append(curr_loaders)  # (train_loader, valid_loader, test_loader)
    if return_datasets:
        return loaders, datasets
    return loaders


def get_splitmnist_models(num_tasks, pretrained=False, weights_dir=None, net_type='lenet'):
    inputs = 1
    outputs = 10 // num_tasks
    models = []
    if pretrained:
        assert weights_dir
    for i in range(1, num_tasks + 1):
        models.append(getModel(net_type, inputs, outputs))
        if pretrained:
            weight_path = weights_dir + f"model_{net_type}_{num_tasks}.{i}.pt"
            models[-1].load_state_dict(torch.load(weight_path))
    return models


def get_mixture_model(num_tasks, weights_dir, net_type='lenet'):
    """
    Current implementation is based on average value of weights
    """
    net = getModel(net_type, 1, 10)
    task_weights = []
    for i in range(1, num_tasks + 1):
        weight_path = weights_dir + f"model_{net_type}_{num_tasks}.{i}.pt"
        task_weights.append(torch.load(weight_path))

    mixture_weights = net.state_dict().copy()
    for key in mixture_weights:
        if key in list(mixture_weights.keys())[:-2]:  # Ignore last fc layer
            concat_weights = torch.cat([w[key].unsqueeze(0) for w in task_weights] , dim=0)
            average_weight = torch.mean(concat_weights, dim=0)
            mixture_weights[key] = average_weight
        else:
            if key.endswith('.log_alpha'):
                concat_weights = torch.cat([w[key].unsqueeze(0) for w in task_weights] , dim=0)
                average_weight = torch.mean(concat_weights, dim=0)
                mixture_weights[key] = average_weight
            else:  # output layer
                raise NotImplementedError

    net.load_state_dict(mixture_weights)
    return net
