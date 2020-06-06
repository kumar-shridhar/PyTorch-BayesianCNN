"""
It is assumed that `weights_dir` contains bayesian models trained on individual
tasks using Bayes by Backprop on trainset.

This script will be used to train MixtureModel having Gaussian Mixture weights
on validset using Cross_Entropy_Loss as a loss function. And test the same on testset.
"""
import sys
sys.path.append('..')

import torch
import torch.nn as nn
from torch import optim
import numpy as np

import metrics
import utils_mixture
from mixture_models import MixtureLeNet
from main_bayesian import train_model, validate_model
import config_mixtures as cfg

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_individual_weights(num_tasks, weights_dir):
    task_weights = []
    for i in range(1, num_tasks + 1):
        weight_path = weights_dir + f"model_lenet_bbb_softplus_{num_tasks}.{i}.pt"
        task_weights.append(torch.load(weight_path, map_location=device))
    return task_weights


if __name__ == '__main__':
    num_tasks = 2
    weights_dir = "checkpoints/MNIST/bayesian/splitted/{}-tasks/".format(num_tasks)
    ckpt_name = weights_dir + f"mixture_model_{num_tasks}.pt"
    lr_start = 0.001
    n_epochs = 100
    train_ens = 10
    valid_ens = 10

    data_loaders = utils_mixture.get_splitmnist_dataloaders(num_tasks)

    individual_weights = get_individual_weights(num_tasks, weights_dir)
    net = MixtureLeNet(10, 1, num_tasks, individual_weights)
    net.cuda()

    # TODO: get shuffled validloaders with incremented labels
    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr_start)
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, verbose=True)

    valid_loss_max = np.Inf
    for epoch in range(n_epochs):
        # training loop
        train_loss = 0.0
        accs = []
        for t in range(num_tasks):
            for inputs, target in data_loaders[t][1]:  # valid_loaders
                optimizer.zero_grad()

                inputs, target = inputs.to(device), target.to(device)
                target += t * 10 // num_tasks  # last fc has 10 nodes

                outputs = torch.zeros(inputs.shape[0], net.num_classes, train_ens).to(device)
                for j in range(train_ens):
                    net_out, _ = net(inputs)
                    outputs[:, :, j] = net_out

                output = outputs.mean(dim=2)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                cfg.distribution_updated = True
                train_loss += loss.item() / len(data_loaders[t][1])
                accs.append(metrics.acc(output.detach(), target))
        train_acc = np.mean(accs)

        # validation loop
        valid_loss = 0.0
        accs = []
        for t in range(num_tasks):
            for inputs, target in data_loaders[t][2]:  # test_loaders
                inputs, target = inputs.to(device), target.to(device)
                target += t * 10 // num_tasks  # last fc has 10 nodes

                outputs = torch.zeros(inputs.shape[0], net.num_classes, train_ens).to(device)
                for j in range(train_ens):
                    net_out, _ = net(inputs)
                    outputs[:, :, j] = net_out

                output = outputs.mean(dim=2)
                loss = criterion(output, target)
                valid_loss += loss.item() / len(data_loaders[t][2])
                accs.append(metrics.acc(output.detach(), target))
        valid_acc = np.mean(accs)

        print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f}'.format(
            epoch, train_loss, train_acc, valid_loss, valid_acc))

        lr_sched.step(valid_loss)

        # save model if validation accuracy has increased
        if valid_loss <= valid_loss_max:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_max, valid_loss))
            torch.save(net.state_dict(), ckpt_name)
            valid_loss_max = valid_loss
