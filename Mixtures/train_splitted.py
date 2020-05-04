import sys
sys.path.append('..')

import os
import argparse
import torch
from torch import nn
import numpy as np
from torch.optim import Adam

import utils
import metrics
import config_mixtures as cfg
import utils_mixture as mix_utils
from main_bayesian import train_model as train_bayesian, validate_model as validate_bayesian
from main_frequentist import train_model as train_frequentist, validate_model as validate_frequentist

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_splitted(num_tasks, bayesian=True, net_type='lenet'):
    assert 10 % num_tasks == 0

    # Hyper Parameter settings
    train_ens = cfg.train_ens
    valid_ens = cfg.valid_ens
    n_epochs = cfg.n_epochs
    lr_start = cfg.lr_start

    if bayesian:
        ckpt_dir = f"checkpoints/MNIST/bayesian/splitted/{num_tasks}-tasks/"
    else:
        ckpt_dir = f"checkpoints/MNIST/frequentist/splitted/{num_tasks}-tasks/"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    loaders, datasets = mix_utils.get_splitmnist_dataloaders(num_tasks, return_datasets=True)
    models = mix_utils.get_splitmnist_models(num_tasks, bayesian=bayesian, pretrained=False, net_type=net_type)

    for task in range(1, num_tasks + 1):
        print(f"Training task-{task}..")
        trainset, testset, _, _ = datasets[task-1]
        train_loader, valid_loader, _ = loaders[task-1]
        net = models[task-1]
        net = net.to(device)
        ckpt_name = ckpt_dir + f"model_{net_type}_{num_tasks}.{task}.pt"

        criterion = (metrics.ELBO(len(trainset)) if bayesian else nn.CrossEntropyLoss()).to(device)
        optimizer = Adam(net.parameters(), lr=lr_start)
        valid_loss_max = np.Inf
        for epoch in range(n_epochs):  # loop over the dataset multiple times
            utils.adjust_learning_rate(optimizer, metrics.lr_linear(epoch, 0, n_epochs, lr_start))

            if bayesian:
                train_loss, train_acc, train_kl = train_bayesian(net, optimizer, criterion, train_loader, num_ens=train_ens)
                valid_loss, valid_acc = validate_bayesian(net, criterion, valid_loader, num_ens=valid_ens)
                print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f} \ttrain_kl_div: {:.4f}'.format(
                    epoch, train_loss, train_acc, valid_loss, valid_acc, train_kl))
            else:
                train_loss, train_acc = train_frequentist(net, optimizer, criterion, train_loader)
                valid_loss, valid_acc = validate_frequentist(net, criterion, valid_loader)
                print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f}'.format(
                    epoch, train_loss, train_acc, valid_loss, valid_acc))

            # save model if validation accuracy has increased
            if valid_loss <= valid_loss_max:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_max, valid_loss))
                torch.save(net.state_dict(), ckpt_name)
                valid_loss_max = valid_loss

        print(f"Done training task-{task}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PyTorch Bayesian Split Model Training")
    parser.add_argument('--num_tasks', default=2, type=int, help='number of tasks')
    parser.add_argument('--net_type', default='lenet', type=str, help='model')
    parser.add_argument('--bayesian', default=1, type=int, help='is_bayesian_model(0/1)')
    args = parser.parse_args()

    train_splitted(args.num_tasks, bool(args.bayesian), args.net_type)
