import os
import argparse
import torch
import numpy as np
import seaborn as sns
from torch.optim import Adam
from torch.utils.data import Dataset
import torchvision.transforms as transforms


import data
import utils
import metrics
import config_bayesian as cfg
from models.BayesianModels.BayesianLeNet import BBBLeNet
from main_bayesian import getModel, train_model, validate_model

mean_var_dir1 = None
mean_var_dir2 = None

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample, label


def split_trainset(trainset, group1_target_list, group2_target_list):
    group1_idx = torch.zeros_like(trainset.targets, dtype=torch.bool)
    for target in group1_target_list:
        group1_idx = group1_idx | (trainset.targets==target)

    group2_idx = torch.zeros_like(trainset.targets, dtype=torch.bool)
    for target in group2_target_list:
        group2_idx = group2_idx | (trainset.targets==target)
    
    group1_data, group1_targets = trainset.data[group1_idx], trainset.targets[group1_idx]
    group2_data, group2_targets = trainset.data[group2_idx], trainset.targets[group2_idx]

    return group1_data, group1_targets, group2_data, group2_targets


def run(dataset, net_type):
    
    # Hyper Parameter settings
    train_ens = cfg.train_ens
    valid_ens = cfg.valid_ens
    n_epochs = cfg.n_epochs
    lr_start = cfg.lr_start
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size

    group1_target_list = [0, 1, 2, 3, 4]
    group2_target_list = [5, 6, 7, 8, 9]

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        ])

    trainset, testset, inputs, outputs = data.getDataset(dataset)

    group1_data, group1_targets, group2_data, group2_targets = \
        split_trainset(trainset, group1_target_list, group2_target_list)

    trainset1 = CustomDataset(group1_data, group1_targets, transform=transform)
    trainset2 = CustomDataset(group2_data, group2_targets, transform=transform)
    
    train_loader1, valid_loader1, test_loader1 = data.getDataloader(
        trainset1, testset, valid_size, batch_size, num_workers)
    train_loader2, valid_loader2, test_loader2 = data.getDataloader(
        trainset2, testset, valid_size, batch_size, num_workers)

    net1 = getModel(net_type, inputs, len(group1_target_list)).to(device)
    net2 = getModel(net_type, inputs, len(group2_target_list)).to(device)

    ckpt_dir = f'checkpoints/{dataset}/bayesian/splitted'
    ckpt_name1 = f'checkpoints/{dataset}/bayesian/splitted/model1_{net_type}.pt'
    ckpt_name2 = f'checkpoints/{dataset}/bayesian/splitted/model2_{net_type}.pt'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    criterion = metrics.ELBO(len(trainset1)).to(device)
    optimizer = Adam(net1.parameters(), lr=lr_start)
    valid_loss_max = np.Inf
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        cfg.curr_epoch_no = epoch
        utils.adjust_learning_rate(optimizer, metrics.lr_linear(epoch, 0, n_epochs, lr_start))

        train_loss, train_acc, train_kl = train_model(net1, optimizer, criterion, train_loader1, num_ens=train_ens)
        valid_loss, valid_acc = validate_model(net1, criterion, valid_loader1, num_ens=valid_ens)

        print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f} \ttrain_kl_div: {:.4f}'.format(
            epoch, train_loss, train_acc, valid_loss, valid_acc, train_kl))

        # save model if validation accuracy has increased
        if valid_loss <= valid_loss_max:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_max, valid_loss))
            torch.save(net1.state_dict(), ckpt_name1)
            valid_loss_max = valid_loss

    print("Done training first Model")

    if cfg.record_mean_var:
        cfg.mean_var_dir = mean_var_dir2

    criterion = metrics.ELBO(len(trainset2)).to(device)
    optimizer = Adam(net2.parameters(), lr=lr_start)
    valid_loss_max = np.Inf
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        cfg.curr_epoch_no = epoch
        utils.adjust_learning_rate(optimizer, metrics.lr_linear(epoch, 0, n_epochs, lr_start))

        train_loss, train_acc, train_kl = train_model(net2, optimizer, criterion, train_loader2, num_ens=train_ens)
        valid_loss, valid_acc = validate_model(net2, criterion, valid_loader2, num_ens=valid_ens)

        print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f} \ttrain_kl_div: {:.4f}'.format(
            epoch, train_loss, train_acc, valid_loss, valid_acc, train_kl))

        # save model if validation accuracy has increased
        if valid_loss <= valid_loss_max:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_max, valid_loss))
            torch.save(net2.state_dict(), ckpt_name2)
            valid_loss_max = valid_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PyTorch Bayesian Split Model Training")
    parser.add_argument('--net_type', default='lenet', type=str, help='model')
    parser.add_argument('--dataset', default='MNIST', type=str, help='dataset = [MNIST/CIFAR10/CIFAR100]')
    args = parser.parse_args()

    if cfg.record_mean_var:
        mean_var_dir1 = f"checkpoints/{args.dataset}/bayesian/splitted/{args.net_type}1/"
        if not os.path.exists(mean_var_dir1):
            os.makedirs(mean_var_dir1, exist_ok=True)
        for file in os.listdir(mean_var_dir1):
            os.remove(mean_var_dir1 + file)

        mean_var_dir2 = f"checkpoints/{args.dataset}/bayesian/splitted/{args.net_type}2/"
        if not os.path.exists(mean_var_dir2):
            os.makedirs(mean_var_dir2, exist_ok=True)
        for file in os.listdir(mean_var_dir2):
            os.remove(mean_var_dir2 + file)
        
        cfg.mean_var_dir = mean_var_dir1

    run(args.dataset, args.net_type)