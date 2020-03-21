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
from main_bayesian import getModel, train_model, validate_model

mean_var_dir1 = None
mean_var_dir2 = None

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run(net_type):
    
    # Hyper Parameter settings
    train_ens = cfg.train_ens
    valid_ens = cfg.valid_ens
    n_epochs = cfg.n_epochs
    lr_start = cfg.lr_start
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size

    trainset1, testset1, _, _ = data.getDataset('SplitMNIST-2.1')
    trainset2, testset2, _, _ = data.getDataset('SplitMNIST-2.2')
    
    train_loader1, valid_loader1, test_loader1 = data.getDataloader(
        trainset1, testset1, valid_size, batch_size, num_workers)
    train_loader2, valid_loader2, test_loader2 = data.getDataloader(
        trainset2, testset2, valid_size, batch_size, num_workers)

    net1 = getModel(net_type, 1, 5).to(device)
    net2 = getModel(net_type, 1, 5).to(device)

    ckpt_dir = f'checkpoints/MNIST/bayesian/splitted-2'
    ckpt_name1 = f'checkpoints/MNIST/bayesian/splitted-2/model_{net_type}_2.1.pt'
    ckpt_name2 = f'checkpoints/MNIST/bayesian/splitted-2/model_{net_type}_2.2.pt'

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
    args = parser.parse_args()

    if cfg.record_mean_var:
        mean_var_dir1 = f"checkpoints/MNIST/bayesian/splitted-2/{args.net_type}_2.1/"
        if not os.path.exists(mean_var_dir1):
            os.makedirs(mean_var_dir1, exist_ok=True)
        for file in os.listdir(mean_var_dir1):
            os.remove(mean_var_dir1 + file)

        mean_var_dir2 = f"checkpoints/MNIST/bayesian/splitted-2/{args.net_type}_2.2/"
        if not os.path.exists(mean_var_dir2):
            os.makedirs(mean_var_dir2, exist_ok=True)
        for file in os.listdir(mean_var_dir2):
            os.remove(mean_var_dir2 + file)
        
        cfg.mean_var_dir = mean_var_dir1

    run(args.net_type)