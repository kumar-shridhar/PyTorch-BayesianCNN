from __future__ import print_function

import os
import time
import argparse

import torch
import numpy as np
from torch.optim import Adam
from torch.nn import functional as F

import data
import utils
import metrics
import config_bayesian as cfg
from models.BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from models.BayesianModels.BayesianAlexNet import BBBAlexNet
from models.BayesianModels.BayesianLeNet import BBBLeNet

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def getModel(net_type, inputs, outputs):
    if (net_type == 'lenet'):
        return BBBLeNet(outputs,inputs)
    elif (net_type == 'alexnet'):
        return BBBAlexNet(outputs, inputs)
    elif (net_type == '3conv3fc'):
        return BBB3Conv3FC(outputs,inputs)
    else:
        raise ValueError('Network should be either [LeNet / AlexNet / 3Conv3FC')


def train_model(net, optimizer, criterion, trainloader, num_ens=1):
    net.train()
    training_loss = 0
    accs = []
    kl_list = []
    for i, (inputs, labels) in enumerate(trainloader, 0):
        optimizer.zero_grad()

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)

        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1)
        
        kl = kl / num_ens
        kl_list.append(kl.item())
        log_outputs = utils.logmeanexp(outputs, dim=2)

        loss = criterion(log_outputs, labels, kl)
        loss.backward()
        optimizer.step()

        accs.append(metrics.logit2acc(log_outputs.data, labels))
        training_loss += loss.cpu().data.numpy()
    return training_loss/len(trainloader), np.mean(accs), np.mean(kl_list)


def validate_model(net, dataloader, num_ens=1):
    """Calculate ensemble accuracy and NLL"""
    net.eval()
    accs = []
    nlls = []
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            outputs[:, :, j] = F.log_softmax(net_out, dim=1).data
        log_outputs = utils.logmeanexp(outputs, dim=2)
        accs.append(metrics.logit2acc(log_outputs, labels))
        nlls.append(F.nll_loss(log_outputs, labels, size_average=False).data.cpu().numpy())
    return np.mean(accs), np.sum(nlls)

"""
def run_an_epoch(net, optimizer, criterion, trainloader, testloader, logger, train_ens, epoch):
    # ELBO evaluation
    net.train()
    training_loss = 0
    steps = 0
    accs = []
    for i, (inputs, labels) in enumerate(trainloader, 0):
        steps += 1
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        for j in range(10):
            outputs = net(inputs).detach()
            training_loss += criterion(outputs, labels).cpu().data.numpy()/10.0
        accs.append(metrics.logit2acc(outputs.data, labels))
    logger.add(epoch, kl=criterion.get_kl(), tr_loss=training_loss/steps, tr_acc=np.mean(accs))
"""

def run(dataset, net_type):

    # Hyper Parameter settings
    train_ens = cfg.train_ens
    valid_ens = cfg.valid_ens
    n_epochs = cfg.n_epochs
    lr_start = cfg.lr_start
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size

    trainset, testset, inputs, outputs = data.getDataset(dataset)
    train_loader, valid_loader, test_loader = data.getDataloader(
        trainset, testset, valid_size, batch_size, num_workers)

    net = getModel(net_type, inputs, outputs).to(device)

    ckpt_dir = f'checkpoints/{dataset}/bayesian'
    ckpt_name = f'checkpoints/{dataset}/bayesian/model_{net_type}.pt'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    criterion = metrics.ELBO(len(trainset)).to(device)
    optimizer = Adam(net.parameters(), lr=lr_start)
    valid_acc_max = 0
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        utils.adjust_learning_rate(optimizer, metrics.lr_linear(epoch, 0, n_epochs, lr_start))

        #run_an_epoch(net, optimizer, criterion, train_loader, test_loader, logger, train_ens, epoch)
        train_loss, train_acc, kl = train_model(net, optimizer, criterion, train_loader, num_ens=train_ens)
        valid_acc, valid_nll = validate_model(net, valid_loader, num_ens=valid_ens)

        print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Accuracy: {:.4f} \tkl_div: {:.4f}'.format(
            epoch, train_loss, train_acc, valid_acc, kl))

        # save model if validation accuracy has increased
        if valid_acc >= valid_acc_max:
            print('Validation accuracy increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_acc_max, valid_acc))
            torch.save(net.state_dict(), ckpt_name)
            valid_acc_max = valid_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PyTorch Bayesian Model Training")
    parser.add_argument('--net_type', default='lenet', type=str, help='model')
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset = [MNIST/CIFAR10/CIFAR100]')
    args = parser.parse_args()

    run(args.dataset, args.net_type)
