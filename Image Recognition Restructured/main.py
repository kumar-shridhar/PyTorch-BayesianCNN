from __future__ import print_function

import os
import argparse

import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam

import data
import utils
import config as cfg
from models.BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from models.BayesianModels.BayesianAlexNet import BBBAlexNet
from models.BayesianModels.BayesianLeNet import BBBLeNet
from models.NonBayesianModels.AlexNet import AlexNet
from models.NonBayesianModels.LeNet import LeNet
from models.NonBayesianModels.ThreeConvThreeFC import ThreeConvThreeFC

# CUDA settings
use_cuda = torch.cuda.is_available()
torch.cuda.set_device(0)


def getModel(net_type, inputs, outputs, IS_BAYESIAN):
    if (net_type == 'lenet'):
        if IS_BAYESIAN:
            return BBBLeNet(outputs,inputs)
        return LeNet(outputs, inputs)
    elif (net_type == 'alexnet'):
        if IS_BAYESIAN:
            return BBBAlexNet(outputs, inputs)
        return AlexNet(outputs, inputs)
    elif (net_type == '3conv3fc'):
        if IS_BAYESIAN:
            return BBB3Conv3FC(outputs,inputs)
        return ThreeConvThreeFC(outputs, inputs)
    else:
        raise ValueError('Network should be either [LeNet / AlexNet / 3Conv3FC')


def train_bayesian(net, optimizer, epoch, train_loader, train_data, beta_type):
    print('Epoch: %d' % epoch)
    net.train()
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs, kl = net.probforward(inputs)
        beta = utils.get_beta(epoch, len(train_data), beta_type)
        loss = utils.elbo(outputs, targets, kl, beta)
        loss.backward()
        optimizer.step()
        pred = torch.max(outputs, dim=1)[1]
        correct += torch.sum(pred.eq(targets)).item()
        total += targets.numel()
    print(f'[TRAIN] Acc: {100.*correct/total:.3f}')


def test_bayesian(net, epoch, test_loader, ckpt_name):
    net.eval()
    correct = 0
    total = 0   
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _ = net.probforward(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            accuracy = 100.*correct/total
        print(f'[TEST] Acc: {accuracy:.3f}')

    torch.save(net.state_dict(), ckpt_name)


def train_frequentist(net, optimizer, criterion, train_loader):
    train_loss = 0.0
    net.train()
    for data, target in train_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
    return train_loss


def valid_frequentist(net, criterion, valid_loader):
    valid_loss = 0.0
    net.eval()
    for data, target in valid_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = net(data)
        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)
    return valid_loss


def run(dataset, net_type, IS_BAYESIAN):

    # Hyper Parameter settings
    n_epochs = cfg.n_epochs
    lr = cfg.lr
    beta_type = cfg.beta_type
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size

    trainset, testset, inputs, outputs = data.getDataset(dataset)
    train_loader, valid_loader, test_loader = data.getDataloader(
        trainset, testset, valid_size, batch_size, num_workers)
    net = getModel(net_type, inputs, outputs, IS_BAYESIAN)

    if IS_BAYESIAN:
        ckpt_dir = f'checkpoints/{dataset}/bayesian'
        ckpt_name = f'checkpoints/{dataset}/bayesian/model_{net_type}.pt'
    else:
        ckpt_dir = f'checkpoints/{dataset}/frequentist'
        ckpt_name = f'checkpoints/{dataset}/frequentist/model_{net_type}.pt'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    if use_cuda:
        net.cuda()

    if IS_BAYESIAN:
        epochs = [80, 60, 40, 20]
        count = 0
        for epoch in epochs:
            optimizer = Adam(net.parameters(), lr=lr)
            for _ in range(epoch):
                train_bayesian(net, optimizer, count, train_loader, trainset, beta_type)
                test_bayesian(net, count, test_loader, ckpt_name)
                count += 1
            lr /= 10
    else:
        # for frequentist
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(net.parameters(), lr=lr)
        valid_loss_min = np.Inf
        for epoch in range(1, n_epochs+1):
            train_loss = train_frequentist(net, optimizer, criterion, train_loader)
            valid_loss = valid_frequentist(net, criterion, valid_loader)

            train_loss = train_loss/len(train_loader.dataset)
            valid_loss = valid_loss/len(valid_loader.dataset)
                
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch, train_loss, valid_loss))
            
            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min, valid_loss))
                torch.save(net.state_dict(), ckpt_name)
                valid_loss_min = valid_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PyTorch Bayesian and Frequentist Model Training")
    parser.add_argument('--net_type', default='alexnet', type=str, help='model')
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset = [MNIST/CIFAR10/CIFAR100]')
    parser.add_argument('--IS_BAYESIAN', default=True, type=utils.str2bool, help='model_type - bayesian/frequentist')
    args = parser.parse_args()

    run(args.dataset, args.net_type, args.IS_BAYESIAN)
