from __future__ import print_function

import os
import time
import argparse

import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.nn import functional as F

import data
import utils
import metrics
from logger import Logger
import config_bayesian as cfg
from models.BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from models.BayesianModels.BayesianAlexNet import BBBAlexNet
from models.BayesianModels.BayesianLeNet import BBBLeNet

# CUDA settings
use_cuda = torch.cuda.is_available()
torch.cuda.set_device(0)


def getModel(net_type, inputs, outputs):
    if (net_type == 'lenet'):
        return BBBLeNet(outputs,inputs)
    elif (net_type == 'alexnet'):
        return BBBAlexNet(outputs, inputs)
    elif (net_type == '3conv3fc'):
        return BBB3Conv3FC(outputs,inputs)
    else:
        raise ValueError('Network should be either [LeNet / AlexNet / 3Conv3FC')


def run_an_epoch(net, optimizer, criterion, trainloader, testloader, logger, train_ens, epoch):
    net.train()
    training_loss = 0
    accs = []
    steps = 0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        steps += 1
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        optimizer.zero_grad()

        outputs = Variable(torch.zeros(inputs.shape[0], net.num_classes, train_ens).cuda())
        for j in range(train_ens):
            outputs[:, :, j] = F.log_softmax(net(inputs), dim=1)
        log_outputs = utils.logmeanexp(outputs, dim=2)

        loss = criterion(log_outputs, labels)
        loss.backward()
        optimizer.step()

        accs.append(metrics.logit2acc(log_outputs.data, labels))
        training_loss += loss.cpu().data.numpy()

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


    # Stochastic test
    net.train()
    acc, nll = utils.evaluate(net, testloader, num_ens=1)
    logger.add(epoch, te_nll_stoch=nll, te_acc_stoch=acc)

    # Test-time averaging
    net.train()
    acc, nll = utils.evaluate(net, testloader, num_ens=10)
    logger.add(epoch, te_nll_ens10=nll, te_acc_ens10=acc)


def run(dataset, net_type):

    # Hyper Parameter settings
    train_ens = cfg.train_ens
    logger_fmt = cfg.logger_fmt

    n_epochs = cfg.n_epochs
    lr_start = cfg.lr_start
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size

    logger = Logger(net_type, fmt=logger_fmt)

    trainset, testset, inputs, outputs = data.getDataset(dataset)
    train_loader, valid_loader, test_loader = data.getDataloader(
        trainset, testset, valid_size, batch_size, num_workers)

    net = getModel(net_type, inputs, outputs)
    if use_cuda:
        net.cuda()
    logger.print(net)

    ckpt_dir = f'checkpoints/{dataset}/bayesian'
    ckpt_name = f'checkpoints/{dataset}/bayesian/model_{net_type}.pt'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    criterion = metrics.ELBO(net, len(trainset)).cuda()
    optimizer = Adam(net.parameters(), lr=lr_start)

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        t0 = time.time()
        utils.adjust_learning_rate(optimizer, metrics.lr_linear(epoch, 0, n_epochs, lr_start))
        run_an_epoch(net, optimizer, criterion, train_loader, test_loader, logger, train_ens, epoch)
        logger.add(epoch, time=time.time()-t0)
        logger.iter_info()
        logger.save(silent=True)
        torch.save(net.state_dict(), logger.checkpoint)
    logger.save()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PyTorch Bayesian Model Training")
    parser.add_argument('--net_type', default='alexnet', type=str, help='model')
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset = [MNIST/CIFAR10/CIFAR100]')
    args = parser.parse_args()

    run(args.dataset, args.net_type)
