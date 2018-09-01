from __future__ import print_function

import os
import sys
import time
import argparse
import datetime
import math
import pickle


import torchvision
import torchvision.transforms as transforms

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import bayesian_config as cf

from utils.BBBlayers import GaussianVariationalInference
from utils.BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from utils.BayesianModels.BayesianAlexNet import BBBAlexNet
from utils.BayesianModels.BayesianLeNet import BBBLeNet
from utils.BayesianModels.BayesianSqueezeNet import BBBSqueezeNet


parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning_rate')
parser.add_argument('--net_type', default='alexnet', type=str, help='model')
#parser.add_argument('--depth', default=28, type=int, help='depth of model')
#parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--num_samples', default=10, type=int, help='Number of samples')
parser.add_argument('--beta_type', default="Blundell", type=str, help='Beta type')
parser.add_argument('--p_logvar_init', default=0, type=int, help='p_logvar_init')
parser.add_argument('--q_logvar_init', default=-10, type=int, help='q_logvar_init')
parser.add_argument('--weight_decay', default=0.0005, type=float, help='weight_decay')
parser.add_argument('--dataset', default='cifar100', type=str, help='dataset = [mnist/cifar10/cifar100/fashionmnist/stl10]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
args = parser.parse_args()

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
torch.cuda.set_device(0)
best_acc = 0
resize=32
start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type

# Data Uplaod
print('\n[Phase 1] : Data Preparation')

transform_train = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])  # meanstd transformation

transform_test = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])


if (args.dataset == 'cifar10'):
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    outputs = 10
    inputs = 3

elif (args.dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    outputs = 100
    inputs = 3

elif (args.dataset == 'mnist'):
    print("| Preparing MNIST dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_test)
    outputs = 10
    inputs = 1

elif (args.dataset == 'fashionmnist'):
    print("| Preparing FASHIONMNIST dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform_test)
    outputs = 10
    inputs = 1
elif (args.dataset == 'stl10'):
    print("| Preparing STL10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.STL10(root='./data',  split='train', download=True, transform=transform_train)
    testset = torchvision.datasets.STL10(root='./data',  split='test', download=False, transform=transform_test)
    outputs = 10
    inputs = 3

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)


# Return network & file name
def getNetwork(args):
    if (args.net_type == 'lenet'):
        net = BBBLeNet(outputs,inputs)
        file_name = 'lenet'
    elif (args.net_type == 'alexnet'):
        net = BBBAlexNet(outputs,inputs)
        file_name = 'alexnet-'
    elif (args.net_type == 'squeezenet'):
        net = BBBSqueezeNet(outputs,inputs)
        file_name = 'squeezenet-'
    elif (args.net_type == '3conv3fc'):
        net = BBB3Conv3FC(outputs,inputs)
        file_name = '3Conv3FC-'
    else:
        print('Error : Network should be either [LeNet / AlexNet /SqueezeNet/ 3Conv3FC')
        sys.exit(0)

    return net, file_name


# Model
print('\n[Phase 2] : Model setup')
if args.resume:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('| Building net type [' + args.net_type + ']...')
    net, file_name = getNetwork(args)

if use_cuda:
    net.cuda()

vi = GaussianVariationalInference(torch.nn.CrossEntropyLoss())

logfile = os.path.join('diagnostics_Bayes{}_{}.txt'.format(args.net_type, args.dataset))

# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    m = math.ceil(len(trainset) / batch_size)
    optimizer = optim.Adam(net.parameters(), lr=cf.learning_rate(args.lr, epoch), weight_decay=args.weight_decay)

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))
    for batch_idx, (inputs_value, targets) in enumerate(trainloader):

        x = inputs_value.view(-1, inputs, resize, resize).repeat(args.num_samples, 1, 1, 1)
        y = targets.repeat(args.num_samples)
        if use_cuda:
            x, y = x.cuda(), y.cuda() # GPU settings

        if args.beta_type is "Blundell":
            beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
        elif args.beta_type is "Soenderby":
            beta = min(epoch / (num_epochs // 4), 1)
        elif args.beta_type is "Standard":
            beta = 1 / m
        else:
            beta = 0
        # Forward Propagation
        x, y = Variable(x), Variable(y)
        outputs, kl = net.probforward(x)
        loss = vi(outputs, y, kl, beta)  # Loss
        optimizer.zero_grad()
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(y.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, num_epochs, batch_idx+1,
                    (len(trainset)//batch_size)+1, loss.data[0], (100*correct/total)/args.num_samples))
        sys.stdout.flush()

    diagnostics_to_write =  {'Epoch': epoch, 'Loss': loss.data[0], 'Accuracy': (100*correct/total)/args.num_samples}
    with open(logfile, 'a') as lf:
        lf.write(str(diagnostics_to_write))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    m = math.ceil(len(testset) / batch_size)
    for batch_idx, (inputs_value, targets) in enumerate(testloader):
        x = inputs_value.view(-1, inputs, resize, resize).repeat(args.num_samples, 1, 1, 1)
        y = targets.repeat(args.num_samples)
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            x, y = Variable(x), Variable(y)
        outputs, kl = net.probforward(x)

        if args.beta_type is "Blundell":
            beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
        elif args.beta_type is "Soenderby":
            beta = min(epoch / (num_epochs // 4), 1)
        elif args.beta_type is "Standard":
            beta = 1 / m
        else:
            beta = 0

        loss = vi(outputs,y,kl,beta)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(y.data).cpu().sum()

    # Save checkpoint when best model
    acc =(100*correct/total)/args.num_samples
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.data[0], acc))
    test_diagnostics_to_write = {'Validation Epoch':epoch, 'Loss':loss.data[0], 'Accuracy': acc}
    with open(logfile, 'a') as lf:
        lf.write(str(test_diagnostics_to_write))

    if acc > best_acc:
        print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
        state = {
                'net':net if use_cuda else net,
                'acc':acc,
                'epoch':epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_point = './checkpoint/'+args.dataset+os.sep
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        torch.save(state, save_point+file_name+'.t7')
        best_acc = acc

print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))

elapsed_time = 0
for epoch in range(start_epoch, start_epoch+num_epochs):
    start_time = time.time()

    train(epoch)
    test(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))

print('\n[Phase 4] : Testing model')
print('* Test results : Acc@1 = %.2f%%' %(best_acc))

















