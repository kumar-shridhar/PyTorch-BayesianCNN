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
from utils.autoaugment import CIFAR10Policy

import torch
import torch.utils.data as data
import numpy as np
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


parser = argparse.ArgumentParser(description='PyTorch Bayesian Model Training')
#parser.add_argument('--lr', default=0.001, type=float, help='learning_rate')
parser.add_argument('--net_type', default='lenet', type=str, help='model')
#parser.add_argument('--depth', default=28, type=int, help='depth of model')
#parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
#parser.add_argument('--num_samples', default=10, type=int, help='Number of samples')
#parser.add_argument('--beta_type', default="Blundell", type=str, help='Beta type')
#parser.add_argument('--p_logvar_init', default=0, type=int, help='p_logvar_init')
#parser.add_argument('--q_logvar_init', default=-10, type=int, help='q_logvar_init')
#parser.add_argument('--weight_decay', default=0.0005, type=float, help='weight_decay')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset = [MNIST/CIFAR10/CIFAR100]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
args = parser.parse_args()

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
torch.cuda.set_device(0)
best_acc = 0
resize=32

# Data Uplaod
print('\n[Phase 1] : Data Preparation')

transform_train = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.RandomCrop(32, padding=4),
    #transforms.RandomHorizontalFlip(),
    #CIFAR10Policy(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])  # meanstd transformation

transform_test = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.RandomCrop(32, padding=4),
    #transforms.RandomHorizontalFlip(),
    #CIFAR10Policy(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])



print("| Preparing {} dataset...".format(args.dataset))
dataset_prep = 'torchvision.datasets.{}'.format(args.dataset)
trainset = dataset_prep(root='./data', train=True, download=True, transform=transform_train)
testset = dataset_prep(root='./data', train=False, download=False, transform=transform_test)


if (dataset_name == 'CIFAR10'):
    outputs = 10
    inputs = 3

elif (dataset_name == 'CIFAR100'):
    outputs = 100
    inputs = 3

elif (dataset_name == 'MNIST'):
    outputs = 10
    inputs = 1

trainloader = torch.utils.data.DataLoader(trainset, batch_size=cf.batch_size, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=cf.batch_size, shuffle=False, num_workers=4)





# Return network & file name
def getNetwork(args):
    if (args.net_type == 'lenet'):
        net = BBBLeNet(outputs,inputs)
        file_name = 'lenet'
    elif (args.net_type == 'alexnet'):
        net = BBBAlexNet(outputs,inputs)
        file_name = 'alexnet-'
    elif (args.net_type == '3conv3fc'):
        net = BBB3Conv3FC(outputs,inputs)
        file_name = '3Conv3FC-'
    else:
        print('Error : Network should be either [LeNet / AlexNet / 3Conv3FC')
        sys.exit(0)

    return net, file_name


# Model
print('\n[Phase 2] : Model setup')
if args.resume:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+str(cf.num_samples)+'.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    cf.start_epoch = checkpoint['epoch']
else:
    print('| Building net type [' + args.net_type + ']...')
    net, file_name = getNetwork(args)

if use_cuda:
    net.cuda()

vi = GaussianVariationalInference(torch.nn.CrossEntropyLoss())

logfile = os.path.join('diagnostics_Bayes{}_{}_{}.txt'.format(args.net_type, args.dataset, cf.num_samples))
value_file = os.path.join("values{}_{}.txt".format(args.net_type, args.dataset))

# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.Adam(net.parameters(), lr=cf.learning_rate(cf.lr, epoch), weight_decay=cf.weight_decay)

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(cf.lr, epoch)))
    m = math.ceil(len(trainset) / cf.batch_size)
    for batch_idx, (inputs_value, targets) in enumerate(trainloader):

        x = inputs_value.view(-1, inputs, resize, resize).repeat(cf.num_samples, 1, 1, 1)
        y = targets.repeat(cf.num_samples)
        if use_cuda:
            x, y = x.cuda(), y.cuda() # GPU settings

        if cf.beta_type is "Blundell":
            beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
        elif cf.beta_type is "Soenderby":
            beta = min(epoch / (cf.num_epochs // 4), 1)
        elif cf.beta_type is "Standard":
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
                %(epoch, cf.num_epochs, batch_idx+1,
                    (len(trainset)//cf.batch_size)+1, loss.data[0], (100*correct/total)/cf.num_samples))
        sys.stdout.flush()

    diagnostics_to_write =  {'Epoch': epoch, 'Loss': loss.data[0], 'Accuracy': (100*correct/total)/cf.num_samples}
    with open(logfile, 'a') as lf:
        lf.write(str(diagnostics_to_write))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    conf=[]
    m = math.ceil(len(testset) / cf.batch_size)
    for batch_idx, (inputs_value, targets) in enumerate(testloader):
        x = inputs_value.view(-1, inputs, resize, resize).repeat(cf.num_samples, 1, 1, 1)
        y = targets.repeat(cf.num_samples)
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            x, y = Variable(x), Variable(y)
        outputs, kl = net.probforward(x)

        if cf.beta_type is "Blundell":
            beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
        elif cf.beta_type is "Soenderby":
            beta = min(epoch / (cf.num_epochs // 4), 1)
        elif cf.beta_type is "Standard":
            beta = 1 / m
        else:
            beta = 0

        loss = vi(outputs,y,kl,beta)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        preds = F.softmax(outputs, dim=1)
        #print(preds)
        results = torch.topk(preds.cpu().data, k=1, dim=1)
        #print(results[0][0].item())
        conf.append(results[0][0].item())
        total += targets.size(0)
        correct += preds
        dicted.eq(y.data).cpu().sum()

    # Save checkpoint when best model
    #print (conf)
    p_hat=np.array(conf)
    #print (p_hat)
    confidence_mean=np.mean(p_hat, axis=0)
    confidence_var=np.var(p_hat, axis=0)
    epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2
    aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)

    print ("Epistemic Uncertainity is: ", epistemic)
    print("Aleatoric Uncertainity is: ", aleatoric)
    print("Mean is: ", confidence_mean)
    print("Variance is: ", confidence_var)
    """
    conv1_var = dict(net.named_parameters())['conv1.qw_logvar'][0]
    print(conv1_var)
    print('--------------------')
    conv1_mean = dict(net.named_parameters())['conv1.qw_mean'][0]
    print(conv1_mean)
    print('--------------------')
    conv1_si = dict(net.named_parameters())['conv1.conv_qw_si'][0]
    print(conv1_si)
    print('--------------------')
    conv1_alpha = dict(net.named_parameters())['conv1.log_alpha'][0]
    print(conv1_alpha)
    print('--------------------')
    """
    acc =(100*correct/total)/cf.num_samples
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.data[0], acc))
    test_diagnostics_to_write = {'Validation Epoch':epoch, 'Loss':loss.data[0], 'Accuracy': acc}
    values_to_write={'Epoch':epoch, 'Confidence Mean: ':confidence_mean, 'Confidence Variance:':confidence_var, 'Epistemic Uncertainity: ': epistemic, 'Aleatoric Uncertainity: ':aleatoric}
    with open(logfile, 'a') as lf:
        lf.write(str(test_diagnostics_to_write))
    with open(value_file, 'a') as lf:
        lf.write(str(values_to_write))

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
        torch.save(state, save_point+file_name+str(cf.num_samples)+'.t7')
        best_acc = acc

print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(cf.num_epochs))
print('| Initial Learning Rate = ' + str(cf.lr))
print('| Optimizer = ' + str(cf.optim_type))

elapsed_time = 0
for epoch in range(cf.start_epoch, cf.start_epoch+cf.num_epochs):
    start_time = time.time()

    train(epoch)
    test(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))

print('\n[Phase 4] : Testing model')
print('* Test results : Acc@1 = %.2f%%' %(best_acc))

















