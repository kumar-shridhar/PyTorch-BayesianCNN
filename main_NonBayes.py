import os
import pickle

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn as nn

from utils.NonBayesianModels.LeNet import LeNet
from utils.NonBayesianModels.ELUN1 import ELUN1
from utils.NonBayesianModels.ExperimentalCNNModel import CNN1
from utils.NonBayesianModels.SqueezeNet import SqueezeNet
from utils.NonBayesianModels.ThreeConvThreeFC import ThreeConvThreeFC


cuda = torch.cuda.is_available()
#torch.cuda.set_device(1)

'''
HYPERPARAMETERS
'''
is_training = True  # set to "False" to only run validation
net = LeNet
batch_size = 512
dataset = 'STL10'  # MNIST, CIFAR-10, CIFAR-100, Monkey species or LSUN
num_epochs = 100
lr = 0.001
weight_decay = 0.0005

if net is LeNet:
    resize = 32
elif net is ThreeConvThreeFC:
    resize = 32
elif net is ELUN1:
    resize = 32
elif net is CNN1:
    resize = 32
elif net is SqueezeNet:
    resize = 224
else:
    pass

'''
LOADING DATASET
'''
if dataset is 'MNIST':
    outputs = 10
    inputs = 1
    transform = transforms.Compose([transforms.Resize((resize, resize)), transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = dsets.FashionMNIST(root="data", download=True, transform=transform)
    val_dataset = dsets.FashionMNIST(root="data", download=True, train=False, transform=transform)

elif dataset is 'CIFAR-100':
    outputs = 100
    inputs = 3
    transform = transforms.Compose([transforms.Resize((resize, resize)), transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    train_dataset = dsets.CIFAR100(root="data", download=True, transform=transform)
    val_dataset = dsets.CIFAR100(root="data", download=True, train=False, transform=transform)

elif dataset is 'CIFAR-10':
    outputs = 10
    inputs = 3
    transform = transforms.Compose([transforms.Resize((resize, resize)), transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    train_dataset = dsets.CIFAR10(root="data", download=True, transform=transform)
    val_dataset = dsets.CIFAR10(root="data", download=True, train=False, transform=transform)

elif dataset is 'Monkeys':
    outputs = 10
    inputs = 3
    transform = transforms.Compose([transforms.Resize((resize, resize)), transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    train_dataset = dsets.ImageFolder(root="data/10-monkey-species/training", transform=transform)
    val_dataset = dsets.ImageFolder(root="data/10-monkey-species/validation", transform=transform)

elif dataset is 'LSUN':
    outputs = 10
    inputs = 3
    transform = transforms.Compose([transforms.Resize((resize, resize)), transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    train_dataset = dsets.LSUN(root="data/lsun", classes="train", transform=transform)
    val_dataset = dsets.LSUN(root="data/lsun", classes="val", transform=transform)
elif dataset is 'STL10':
    outputs = 10
    inputs = 3
    transform = transforms.Compose([transforms.Resize((resize, resize)), transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    train_dataset = dsets.STL10(root="data/", transform=transform,download=True)
    val_dataset = dsets.STL10(root="data/", transform=transform,download=True)

'''
MAKING DATASET ITERABLE
'''

print('length of training dataset:', len(train_dataset))
n_iterations = num_epochs * (len(train_dataset) / batch_size)
n_iterations = int(n_iterations)
print('Number of iterations: ', n_iterations)

loader_train = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

loader_val = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)


'''
INSTANTIATE MODEL
'''

model = net(num_classes=outputs, inputs=inputs)
model = torch.nn.DataParallel(model, device_ids=[0]).cuda()

if cuda:
    model.cuda()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

logfile = os.path.join('diagnostics_MLE_F.txt')
with open(logfile, 'w') as lf:
    lf.write('')


def run_epoch(loader):
    accuracies = []
    losses = []

    for i, (images, labels) in enumerate(loader):

        x = images.view(-1, inputs, resize, resize)
        y = labels

        if cuda:
            x = x.cuda()
            y = y.cuda()

        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)

        if is_training:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        _, predicted = outputs.max(1)
        accuracy = (predicted.data.cpu() == y.cpu()).float().mean()

        accuracies.append(accuracy)
        losses.append(loss.item())

    diagnostics = {'loss': sum(losses) / len(losses),
                   'acc': sum(accuracies) / len(accuracies)}

    return diagnostics


for epoch in range(num_epochs):
    if is_training is True:
        diagnostics_train = run_epoch(loader_train)
        diagnostics_val = run_epoch(loader_val)
        diagnostics_train = dict({"type": "train", "epoch": epoch}, **diagnostics_train)
        diagnostics_val = dict({"type": "validation", "epoch": epoch}, **diagnostics_val)
        print(diagnostics_train)
        print(diagnostics_val)

        with open(logfile, 'a') as lf:
            lf.write(str(diagnostics_train))
            lf.write(str(diagnostics_val))
    else:
        diagnostics_val = run_epoch(loader_val)
        diagnostics_val = dict({"type": "validation", "epoch": epoch}, **diagnostics_val)
        print(diagnostics_val)

        with open(logfile, 'a') as lf:
            lf.write(str(diagnostics_val))

'''
SAVE PARAMETERS
'''
if is_training:
    weightsfile = os.path.join("weights_MLE.pkl")
    with open(weightsfile, "wb") as wf:
        pickle.dump(model.state_dict(), wf)

