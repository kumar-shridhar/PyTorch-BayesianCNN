import os
import math
import pickle

import torch.cuda
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.datasets as dsets

from utils.BBBlayers import GaussianVariationalInference
from utils.BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from utils.BayesianModels.BayesianAlexNet import BBBAlexNet
from utils.BayesianModels.BayesianELUN1 import BBBELUN1
from utils.BayesianModels.BayesianExperimentalCNNModel import BBBCNN1
from utils.BayesianModels.BayesianLeNet import BBBLeNet
from utils.BayesianModels.BayesianSqueezeNet import BBBSqueezeNet

#from utils.BayesianDataParallel.BBBDataParallel import DataParallel

cuda = torch.cuda.is_available()
#print (cuda)
torch.cuda.set_device(0)

'''
HYPERPARAMETERS
'''
is_training = True  # set to "False" to only run validation
num_samples = 10  # because of Casper's trick
batch_size = 128
beta_type = "Blundell"
net = BBBLeNet
dataset = 'STL10'  # MNIST, CIFAR-10, CIFAR-100 or Monkey species
num_epochs = 100
p_logvar_init = 0
q_logvar_init = -10
lr = 0.001
weight_decay = 0.0005

# dimensions of input and output
if dataset is 'MNIST':    # train with MNIST version
    outputs = 10
    inputs = 1
elif dataset is 'CIFAR-10':  # train with CIFAR-10
    outputs = 10
    inputs = 3
elif dataset is 'CIFAR-100':    # train with CIFAR-100
    outputs = 100
    inputs = 3
elif dataset is 'Monkeys':    # train with Monkey species
    outputs = 10
    inputs = 3
elif dataset is 'STL10':
    outputs=10
    inputs=3
else:
    pass

if net is BBBLeNet:
    resize = 32
elif net is BBB3Conv3FC:
    resize = 32
elif net is BBBAlexNet:
    resize = 32
elif net is BBBELUN1:
    resize = 32
elif net is BBBCNN1:
    resize = 32
elif net is BBBSqueezeNet:
    resize = 224
else:
    pass

'''
LOADING DATASET
'''

if dataset is 'MNIST':
    transform = transforms.Compose([transforms.Resize((resize, resize)), transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = dsets.FashionMNIST(root="data", download=True, transform=transform)
    val_dataset = dsets.FashionMNIST(root="data", download=True, train=False, transform=transform)

elif dataset is 'CIFAR-100':
    transform = transforms.Compose([transforms.Resize((resize, resize)), transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    train_dataset = dsets.CIFAR100(root="data", download=True, transform=transform)
    val_dataset = dsets.CIFAR100(root='data', download=True, train=False, transform=transform)

elif dataset is 'CIFAR-10':
    transform = transforms.Compose([transforms.Resize((resize, resize)), transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    train_dataset = dsets.CIFAR10(root="data", download=True, transform=transform)
    val_dataset = dsets.CIFAR10(root='data', download=True, train=False, transform=transform)

elif dataset is 'Monkeys':
    transform = transforms.Compose([transforms.Resize((resize, resize)), transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    train_dataset = dsets.ImageFolder(root="data/10-monkey-species/training", transform=transform)
    val_dataset = dsets.ImageFolder(root="data/10-monkey-species/validation", transform=transform)
elif dataset is 'STL10':
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

model = net(outputs=outputs, inputs=inputs)
#model = DataParallel(model, device_ids=[0,1]).cuda()

if cuda:
    model.cuda()

#model = torch.nn.DataParallel(model)

'''
INSTANTIATE VARIATIONAL INFERENCE AND OPTIMISER
'''
vi = GaussianVariationalInference(torch.nn.CrossEntropyLoss())
optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

'''
check parameter matrix shapes
'''

# how many parameter matrices do we have?
print('Number of parameter matrices: ', len(list(model.parameters())))

for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())

'''
TRAIN MODEL
'''

logfile = os.path.join('diagnostics_{}_{}.txt'.format(net, dataset))
with open(logfile, 'w') as lf:
    lf.write('')


def run_epoch(loader, epoch, is_training=False):
    m = math.ceil(len(loader.dataset) / loader.batch_size)

    accuracies = []
    likelihoods = []
    kls = []
    losses = []

    for i, (images, labels) in enumerate(loader):
        # Repeat samples (Casper's trick)
        x = images.view(-1, inputs, resize, resize).repeat(num_samples, 1, 1, 1)
        y = labels.repeat(num_samples)

        if cuda:
            x = x.cuda()
            y = y.cuda()

        if beta_type is "Blundell":
            beta = 2 ** (m - (i + 1)) / (2 ** m - 1)
        elif beta_type is "Soenderby":
            beta = min(epoch / (num_epochs//4), 1)
        elif beta_type is "Standard":
            beta = 1 / m
        else:
            beta = 0

        logits, kl = model.probforward(x)
        loss = vi(logits, y, kl, beta)
        ll = -loss.data.mean() + beta*kl.data.mean()

        if is_training:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        _, predicted = logits.max(1)
        accuracy = (predicted.data.cpu() == y.cpu()).float().mean()

        accuracies.append(accuracy)
        losses.append(loss.data.mean())
        kls.append(beta*kl.data.mean())
        likelihoods.append(ll)

    diagnostics = {'loss': sum(losses)/len(losses),
                   'acc': sum(accuracies)/len(accuracies),
                   'kl': sum(kls)/len(kls),
                   'likelihood': sum(likelihoods)/len(likelihoods)}

    return diagnostics


for epoch in range(num_epochs):
    if is_training is True:
        diagnostics_train = run_epoch(loader_train, epoch, is_training=True)
        diagnostics_val = run_epoch(loader_val, epoch)
        diagnostics_train = dict({"type": "train", "epoch": epoch}, **diagnostics_train)
        diagnostics_val = dict({"type": "validation", "epoch": epoch}, **diagnostics_val)
        print(diagnostics_train)
        print(diagnostics_val)

        with open(logfile, 'a') as lf:
            lf.write(str(diagnostics_train))
            lf.write(str(diagnostics_val))
    else:
        diagnostics_val = run_epoch(loader_val, epoch)
        diagnostics_val = dict({"type": "validation", "epoch": epoch}, **diagnostics_val)
        print(diagnostics_val)

        with open(logfile, 'a') as lf:
            lf.write(str(diagnostics_val))

'''
SAVE PARAMETERS
'''
if is_training:
    weightsfile = os.path.join("weights_{}_{}.pkl".format(net,dataset))
    with open(weightsfile, "wb") as wf:
        pickle.dump(model.state_dict(), wf)

