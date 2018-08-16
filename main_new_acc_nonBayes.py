import os
import pickle

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn as nn

from utils.NonBayesianModels.AlexNet import AlexNet
from utils.NonBayesianModels.LeNet import LeNet
from utils.NonBayesianModels.ELUN1 import ELUN1
from utils.NonBayesianModels.ExperimentalCNNModel import CNN1
from utils.NonBayesianModels.SqueezeNet import SqueezeNet
from utils.NonBayesianModels.ThreeConvThreeFC import ThreeConvThreeFC

cuda = torch.cuda.is_available()
torch.cuda.set_device(1)
#device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

'''
HYPERPARAMETERS
'''
is_training = True  # set to "False" to only run validation
net = LeNet
batch_size = 1024
dataset = 'CIFAR-100'  # MNIST, CIFAR-10, CIFAR-100, Monkey species or LSUN
num_epochs = 1000
lr = 0.001
weight_decay = 0.0005

if net is LeNet:
    resize = 32
elif net is ThreeConvThreeFC:
    resize = 32
elif net is AlexNet:
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
#model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
model = net(outputs=outputs, inputs=inputs).cuda()


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

total_step = len(loader_train)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(loader_train):
        images = images.cuda()
        labels = labels.cuda()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in loader_val:
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy on test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model_new_acc.ckpt')
