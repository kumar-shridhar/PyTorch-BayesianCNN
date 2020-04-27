import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


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


def extract_classes(dataset, classes):
    idx = torch.zeros_like(dataset.targets, dtype=torch.bool)
    for target in classes:
        idx = idx | (dataset.targets==target)

    data, targets = dataset.data[idx], dataset.targets[idx]
    return data, targets


def getDataset(dataset):
    transform_split_mnist = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        ])

    transform_mnist = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        ])

    transform_cifar = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    if(dataset == 'CIFAR10'):
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)
        num_classes = 10
        inputs=3

    elif(dataset == 'CIFAR100'):
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_cifar)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_cifar)
        num_classes = 100
        inputs = 3
        
    elif(dataset == 'MNIST'):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
        num_classes = 10
        inputs = 1

    elif(dataset == 'SplitMNIST-2.1'):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)

        train_data, train_targets = extract_classes(trainset, [0, 1, 2, 3, 4])
        test_data, test_targets = extract_classes(testset, [0, 1, 2, 3, 4])

        trainset = CustomDataset(train_data, train_targets, transform=transform_split_mnist)
        testset = CustomDataset(test_data, test_targets, transform=transform_split_mnist)
        num_classes = 5
        inputs = 1

    elif(dataset == 'SplitMNIST-2.2'):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)

        train_data, train_targets = extract_classes(trainset, [5, 6, 7, 8, 9])
        test_data, test_targets = extract_classes(testset, [5, 6, 7, 8, 9])
        train_targets -= 5 # Mapping target 5-9 to 0-4
        test_targets -= 5 # Hence, add 5 after prediction

        trainset = CustomDataset(train_data, train_targets, transform=transform_split_mnist)
        testset = CustomDataset(test_data, test_targets, transform=transform_split_mnist)
        num_classes = 5
        inputs = 1

    elif(dataset == 'SplitMNIST-5.1'):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)

        train_data, train_targets = extract_classes(trainset, [0, 1])
        test_data, test_targets = extract_classes(testset, [0, 1])

        trainset = CustomDataset(train_data, train_targets, transform=transform_split_mnist)
        testset = CustomDataset(test_data, test_targets, transform=transform_split_mnist)
        num_classes = 2
        inputs = 1

    elif(dataset == 'SplitMNIST-5.2'):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)

        train_data, train_targets = extract_classes(trainset, [2, 3])
        test_data, test_targets = extract_classes(testset, [2, 3])
        train_targets -= 2 # Mapping target 2-3 to 0-1
        test_targets -= 2 # Hence, add 2 after prediction

        trainset = CustomDataset(train_data, train_targets, transform=transform_split_mnist)
        testset = CustomDataset(test_data, test_targets, transform=transform_split_mnist)
        num_classes = 2
        inputs = 1

    elif(dataset == 'SplitMNIST-5.3'):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)

        train_data, train_targets = extract_classes(trainset, [4, 5])
        test_data, test_targets = extract_classes(testset, [4, 5])
        train_targets -= 4 # Mapping target 4-5 to 0-1
        test_targets -= 4 # Hence, add 4 after prediction

        trainset = CustomDataset(train_data, train_targets, transform=transform_split_mnist)
        testset = CustomDataset(test_data, test_targets, transform=transform_split_mnist)
        num_classes = 2
        inputs = 1

    elif(dataset == 'SplitMNIST-5.4'):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)

        train_data, train_targets = extract_classes(trainset, [6, 7])
        test_data, test_targets = extract_classes(testset, [6, 7])
        train_targets -= 6 # Mapping target 6-7 to 0-1
        test_targets -= 6 # Hence, add 6 after prediction

        trainset = CustomDataset(train_data, train_targets, transform=transform_split_mnist)
        testset = CustomDataset(test_data, test_targets, transform=transform_split_mnist)
        num_classes = 2
        inputs = 1

    elif(dataset == 'SplitMNIST-5.5'):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)

        train_data, train_targets = extract_classes(trainset, [8, 9])
        test_data, test_targets = extract_classes(testset, [8, 9])
        train_targets -= 8 # Mapping target 8-9 to 0-1
        test_targets -= 8 # Hence, add 8 after prediction

        trainset = CustomDataset(train_data, train_targets, transform=transform_split_mnist)
        testset = CustomDataset(test_data, test_targets, transform=transform_split_mnist)
        num_classes = 2
        inputs = 1

    return trainset, testset, inputs, num_classes


def getDataloader(trainset, testset, valid_size, batch_size, num_workers):
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
        sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
        num_workers=num_workers)

    return train_loader, valid_loader, test_loader
