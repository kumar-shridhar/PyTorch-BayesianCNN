import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms

import data
import utils
from models.BayesianModels.BayesianLeNet import BBBLeNet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    ])

net = BBBLeNet(10, 1)
net.load_state_dict(torch.load('checkpoints/MNIST/bayesian/model_lenet.pt'))
net.cuda()

trainset, testset, inputs, outputs = data.getDataset('MNIST')
mnist_loader, valid_loader, test_loader = data.getDataloader(
    trainset, testset, 0.2, 256, 4)

data_path = '../notMNIST_small/'
train_dataset = torchvision.datasets.ImageFolder(
    root=data_path, transform=transform)
not_minst_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=256, num_workers=0, shuffle=True)

net.train()

epistemic_scale = 1e-12
aleatoric_scale = 1e-4

for mini_batch, (inputs, labels) in enumerate(mnist_loader, 1):
    inputs, labels = inputs.to(device), labels.to(device)

    if mini_batch==6: # Till 5 batches
        break

    epistemic_softmax = []
    aleatoric_softmax = []
    epistemic_normalized = []
    aleatoric_normalized = []

    for i in range(labels.shape[0]):
        uncertainty_softmax = list(utils.calc_uncertainty_softmax(net, inputs[i]))
        uncertainty_softmax[0] /= epistemic_scale
        uncertainty_softmax[1] /= aleatoric_scale

        uncertainty_normalized = list(utils.calc_uncertainty_normalized(net, inputs[i]))
        uncertainty_normalized[0] /= epistemic_scale
        uncertainty_normalized[1] /= aleatoric_scale

        epistemic_softmax.append(uncertainty_softmax[0])
        aleatoric_softmax.append(uncertainty_softmax[1])
        epistemic_normalized.append(uncertainty_normalized[0])
        aleatoric_normalized.append(uncertainty_normalized[1])

    print("MNIST Softmax batch-{}: Epistemic:{:06} Aleatoric:{:06}".format(
        mini_batch, np.mean(epistemic_softmax), np.mean(aleatoric_softmax)))
    print("MNIST Normalized batch-{}: Epistemic:{:06} Aleatoric:{:06}".format(
        mini_batch, np.mean(epistemic_normalized), np.mean(aleatoric_normalized)))

print("")

for mini_batch, (inputs, labels) in enumerate(not_minst_loader, 1):
    inputs = inputs[:, 0, :, :].unsqueeze(1)
    inputs, labels = inputs.to(device), labels.to(device)

    if mini_batch==6: # Till 5 batches
        break

    epistemic_softmax = []
    aleatoric_softmax = []
    epistemic_normalized = []
    aleatoric_normalized = []

    for i in range(labels.shape[0]):
        uncertainty_softmax = list(utils.calc_uncertainty_softmax(net, inputs[i]))
        uncertainty_softmax[0] /= epistemic_scale
        uncertainty_softmax[1] /= aleatoric_scale

        uncertainty_normalized = list(utils.calc_uncertainty_normalized(net, inputs[i]))
        uncertainty_normalized[0] /= epistemic_scale
        uncertainty_normalized[1] /= aleatoric_scale

        epistemic_softmax.append(uncertainty_softmax[0])
        aleatoric_softmax.append(uncertainty_softmax[1])
        epistemic_normalized.append(uncertainty_normalized[0])
        aleatoric_normalized.append(uncertainty_normalized[1])

    print("notMNIST Softmax batch-{}: Epistemic:{:06} Aleatoric:{:06}".format(
        mini_batch, np.mean(epistemic_softmax), np.mean(aleatoric_softmax)))
    print("notMNIST Normalized batch-{}: Epistemic:{:06} Aleatoric:{:06}".format(
        mini_batch, np.mean(epistemic_normalized), np.mean(aleatoric_normalized)))
