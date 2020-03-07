import argparse
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms

import data
import utils
from main_bayesian import getModel


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epistemic_scale = 1e-12
aleatoric_scale = 1e-4


def calculate_uncertainty_over_batch(inputs, labels, net):
    inputs, labels = inputs.to(device), labels.to(device)

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

    return np.mean(epistemic_softmax), np.mean(aleatoric_softmax), \
        np.mean(epistemic_normalized), np.mean(aleatoric_normalized)


def run(net_type, weight_path, not_mnist_dir, num_batches):
    net = getModel(net_type, 1, 10)
    net.load_state_dict(torch.load(weight_path))
    net.to(device)
    net.train()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        ])

    trainset, testset, _, _ = data.getDataset('MNIST')
    mnist_loader, valid_loader, test_loader = data.getDataloader(
        trainset, testset, 0.2, 256, 4)

    train_dataset = torchvision.datasets.ImageFolder(
        root=not_mnist_dir, transform=transform)
    not_mnist_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, num_workers=4, shuffle=True)

    for mini_batch, (inputs, labels) in enumerate(mnist_loader, 1):
        inputs, labels = inputs.to(device), labels.to(device)
        uncertainty = calculate_uncertainty_over_batch(inputs, labels, net)

        print("MNIST Softmax batch-{}: Epistemic:{:.4f} Aleatoric:{:.4f}".format(
            mini_batch, *uncertainty[:2]))
        print("MNIST Normalized batch-{}: Epistemic:{:.4f} Aleatoric:{:.4f}".format(
            mini_batch, *uncertainty[2:]))

        if mini_batch==num_batches: # Till num_batches batches
            break
    print('')
    for mini_batch, (inputs, labels) in enumerate(not_mnist_loader, 1):
        inputs = inputs[:, 0, :, :].unsqueeze(1)
        inputs, labels = inputs.to(device), labels.to(device)
        uncertainty = calculate_uncertainty_over_batch(inputs, labels, net)

        print("notMNIST Softmax batch-{}: Epistemic:{:.4f} Aleatoric:{:.4f}".format(
            mini_batch, *uncertainty[:2]))
        print("notMNIST Normalized batch-{}: Epistemic:{:.4f} Aleatoric:{:.4f}".format(
            mini_batch, *uncertainty[2:]))

        if mini_batch==num_batches: # Till num_batches batches
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PyTorch Uncertainty Estimation b/w MNIST and notMNIST")
    parser.add_argument('--net_type', default='lenet', type=str, help='model')
    parser.add_argument('--weights_path', default='checkpoints/MNIST/bayesian/model_lenet.pt', type=str, help='weights for model')
    parser.add_argument('--not_mnist_dir', default='data/notMNIST_small/', type=str, help='weights for model')
    parser.add_argument('--num_batches', default=1, type=int, help='numberof batches')
    args = parser.parse_args()

    run(args.net_type, args.weights_path, args.not_mnist_dir, args.num_batches)
