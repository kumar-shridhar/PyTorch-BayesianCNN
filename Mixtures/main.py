"""
It is assumed that `weights_dir` contains bayesian models trained on individual
tasks using Bayes by Backprop on trainset.

This script will be used to train MixtureModel having Gaussian Mixture weights
on validset using Cross_Entropy_Loss as a loss function. And test the same on testset.
"""
import sys
sys.path.append('..')

import copy
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

import metrics
import utils_mixture
from mixture_models import MixtureLeNet
from main_bayesian import train_model, validate_model
import config_mixtures as cfg

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def densite_theorique(pi, mu, sigma, x):
    K = mu.shape[0]
    y = 0
    for i in range(K):
        y += pi[i] * sps.norm.pdf(x, loc=mu[i], scale=sigma[i])
    return y.reshape(x.shape)


def plot_gmm(model, layer, node, interval=None, bias=False):
    if layer.startswith('conv'):
        assert len(node) == 4 or len(node) == 3
    if layer.startswith('fc') or layer == "classifier":
        assert len(node) == 2 or len(node) == 1

    node = tuple(list(node) + [slice(None),])
    pi = getattr(model, layer).W_pi.data[node]
    mu = getattr(model, layer).W_mu.data[node] if not bias else getattr(model, layer).bias_mu.data[node]
    rho = getattr(model, layer).W_rho.data[node] if not bias else getattr(model, layer).bias_rho.data[node]

    pi = pi / pi.sum()
    sigma = torch.log1p(torch.exp(rho))

    if not interval:
        interval = (-1, 1)

    x = np.linspace(*interval, num=1000)
    pdf = densite_theorique(pi.cpu(), mu.cpu(), sigma.cpu(), x)
    pdf = pdf / pdf.sum()
    plt.plot(x, pdf)
    plt.show()


def plot_gmm_initial_final(model_init, model_final, layer, node, interval=None, bias=False):
    if layer.startswith('conv'):
        assert len(node) == 4 or len(node) == 3
    if layer.startswith('fc') or layer == "classifier":
        assert len(node) == 2 or len(node) == 1

    node = tuple(list(node) + [slice(None),])
    pi = getattr(model_init, layer).W_pi.data[node]
    mu = getattr(model_init, layer).W_mu.data[node] if not bias else getattr(model_init, layer).bias_mu.data[node]
    rho = getattr(model_init, layer).W_rho.data[node] if not bias else getattr(model_init, layer).bias_rho.data[node]

    pi = pi / pi.sum()
    sigma = torch.log1p(torch.exp(rho))

    print("Initial pi:", pi)
    print("Initial mu:", mu)
    print("Initial sigma:", sigma)

    if not interval:
        interval = (-1, 1)

    x = np.linspace(*interval, num=1000)
    pdf_init = densite_theorique(pi.cpu(), mu.cpu(), sigma.cpu(), x)
    pdf_init = pdf_init / pdf_init.sum()
    plt.plot(x, pdf_init, c='b', label='Before Mixture Training')


    pi = getattr(model_final, layer).W_pi.data[node]
    mu = getattr(model_final, layer).W_mu.data[node] if not bias else getattr(model_final, layer).bias_mu.data[node]
    rho = getattr(model_final, layer).W_rho.data[node] if not bias else getattr(model_final, layer).bias_rho.data[node]

    pi = pi / pi.sum()
    sigma = torch.log1p(torch.exp(rho))

    print("Final pi:", pi)
    print("Final mu:", mu)
    print("Final sigma:", sigma)

    if not interval:
        interval = (-1, 1)

    x = np.linspace(*interval, num=1000)
    pdf_final = densite_theorique(pi.cpu(), mu.cpu(), sigma.cpu(), x)
    pdf_final = pdf_final / pdf_final.sum()
    plt.plot(x, pdf_final, c='r', label='After Mixture Training')
    
    plt.legend(loc="upper left")
    plt.title(f"Distribution for weight ({node[0]}, {node[1]}) in {layer}")
    plt.xlabel("Value")
    plt.ylabel("Probability Density")

    plt.show()


def plot_gaussian_initial_final(model_init, model_final, node, interval=None, bias=False):
    layer = 'fc3'
    assert len(node) == 2 or len(node) == 1

    node = tuple(node)
    pi = torch.tensor([1])
    mu = getattr(model_init, layer).W_mu.data[node] if not bias else getattr(model_init, layer).bias_mu.data[node]
    rho = getattr(model_init, layer).W_rho.data[node] if not bias else getattr(model_init, layer).bias_rho.data[node]

    sigma = torch.log1p(torch.exp(rho))

    print("Initial pi:", pi)
    print("Initial mu:", mu)
    print("Initial sigma:", sigma)

    if not interval:
        interval = (-1, 1)

    x = np.linspace(*interval, num=1000)
    pdf_init = densite_theorique(pi.cpu(), mu.unsqueeze(0).cpu(), sigma.unsqueeze(0).cpu(), x)
    pdf_init = pdf_init / pdf_init.sum()
    plt.plot(x, pdf_init, c='b', label='initial')


    pi = torch.tensor([1])
    mu = getattr(model_final, layer).W_mu.data[node] if not bias else getattr(model_final, layer).bias_mu.data[node]
    rho = getattr(model_final, layer).W_rho.data[node] if not bias else getattr(model_final, layer).bias_rho.data[node]

    sigma = torch.log1p(torch.exp(rho))

    print("Final pi:", pi)
    print("Final mu:", mu)
    print("Final sigma:", sigma)

    if not interval:
        interval = (-1, 1)

    x = np.linspace(*interval, num=1000)
    pdf_final = densite_theorique(pi.cpu(), mu.unsqueeze(0).cpu(), sigma.unsqueeze(0).cpu(), x)
    pdf_final = pdf_final / pdf_final.sum()
    plt.plot(x, pdf_final, c='r', label='final')

    plt.legend(loc="upper left")
    plt.title(f"Distribution for weight ({node[0]}, {node[1]}) in fc3 (last layer)")
    plt.xlabel("Value")
    plt.ylabel("Probability Density")

    plt.show()


def get_individual_weights(num_tasks, weights_dir):
    task_weights = []
    for i in range(1, num_tasks + 1):
        weight_path = weights_dir + f"model_lenet_bbb_softplus_{num_tasks}.{i}.pt"
        task_weights.append(torch.load(weight_path, map_location=device))
    return task_weights


if __name__ == '__main__':
    num_tasks = 5
    weights_dir = "checkpoints/MNIST/bayesian/splitted/{}-tasks/".format(num_tasks)
    ckpt_name = weights_dir + f"mixture_model_{num_tasks}.pt"
    lr_start = 0.001
    n_epochs = 50
    train_ens = 10
    valid_ens = 10

    pickle_path = "checkpoints/MNIST/bayesian/splitted/{}-tasks/loaders.pkl".format(num_tasks)
    data_loaders, _ = utils_mixture.unpickle_dataloaders(pickle_path)

    individual_weights = get_individual_weights(num_tasks, weights_dir)
    net = MixtureLeNet(10, 1, num_tasks, individual_weights)
    net.cuda()

    net_init = copy.deepcopy(net)

    # TODO: get shuffled validloaders with incremented labels
    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr_start)
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, verbose=True)

    tl = []
    vl = []
    ta = []
    va = []

    valid_loss_max = np.Inf
    for epoch in range(n_epochs):
        # training loop
        train_loss = 0.0
        accs = []
        for t in range(num_tasks):
            for inputs, target in data_loaders[t][1]:  # valid_loaders
                optimizer.zero_grad()

                inputs, target = inputs.to(device), target.to(device)
                target += t * 10 // num_tasks  # last fc has 10 nodes

                outputs = torch.zeros(inputs.shape[0], net.num_classes, train_ens).to(device)
                for j in range(train_ens):
                    net_out, _ = net(inputs)
                    outputs[:, :, j] = net_out

                output = outputs.mean(dim=2)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                cfg.distribution_updated = True
                train_loss += loss.item() / len(data_loaders[t][1])
                accs.append(metrics.acc(output.detach(), target))
        train_acc = np.mean(accs)

        # validation loop
        valid_loss = 0.0
        accs = []
        for t in range(num_tasks):
            for inputs, target in data_loaders[t][2]:  # test_loaders
                inputs, target = inputs.to(device), target.to(device)
                target += t * 10 // num_tasks  # last fc has 10 nodes

                outputs = torch.zeros(inputs.shape[0], net.num_classes, train_ens).to(device)
                for j in range(train_ens):
                    net_out, _ = net(inputs)
                    outputs[:, :, j] = net_out

                output = outputs.mean(dim=2)
                loss = criterion(output, target)
                valid_loss += loss.item() / len(data_loaders[t][2])
                accs.append(metrics.acc(output.detach(), target))
        valid_acc = np.mean(accs)

        print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f}'.format(
            epoch, train_loss, train_acc, valid_loss, valid_acc))

        tl.append(train_loss)
        vl.append(valid_loss)
        ta.append(train_acc.item())
        va.append(valid_acc.item())

        lr_sched.step(valid_loss)

        # save model if validation accuracy has increased
        if valid_loss <= valid_loss_max:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_max, valid_loss))
            torch.save(net.state_dict(), ckpt_name)
            valid_loss_max = valid_loss

    # layer = 'fc2'
    # print("Distributions of layer", layer)
    # for i in range(10):
    #     node = [np.random.randint(0, 84), np.random.randint(0, 120)]
    #     print("Node:", node)
    #     plot_gmm_initial_final(net_init, net, layer, node)
    #     print()

    # print("Distributions for layer fc3")
    # for i in range(10):
    #     node = [np.random.randint(0, 10), np.random.randint(0, 84)]
    #     print("Node:", node)
    #     plot_gaussian_initial_final(net_init, net, node)
    #     print()

    plt.plot(range(1, n_epochs + 1), tl, c='r', label='Training Loss')
    plt.plot(range(1, n_epochs + 1), vl, c='b', label='Validation Loss')
    plt.legend(loc="upper left")
    plt.title(f"Mixture Model Loss {num_tasks} tasks\n lr = {lr_start}    num_ensemble = 10")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    ta = np.array(ta) * 100
    va = np.array(va) * 100

    plt.plot(range(1, n_epochs + 1), ta, c='r', label='Training Accuracy')
    plt.plot(range(1, n_epochs + 1), va, c='b', label='Validation Accuracy')
    plt.legend(loc="upper left")
    plt.title(f"Mixture Model Accuracy(%) {num_tasks} tasks\n lr = {lr_start}    num_ensemble = 10")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy(%)")
    plt.show()
