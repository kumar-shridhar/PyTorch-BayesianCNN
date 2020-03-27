import os
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

import data
import utils
import metrics
from main_bayesian import getModel
import config_mixtures as cfg


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Pass(nn.Module):
    def __init__(self):
        super(Pass, self).__init__()

    def forward(self, x):
        return x


def _get_splitmnist_datasets(num_tasks):
    datasets = []
    for i in range(1, num_tasks + 1):
        name = 'SplitMNIST-{}.{}'.format(num_tasks, i)
        datasets.append(data.getDataset(name))
    return datasets


def get_splitmnist_dataloaders(num_tasks, return_datasets=False):
    loaders = []
    datasets = _get_splitmnist_datasets(num_tasks)
    for i in range(1, num_tasks + 1):
        trainset, testset, _, _ = datasets[i-1]
        curr_loaders = data.getDataloader(
            trainset, testset, cfg.valid_size, cfg.batch_size, cfg.num_workers)
        loaders.append(curr_loaders)  # (train_loader, valid_loader, test_loader)
    if return_datasets:
        return loaders, datasets
    return loaders


def get_splitmnist_models(num_tasks, pretrained=False, weights_dir=None, net_type='lenet'):
    inputs = 1
    outputs = 10 // num_tasks
    models = []
    if pretrained:
        assert weights_dir
    for i in range(1, num_tasks + 1):
        models.append(getModel(net_type, inputs, outputs))
        if pretrained:
            weight_path = weights_dir + f"model_{net_type}_{num_tasks}.{i}.pt"
            models[-1].load_state_dict(torch.load(weight_path))
    return models


def get_mixture_model(num_tasks, weights_dir, net_type='lenet', include_last_layer=True):
    """
    Current implementation is based on average value of weights
    """
    net = getModel(net_type, 1, 5)
    if not include_last_layer:
        net.fc3 = Pass()

    task_weights = []
    for i in range(1, num_tasks + 1):
        weight_path = weights_dir + f"model_{net_type}_{num_tasks}.{i}.pt"
        task_weights.append(torch.load(weight_path))

    mixture_weights = net.state_dict().copy()
    layer_list = list(mixture_weights.keys())

    for key in mixture_weights:
        if key in layer_list:
            concat_weights = torch.cat([w[key].unsqueeze(0) for w in task_weights] , dim=0)
            average_weight = torch.mean(concat_weights, dim=0)
            mixture_weights[key] = average_weight

    net.load_state_dict(mixture_weights)
    return net


def calculate_accuracy(net, validloader, offset=0, num_ens=1):
    net.eval()
    accs = []

    for i, (inputs, labels) in enumerate(validloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)
        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1).data

        log_outputs = utils.logmeanexp(outputs, dim=2)
        accs.append(metrics.acc(log_outputs, labels))

    return np.mean(accs)


def predict_using_epistemic_uncertainty_with_mixture_model(model, fc3_1, fc3_2, valid_loader, T=10):  # over a batch (a pass)
    accs = []
    total_epistemic_1 = 0.0
    total_epistemic_2 = 0.0
    set_1_selected = 0
    set_2_selected = 0

    for i, (inputs, labels) in enumerate(valid_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = []
        for i in range(inputs.shape[0]):  # loop over batch
            input_image = inputs[i].unsqueeze(0)

            p_hat_1 = []
            p_hat_2 = []
            preds_1 = []
            preds_2 = []
            for t in range(T):
                net_out_mix, _ = model(input_image)

                # set_1
                net_out_1 = fc3_1(net_out_mix)
                preds_1.append(net_out_1)
                prediction = F.softplus(net_out_1)
                prediction = prediction / torch.sum(prediction, dim=1)
                p_hat_1.append(prediction.cpu().detach())

                # set_2
                net_out_2 = fc3_2(net_out_mix)
                preds_2.append(net_out_2)
                prediction = F.softplus(net_out_2)
                prediction = prediction / torch.sum(prediction, dim=1)
                p_hat_2.append(prediction.cpu().detach())

            # set_1
            p_hat = torch.cat(p_hat_1, dim=0).numpy()
            p_bar = np.mean(p_hat, axis=0)

            preds = torch.cat(preds_1, dim=0)
            pred_set_1 = torch.sum(preds, dim=0).unsqueeze(0)

            temp = p_hat - np.expand_dims(p_bar, 0)
            epistemic = np.dot(temp.T, temp) / T
            epistemic_set_1 = np.sum(np.diag(epistemic)).item()
            total_epistemic_1 += epistemic_set_1

            # set_2
            p_hat = torch.cat(p_hat_2, dim=0).numpy()
            p_bar = np.mean(p_hat, axis=0)

            preds = torch.cat(preds_2, dim=0)
            pred_set_2 = torch.sum(preds, dim=0).unsqueeze(0)

            temp = p_hat - np.expand_dims(p_bar, 0)
            epistemic = np.dot(temp.T, temp) / T
            epistemic_set_2 = np.sum(np.diag(epistemic)).item()
            total_epistemic_2 += epistemic_set_2

            if epistemic_set_1 > epistemic_set_2:
                set_2_selected += 1
                outputs.append(pred_set_2)
            else:
                set_1_selected += 1
                outputs.append(pred_set_1)

        outputs = torch.cat(outputs, dim=0)
        accs.append(metrics.acc(outputs.detach(), labels))

    return np.mean(accs), set_1_selected/(set_1_selected + set_2_selected), \
        set_2_selected/(set_1_selected + set_2_selected), total_epistemic_1, total_epistemic_2


def predict_using_epistemic_uncertainty_without_mixture_model(net_1, net_2, valid_loader, T=10):  # over a batch (a pass)
    accs = []
    total_epistemic_1 = 0.0
    total_epistemic_2 = 0.0
    set_1_selected = 0
    set_2_selected = 0

    for i, (inputs, labels) in enumerate(valid_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = []
        for i in range(inputs.shape[0]):  # loop over batch
            input_image = inputs[i].unsqueeze(0)

            p_hat = []
            preds = []
            for t in range(T):
                net_out, _ = net_1(input_image)
                preds.append(net_out)
                prediction = F.softplus(net_out)
                prediction = prediction / torch.sum(prediction, dim=1)
                p_hat.append(prediction.cpu().detach())

            p_hat = torch.cat(p_hat, dim=0).numpy()
            p_bar = np.mean(p_hat, axis=0)

            preds = torch.cat(preds, dim=0)
            pred_set_1 = torch.sum(preds, dim=0).unsqueeze(0)

            temp = p_hat - np.expand_dims(p_bar, 0)
            epistemic_set_1 = np.dot(temp.T, temp) / T
            epistemic_set_1 = np.sum(np.diag(epistemic_set_1)).item()
            total_epistemic_1 += epistemic_set_1

            p_hat = []
            preds = []
            for t in range(T):
                net_out, _ = net_2(input_image)
                preds.append(net_out)
                prediction = F.softplus(net_out)
                prediction = prediction / torch.sum(prediction, dim=1)
                p_hat.append(prediction.cpu().detach())

            p_hat = torch.cat(p_hat, dim=0).numpy()
            p_bar = np.mean(p_hat, axis=0)

            preds = torch.cat(preds, dim=0)
            pred_set_2 = torch.sum(preds, dim=0).unsqueeze(0)

            temp = p_hat - np.expand_dims(p_bar, 0)
            epistemic_set_2 = np.dot(temp.T, temp) / T
            epistemic_set_2 = np.sum(np.diag(epistemic_set_2)).item()
            total_epistemic_2 += epistemic_set_2

            if epistemic_set_1 > epistemic_set_2:
                set_2_selected += 1
                outputs.append(pred_set_2)
            else:
                set_1_selected += 1
                outputs.append(pred_set_1)

        outputs = torch.cat(outputs, dim=0)
        accs.append(metrics.acc(outputs.detach(), labels))

    return np.mean(accs), set_1_selected/(set_1_selected + set_2_selected), \
        set_2_selected/(set_1_selected + set_2_selected), total_epistemic_1, total_epistemic_2


def predict_using_epistemic_uncertainty_single_model(model, valid_loader, T=10):
    accs = []
    total_epistemic = 0.0

    for i, (inputs, labels) in enumerate(valid_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = []
        for i in range(inputs.shape[0]):  # loop over batch
            input_image = inputs[i].unsqueeze(0)

            p_hat = []
            preds = []
            for t in range(T):
                net_out, _ = model(input_image)
                preds.append(net_out)
                prediction = F.softplus(net_out)
                prediction = prediction / torch.sum(prediction, dim=1)
                p_hat.append(prediction.cpu().detach())

            p_hat = torch.cat(p_hat, dim=0).numpy()
            p_bar = np.mean(p_hat, axis=0)

            preds = torch.cat(preds, dim=0)
            pred = torch.sum(preds, dim=0).unsqueeze(0)

            temp = p_hat - np.expand_dims(p_bar, 0)
            epistemic = np.dot(temp.T, temp) / T
            epistemic = np.sum(np.diag(epistemic)).item()
            total_epistemic += epistemic

            outputs.append(pred)

        outputs = torch.cat(outputs, dim=0)
        accs.append(metrics.acc(outputs.detach(), labels))

    return np.mean(accs), total_epistemic
