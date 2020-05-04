import sys
sys.path.append('..')

import os
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

import data
import utils
import metrics
from main_bayesian import getModel as getBayesianModel
from main_frequentist import getModel as getFrequentistModel
import config_mixtures as cfg
import uncertainty_estimation as ue


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


def get_splitmnist_models(num_tasks, bayesian=True, pretrained=False, weights_dir=None, net_type='lenet'):
    inputs = 1
    outputs = 10 // num_tasks
    models = []
    if pretrained:
        assert weights_dir
    for i in range(1, num_tasks + 1):
        if bayesian:
            model = getBayesianModel(net_type, inputs, outputs)
        else:
            model = getFrequentistModel(net_type, inputs, outputs)
        models.append(model)
        if pretrained:
            weight_path = weights_dir + f"model_{net_type}_{num_tasks}.{i}.pt"
            models[-1].load_state_dict(torch.load(weight_path))
    return models


def get_mixture_model(num_tasks, weights_dir, net_type='lenet', include_last_layer=True):
    """
    Current implementation is based on average value of weights
    """
    net = getBayesianModel(net_type, 1, 5)
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


def predict_regular(net, validloader, bayesian=True, num_ens=10):
    """
    For both Bayesian and Frequentist models
    """
    net.eval()
    accs = []

    for i, (inputs, labels) in enumerate(validloader):
        inputs, labels = inputs.to(device), labels.to(device)
        if bayesian:
            outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)
            for j in range(num_ens):
                net_out, _ = net(inputs)
                outputs[:, :, j] = F.log_softmax(net_out, dim=1).data

            log_outputs = utils.logmeanexp(outputs, dim=2)
            accs.append(metrics.acc(log_outputs, labels))
        else:
            output = net(inputs)
            accs.append(metrics.acc(output.detach(), labels))

    return np.mean(accs)


def predict_using_uncertainty_separate_models(net1, net2, valid_loader, uncertainty_type='epistemic_softmax', T=25):
    """
    For Bayesian models
    """
    accs = []
    total_u1 = 0.0
    total_u2 = 0.0
    set1_selected = 0
    set2_selected = 0

    epi_or_ale, soft_or_norm = uncertainty_type.split('_')
    soft_or_norm = True if soft_or_norm=='normalized' else False

    for i, (inputs, labels) in enumerate(valid_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        pred1, epi1, ale1 = ue.get_uncertainty_per_batch(net1, inputs, T=T, normalized=soft_or_norm)
        pred2, epi2, ale2 = ue.get_uncertainty_per_batch(net2, inputs, T=T, normalized=soft_or_norm)

        if epi_or_ale=='epistemic':
            u1 = np.sum(epi1, axis=1)
            u2 = np.sum(epi2, axis=1)
        elif epi_or_ale=='aleatoric':
            u1 = np.sum(ale1, axis=1)
            u2 = np.sum(ale2, axis=1)
        elif epi_or_ale=='both':
            u1 = np.sum(epi1, axis=1) + np.sum(ale1, axis=1)
            u2 = np.sum(epi2, axis=1) + np.sum(ale2, axis=1)
        else:
            raise ValueError("Not correct uncertainty type")

        total_u1 += np.sum(u1).item()
        total_u2 += np.sum(u2).item()

        set1_preferred = u2 > u1  # idx where set1 has less uncertainty
        set1_preferred = np.expand_dims(set1_preferred, 1)
        preds = np.where(set1_preferred, pred1, pred2)

        set1_selected += np.sum(set1_preferred)
        set2_selected += np.sum(~set1_preferred)

        accs.append(metrics.acc(torch.tensor(preds), labels))

    return np.mean(accs), set1_selected/(set1_selected + set2_selected), \
        set2_selected/(set1_selected + set2_selected), total_u1, total_u2


def predict_using_confidence_separate_models(net1, net2, valid_loader):
    """
    For Frequentist models
    """
    accs = []
    set1_selected = 0
    set2_selected = 0

    for i, (inputs, labels) in enumerate(valid_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        pred1 = F.softmax(net1(inputs), dim=1)
        pred2 = F.softmax(net2(inputs), dim=1)

        set1_preferred = pred1.max(dim=1)[0] > pred2.max(dim=1)[0]  # idx where set1 has more confidence
        preds = torch.where(set1_preferred.unsqueeze(1), pred1, pred2)

        set1_selected += torch.sum(set1_preferred).float().item()
        set2_selected += torch.sum(~set1_preferred).float().item()

        accs.append(metrics.acc(preds.detach(), labels))

    return np.mean(accs), set1_selected/(set1_selected + set2_selected), \
        set2_selected/(set1_selected + set2_selected)


def wip_predict_using_epistemic_uncertainty_with_mixture_model(model, fc3_1, fc3_2, valid_loader, T=10):
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
