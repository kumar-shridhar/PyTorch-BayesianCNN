import sys
sys.path.append('..')

import os
import ot
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
from layers import BBB_Linear, BBB_LRT_Linear


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_task_accuracy(outputs, targets, task_out, task_target):
    return np.mean((outputs.argmax(axis=1) == targets) & (task_out == task_target))


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


def get_splitmnist_models(num_tasks, bayesian=True, pretrained=False, weights_dir=None, priors=None,
                          net_type='lenet', layer_type='lrt', activation_type='softplus'):
    inputs = 1
    outputs = 10 // num_tasks
    if bayesian:
        assert layer_type
        assert activation_type
    models = []
    if pretrained:
        assert weights_dir
    for i in range(1, num_tasks + 1):
        if bayesian:
            model = getBayesianModel(net_type, inputs, outputs, priors, layer_type, activation_type)
        else:
            model = getFrequentistModel(net_type, inputs, outputs)
        models.append(model)
        if pretrained:
            if bayesian:
                weight_path = weights_dir + f"model_{net_type}_{layer_type}_{activation_type}_{num_tasks}.{i}.pt"
            else:
                weight_path = weights_dir + f"model_{net_type}_{num_tasks}.{i}.pt"
            models[-1].load_state_dict(torch.load(weight_path))
    return models


def predict_regular(net, validloader, bayesian=True, num_ens=10):
    """
    For both Bayesian and Frequentist models
    """
    if bayesian:
        net.train()
    else:
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


def predict_using_uncertainty_multi_model(nets, valid_loader, task_id, uncertainty_type='epistemic_softmax', T=25):
    """
    For Bayesian models
    """
    accs = []
    num_tasks = len(nets)
    model_selected = [0.0] * num_tasks
    total_uncertainty = [0.0] * num_tasks

    epi_or_ale, soft_or_norm = uncertainty_type.split('_')
    soft_or_norm = True if soft_or_norm=='normalized' else False

    for i, (inputs, labels) in enumerate(valid_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        preds = []
        uncerts = []

        for t in range(num_tasks):
            net = nets[t]
            net.cuda()
            # (batch, categories); (batch, categories); (batch, categories)
            pred, epi, ale = ue.get_uncertainty_per_batch(net, inputs, T=T, normalized=soft_or_norm)
            preds.append(torch.tensor(pred))
            if epi_or_ale=='epistemic':
                uncerts.append(torch.tensor(epi))
            elif epi_or_ale=='aleatoric':
                uncerts.append(torch.tensor(ale))
            elif epi_or_ale=='both':
                uncerts.append(torch.tensor(epi + ale))
            else:
                raise ValueError("Not correct uncertainty type")

        preds = torch.stack(preds)  # (nets, batch, categories)
        uncerts = torch.stack(uncerts)  # (nets, batch, categories)
        uncerts = torch.sum(uncerts, dim=2) # (nets, batch)

        model_preferred = torch.argmin(uncerts, dim=0).numpy()  # model which has the least uncertainty (model_idx)

        for t in range(num_tasks):
            total_uncertainty[t] += torch.sum(uncerts[t]).item()
            model_selected[t] += np.sum(model_preferred == t)

        preds = preds[model_preferred, range(inputs.shape[0]), :]

        task_target = np.ndarray((inputs.shape[0],), dtype=np.int32)
        task_target.fill(task_id)
        accs.append(get_task_accuracy(preds.cpu().numpy(), labels.cpu().numpy(), model_preferred, task_target))

    return np.mean(accs), [m/np.sum(model_selected) for m in model_selected], total_uncertainty


def predict_using_confidence_multi_model(nets, valid_loader, task_id):
    """
    For Frequentist models
    """
    accs = []
    num_tasks = len(nets)
    model_selected = [0.0] * num_tasks

    for i, (inputs, labels) in enumerate(valid_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        preds = []
        for t in range(num_tasks):
            net = nets[t]
            net.cuda()
            conf = F.softmax(net(inputs), dim=1)
            preds.append(conf)

        preds = torch.cat([p.unsqueeze(0) for p in preds], dim=0)  # (nets, batch, categories)

        model_preferred = torch.max(preds, dim=2)[0].argmax(dim=0).cpu().detach().numpy()  # model which has the most confidence (model_idx)
        for t in range(num_tasks):
            model_selected[t] += np.sum(model_preferred == t)

        preds = preds[model_preferred, range(inputs.shape[0]), :]

        task_target = np.ndarray((inputs.shape[0],), dtype=np.int32)
        task_target.fill(task_id)
        accs.append(get_task_accuracy(preds.detach().cpu().numpy(), labels.cpu().numpy(), model_preferred, task_target))

    return np.mean(accs), [m/np.sum(model_selected) for m in model_selected]


def _get_barycentre_params(mu, sigma, average=False):
    """
    mu: (num_tasks,):: mean of nodes from respective models
    sigma: (num_tasks,):: std_dev of nodes from respective models
    average: boolean:: if average apply same weights else apply weights based on std_dev \
                       lower std_dev => higher weight

    returns: mu_t, sigma_t
    """
    assert type(mu) is np.ndarray
    assert type(sigma) is np.ndarray

    if average:
        num_gaussians = mu.shape[0]
        weights = np.array([1/num_gaussians] * num_gaussians)
    else:
        weights = 1 / sigma
        weights /= weights.sum()

    mu_t = np.dot(weights, mu)
    sigma_t = np.dot(weights, sigma)

    return mu_t, sigma_t


def _get_mixture_weights(mu_layers, rho_layers, average=False):
    assert type(mu_layers) is list
    assert type(rho_layers) is list

    init_weights_shape = mu_layers[0].shape
    sigma_layers = [torch.log1p(torch.exp(layer_rho)) for layer_rho in rho_layers]

    mu_layers = [layer_mu.flatten().cpu().numpy() for layer_mu in mu_layers]
    sigma_layers = [layer_sigma.flatten().cpu().numpy() for layer_sigma in sigma_layers]

    mu_mix = np.empty_like(mu_layers[0])
    sigma_mix = np.empty_like(sigma_layers[0])

    for i in range(mu_layers[0].shape[0]):
        mu = np.array([layer[i] for layer in mu_layers])
        sigma = np.array([layer[i] for layer in sigma_layers])
        mu_mix[i], sigma_mix[i] = _get_barycentre_params(mu, sigma, average=average)

    mu_mix = torch.from_numpy(mu_mix.reshape(init_weights_shape))
    sigma_mix = torch.from_numpy(sigma_mix.reshape(init_weights_shape))
    rho_mix = torch.log(torch.expm1(sigma_mix))

    return mu_mix, rho_mix


def get_mixture_model(num_tasks, weights_dir, average=False, net_type='lenet', layer_type='lrt', activation_type='softplus'):

    inputs = 1  # MNIST
    outputs = 10 // num_tasks
    net = getBayesianModel(net_type, inputs, outputs, None, layer_type, activation_type)

    task_weights = []
    for i in range(1, num_tasks + 1):
        weight_path = weights_dir + f"model_{net_type}_{layer_type}_{activation_type}_{num_tasks}.{i}.pt"
        task_weights.append(torch.load(weight_path))

    mixture_weights = net.state_dict().copy()
    layer_list = list(mixture_weights.keys())
    assert len(layer_list) % 2 == 0

    # Last layer
    last_layer_name = layer_list[-1].split('.')[0]
    last_shape = tuple(mixture_weights[last_layer_name + '.W_mu'].shape)[::-1]
    count = sum([1 if layer.startswith(last_layer_name) else 0 for layer in layer_list])
    bias_last = True if count==4 else False

    num_mix_layers = len(layer_list) - count  # Leave out last layer
    for i in range(0, num_mix_layers, 2):
        key_mu = layer_list[i]
        key_rho = layer_list[i+1]
        mu_layers = [tw[key_mu] for tw in task_weights]
        rho_layers = [tw[key_rho] for tw in task_weights]
        mixture_weights[key_mu], mixture_weights[key_rho] = \
            _get_mixture_weights(mu_layers, rho_layers, average=average)

    heads = []
    for i in range(num_tasks):
        if layer_type == 'bbb':
            heads.append(BBB_Linear(*last_shape, bias=bias_last))
        elif layer_type == 'lrt':
            heads.append(BBB_LRT_Linear(*last_shape, bias=bias_last))
        else:
            raise ValueError

        heads[-1].W_mu = torch.nn.Parameter(task_weights[i][last_layer_name + '.W_mu'])
        heads[-1].W_rho = torch.nn.Parameter(task_weights[i][last_layer_name + '.W_rho'])
        if bias_last:
            heads[-1].bias_mu = torch.nn.Parameter(task_weights[i][last_layer_name + '.bias_mu'])
            heads[-1].bias_rho = torch.nn.Parameter(task_weights[i][last_layer_name + '.bias_rho'])

    net.load_state_dict(mixture_weights)
    setattr(net, last_layer_name, Pass())

    return net, heads


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
