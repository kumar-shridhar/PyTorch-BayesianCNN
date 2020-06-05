import sys
sys.path.append("..")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.nn import Parameter

from layers.misc import ModuleWrapper
import config_mixtures as cfg


def GaussianMixtureModel(pi, mu, sigma):
    """
    pi (torch.tensor): (features, num_tasks)
    mu (torch.tensor): (features, num_tasks)
    sigma (torch.tensor): (features, num_tasks)
    """
    pi = pi.div(pi.sum(dim=-1, keepdims=True))
    mixture = D.Categorical(pi)
    components = D.Normal(mu, sigma)
    return D.MixtureSameFamily(mixture, components)


class MixtureLinear(ModuleWrapper):
    def __init__(self, in_features, out_features, num_tasks, W_mu_individual, W_rho_individual,
                 bias=True, bias_mu_individual=None, bias_rho_individual=None):
        """
        W_mu_individual and W_rho_individual are lists containing
        W_mus and W_rhos of individual models
        """
        super(MixtureLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gmm = None

        self.W_pi = Parameter(torch.empty((out_features, in_features, num_tasks), device=self.device))
        self.W_mu = Parameter(torch.empty((out_features, in_features, num_tasks), device=self.device))
        self.W_rho = Parameter(torch.empty((out_features, in_features, num_tasks), device=self.device))

        if self.use_bias:
            self.bias_pi = Parameter(torch.empty((out_features, num_tasks), device=self.device))
            self.bias_mu = Parameter(torch.empty((out_features, num_tasks), device=self.device))
            self.bias_rho = Parameter(torch.empty((out_features, num_tasks), device=self.device))
        else:
            self.register_parameter('bias_pi', None)
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters(num_tasks, W_mu_individual, W_rho_individual, bias_mu_individual, bias_rho_individual)

    def reset_parameters(self, num_tasks, W_mu_individual, W_rho_individual, bias_mu_individual, bias_rho_individual):
        self.W_pi.data.fill_(1. / num_tasks)
        for i in range(num_tasks):
            self.W_mu.data[..., i] = W_mu_individual[i].data
            self.W_rho.data[..., i] = W_rho_individual[i].data

        if self.use_bias:
            self.bias_pi.data.fill_(1. / num_tasks)
            for i in range(num_tasks):
                self.bias_mu.data[:, i] = bias_mu_individual[i].data
                self.bias_rho.data[:, i] = bias_rho_individual[i].data

        cfg.distribution_updated = True

    def forward(self, input, sample=True):
        W_sigma = torch.log1p(torch.exp(self.W_rho))
        if self.gmm is None or cfg.distribution_updated:
            self.gmm = GaussianMixtureModel(self.W_pi, self.W_mu, W_sigma)
            cfg.distribution_updated = False

        weight = self.gmm.sample()

        if self.use_bias:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            gmm_bias = GaussianMixtureModel(self.bias_pi, self.bias_mu, bias_sigma)
            bias = gmm_bias.sample()
        else:
            bias = None

        return F.linear(input, weight, bias)


class MixtureConv2d(ModuleWrapper):
    def __init__(self, in_channels, out_channels, kernel_size,
                 num_tasks, W_mu_individual, W_rho_individual,
                 bias=True, bias_mu_individual=None, bias_rho_individual=None,
                 stride=1, padding=0, dilation=1):
        """
        W_mu_individual and W_rho_individual are lists containing
        W_mus and W_rhos of individual models
        """
        super(MixtureConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gmm = None

        self.W_pi = Parameter(torch.empty((out_channels, in_channels, *self.kernel_size, num_tasks), device=self.device))
        self.W_mu = Parameter(torch.empty((out_channels, in_channels, *self.kernel_size, num_tasks), device=self.device))
        self.W_rho = Parameter(torch.empty((out_channels, in_channels, *self.kernel_size, num_tasks), device=self.device))

        if self.use_bias:
            self.bias_pi = Parameter(torch.empty((out_channels, num_tasks), device=self.device))
            self.bias_mu = Parameter(torch.empty((out_channels, num_tasks), device=self.device))
            self.bias_rho = Parameter(torch.empty((out_channels, num_tasks), device=self.device))
        else:
            self.register_parameter('bias_pi', None)
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters(num_tasks, W_mu_individual, W_rho_individual, bias_mu_individual, bias_rho_individual)

    def reset_parameters(self, num_tasks, W_mu_individual, W_rho_individual, bias_mu_individual, bias_rho_individual):
        self.W_pi.data.fill_(1. / num_tasks)
        for i in range(num_tasks):
            self.W_mu.data[..., i] = W_mu_individual[i].data
            self.W_rho.data[..., i] = W_rho_individual[i].data

        if self.use_bias:
            self.bias_pi.data.fill_(1. / num_tasks)
            for i in range(num_tasks):
                self.bias_mu.data[:, i] = bias_mu_individual[i].data
                self.bias_rho.data[:, i] = bias_rho_individual[i].data

        cfg.distribution_updated = True

    def forward(self, input, sample=True):
        W_sigma = torch.log1p(torch.exp(self.W_rho))
        if self.gmm is None or cfg.distribution_updated:
            self.gmm = GaussianMixtureModel(self.W_pi, self.W_mu, W_sigma)
            cfg.distribution_updated = False

        weight = self.gmm.sample()

        if self.use_bias:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            gmm_bias = GaussianMixtureModel(self.bias_pi, self.bias_mu, bias_sigma)
            bias = gmm_bias.sample()
        else:
            bias = None

        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
