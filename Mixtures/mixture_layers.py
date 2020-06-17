import sys
sys.path.append("..")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.nn import Parameter

from layers.misc import ModuleWrapper


def _sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def _gumbel_softmax_sample(logits):
    temperature = 0.1
    sample = _sample_gumbel(logits.size()[-1])
    if logits.is_cuda:
        sample = sample.cuda()
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)


def _gumbel_softmax(pi, hard):
    logits = torch.log(pi)
    y = _gumbel_softmax_sample(logits)
    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard


def sample_gmm(pi, mu, sigma):
    idx = _gumbel_softmax(pi, hard=True)
    sample_shape = mu.shape[:-1]
    mu = (mu * idx).sum(dim=-1)
    sigma = (sigma * idx).sum(dim=-1)
    eps = torch.randn_like(mu).cuda()
    sample = mu + sigma * eps
    return sample


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
        W_sigma = torch.log1p(torch.exp(self.W_rho))
        self.W_pi.data = 1 / W_sigma
        for i in range(num_tasks):
            self.W_mu.data[..., i] = W_mu_individual[i].data
            self.W_rho.data[..., i] = W_rho_individual[i].data

        if self.use_bias:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            self.bias_pi.data = 1 / bias_sigma
            for i in range(num_tasks):
                self.bias_mu.data[:, i] = bias_mu_individual[i].data
                self.bias_rho.data[:, i] = bias_rho_individual[i].data

    def forward(self, input):
        W_sigma = torch.log1p(torch.exp(self.W_rho))
        if self.use_bias:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        weight = sample_gmm(self.W_pi, self.W_mu, W_sigma)
        if self.use_bias:
            bias = sample_gmm(self.bias_pi, self.bias_mu, bias_sigma)
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
        W_sigma = torch.log1p(torch.exp(self.W_rho))
        self.W_pi.data = 1 / W_sigma
        for i in range(num_tasks):
            self.W_mu.data[..., i] = W_mu_individual[i].data
            self.W_rho.data[..., i] = W_rho_individual[i].data

        if self.use_bias:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            self.bias_pi.data = 1 / bias_sigma
            for i in range(num_tasks):
                self.bias_mu.data[:, i] = bias_mu_individual[i].data
                self.bias_rho.data[:, i] = bias_rho_individual[i].data

    def forward(self, input):
        W_sigma = torch.log1p(torch.exp(self.W_rho))
        if self.use_bias:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        weight = sample_gmm(self.W_pi, self.W_mu, W_sigma)
        if self.use_bias:
            bias = sample_gmm(self.bias_pi, self.bias_mu, bias_sigma)
        else:
            bias = None

        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)


class MixtureClassifier(ModuleWrapper):
    def __init__(self, in_features, out_features, num_tasks, W_mu_individual, W_rho_individual,
                 bias=True, bias_mu_individual=None, bias_rho_individual=None):
        super(MixtureClassifier, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.W_mu = Parameter(torch.empty((out_features, in_features), device=self.device))
        self.W_rho = Parameter(torch.empty((out_features, in_features), device=self.device))

        if self.use_bias:
            self.bias_mu = Parameter(torch.empty((out_features), device=self.device))
            self.bias_rho = Parameter(torch.empty((out_features), device=self.device))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters(num_tasks, W_mu_individual, W_rho_individual, bias_mu_individual, bias_rho_individual)

    def reset_parameters(self, num_tasks, W_mu_individual, W_rho_individual, bias_mu_individual, bias_rho_individual):
        for i in range(num_tasks):
            self.W_mu.data[i * 10 // num_tasks:(i+1) * 10 // num_tasks, :] = W_mu_individual[i].data
            self.W_rho.data[i * 10 // num_tasks:(i+1) * 10 // num_tasks, :] = W_rho_individual[i].data

        if self.use_bias:
            for i in range(num_tasks):
                self.bias_mu.data[i * 10 // num_tasks:(i+1) * 10 // num_tasks] = bias_mu_individual[i].data
                self.bias_rho.data[i * 10 // num_tasks:(i+1) * 10 // num_tasks] = bias_rho_individual[i].data

    def forward(self, input, sample=True):
        W_eps = torch.empty(self.W_mu.size()).normal_(0, 1).to(self.device)
        W_sigma = torch.log1p(torch.exp(self.W_rho))
        weight = self.W_mu + W_eps * W_sigma

        if self.use_bias:
            bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1).to(self.device)
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias = self.bias_mu + bias_eps * self.bias_sigma
        else:
            bias = None

        return F.linear(input, weight, bias)
