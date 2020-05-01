import sys
sys.path.append("..")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import utils
from metrics import KL_DIV
import config_bayesian as cfg
from .misc import ModuleWrapper


class BBBConv2d(ModuleWrapper):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True):
        super(BBBConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.prior_mu = 0
        self.prior_sigma = 0.1

        self.W_mu = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.W_sigma = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if self.use_bias:
            self.bias_mu = Parameter(torch.Tensor(out_channels))
            self.bias_sigma = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_sigma', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(0, 0.1)
        self.W_sigma.data.normal_(0.05, 0.1)

        if self.use_bias:
            self.bias_mu.data.normal_(0, 0.1)
            self.bias_sigma.data.normal_(0.05, 0.1)

    def forward(self, x):

        W_var = 1e-6 + F.softplus(self.W_sigma) ** 2
        bias_var = 1e-6 + F.softplus(self.bias_sigma) ** 2

        act_mu = F.conv2d(
            x, self.W_mu, self.bias_mu, self.stride, self.padding, self.dilation, self.groups)
        act_var = 1e-16 + F.conv2d(
            x ** 2, W_var, bias_std, self.stride, self.padding, self.dilation, self.groups)
        act_std = torch.sqrt(act_var)

        if self.training or sample:
            eps = torch.empty(act_mu.size()).normal_(0, 1).to(self.device)
            return act_mean + act_std * eps
        else:
            return act_mean

    def kl_loss(self):
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl
