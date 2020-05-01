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


class BBBLinear(ModuleWrapper):
    
    def __init__(self, in_features, out_features, bias=True):
        super(BBBLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.prior_mu = 0
        self.prior_sigma = 0.1

        self.W_mu = Parameter(torch.Tensor(out_features, in_features))
        self.W_sigma = Parameter(torch.Tensor(out_features, in_features))
        if self.use_bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_sigma = Parameter(torch.Tensor(out_features))
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

    def forward(self, x, sample=True):

        W_var = 1e-6 + F.softplus(self.W_sigma) ** 2
        bias_var = 1e-6 + F.softplus(self.bias_sigma) ** 2

        act_mu = F.linear(x, self.W_mu, self.bias_mu)
        act_var = 1e-16 + F.linear(x ** 2, W_var, bias_std)
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
