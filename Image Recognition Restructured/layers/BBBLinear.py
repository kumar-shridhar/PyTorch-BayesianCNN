import sys
sys.path.append("..")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable

import metrics
from .misc import ModuleWrapper


class BBBLinear(ModuleWrapper):
    
    def __init__(self, in_features, out_features, alpha_shape=(1, 1), bias=True):
        super(BBBLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha_shape = alpha_shape
        self.W = Parameter(torch.Tensor(out_features, in_features))
        self.log_alpha = Parameter(torch.Tensor(*alpha_shape))
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.kl_fun = metrics.kl_ard

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.log_alpha.data.fill_(-5.0)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):

        lrt_mean = F.linear(x, self.W)
        if self.bias is not None:
            lrt_mean = lrt_mean + self.bias

        sigma2 = torch.exp(self.log_alpha) * self.W * self.W

        lrt_std = torch.sqrt(1e-16 + F.linear(x * x, sigma2))
        if self.training:
            eps = Variable(lrt_std.data.new(lrt_std.size()).normal_())
        else:
            eps = 0.0
        return lrt_mean + lrt_std * eps

    def kl_reg(self):
        return self.W.nelement() * self.kl_fun(self.log_alpha) / self.log_alpha.nelement()

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', alpha_shape=' + str(self.alpha_shape) \
               + ', bias=' + str(self.bias is not None) + ')' ', bias=' + str(self.bias is not None) + ')'
