import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class BBBConv2d(nn.Module):
    def __init__(self, q_logvar_init, p_logvar_init, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(BBBConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.q_logvar_init = q_logvar_init
        self.p_logvar_init = p_logvar_init
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.mu_weight = Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.sigma_weight = Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.register_buffer('eps_weight', torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        n *= self.kernel_size ** 2
        stdv = 1.0 / math.sqrt(n)
        self.mu_weight.data.uniform_(-stdv, stdv)
        self.sigma_weight.data.fill_(self.p_logvar_init)

    def forward(self, input):
        raise NotImplementedError()


    def convprobforward(self, input):
        sig_weight = torch.exp(self.sigma_weight)
        weight = self.mu_weight + sig_weight * self.eps_weight.normal_()
        kl_ = math.log(self.q_logvar_init) - self.sigma_weight + (sig_weight**2 + self.mu_weight**2) / (2 * self.q_logvar_init ** 2) - 0.5
        bias = None
        
        out = F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        kl = kl_.sum() 
        return out, kl


class BBBLinearFactorial(nn.Module):
    def __init__(self, q_logvar_init, p_logvar_init, in_features, out_features, bias=False):
        super(BBBLinearFactorial, self).__init__()
        self.q_logvar_init = q_logvar_init
        self.in_features = in_features
        self.out_features = out_features
        self.p_logvar_init = p_logvar_init
        self.mu_weight = Parameter(torch.Tensor(out_features, in_features))
        self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('eps_weight', torch.Tensor(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.mu_weight.size(1))
        self.mu_weight.data.uniform_(-stdv, stdv)
        self.sigma_weight.data.fill_(self.p_logvar_init)
        self.eps_weight.data.zero_()

    def forward(self, input):
        raise NotImplementedError()
        

    def fcprobforward(self, input):
        sig_weight = torch.exp(self.sigma_weight)
        weight = self.mu_weight + sig_weight * self.eps_weight.normal_()
        kl_ = math.log(self.q_logvar_init) - self.sigma_weight + (sig_weight**2 + self.mu_weight**2) / (2 * self.q_logvar_init ** 2) - 0.5
        bias = None
        out = F.linear(input, weight, bias)
        kl = kl_.sum() 
        return out, kl
