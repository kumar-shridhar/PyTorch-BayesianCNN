import numpy as np
import torch.nn.functional as F
from torch import nn
import torch


class ELBO(nn.Module):
    def __init__(self, net, train_size):
        super(ELBO, self).__init__()
        self.train_size = train_size
        self.net = net

    def forward(self, input, target, kl_weight=1.0):
        assert not target.requires_grad
        kl = 0.0
        for module in self.net.modules():
            if hasattr(module, 'kl_reg'):
                kl = kl + module.kl_reg()
        return F.cross_entropy(input, target, size_average=True) * self.train_size + kl_weight * kl

    def get_kl(self):
        kl = 0.0
        for module in self.net.modules():
            if hasattr(module, 'kl_reg'):
                kl = kl + module.kl_reg()
        return kl


def lr_linear(epoch_num, decay_start, total_epochs, start_value):
    if epoch_num < decay_start:
        return start_value
    return start_value*float(total_epochs-epoch_num)/float(total_epochs-decay_start)


def logit2acc(outputs, targets):
    return np.mean(outputs.cpu().numpy().argmax(axis=1) == targets.data.cpu().numpy())


def kl_ard(log_alpha):
    return 0.5 * torch.sum(torch.log1p(torch.exp(-log_alpha)))

