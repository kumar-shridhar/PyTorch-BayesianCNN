import torch
from torch import nn


class ModuleWrapper(nn.Module):
    """Wrapper for nn.Module with support for arbitrary flags and a universal forward pass"""

    def __init__(self):
        super(ModuleWrapper, self).__init__()

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def forward(self, x):
        for module in self.children():
            x = module(x)

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl += module.kl_loss()

        return x, kl


class FlattenLayer(ModuleWrapper):

    def __init__(self, num_features):
        super(FlattenLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.view(-1, self.num_features)


class Posterior(nn.Module):
    def __init__(self, mu, rho, device):
        super(Posterior, self).__init__()
        self.mu = mu
        self.rho = rho
        self.device = device

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    @property
    def eps(self):
        return torch.distributions.Normal(0, 1).sample(self.mu.size()).to(self.device)

    def sample(self):
        posterior_sample = self.mu + self.sigma * self.eps
        return posterior_sample
