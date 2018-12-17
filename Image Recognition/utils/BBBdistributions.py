import torch
import math
import numpy as np
import torch.nn as nn


class Distribution(object):
    """
    Base class for torch-based probability distributions.
    """
    def pdf(self, x):
        raise NotImplementedError

    def logpdf(self, x):
        raise NotImplementedError

    def cdf(self, x):
        raise NotImplementedError

    def logcdf(self, x):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class Normal(Distribution):
    # scalar version
    def __init__(self, mu, logvar):
        self.mu = mu
        self.logvar = logvar
        self.shape = mu.size()

        super(Normal, self).__init__()

    def logpdf(self, x):
        c = - float(0.5 * math.log(2 * math.pi))
        return c - 0.5 * self.logvar - (x - self.mu).pow(2) / (2 * torch.exp(self.logvar))

    def pdf(self, x):
        return torch.exp(self.logpdf(x))

    def sample(self):
        if self.mu.is_cuda:
            eps = torch.cuda.FloatTensor(self.shape).normal_()
        else:
            eps = torch.FloatTensor(self.shape).normal_()
        # local reparameterization trick
        return self.mu + torch.exp(0.5 * self.logvar) * eps

    def entropy(self):
        return 0.5 * math.log(2. * math.pi * math.e) + 0.5 * self.logvar


class FixedNormal(Distribution):
    # takes mu and logvar as float values and assumes they are shared across all weights
    def __init__(self, mu, logvar):
        self.mu = mu
        self.logvar = logvar
        super(FixedNormal, self).__init__()

    def logpdf(self, x):
        c = - float(0.5 * math.log(2 * math.pi))
        return c - 0.5 * self.logvar - (x - self.mu).pow(2) / (2 * math.exp(self.logvar))


class Normalout(Distribution):
    # scalar version
    def __init__(self, mu, std):
        self.mu = mu
        self.std = std
        self.shape = mu.size()

        super(Normalout, self).__init__()

    def logpdf(self, x):
        c = - float(0.5 * math.log(2 * math.pi))
        return c - 0.5 * self.std - (x - self.mu).pow(2) / (2 * torch.exp(self.std))

    def pdf(self, x):
        return torch.exp(self.logpdf(x))

    def sample(self):
        if self.mu.is_cuda:
            eps = torch.cuda.FloatTensor(self.shape).normal_()
        else:
            eps = torch.FloatTensor(self.shape).normal_()
        # local reparameterization trick
        return self.mu + torch.exp(0.5 * self.std) * eps

    def entropy(self):
        return 0.5 * math.log(2. * math.pi * math.e) + 0.5 * self.std


class FixedMixtureNormal(nn.Module):
    # scale mixture Gaussian prior (with scale mixture factor pi)
    def __init__(self, mu, logvar, pi):
        super(FixedMixtureNormal, self).__init__()
        # Ensure convex combination
        assert sum(pi) - 1 < 0.0001
        self.mu = nn.Parameter(torch.from_numpy(np.array(mu)).float(), requires_grad=False)
        self.logvar = nn.Parameter(torch.from_numpy(np.array(logvar)).float(), requires_grad=False)
        self.pi = nn.Parameter(torch.from_numpy(np.array(pi)).float(), requires_grad=False)

    def _component_logpdf(self, x):
        ndim = len(x.size())
        shape_expand = ndim * (None,)
        x = x.unsqueeze(-1)

        c = - float(0.5 * math.log(2 * math.pi))
        mu = self.mu[shape_expand]
        logvar = self.logvar[shape_expand]
        pi = self.pi[shape_expand]

        return c - 0.5 * logvar - (x - mu).pow(2) / 2 * torch.exp(logvar)

    def logpdf(self, x):
        ndim = len(x.size())
        shape_expand = ndim * (None,)
        pi = self.pi[shape_expand]
        px = torch.exp(self._component_logpdf(x))  # ... x num_components
        return torch.log(torch.sum(pi * px, -1))


def distribution_selector(mu, logvar, pi):
    if isinstance(logvar, (list, tuple)) and isinstance(pi, (list, tuple)):
        assert len(logvar) == len(pi)
        num_components = len(logvar)
        if not isinstance(mu, (list, tuple)):
            mu = (mu,) * num_components
        return FixedMixtureNormal(mu, logvar, pi)
    else:
        return FixedNormal(mu, logvar)
