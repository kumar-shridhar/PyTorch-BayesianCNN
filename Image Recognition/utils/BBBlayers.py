import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from .BBBdistributions import Normal, distribution_selector
from torch.nn.modules.utils import _pair

cuda = torch.cuda.is_available()


class FlattenLayer(nn.Module):

    def __init__(self, num_features):
        super(FlattenLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.view(-1, self.num_features)


class _ConvNd(nn.Module):
    """
    Describes a Bayesian convolutional layer with
    a distribution over each of the weights and biases
    in the layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, output_padding, groups, p_logvar_init=-3, p_pi=1.0, q_logvar_init=-5):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.groups = groups

        # initialize log variance of p and q
        self.p_logvar_init = p_logvar_init
        self.q_logvar_init = q_logvar_init

        self.weight = Parameter(torch.Tensor(out_channels, in_channels// groups, *kernel_size))

        # approximate posterior weights...
        # self.qw_mean = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        # self.qw_logvar = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))

        # optionally add bias
        # self.qb_mean = Parameter(torch.Tensor(out_channels))
        # self.qb_logvar = Parameter(torch.Tensor(out_channels))

        # ...and output...
        self.conv_qw_mean = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        self.conv_qw_std = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))

        # ...as normal distributions
        # self.qw = Normal(mu=self.qw_mean, logvar=self.qw_logvar)
        # self.qb = Normal(mu=self.qb_mean, logvar=self.qb_logvar)

        self.conv_qw = Normal(mu=self.conv_qw_mean, logvar=self.conv_qw_std)

        # initialise
        self.log_alpha = Parameter(torch.Tensor(1, 1))

        # prior model
        # (does not have any trainable parameters so we use fixed normal or fixed mixture normal distributions)
        self.pw = distribution_selector(mu=0.0, logvar=p_logvar_init, pi=p_pi)
        # self.pb = distribution_selector(mu=0.0, logvar=p_logvar_init, pi=p_pi)

        # initialize all parameters
        self.reset_parameters()

    def reset_parameters(self):
        # initialise (learnable) approximate posterior parameters
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)

        self.weight.data.uniform_(-stdv, stdv)
        # self.qw_mean.data.uniform_(-stdv, stdv)
        # self.qw_logvar.data.uniform_(-stdv, stdv).add_(self.q_logvar_init)

        # self.qb_mean.data.uniform_(-stdv, stdv)
        # self.qb_logvar.data.uniform_(-stdv, stdv).add_(self.q_logvar_init)

        self.conv_qw_mean.data.uniform_(-stdv, stdv)
        self.conv_qw_std.data.uniform_(-stdv, stdv).add_(self.q_logvar_init)
        self.log_alpha.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class BBBConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(BBBConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, _pair(0), groups)

    def forward(self, input):
        raise NotImplementedError()

    def convprobforward(self, input):
        """
        Convolutional probabilistic forwarding method.
        :param input: data tensor
        :return: output, KL-divergence
        """

        # local reparameterization trick for convolutional layer

        conv_qw_mean = F.conv2d(input=input, weight=self.weight, stride=self.stride, padding=self.padding,
                                     dilation=self.dilation, groups=self.groups)
        conv_qw_std = torch.sqrt(1e-8 + F.conv2d(input=input.pow(2), weight=torch.exp(self.log_alpha)*self.weight.pow(2),
                                                 stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups))

        if cuda:
            conv_qw_mean.cuda()
            conv_qw_std.cuda()

        # sample from output
        if cuda:
            #output = conv_qw_mean + conv_qw_std * (torch.randn(conv_qw_mean.size())).cuda()
            output = conv_qw_mean + conv_qw_std * torch.cuda.FloatTensor(conv_qw_mean.size()).normal_()
        else:
            output = conv_qw_mean + conv_qw_std * (torch.randn(conv_qw_mean.size()))

        if cuda:
            output.cuda()

        conv_qw = Normal(mu=conv_qw_mean, logvar=conv_qw_std)

        #self.conv_qw_mean = Parameter(torch.Tensor(conv_qw_mean.cpu()))
        #self.conv_qw_std = Parameter(torch.Tensor(conv_qw_std.cpu()))

        w_sample = conv_qw.sample()

        # KL divergence
        qw_logpdf = conv_qw.logpdf(w_sample)

        kl = torch.sum(qw_logpdf - self.pw.logpdf(w_sample))

        return output, kl


class BBBLinearFactorial(nn.Module):
    """
    Describes a Linear fully connected Bayesian layer with
    a distribution over each of the weights and biases
    in the layer.
    """
    def __init__(self, in_features, out_features, p_logvar_init=-3, p_pi=1.0, q_logvar_init=-5):
        # p_logvar_init, p_pi can be either
        # (list/tuples): prior model is a mixture of Gaussians components=len(p_pi)=len(p_logvar_init)
        # float: Gussian distribution
        # q_logvar_init: float, the approximate posterior is currently always a factorized gaussian
        super(BBBLinearFactorial, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.p_logvar_init = p_logvar_init
        self.q_logvar_init = q_logvar_init

        self.weight = Parameter(torch.Tensor(out_features, in_features))

        # Approximate posterior weights...
        # self.qw_mean = Parameter(torch.Tensor(out_features, in_features))
        # self.qw_logvar = Parameter(torch.Tensor(out_features, in_features))

        # optionally add bias
        # self.qb_mean = Parameter(torch.Tensor(out_features))
        # self.qb_logvar = Parameter(torch.Tensor(out_features))

        # ...and output...
        self.fc_qw_mean = Parameter(torch.Tensor(out_features, in_features))
        self.fc_qw_std = Parameter(torch.Tensor(out_features, in_features))

        # ...as normal distributions
        # self.qw = Normal(mu=self.qw_mean, logvar=self.qw_logvar)
        # self.qb = Normal(mu=self.qb_mean, logvar=self.qb_logvar)
        self.fc_qw = Normal(mu=self.fc_qw_mean, logvar=self.fc_qw_std)

        # initialise
        self.log_alpha = Parameter(torch.Tensor(1, 1))

        # prior model
        self.pw = distribution_selector(mu=0.0, logvar=p_logvar_init, pi=p_pi)
        # self.pb = distribution_selector(mu=0.0, logvar=p_logvar_init, pi=p_pi)

        # initialize all paramaters
        self.reset_parameters()

    def reset_parameters(self):
        # initialize (trainable) approximate posterior parameters
        stdv = 10. / math.sqrt(self.in_features)

        self.weight.data.uniform_(-stdv, stdv)
        # self.qw_mean.data.uniform_(-stdv, stdv)
        # self.qw_logvar.data.uniform_(-stdv, stdv).add_(self.q_logvar_init)
        # self.qb_mean.data.uniform_(-stdv, stdv)
        # self.qb_logvar.data.uniform_(-stdv, stdv).add_(self.q_logvar_init)
        self.fc_qw_mean.data.uniform_(-stdv, stdv)
        self.fc_qw_std.data.uniform_(-stdv, stdv).add_(self.q_logvar_init)
        self.log_alpha.data.uniform_(-stdv, stdv)

    def forward(self, input):
        raise NotImplementedError()

    def fcprobforward(self, input):
        """
        Probabilistic forwarding method.
        :param input: data tensor
        :return: output, kl-divergence
        """

        fc_qw_mean = F.linear(input=input, weight=self.weight)
        fc_qw_si = torch.sqrt(1e-8 + F.linear(input=input.pow(2), weight=torch.exp(self.log_alpha)*self.weight.pow(2)))

        if cuda:
            fc_qw_mean.cuda()
            fc_qw_si.cuda()

        # sample from output
        if cuda:
            #output = fc_qw_mean + fc_qw_si * (torch.randn(fc_qw_mean.size())).cuda()
            output = fc_qw_mean + fc_qw_si * torch.cuda.FloatTensor(fc_qw_mean.size()).normal_()
        else:
            output = fc_qw_mean + fc_qw_si * (torch.randn(fc_qw_mean.size()))

        if cuda:
            output.cuda()

        self.fc_qw_mean = Parameter(torch.Tensor(fc_qw_mean.cpu()))
        self.fc_qw_std = Parameter(torch.Tensor(fc_qw_si.cpu()))

        w_sample = self.fc_qw.sample()

        # KL divergence
        qw_logpdf = self.fc_qw.logpdf(w_sample)

        kl = torch.sum(qw_logpdf - self.pw.logpdf(w_sample))

        return output, kl

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GaussianVariationalInference(nn.Module):
    def __init__(self, loss=nn.CrossEntropyLoss()):
        super(GaussianVariationalInference, self).__init__()
        self.loss = loss

    def forward(self, logits, y, kl, beta):
        logpy = self.loss(logits, y)

        loss = logpy + beta * kl  # ELBO

        return loss
