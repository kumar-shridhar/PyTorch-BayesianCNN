import math
import torch.nn as nn
from layers.BBBConv import BBBConv2d
from layers.BBBLinear import  BBBLinearFactorial
from layers.misc import FlattenLayer


class BBBLeNet(nn.Module):
    '''The architecture of LeNet with Bayesian Layers'''

    def __init__(self, outputs, inputs):
        super(BBBLeNet, self).__init__()

        self.q_logvar_init = 0.05
        self.p_logvar_init = math.log(0.05)

        self.conv1 = BBBConv2d(self.q_logvar_init, self.p_logvar_init, inputs, 6, 5, stride=1)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BBBConv2d(self.q_logvar_init, self.p_logvar_init, 6, 16, 5, stride=1)
        self.soft2 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = FlattenLayer(5 * 5 * 16)
        self.fc1 = BBBLinearFactorial(self.q_logvar_init, self.p_logvar_init, 5 * 5 * 16, 120)
        self.soft3 = nn.Softplus()

        self.fc2 = BBBLinearFactorial(self.q_logvar_init, self.p_logvar_init, 120, 84)
        self.soft4 = nn.Softplus()

        self.fc3 = BBBLinearFactorial(self.q_logvar_init, self.p_logvar_init, 84, outputs)

        layers = [self.conv1, self.soft1, self.pool1, self.conv2, self.soft2, self.pool2,
                  self.flatten, self.fc1, self.soft3, self.fc2, self.soft4, self.fc3]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        'Forward pass with Bayesian weights'
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, _kl, = layer.fcprobforward(x)
                kl += _kl
            else:
                x = layer(x)
        logits = x
        return logits, kl
