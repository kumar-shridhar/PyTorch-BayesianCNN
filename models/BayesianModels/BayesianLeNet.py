import math
import torch.nn as nn
from layers.BBBConv import BBBConv2d
from layers.BBBLinear import  BBBLinear
from layers.misc import FlattenLayer, ModuleWrapper


class BBBLeNet(ModuleWrapper):
    '''The architecture of LeNet with Bayesian Layers'''

    def __init__(self, outputs, inputs):
        super(BBBLeNet, self).__init__()

        self.num_classes = outputs

        self.conv1 = BBBConv2d(inputs, 6, 5, alpha_shape=(1,1), padding=0, bias=False, name='conv1')
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BBBConv2d(6, 16, 5, alpha_shape=(1,1), padding=0, bias=False, name='conv2')
        self.soft2 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = FlattenLayer(5 * 5 * 16)
        self.fc1 = BBBLinear(5 * 5 * 16, 120, alpha_shape=(1,1), bias=False, name='fc1')
        self.soft3 = nn.Softplus()

        self.fc2 = BBBLinear(120, 84, alpha_shape=(1,1), bias=False, name='fc2')
        self.soft4 = nn.Softplus()

        self.fc3 = BBBLinear(84, outputs, alpha_shape=(1,1), bias=False, name='fc3')
