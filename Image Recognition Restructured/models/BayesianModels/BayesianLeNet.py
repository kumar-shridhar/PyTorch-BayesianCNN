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

        self.conv1 = BBBConv2d(inputs, 6, 5, alpha_shape=(1,1), padding=0, bias=False)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BBBConv2d(6, 16, 5, alpha_shape=(1,1), padding=0, bias=False)
        self.soft2 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = FlattenLayer(5 * 5 * 16)
        self.fc1 = BBBLinear(5 * 5 * 16, 120, alpha_shape=(1,1), bias=False)
        self.soft3 = nn.Softplus()

        self.fc2 = BBBLinear(120, 84, alpha_shape=(1,1), bias=False)
        self.soft4 = nn.Softplus()

        self.fc3 = BBBLinear(84, outputs, alpha_shape=(1,1), bias=False)


class LeNet5(ModuleWrapper):
    
    def __init__(self, outputs, inputs):
        super(LeNet5, self).__init__()

        self.num_classes = outputs
        nonlinearity = nn.ReLU

        # Conv-BN-Tanh-Pool
        self.conv1 = BBBConv2d(inputs, 20, 5, alpha_shape=(1,1), padding=0, bias=False)
  
        self.bn1 = nn.BatchNorm2d(20)
        self.relu1 = nonlinearity()
        self.pool1 = nn.MaxPool2d(2, padding=0)

        self.conv2 = BBBConv2d(20, 50, 5, alpha_shape=(1,1), padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(50)
        self.relu2 = nonlinearity()
        self.pool2 = nn.MaxPool2d(2, padding=0)

        self.flatten = FlattenLayer(800)

        self.dense1 = BBBLinear(800, 500, alpha_shape=(1,1), bias=False)
        self.bn3 = nn.BatchNorm1d(500)
        self.relu3 = nonlinearity()

        self.dense2 = BBBLinear(500, outputs, alpha_shape=(1,1), bias=False)
