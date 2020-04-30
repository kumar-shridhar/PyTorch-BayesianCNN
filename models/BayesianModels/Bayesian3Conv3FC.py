import math
import torch.nn as nn
from layers.BBBConv import BBBConv2d
from layers.BBBLinear import BBBLinear
from layers.misc import FlattenLayer, ModuleWrapper

class BBB3Conv3FC(ModuleWrapper):
    """

    Simple Neural Network having 3 Convolution
    and 3 FC layers with Bayesian layers.
    """
    def __init__(self, outputs, inputs):
        super(BBB3Conv3FC, self).__init__()

        self.num_classes = outputs

        self.conv1 = BBBConv2d(inputs, 32, 5, padding=2, bias=True)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = BBBConv2d(32, 64, 5, padding=2, bias=True)
        self.soft2 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = BBBConv2d(64, 128, 5, padding=1, bias=True)
        self.soft3 = nn.Softplus()
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flatten = FlattenLayer(2 * 2 * 128)
        self.fc1 = BBBLinear(2 * 2 * 128, 1000, bias=True)
        self.soft5 = nn.Softplus()

        self.fc2 = BBBLinear(1000, 1000, bias=True)
        self.soft6 = nn.Softplus()

        self.fc3 = BBBLinear(1000, outputs, bias=True)
