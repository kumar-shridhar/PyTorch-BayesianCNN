import torch.nn as nn
import math
from layers.BBBConv import BBBConv2d
from layers.BBBLinear import BBBLinear
from layers.misc import FlattenLayer, ModuleWrapper


class BBBAlexNet(ModuleWrapper):
    '''The architecture of AlexNet with Bayesian Layers'''

    def __init__(self, outputs, inputs):
        super(BBBAlexNet, self).__init__()

        self.num_classes = outputs

        self.conv1 = BBBConv2d(inputs, 64, 11, stride=4, padding=5, bias=True)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BBBConv2d(64, 192, 5, padding=2, bias=True)
        self.soft2 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = BBBConv2d(192, 384, 3, padding=1, bias=True)
        self.soft3 = nn.Softplus()

        self.conv4 = BBBConv2d(384, 256, 3, padding=1, bias=True)
        self.soft4 = nn.Softplus()

        self.conv5 = BBBConv2d(256, 128, 3, padding=1, bias=True)
        self.soft5 = nn.Softplus()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = FlattenLayer(1 * 1 * 128)
        self.classifier = BBBLinear(1 * 1 * 128, outputs, bias=True)
