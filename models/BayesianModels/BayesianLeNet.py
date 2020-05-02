import math
import torch.nn as nn
from layers import BBB_LRT_Linear, BBB_LRT_Conv2d
from layers import BBB_MCMF_LRT_Linear, BBB_MCMF_LRT_Conv2d
from layers import FlattenLayer, ModuleWrapper


class BBBLeNet(ModuleWrapper):
    '''The architecture of LeNet with Bayesian Layers'''

    def __init__(self, outputs, inputs, layer_type='mcmf_lrt'):
        super(BBBLeNet, self).__init__()

        self.num_classes = outputs

        if layer_type=='mcmf_lrt':
            BBBLinear = BBB_MCMF_LRT_Linear
            BBBConv2d = BBB_MCMF_LRT_Conv2d
        elif layer_type=='lrt':
            BBBLinear = BBB_LRT_Linear
            BBBConv2d = BBB_LRT_Conv2d
        else:
            raise ValueError("Undefined layer_type")

        self.conv1 = BBBConv2d(inputs, 6, 5, padding=0, bias=True)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BBBConv2d(6, 16, 5, padding=0, bias=True)
        self.soft2 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = FlattenLayer(5 * 5 * 16)
        self.fc1 = BBBLinear(5 * 5 * 16, 120, bias=True)
        self.soft3 = nn.Softplus()

        self.fc2 = BBBLinear(120, 84, bias=True)
        self.soft4 = nn.Softplus()

        self.fc3 = BBBLinear(84, outputs, bias=True)
