import torch.nn as nn
import torch.nn.functional as F
from layers import BBB_Linear, BBB_Conv2d
from layers import BBB_LRT_Linear, BBB_LRT_Conv2d
from layers import FlattenLayer, ModuleWrapper
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=2, padding_mode='replicate') # padding mode same as original Caffe code
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=2, padding_mode='replicate')
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2, padding_mode='replicate')
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    

class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, 1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.sigmoid(self.pixel_shuffle(self.conv3(x)))
        return x

class BayesianNet(ModuleWrapper):
    def __init__(self, inputs, priors, layer_type='lrt', activation_type='softplus'):
        super(BayesianNet, self).__init__()

        self.layer_type = layer_type
        self.priors = priors

        if layer_type=='lrt':
            BBBLinear = BBB_LRT_Linear
            BBBConv2d = BBB_LRT_Conv2d
        elif layer_type=='bbb':
            BBBLinear = BBB_Linear
            BBBConv2d = BBB_Conv2d
        else:
            raise ValueError("Undefined layer_type")
        
        if activation_type=='softplus':
            self.act = nn.Softplus
        elif activation_type=='relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")
        self.conv1 = BBBConv2d(1, 64, 5, padding=2, bias=True, priors=self.priors)
        # self.act1 = self.act()
        # self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1)


        # self.conv2 = BBBConv2d(64, 64, 3, padding=1, bias=True, priors=self.priors)
        self.conv2 = BBBConv2d(64, 32, 3, padding=1, bias=True, priors=self.priors)
        # self.act2 = self.act()
        # self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1)

        self.conv3 = BBBConv2d(32, 1 * (1 ** 2), 3, padding=1, bias=True, priors=self.priors)
        # self.act3 = self.act()
        # self.pool3 = nn.MaxPool2d(kernel_size=3, stride=1)

        # self.flatten = FlattenLayer(2 * 2 * 128)
        # self.fc1 = BBBLinear(2 * 2 * 128, 1000, bias=True, priors=self.priors)
        # self.act4 = self.act()

        # self.fc2 = BBBLinear(1000, 1000, bias=True, priors=self.priors)
        # self.act5 = self.act()

        # self.fc3 = BBBLinear(1000, 1, bias=True, priors=self.priors)


        self.pixel_shuffle = nn.PixelShuffle(1)

    # TODO: should this one be overwritten? 
    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.sigmoid(self.pixel_shuffle(self.conv3(x)))
        return x
