import torch.nn as nn
from utils.BBBlayers import BBBConv2d, BBBLinearFactorial, FlattenLayer

class BBBELUN2(nn.Module):
    """
    Bayesian Exponential Learning Unit 2 Network
    taken from : https://arxiv.org/pdf/1511.07289.pdf

    """
    def __init__(self, outputs, inputs):
        super(BBBELUN2, self).__init__()
        self.conv1 = BBBConv2d(inputs, 96, 6, stride=1)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BBBConv2d(96, 512, 3, stride=1)
        self.soft2 = nn.Softplus()
        self.conv3 = BBBConv2d(512, 512, 3, stride=1)
        self.soft3 = nn.Softplus()
        self.conv4 = BBBConv2d(512, 512, 3, stride=1)
        self.soft4 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = BBBConv2d(512, 768, 3, stride=1)
        self.soft5 = nn.Softplus()
        self.conv6 = BBBConv2d(768, 768, 3, stride=1)
        self.soft6 = nn.Softplus()
        self.conv7 = BBBConv2d(768, 768, 2, stride=1)
        self.soft7 = nn.Softplus()
        self.conv8 = BBBConv2d(768, 768, 2, stride=1)
        self.soft8 = nn.Softplus()
        self.conv9 = BBBConv2d(768, 768, 1, stride=1)
        self.soft9 = nn.Softplus()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv10 = BBBConv2d(768, 1024, 3, stride=1)
        self.soft10 = nn.Softplus()
        self.conv11 = BBBConv2d(1024, 1024, 3, stride=1)
        self.soft11 = nn.Softplus()
        self.conv12 = BBBConv2d(1024, 1024, 3, stride=1)
        self.soft12 = nn.Softplus()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = FlattenLayer(8 * 8 * 1024)
        self.fc1 = BBBLinearFactorial(8 * 8 * 1024, 4096)
        self.soft13 = nn.Softplus()

        self.fc2 = BBBLinearFactorial(4096, 4096)
        self.soft14 = nn.Softplus()

        self.fc3 = BBBLinearFactorial(4096, outputs)

        layers = [self.conv1, self.soft1, self.pool1, self.conv2, self.soft2, self.conv3, self.soft3, self.conv4,
                  self.soft4, self.pool2, self.conv5, self.soft5, self.conv6, self.soft6, self.conv7, self.soft7,
                  self.conv8, self.soft8, self.conv9, self.soft9, self.pool3, self.conv10, self.soft10, self.conv11,
                  self.soft11, self.conv12, self.soft12,  self.pool4, self.flatten, self.fc1, self.soft13, self.fc2,
                  self.soft14, self.fc3]

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
        print('logits', logits)
        return logits, kl
