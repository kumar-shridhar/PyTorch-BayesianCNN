import torch.nn as nn
from utils.BBBlayers import BBBConv2d, BBBLinearFactorial, FlattenLayer


class BBBCNN1(nn.Module):
    """
    Experimental self-defined Bayesian CNN model

    """
    def __init__(self, outputs, inputs):
        super(BBBCNN1, self).__init__()
        self.conv1 = BBBConv2d(inputs, 92, 3, stride=1)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv2 = BBBConv2d(92, 384, 1, stride=1)
        self.soft2 = nn.Softplus()
        self.conv3 = BBBConv2d(384, 384, 2, stride=1)
        self.soft3 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv4 = BBBConv2d(384, 640, 2, stride=1)
        self.soft4 = nn.Softplus()
        self.conv5 = BBBConv2d(640, 640, 2, stride=1)
        self.soft5 = nn.Softplus()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv6 = BBBConv2d(640, 640, 1, stride=1)
        self.soft6 = nn.Softplus()
        self.conv7 = BBBConv2d(640, 768, 2, stride=1)
        self.soft7 = nn.Softplus()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv8 = BBBConv2d(768, 768, 2, stride=1)
        self.soft8 = nn.Softplus()
        self.conv9 = BBBConv2d(768, 768, 2, stride=1)
        self.soft9 = nn.Softplus()
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv10 = BBBConv2d(768, 768, 1, stride=1)
        self.soft10 = nn.Softplus()
        self.conv11 = BBBConv2d(768, 640, 2, stride=1)
        self.soft11 = nn.Softplus()
        self.conv12 = BBBConv2d(640, 384, 2, stride=1)
        self.soft12 = nn.Softplus()
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = FlattenLayer(8 * 8 * 384)
        self.fc1 = BBBLinearFactorial(8 * 8 * 384, outputs)

        layers = [self.conv1, self.soft1, self.pool1, self.conv2, self.soft2, self.conv3, self.soft3, self.pool2, self.conv4,
                  self.soft4, self.conv5, self.soft5, self.pool3, self.conv6, self.soft6, self.conv7, self.soft7, self.pool4,
                  self.conv8, self.soft8, self.conv9, self.soft9, self.pool5, self.conv10, self.soft10, self.conv11,
                  self.soft11, self.conv12, self.soft12, self.pool6, self.flatten, self.fc1]

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


