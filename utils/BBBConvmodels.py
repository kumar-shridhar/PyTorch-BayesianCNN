import torch
import torch.nn as nn
from .BBBlayers import BBBConv2d, BBBLinearFactorial, FlattenLayer


# Bayesian LeNet
class BBBLeNet(nn.Module):
    def __init__(self, outputs, inputs):
        super(BBBLeNet, self).__init__()
        self.conv1 = BBBConv2d(inputs, 6, 5, stride=1)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BBBConv2d(6, 16, 5, stride=1)
        self.soft2 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = FlattenLayer(5 * 5 * 16)
        self.fc1 = BBBLinearFactorial(5 * 5 * 16, 120)
        self.soft3 = nn.Softplus()

        self.fc2 = BBBLinearFactorial(120, 84)
        self.soft4 = nn.Softplus()

        self.fc3 = BBBLinearFactorial(84, outputs)

        layers = [self.conv1, self.soft1, self.pool1, self.conv2, self.soft2, self.pool2,
                  self.flatten, self.fc1, self.soft3, self.fc2, self.soft4, self.fc3]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
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


# Bayesian 3Conv3FC
class BBB3Conv3FC(nn.Module):
    """
    To train on CIFAR-10:
    https://arxiv.org/pdf/1207.0580.pdf
    """
    def __init__(self, outputs, inputs):
        super(BBB3Conv3FC, self).__init__()
        self.conv1 = BBBConv2d(inputs, 32, 5, stride=1, padding=2)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = BBBConv2d(32, 64, 5, stride=1, padding=2)
        self.soft2 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = BBBConv2d(64, 128, 5, stride=1, padding=1)
        self.soft3 = nn.Softplus()
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flatten = FlattenLayer(2 * 2 * 128)
        self.fc1 = BBBLinearFactorial(2 * 2 * 128, 1000)
        self.soft5 = nn.Softplus()

        self.fc2 = BBBLinearFactorial(1000, 1000)
        self.soft6 = nn.Softplus()

        self.fc3 = BBBLinearFactorial(1000, outputs)

        layers = [self.conv1, self.soft1, self.pool1, self.conv2, self.soft2, self.pool2,
                  self.conv3, self.soft3, self.pool3, self.flatten, self.fc1, self.soft5,
                  self.fc2, self.soft6, self.fc3]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
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


# Bayesian ELU-Network 1
class BBBELUN1(nn.Module):
    """
    To train on CIFAR-100:
    https://arxiv.org/pdf/1511.07289.pdf
    """
    def __init__(self, outputs, inputs):
        super(BBBELUN1, self).__init__()
        self.conv1 = BBBConv2d(inputs, 384, 3, stride=1)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv2 = BBBConv2d(384, 384, 1, stride=1)
        self.soft2 = nn.Softplus()
        self.conv3 = BBBConv2d(384, 384, 2, stride=1)
        self.soft3 = nn.Softplus()
        self.conv4 = BBBConv2d(384, 640, 2, stride=1)
        self.soft4 = nn.Softplus()
        self.conv5 = BBBConv2d(640, 640, 2, stride=1)
        self.soft5 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv6 = BBBConv2d(640, 640, 1, stride=1)
        self.soft6 = nn.Softplus()
        self.conv7 = BBBConv2d(640, 768, 2, stride=1)
        self.soft7 = nn.Softplus()
        self.conv8 = BBBConv2d(768, 768, 2, stride=1)
        self.soft8 = nn.Softplus()
        self.conv9 = BBBConv2d(768, 768, 2, stride=1)
        self.soft9 = nn.Softplus()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv10 = BBBConv2d(768, 768, 1, stride=1)
        self.soft10 = nn.Softplus()
        self.conv11 = BBBConv2d(768, 896, 2, stride=1)
        self.soft11 = nn.Softplus()
        self.conv12 = BBBConv2d(896, 896, 2, stride=1)
        self.soft12 = nn.Softplus()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv13 = BBBConv2d(896, 896, 3, stride=1)
        self.soft13 = nn.Softplus()
        self.conv14 = BBBConv2d(896, 1024, 2, stride=1)
        self.soft14 = nn.Softplus()
        self.conv15 = BBBConv2d(1024, 1024, 2, stride=1)
        self.soft15 = nn.Softplus()
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv16 = BBBConv2d(1024, 1024, 1, stride=1)
        self.soft16 = nn.Softplus()
        self.conv17 = BBBConv2d(1024, 1152, 2, stride=1)
        self.soft17 = nn.Softplus()
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv18 = BBBConv2d(1152, 1152, 2, stride=1)
        self.soft18 = nn.Softplus()
        self.pool7 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = FlattenLayer(2 * 2 * 1152)
        self.fc1 = BBBLinearFactorial(2 * 2 * 1152, outputs)

        layers = [self.conv1, self.soft1, self.pool1, self.conv2, self.soft2, self.conv3, self.soft3, self.conv4,
                  self.soft4, self.conv5, self.soft5, self.pool2, self.conv6, self.soft6, self.conv7, self.soft7,
                  self.conv8, self.soft8, self.conv9, self.soft9, self.pool3, self.conv10, self.soft10, self.conv11,
                  self.soft11, self.conv12, self.soft12, self.pool4, self.conv13, self.soft13, self.conv14,
                  self.soft14, self.conv15, self.soft15, self.pool5, self.conv16, self.soft16,
                  self.conv17, self.soft17, self.pool6, self.conv18, self.soft18, self.pool7, self.flatten, self.fc1]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
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


# Bayesian self-defined CNN
class BBBCNN1(nn.Module):
    """
    To train on CIFAR-100:
    https://arxiv.org/pdf/1511.07289.pdf
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
                print('x', x.size())
        logits = x
        print('logits', logits)
        return logits, kl


# Bayesian ELU-Network 2
class BBBELUN2(nn.Module):
    """
    To train on ImageNet:
    https://arxiv.org/pdf/1511.07289.pdf
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
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl
                print('x_conv size', x.size())

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, _kl, = layer.fcprobforward(x)
                kl += _kl
                print('x_fc size', x.size())
            else:
                x = layer(x)
                print('x size', x.size())
        logits = x
        print('logits', logits)
        return logits, kl
