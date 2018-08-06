import torch.nn as nn
from utils.BBBlayers import BBBConv2d, FlattenLayer, BBBLinearFactorial


class BBBSqueezeNet(nn.Module):
    """
    SqueezeNet with slightly modified Fire modules and Bayesian layers.
    """
    def __init__(self, outputs, inputs):
        super(BBBSqueezeNet, self).__init__()

        self.conv1 = BBBConv2d(inputs, 64, kernel_size=3, stride=2)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # Fire module 1
        self.squeeze1 = BBBConv2d(64, 16, kernel_size=1)
        self.squeeze_activation1 = nn.Softplus()
        self.expand3x3_1 = BBBConv2d(16, 128, kernel_size=3, padding=1)
        self.expand3x3_activation1 = nn.Softplus()

        # Fire module 2
        self.squeeze2 = BBBConv2d(128, 16, kernel_size=1)
        self.squeeze_activation2 = nn.Softplus()
        self.expand3x3_2 = BBBConv2d(16, 128, kernel_size=3, padding=1)
        self.expand3x3_activation2 = nn.Softplus()

        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # Fire module 3
        self.squeeze3 = BBBConv2d(128, 32, kernel_size=1)
        self.squeeze_activation3 = nn.Softplus()
        self.expand3x3_3 = BBBConv2d(32, 256, kernel_size=3, padding=1)
        self.expand3x3_activation3 = nn.Softplus()

        # Fire module 4
        self.squeeze4 = BBBConv2d(256, 32, kernel_size=1)
        self.squeeze_activation4 = nn.Softplus()
        self.expand3x3_4 = BBBConv2d(32, 256, kernel_size=3, padding=1)
        self.expand3x3_activation4 = nn.Softplus()

        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # Fire module 5
        self.squeeze5 = BBBConv2d(256, 48, kernel_size=1)
        self.squeeze_activation5 = nn.Softplus()
        self.expand3x3_5 = BBBConv2d(48, 384, kernel_size=3, padding=1)
        self.expand3x3_activation5 = nn.Softplus()

        # Fire module 6
        self.squeeze6 = BBBConv2d(384, 48, kernel_size=1)
        self.squeeze_activation6 = nn.Softplus()
        self.expand3x3_6 = BBBConv2d(48, 384, kernel_size=3, padding=1)
        self.expand3x3_activation6 = nn.Softplus()

        # Fire module 7
        self.squeeze7 = BBBConv2d(384, 64, kernel_size=1)
        self.squeeze_activation7 = nn.Softplus()
        self.expand3x3_7 = BBBConv2d(64, 512, kernel_size=3, padding=1)
        self.expand3x3_activation7 = nn.Softplus()

        # Fire module 8
        self.squeeze8 = BBBConv2d(512, 64, kernel_size=1)
        self.squeeze_activation8 = nn.Softplus()
        self.expand3x3_8 = BBBConv2d(64, 512, kernel_size=3, padding=1)
        self.expand3x3_activation8 = nn.Softplus()

        self.drop1 = nn.Dropout(p=0.5)
        self.conv2 = BBBConv2d(512, outputs, kernel_size=1)
        self.soft2 = nn.Softplus()
        self.flatten = FlattenLayer(13 * 13 * 100)
        self.fc1 = BBBLinearFactorial(13 * 13 * 100, outputs)

        layers = [self.conv1, self.soft1, self.pool1,
                  self.squeeze1, self.squeeze_activation1, self.expand3x3_1, self.expand3x3_activation1,
                  self.squeeze2, self.squeeze_activation2, self.expand3x3_2, self.expand3x3_activation2,
                  self.pool2,
                  self.squeeze3, self.squeeze_activation3, self.expand3x3_3, self.expand3x3_activation3,
                  self.squeeze4, self.squeeze_activation4, self.expand3x3_4, self.expand3x3_activation4,
                  self.pool3,
                  self.squeeze5, self.squeeze_activation5, self.expand3x3_5, self.expand3x3_activation5,
                  self.squeeze6, self.squeeze_activation6, self.expand3x3_6, self.expand3x3_activation6,
                  self.squeeze7, self.squeeze_activation7, self.expand3x3_7, self.expand3x3_activation7,
                  self.squeeze8, self.squeeze_activation8, self.expand3x3_8, self.expand3x3_activation8,
                  self.drop1, self.conv2, self.soft2, self.flatten, self.fc1]

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
