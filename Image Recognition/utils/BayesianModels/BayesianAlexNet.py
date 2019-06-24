
import torch.nn as nn
# from utils.BBBlayers import BBBConv2d, BBBLinearFactorial, FlattenLayer
# from utils.conv2d import BBBConv2d
# from utils.linear import BBBLinearFactorial
import math
from utils.BBBlayers import BBBConv2d, BBBLinearFactorial


class BBBAlexNet(nn.Module):
    '''The architecture of AlexNet with Bayesian Layers'''

    def __init__(self, outputs, inputs):
        super(BBBAlexNet, self).__init__()

        self.q_logvar_init = 0.05
        self.p_logvar_init = math.log(0.05)
 
        self.classifier = BBBLinearFactorial(self.q_logvar_init, self.p_logvar_init, 1* 1 * 128, outputs)

        self.conv1 = BBBConv2d(self.q_logvar_init, self.p_logvar_init, inputs, 64, kernel_size=11, stride=4, padding=5)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BBBConv2d(self.q_logvar_init,  self.p_logvar_init, 64, 192, kernel_size=5, padding=2)
        self.soft2 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = BBBConv2d(self.q_logvar_init, self.p_logvar_init, 192, 384, kernel_size=3, padding=1)
        self.soft3 = nn.Softplus()

        self.conv4 = BBBConv2d(self.q_logvar_init, self.p_logvar_init, 384, 256, kernel_size=3, padding=1)
        self.soft4 = nn.Softplus()

        self.conv5 = BBBConv2d(self.q_logvar_init, self.p_logvar_init, 256, 128, kernel_size=3, padding=1)
        self.soft5 = nn.Softplus()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.flatten = FlattenLayer(1 * 1 * 128)
        # self.fc1 = BBBLinearFactorial(q_logvar_init, N, p_logvar_init, 1* 1 * 128, outputs)


        layers = [self.conv1, self.soft1, self.pool1, self.conv2, self.soft2, self.pool2, self.conv3, self.soft3,
                  self.conv4, self.soft4, self.conv5, self.soft5, self.pool3]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
            else:
                x = layer.forward(x)
        x = x.view(x.size(0), -1)
        x, _kl = self.classifier.fcprobforward(x)
        kl += _kl
        logits = x
        return logits, kl

    # def probforward(self, x):
    #     'Forward pass with Bayesian weights'
    #     kl = 0
    #     for layer in self.layers:
    #         if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
    #             x, _kl, = layer.convprobforward(x)
    #             kl += _kl

    #         elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
    #             x, _kl, = layer.fcprobforward(x)
    #             kl += _kl
    #         else:
    #             x = layer(x)
    #     logits = x
    #     return logits, kl


