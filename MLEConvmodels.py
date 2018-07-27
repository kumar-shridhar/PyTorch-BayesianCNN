import torch
import torch.nn as nn
from utils.BBBlayers import FlattenLayer


# LeNet
class LeNet(nn.Module):
    def __init__(self, outputs, inputs):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(inputs, 6, 5, stride=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 5, stride=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            FlattenLayer(5 * 5 * 16),
            nn.Linear(5 * 5 * 16, 120),
            nn.Softplus(),
            nn.Linear(120, 84),
            nn.Softplus(),
            nn.Linear(84, outputs)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# 3Conv3FC
class _3Conv3FC(nn.Module):
    """
    To train on CIFAR-10:
    https://arxiv.org/pdf/1207.0580.pdf
    """
    def __init__(self, outputs, inputs):
        super(_3Conv3FC, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(inputs, 32, 5, stride=1, padding=2),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, 5, stride=1, padding=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            FlattenLayer(2 * 2 * 128),
            nn.Linear(2 * 2 * 128, 1000),
            nn.Softplus(),
            nn.Linear(1000, 1000),
            nn.Softplus(),
            nn.Linear(1000, outputs)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        print('X', x)
        return x


# Bayesian self-defined CNN
class CNN1(nn.Module):
    """
    To train on CIFAR-100
    """
    def __init__(self, outputs, inputs):
        super(CNN1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(inputs, 92, 3, stride=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(92, 384, 1, stride=1),
            nn.Softplus(),
            nn.Conv2d(384, 384, 2, stride=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(384, 640, 2, stride=1),
            nn.Softplus(),
            nn.Conv2d(640, 640, 2, stride=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(640, 640, 1, stride=1),
            nn.Softplus(),
            nn.Conv2d(640, 768, 2, stride=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(768, 768, 2, stride=1),
            nn.Softplus(),
            nn.Conv2d(768, 768, 2, stride=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(768, 768, 1, stride=1),
            nn.Softplus(),
            nn.Conv2d(768, 640, 2, stride=1),
            nn.Softplus(),
            nn.Conv2d(640, 384, 2, stride=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            FlattenLayer(8 * 8 * 384),
            nn.Linear(8 * 8 * 384, outputs)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        print('X', x)
        return x


# ELU-Network 1
class ELUN1(nn.Module):
    """
    To train on CIFAR-100:
    https://arxiv.org/pdf/1511.07289.pdf
    """
    def __init__(self, outputs, inputs):
        super(ELUN1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(inputs, 384, 3, stride=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(384, 384, 1, stride=1),
            nn.Softplus(),
            nn.Conv2d(384, 384, 2, stride=1),
            nn.Softplus(),
            nn.Conv2d(384, 640, 2, stride=1),
            nn.Softplus(),
            nn.Conv2d(640, 640, 2, stride=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(640, 640, 1, stride=1),
            nn.Softplus(),
            nn.Conv2d(640, 768, 2, stride=1),
            nn.Softplus(),
            nn.Conv2d(768, 768, 2, stride=1),
            nn.Softplus(),
            nn.Conv2d(768, 768, 2, stride=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(768, 768, 1, stride=1),
            nn.Softplus(),
            nn.Conv2d(768, 896, 2, stride=1),
            nn.Softplus(),
            nn.Conv2d(896, 896, 2, stride=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(896, 896, 3, stride=1),
            nn.Softplus(),
            nn.Conv2d(896, 1024, 2, stride=1),
            nn.Softplus(),
            nn.Conv2d(1024, 1024, 2, stride=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(1024, 1024, 1, stride=1),
            nn.Softplus(),
            nn.Conv2d(1024, 1152, 2, stride=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(1152, 1152, 2, stride=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            FlattenLayer(2 * 2 * 1152),
            nn.Linear(2 * 2 * 1152, outputs)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ELU-Network 2
class ELUN2(nn.Module):
    """
    To train on CIFAR-100:
    https://arxiv.org/pdf/1511.07289.pdf
    """
    def __init__(self, outputs, inputs):
        super(ELUN2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(inputs, 96, 6, stride=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(96, 512, 3, stride=1),
            nn.Softplus(),
            nn.Conv2d(512, 512, 3, stride=1),
            nn.Softplus(),
            nn.Conv2d(512, 512, 3, stride=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 768, 3, stride=1),
            nn.Softplus(),
            nn.Conv2d(768, 768, 3, stride=1),
            nn.Softplus(),
            nn.Conv2d(768, 768, 2, stride=1),
            nn.Softplus(),
            nn.Conv2d(768, 768, 2, stride=1),
            nn.Softplus(),
            nn.Conv2d(768, 768, 1, stride=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(768, 1024, 3, stride=1),
            nn.Softplus(),
            nn.Conv2d(1024, 1024, 3, stride=1),
            nn.Softplus(),
            nn.Conv2d(1024, 1024, 3, stride=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            FlattenLayer(8 * 8 * 1024),
            nn.Linear(8 * 8 * 1024, 4096),
            nn.Softplus(),
            nn.Linear(4096, 4096),
            nn.Softplus(),
            nn.Linear(4096, outputs)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
