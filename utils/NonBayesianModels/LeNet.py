import torch.nn as nn
from utils.BBBlayers import FlattenLayer


# LeNet
class LeNet(nn.Module):
    def __init__(self, outputs, inputs):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(inputs, 64, 5, stride=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, 5, stride=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            FlattenLayer(5 * 5 * 32),
            nn.Linear(5 * 5 * 32, 256),
            nn.Softplus(),
            nn.Linear(256, 128),
            nn.Softplus(),
            nn.Linear(128, outputs)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x