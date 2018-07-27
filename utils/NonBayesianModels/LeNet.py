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