import torch.nn as nn
from utils.BBBlayers import FlattenLayer

class ELUN1(nn.Module):
    """
    Exponential Learning Unit 1 Network
    taken from : https://arxiv.org/pdf/1511.07289.pdf
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
