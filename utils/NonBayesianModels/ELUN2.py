import torch.nn as nn
from utils.BBBlayers import FlattenLayer

class ELUN2(nn.Module):
    """
    Exponential Learning Unit 2 Network
    taken from : https://arxiv.org/pdf/1511.07289.pdf
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