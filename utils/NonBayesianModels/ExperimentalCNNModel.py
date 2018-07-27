import torch.nn as nn
from utils.BBBlayers import FlattenLayer

class CNN1(nn.Module):
    """
    Experimental self-defined CNN Model
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