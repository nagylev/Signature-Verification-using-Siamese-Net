import torch
from torch import nn


class SigSiameseNet(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.cnnModel = nn.Sequential(

            nn.Conv2d(1, 96, kernel_size=11),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),

            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, padding_mode='zeros'),

            nn.ReLU(),
            nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),

            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

            nn.Flatten(),
            nn.Linear(23296, 1024),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),

            nn.Linear(1024, 128),
            nn.ReLU()
        )

    def forward(self, in1, in2):
        in1 = self.cnnModel(in1)
        in2 = self.cnnModel(in2)
        return in1, in2

