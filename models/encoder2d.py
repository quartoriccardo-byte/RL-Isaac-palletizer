
import torch
import torch.nn as nn

class Encoder2D(nn.Module):
    def __init__(self, in_channels=1, features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(features),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(features),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.out_ch = features # Expose output channels for dynamic policy head

    def forward(self, x):
        return self.net(x)
