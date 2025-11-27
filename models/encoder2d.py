
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=stride) if (in_ch != out_ch or stride!=1) else nn.Identity()

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        s = self.skip(x)
        return F.relu(y + s)

class Encoder2D(nn.Module):
    def __init__(self, in_ch:int, base:int=32):
        super().__init__()
        C = base
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, C, 3, padding=1),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True)
        )
        self.layer1 = ResidualBlock(C, C, stride=1)
        self.layer2 = ResidualBlock(C, C*2, stride=2)  # 16x16
        self.layer3 = ResidualBlock(C*2, C*4, stride=2)  # 8x8
        self.out_ch = C*4

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
