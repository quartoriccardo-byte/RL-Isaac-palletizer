
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )

class UNet2D(nn.Module):
    def __init__(self, in_ch:int, base:int=32):
        super().__init__()
        c1, c2, c3, c4 = base, base*2, base*4, base*8
        self.down1 = conv_block(in_ch, c1)
        self.pool1 = nn.MaxPool2d(2)  # 16x16
        self.down2 = conv_block(c1, c2)
        self.pool2 = nn.MaxPool2d(2)  # 8x8
        self.down3 = conv_block(c2, c3)
        self.pool3 = nn.MaxPool2d(2)  # 4x4
        self.bridge = conv_block(c3, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)  # 8x8
        self.dec3 = conv_block(c3+c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)  # 16x16
        self.dec2 = conv_block(c2+c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)  # 32x32
        self.dec1 = conv_block(c1+c1, c1)
        self.out = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        b = self.bridge(p3)
        u3 = self.up3(b)
        c3 = torch.cat([u3, d3], dim=1)
        d = self.dec3(c3)
        u2 = self.up2(d)
        c2 = torch.cat([u2, d2], dim=1)
        d = self.dec2(c2)
        u1 = self.up1(d)
        c1 = torch.cat([u1, d1], dim=1)
        d = self.dec1(c1)
        m = torch.sigmoid(self.out(d))
        return m.squeeze(1)  # (B, L, W)
