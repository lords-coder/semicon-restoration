import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.block(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()

        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class RestoreNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, channels=64, num_blocks=8):
        super().__init__()
        
        self.conv_first = nn.Conv2d(in_channels, channels, 3, 1, 1)
        
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        
        self.attention = ChannelAttention(channels)
        
        self.conv_mid = nn.Conv2d(channels, channels, 3, 1, 1)
        
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            
        )
        
        self.conv_last = nn.Conv2d(channels, out_channels, 3, 1, 1)

    def forward(self, x):
        feat = self.conv_first(x)
        residual = feat
        out = self.blocks(feat)
        out = self.attention(out)
        out = self.conv_mid(out) + residual
        out = self.upsample(out)
        out = self.conv_last(out)
        return out