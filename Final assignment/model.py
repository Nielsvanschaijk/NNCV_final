import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Model(nn.Module):
    """
    Multi-head UNet model (used in training).
    Codalab will import this class and call forward(x), which must return a single output.
    So we return the output of one head — e.g., small head.
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.inc   = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        # Small head (8 classes)
        self.up1_small = Up(1024, 256)
        self.up2_small = Up(512, 128)
        self.up3_small = Up(256, 64)
        self.up4_small = Up(128, 64)
        self.outc_small = OutConv(64, 8)

        # Medium head (5 classes) — not used in forward
        self.up1_medium = Up(1024, 256)
        self.up2_medium = Up(512, 128)
        self.up3_medium = Up(256, 64)
        self.up4_medium = Up(128, 64)
        self.outc_medium = OutConv(64, 5)

        # Big head (9 classes) — not used in forward
        self.up1_big = Up(1024, 256)
        self.up2_big = Up(512, 128)
        self.up3_big = Up(256, 64)
        self.up4_big = Up(128, 64)
        self.outc_big = OutConv(64, 9)

    def forward(self, x):
        # Shared encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Use SMALL head only for output (matches model checkpoint!)
        xs = self.up1_small(x5, x4)
        xs = self.up2_small(xs, x3)
        xs = self.up3_small(xs, x2)
        xs = self.up4_small(xs, x1)
        out = self.outc_small(xs)

        return out  # shape [B, 8, H, W]
