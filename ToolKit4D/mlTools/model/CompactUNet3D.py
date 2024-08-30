# Peiyi Leng; edsml-pl1023
import torch.nn as nn
import torch.nn.functional as F
import torch


class SmallDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SmallDoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class SmallDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SmallDown, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool3d(2),
            SmallDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class SmallUp(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(SmallUp, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2,
                                         kernel_size=2, stride=2)

        self.conv = SmallDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class CompactUNet3D(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(CompactUNet3D, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        # Reduce the number of filters in each layer
        self.inc = SmallDoubleConv(n_channels, 32)  # Reduced from 64 to 32
        self.down1 = SmallDown(32, 64)              # Reduced from 128 to 64
        self.down2 = SmallDown(64, 128)             # Reduced from 256 to 128
        self.down3 = SmallDown(128, 256)            # Reduced from 512 to 256
        factor = 2 if bilinear else 1
        self.down4 = SmallDown(256, 512 // factor)  # Reduced from 1024 to 512
        self.up1 = SmallUp(512, 256 // factor, bilinear)
        self.up2 = SmallUp(256, 128 // factor, bilinear)
        self.up3 = SmallUp(128, 64 // factor, bilinear)
        self.up4 = SmallUp(64, 32, bilinear)

        # Final layers for regression
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Apply global average pooling and then the fully connected layer
        x = self.global_avg_pool(x)  # Output shape: [batch_size, 32, 1, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten the output to [batch_size, 32]
        x = self.fc(x)  # Output shape: [batch_size, 1]

        return x
