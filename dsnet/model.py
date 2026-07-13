import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two convolution, batch-normalization, and ReLU operations."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DSNet(nn.Module):
    """
    DSNet-style encoder-decoder network with deep supervision.

    Input
    -----
    Tensor of shape [batch_size, 3, height, width].

    Output
    ------
    Tuple containing:
        final, aux2, aux3, aux4

    Every output contains raw logits. Sigmoid is applied only during
    prediction or metric calculation.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        # Final prediction
        self.final_output = nn.Conv2d(64, out_channels, kernel_size=1)

        # Deep-supervision predictions
        self.aux2_output = nn.Conv2d(128, out_channels, kernel_size=1)
        self.aux3_output = nn.Conv2d(256, out_channels, kernel_size=1)
        self.aux4_output = nn.Conv2d(512, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        bottleneck = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.up4(bottleneck)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        final = self.final_output(d1)
        output_size = final.shape[2:]

        aux2 = F.interpolate(
            self.aux2_output(d2),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        aux3 = F.interpolate(
            self.aux3_output(d3),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        aux4 = F.interpolate(
            self.aux4_output(d4),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        return final, aux2, aux3, aux4
