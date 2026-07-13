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


class FeatureGate(nn.Module):
    """Learns channel-wise feature importance."""

    def __init__(self, channels: int):
        super().__init__()

        reduced_channels = max(channels // 4, 1)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                channels,
                reduced_channels,
                kernel_size=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                reduced_channels,
                channels,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.gate(x)
        return x * weights


class DDSM(nn.Module):
    """
    Domain-Distinct Selected Module.

    Extracts and selects domain-specific features.
    """

    def __init__(self, channels: int):
        super().__init__()

        self.conv = ConvBlock(
            channels,
            channels,
        )
        self.gate = FeatureGate(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.gate(x)


class DUSM(nn.Module):
    """
    Domain-Universal Selected Module.

    Combines 1x1 and 3x3 convolutional features and applies
    channel-wise selection.
    """

    def __init__(self, channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.conv3 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.gate = FeatureGate(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x) + self.conv3(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return self.gate(x)


class FusionBlock(nn.Module):
    """Fuses distinct and universal features."""

    def __init__(self, channels: int):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        distinct_features: torch.Tensor,
        universal_features: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat(
            [distinct_features, universal_features],
            dim=1,
        )
        return self.fusion(combined)


class DS2NetSingleDomain(nn.Module):
    """
    DS2Net-style single-domain segmentation model.

    This implementation adapts DS2Net-style feature-selection
    modules for the ovarian ultrasound dataset.

    Input
    -----
    Tensor of shape [batch_size, 3, height, width].

    Output
    ------
    Tuple containing:
        final, aux2, aux3, aux4

    Every output contains raw logits.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        self.bottleneck = ConvBlock(512, 1024)

        # DS2Net-style feature-selection modules
        self.ddsm1 = DDSM(64)
        self.dusm1 = DUSM(64)
        self.fuse1 = FusionBlock(64)

        self.ddsm2 = DDSM(128)
        self.dusm2 = DUSM(128)
        self.fuse2 = FusionBlock(128)

        self.ddsm3 = DDSM(256)
        self.dusm3 = DUSM(256)
        self.fuse3 = FusionBlock(256)

        self.ddsm4 = DDSM(512)
        self.dusm4 = DUSM(512)
        self.fuse4 = FusionBlock(512)

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            kernel_size=2,
            stride=2,
        )
        self.dec4 = ConvBlock(1024, 512)

        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2,
        )
        self.dec3 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2,
        )
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2,
        )
        self.dec1 = ConvBlock(128, 64)

        # Final and auxiliary outputs
        self.final_output = nn.Conv2d(
            64,
            out_channels,
            kernel_size=1,
        )
        self.aux2_output = nn.Conv2d(
            128,
            out_channels,
            kernel_size=1,
        )
        self.aux3_output = nn.Conv2d(
            256,
            out_channels,
            kernel_size=1,
        )
        self.aux4_output = nn.Conv2d(
            512,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        bottleneck = self.bottleneck(
            self.pool(e4)
        )

        # Selected skip features
        s1 = self.fuse1(
            self.ddsm1(e1),
            self.dusm1(e1),
        )
        s2 = self.fuse2(
            self.ddsm2(e2),
            self.dusm2(e2),
        )
        s3 = self.fuse3(
            self.ddsm3(e3),
            self.dusm3(e3),
        )
        s4 = self.fuse4(
            self.ddsm4(e4),
            self.dusm4(e4),
        )

        # Decoder
        d4 = self.up4(bottleneck)
        d4 = self.dec4(
            torch.cat([d4, s4], dim=1)
        )

        d3 = self.up3(d4)
        d3 = self.dec3(
            torch.cat([d3, s3], dim=1)
        )

        d2 = self.up2(d3)
        d2 = self.dec2(
            torch.cat([d2, s2], dim=1)
        )

        d1 = self.up1(d2)
        d1 = self.dec1(
            torch.cat([d1, s1], dim=1)
        )

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


# Optional alias for compatibility with older code.
DS2Net = DS2NetSingleDomain
