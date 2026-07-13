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


class AttentionGate(nn.Module):
    """Filters an encoder skip connection using a decoder gating signal."""

    def __init__(
        self,
        gating_channels: int,
        skip_channels: int,
        intermediate_channels: int,
    ):
        super().__init__()

        self.gating_projection = nn.Sequential(
            nn.Conv2d(
                gating_channels,
                intermediate_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(intermediate_channels),
        )

        self.skip_projection = nn.Sequential(
            nn.Conv2d(
                skip_channels,
                intermediate_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(intermediate_channels),
        )

        self.attention_map = nn.Sequential(
            nn.Conv2d(
                intermediate_channels,
                1,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        gating_signal: torch.Tensor,
        skip_feature: torch.Tensor,
    ) -> torch.Tensor:
        combined = self.relu(
            self.gating_projection(gating_signal)
            + self.skip_projection(skip_feature)
        )

        attention_coefficients = self.attention_map(combined)

        return skip_feature * attention_coefficients


class AttentionUNet(nn.Module):
    """
    Attention U-Net with deep supervision.

    Input
    -----
    Tensor of shape [batch_size, 3, height, width].

    Output
    ------
    Tuple containing:
        final, aux2, aux3, aux4

    All outputs contain raw logits.
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

        # Decoder level 4
        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            kernel_size=2,
            stride=2,
        )
        self.att4 = AttentionGate(
            gating_channels=512,
            skip_channels=512,
            intermediate_channels=256,
        )
        self.dec4 = ConvBlock(1024, 512)

        # Decoder level 3
        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2,
        )
        self.att3 = AttentionGate(
            gating_channels=256,
            skip_channels=256,
            intermediate_channels=128,
        )
        self.dec3 = ConvBlock(512, 256)

        # Decoder level 2
        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2,
        )
        self.att2 = AttentionGate(
            gating_channels=128,
            skip_channels=128,
            intermediate_channels=64,
        )
        self.dec2 = ConvBlock(256, 128)

        # Decoder level 1
        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2,
        )
        self.att1 = AttentionGate(
            gating_channels=64,
            skip_channels=64,
            intermediate_channels=32,
        )
        self.dec1 = ConvBlock(128, 64)

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

        # Decoder level 4
        d4 = self.up4(bottleneck)
        attended_e4 = self.att4(
            gating_signal=d4,
            skip_feature=e4,
        )
        d4 = self.dec4(
            torch.cat([d4, attended_e4], dim=1)
        )

        # Decoder level 3
        d3 = self.up3(d4)
        attended_e3 = self.att3(
            gating_signal=d3,
            skip_feature=e3,
        )
        d3 = self.dec3(
            torch.cat([d3, attended_e3], dim=1)
        )

        # Decoder level 2
        d2 = self.up2(d3)
        attended_e2 = self.att2(
            gating_signal=d2,
            skip_feature=e2,
        )
        d2 = self.dec2(
            torch.cat([d2, attended_e2], dim=1)
        )

        # Decoder level 1
        d1 = self.up1(d2)
        attended_e1 = self.att1(
            gating_signal=d1,
            skip_feature=e1,
        )
        d1 = self.dec1(
            torch.cat([d1, attended_e1], dim=1)
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
