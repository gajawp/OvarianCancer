import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, stride=stride),
            ConvBNReLU(out_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = ConvBNReLU(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = self.conv(self.pool(x))
        return F.interpolate(
            x,
            size=size,
            mode="bilinear",
            align_corners=False,
        )


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=256,
        dilation_rates=(6, 12, 18),
    ):
        super().__init__()

        self.branches = nn.ModuleList([
            ConvBNReLU(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
            ),
            ASPPConv(in_channels, out_channels, dilation_rates[0]),
            ASPPConv(in_channels, out_channels, dilation_rates[1]),
            ASPPConv(in_channels, out_channels, dilation_rates[2]),
            ASPPPooling(in_channels, out_channels),
        ])

        self.project = nn.Sequential(
            ConvBNReLU(
                out_channels * 5,
                out_channels,
                kernel_size=1,
                padding=0,
            ),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        features = [branch(x) for branch in self.branches]
        return self.project(torch.cat(features, dim=1))


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+-style semantic segmentation model.

    Returns one raw-logit segmentation tensor.
    """

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.encoder1 = EncoderBlock(in_channels, 64, stride=2)
        self.encoder2 = EncoderBlock(64, 128, stride=2)
        self.encoder3 = EncoderBlock(128, 256, stride=2)
        self.encoder4 = EncoderBlock(256, 512, stride=2)

        self.aspp = ASPP(512, 256)

        self.low_level_projection = ConvBNReLU(
            128,
            48,
            kernel_size=1,
            padding=0,
        )

        self.decoder = nn.Sequential(
            ConvBNReLU(256 + 48, 256),
            ConvBNReLU(256, 256),
        )

        self.classifier = nn.Conv2d(
            256,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        x1 = self.encoder1(x)
        low_level = self.encoder2(x1)
        x3 = self.encoder3(low_level)
        high_level = self.encoder4(x3)

        high_level = self.aspp(high_level)
        high_level = F.interpolate(
            high_level,
            size=low_level.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        low_level = self.low_level_projection(low_level)

        x = torch.cat([high_level, low_level], dim=1)
        x = self.decoder(x)
        logits = self.classifier(x)

        return F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
