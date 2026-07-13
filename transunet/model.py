import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
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

    def forward(self, x):
        return self.block(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_dim,
        dropout=0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens):
        normalized = self.norm1(tokens)

        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )

        tokens = tokens + self.dropout1(
            attention_output
        )

        tokens = tokens + self.mlp(
            self.norm2(tokens)
        )

        return tokens


class TransformerBottleneck(nn.Module):
    def __init__(
        self,
        in_channels=512,
        embed_dim=512,
        num_heads=8,
        depth=4,
        mlp_dim=1024,
        dropout=0.1,
        max_tokens=256,
    ):
        super().__init__()

        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=1,
            bias=False,
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                max_tokens,
                embed_dim,
            )
        )

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.final_norm = nn.LayerNorm(
            embed_dim
        )

    def forward(self, x):
        x = self.projection(x)

        batch_size, channels, height, width = x.shape

        tokens = (
            x.flatten(2)
            .transpose(1, 2)
            .contiguous()
        )

        token_count = tokens.shape[1]

        if token_count > self.position_embedding.shape[1]:
            raise ValueError(
                "Transformer bottleneck received more tokens "
                "than the configured positional embedding supports."
            )

        tokens = tokens + self.position_embedding[
            :,
            :token_count,
            :,
        ]

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.final_norm(tokens)

        feature_map = (
            tokens.transpose(1, 2)
            .contiguous()
            .reshape(
                batch_size,
                channels,
                height,
                width,
            )
        )

        return feature_map


class TransUNet(nn.Module):
    """
    Lightweight hybrid CNN-Transformer U-Net.

    Input:
        [batch_size, 3, height, width]

    Output:
        Raw segmentation logits with shape
        [batch_size, 1, height, width]
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
    ):
        super().__init__()

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # CNN encoder
        self.enc1 = ConvBlock(
            in_channels,
            64,
        )
        self.enc2 = ConvBlock(
            64,
            128,
        )
        self.enc3 = ConvBlock(
            128,
            256,
        )
        self.enc4 = ConvBlock(
            256,
            512,
        )

        # Transformer bottleneck at 1/16 resolution.
        # For 256x256 inputs, the bottleneck is 16x16 = 256 tokens.
        self.transformer = TransformerBottleneck(
            in_channels=512,
            embed_dim=512,
            num_heads=8,
            depth=4,
            mlp_dim=1024,
            dropout=0.1,
            max_tokens=256,
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            512,
            512,
            kernel_size=2,
            stride=2,
        )
        self.dec4 = ConvBlock(
            1024,
            512,
        )

        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2,
        )
        self.dec3 = ConvBlock(
            512,
            256,
        )

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2,
        )
        self.dec2 = ConvBlock(
            256,
            128,
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2,
        )
        self.dec1 = ConvBlock(
            128,
            64,
        )

        self.classifier = nn.Conv2d(
            64,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        e1 = self.enc1(x)
        e2 = self.enc2(
            self.pool(e1)
        )
        e3 = self.enc3(
            self.pool(e2)
        )
        e4 = self.enc4(
            self.pool(e3)
        )

        bottleneck_input = self.pool(e4)
        bottleneck = self.transformer(
            bottleneck_input
        )

        d4 = self.up4(bottleneck)
        d4 = self.dec4(
            torch.cat(
                [d4, e4],
                dim=1,
            )
        )

        d3 = self.up3(d4)
        d3 = self.dec3(
            torch.cat(
                [d3, e3],
                dim=1,
            )
        )

        d2 = self.up2(d3)
        d2 = self.dec2(
            torch.cat(
                [d2, e2],
                dim=1,
            )
        )

        d1 = self.up1(d2)
        d1 = self.dec1(
            torch.cat(
                [d1, e1],
                dim=1,
            )
        )

        logits = self.classifier(d1)

        if logits.shape[-2:] != input_size:
            logits = F.interpolate(
                logits,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )

        return logits
