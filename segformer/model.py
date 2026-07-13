import torch
import torch.nn as nn
import torch.nn.functional as F


class OverlapPatchEmbedding(nn.Module):
    """
    Converts an input feature map into overlapping patch tokens.

    The convolution creates overlapping patches, and LayerNorm
    normalizes the resulting token embeddings.
    """

    def __init__(
        self,
        in_channels,
        embed_dim,
        patch_size,
        stride,
    ):
        super().__init__()

        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
            bias=False,
        )

        self.normalization = nn.LayerNorm(
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

        tokens = self.normalization(
            tokens
        )

        return tokens, height, width


class EfficientSelfAttention(nn.Module):
    """
    Multi-head self-attention with spatial reduction.

    Spatial reduction decreases the number of key and value tokens,
    which makes attention more memory-efficient.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        sr_ratio=1,
        dropout=0.0,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(
            embed_dim,
            embed_dim,
        )

        self.key_value = nn.Linear(
            embed_dim,
            embed_dim * 2,
        )

        self.sr_ratio = sr_ratio

        if sr_ratio > 1:
            self.spatial_reduction = nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=sr_ratio,
                stride=sr_ratio,
            )

            self.spatial_normalization = nn.LayerNorm(
                embed_dim
            )

        self.attention_dropout = nn.Dropout(
            dropout
        )

        self.output_projection = nn.Linear(
            embed_dim,
            embed_dim,
        )

        self.output_dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        tokens,
        height,
        width,
    ):
        batch_size, token_count, channels = tokens.shape

        query = self.query(
            tokens
        )

        query = (
            query.reshape(
                batch_size,
                token_count,
                self.num_heads,
                self.head_dim,
            )
            .permute(
                0,
                2,
                1,
                3,
            )
            .contiguous()
        )

        if self.sr_ratio > 1:
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

            reduced_features = self.spatial_reduction(
                feature_map
            )

            reduced_tokens = (
                reduced_features.flatten(2)
                .transpose(1, 2)
                .contiguous()
            )

            reduced_tokens = self.spatial_normalization(
                reduced_tokens
            )

            key_value_input = reduced_tokens

        else:
            key_value_input = tokens

        key_value = self.key_value(
            key_value_input
        )

        key_value = (
            key_value.reshape(
                batch_size,
                -1,
                2,
                self.num_heads,
                self.head_dim,
            )
            .permute(
                2,
                0,
                3,
                1,
                4,
            )
            .contiguous()
        )

        key = key_value[0].contiguous()
        value = key_value[1].contiguous()

        attention_scores = (
            query
            @ key.transpose(-2, -1)
        ) * self.scale

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )

        attention_weights = self.attention_dropout(
            attention_weights
        )

        output = attention_weights @ value

        output = (
            output.transpose(1, 2)
            .contiguous()
            .reshape(
                batch_size,
                token_count,
                channels,
            )
        )

        output = self.output_projection(
            output
        )

        return self.output_dropout(
            output
        )


class MixFFN(nn.Module):
    """
    SegFormer Mix-FFN block.

    A depthwise convolution is inserted between two linear layers
    to preserve local spatial information.
    """

    def __init__(
        self,
        embed_dim,
        hidden_dim,
        dropout=0.0,
    ):
        super().__init__()

        self.linear1 = nn.Linear(
            embed_dim,
            hidden_dim,
        )

        self.depthwise_convolution = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )

        self.activation = nn.GELU()

        self.dropout1 = nn.Dropout(
            dropout
        )

        self.linear2 = nn.Linear(
            hidden_dim,
            embed_dim,
        )

        self.dropout2 = nn.Dropout(
            dropout
        )

    def forward(
        self,
        tokens,
        height,
        width,
    ):
        batch_size, _, _ = tokens.shape

        x = self.linear1(
            tokens
        )

        hidden_dim = x.shape[-1]

        x = (
            x.transpose(1, 2)
            .contiguous()
            .reshape(
                batch_size,
                hidden_dim,
                height,
                width,
            )
        )

        x = self.depthwise_convolution(
            x
        )

        x = (
            x.flatten(2)
            .transpose(1, 2)
            .contiguous()
        )

        x = self.activation(
            x
        )

        x = self.dropout1(
            x
        )

        x = self.linear2(
            x
        )

        return self.dropout2(
            x
        )


class TransformerBlock(nn.Module):
    """
    Transformer block containing efficient attention and Mix-FFN.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_ratio,
        sr_ratio,
        dropout=0.0,
    ):
        super().__init__()

        self.normalization1 = nn.LayerNorm(
            embed_dim
        )

        self.attention = EfficientSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            dropout=dropout,
        )

        self.normalization2 = nn.LayerNorm(
            embed_dim
        )

        self.feed_forward = MixFFN(
            embed_dim=embed_dim,
            hidden_dim=int(
                embed_dim * mlp_ratio
            ),
            dropout=dropout,
        )

    def forward(
        self,
        tokens,
        height,
        width,
    ):
        tokens = tokens + self.attention(
            self.normalization1(tokens),
            height,
            width,
        )

        tokens = tokens + self.feed_forward(
            self.normalization2(tokens),
            height,
            width,
        )

        return tokens


class MiTStage(nn.Module):
    """
    One stage of the Mix Transformer encoder.
    """

    def __init__(
        self,
        in_channels,
        embed_dim,
        patch_size,
        stride,
        depth,
        num_heads,
        mlp_ratio,
        sr_ratio,
        dropout=0.0,
    ):
        super().__init__()

        self.patch_embedding = OverlapPatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            stride=stride,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    sr_ratio=sr_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.normalization = nn.LayerNorm(
            embed_dim
        )

    def forward(self, x):
        tokens, height, width = self.patch_embedding(
            x
        )

        for block in self.blocks:
            tokens = block(
                tokens,
                height,
                width,
            )

        tokens = self.normalization(
            tokens
        )

        batch_size, _, channels = tokens.shape

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


class SegFormerEncoder(nn.Module):
    """
    Lightweight MiT-B0-style hierarchical transformer encoder.
    """

    def __init__(
        self,
        in_channels=3,
    ):
        super().__init__()

        embed_dimensions = (
            32,
            64,
            160,
            256,
        )

        self.stage1 = MiTStage(
            in_channels=in_channels,
            embed_dim=embed_dimensions[0],
            patch_size=7,
            stride=4,
            depth=2,
            num_heads=1,
            mlp_ratio=4,
            sr_ratio=8,
        )

        self.stage2 = MiTStage(
            in_channels=embed_dimensions[0],
            embed_dim=embed_dimensions[1],
            patch_size=3,
            stride=2,
            depth=2,
            num_heads=2,
            mlp_ratio=4,
            sr_ratio=4,
        )

        self.stage3 = MiTStage(
            in_channels=embed_dimensions[1],
            embed_dim=embed_dimensions[2],
            patch_size=3,
            stride=2,
            depth=2,
            num_heads=5,
            mlp_ratio=4,
            sr_ratio=2,
        )

        self.stage4 = MiTStage(
            in_channels=embed_dimensions[2],
            embed_dim=embed_dimensions[3],
            patch_size=3,
            stride=2,
            depth=2,
            num_heads=8,
            mlp_ratio=4,
            sr_ratio=1,
        )

    def forward(self, x):
        feature1 = self.stage1(
            x
        )

        feature2 = self.stage2(
            feature1
        )

        feature3 = self.stage3(
            feature2
        )

        feature4 = self.stage4(
            feature3
        )

        return (
            feature1,
            feature2,
            feature3,
            feature4,
        )


class SegFormerDecoder(nn.Module):
    """
    Lightweight multi-scale SegFormer decoder.
    """

    def __init__(
        self,
        encoder_channels=(32, 64, 160, 256),
        decoder_dim=256,
        out_channels=1,
    ):
        super().__init__()

        self.projections = nn.ModuleList(
            [
                nn.Conv2d(
                    input_channels,
                    decoder_dim,
                    kernel_size=1,
                    bias=False,
                )
                for input_channels in encoder_channels
            ]
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                decoder_dim * 4,
                decoder_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                decoder_dim
            ),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        self.classifier = nn.Conv2d(
            decoder_dim,
            out_channels,
            kernel_size=1,
        )

    def forward(self, features):
        target_size = features[0].shape[-2:]

        projected_features = []

        for feature, projection in zip(
            features,
            self.projections,
        ):
            projected = projection(
                feature
            )

            if projected.shape[-2:] != target_size:
                projected = F.interpolate(
                    projected,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )

            projected_features.append(
                projected
            )

        fused_features = torch.cat(
            projected_features,
            dim=1,
        ).contiguous()

        fused_features = self.fusion(
            fused_features
        )

        return self.classifier(
            fused_features
        )


class SegFormer(nn.Module):
    """
    Lightweight MiT-B0-style SegFormer for binary segmentation.

    Input
    -----
    Tensor with shape:
        [batch_size, 3, height, width]

    Output
    ------
    Raw segmentation logits with shape:
        [batch_size, 1, height, width]
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
    ):
        super().__init__()

        self.encoder = SegFormerEncoder(
            in_channels=in_channels
        )

        self.decoder = SegFormerDecoder(
            encoder_channels=(
                32,
                64,
                160,
                256,
            ),
            decoder_dim=256,
            out_channels=out_channels,
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        encoder_features = self.encoder(
            x
        )

        logits = self.decoder(
            encoder_features
        )

        logits = F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return logits