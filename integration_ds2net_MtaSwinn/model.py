# integration_ds2net_MtaSwinn/model.py

import torch
import torch.nn as nn

from torchvision.models import (
    Swin_T_Weights,
    swin_t,
)


class TokenAttentionPooling(nn.Module):
    """
    Learn attention weights over the final Swin feature tokens.

    Input:
        [batch_size, height, width, channels]

    Output:
        [batch_size, channels]
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(
                feature_dim,
                hidden_dim,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(
        self,
        feature_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if feature_map.ndim != 4:
            raise ValueError(
                "Expected Swin feature map with shape "
                "[B, H, W, C], but received "
                f"{tuple(feature_map.shape)}"
            )

        batch_size, height, width, channels = (
            feature_map.shape
        )

        tokens = feature_map.reshape(
            batch_size,
            height * width,
            channels,
        )

        attention_logits = self.attention(
            tokens
        )

        attention_weights = torch.softmax(
            attention_logits,
            dim=1,
        )

        pooled_features = torch.sum(
            tokens * attention_weights,
            dim=1,
        )

        return (
            pooled_features,
            attention_weights,
        )


class MTASwinClassifier(nn.Module):
    """
    Segmentation-guided MTA-Swin classifier.

    MTA in this implementation refers to learnable
    multi-token attention over the final Swin tokens.

    The DS2Net ROI is supplied as the image input.
    """

    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = True,
        dropout: float = 0.30,
        attention_hidden_dim: int = 256,
    ):
        super().__init__()

        weights = (
            Swin_T_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )

        backbone = swin_t(
            weights=weights
        )

        self.features = backbone.features
        self.norm = backbone.norm

        feature_dim = (
            backbone.head.in_features
        )

        self.feature_dim = feature_dim

        self.token_attention = (
            TokenAttentionPooling(
                feature_dim=feature_dim,
                hidden_dim=attention_hidden_dim,
            )
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(
                feature_dim,
                feature_dim // 2,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                feature_dim // 2,
                num_classes,
            ),
        )

    def forward(
        self,
        images: torch.Tensor,
        return_attention: bool = False,
    ):
        feature_map = self.features(
            images
        )

        feature_map = self.norm(
            feature_map
        )

        (
            pooled_features,
            attention_weights,
        ) = self.token_attention(
            feature_map
        )

        logits = self.classifier(
            pooled_features
        )

        if return_attention:
            return {
                "logits": logits,
                "attention_weights": (
                    attention_weights
                ),
                "features": pooled_features,
            }

        return logits

    def freeze_backbone(
        self,
    ) -> None:
        for parameter in (
            self.features.parameters()
        ):
            parameter.requires_grad = False

        for parameter in (
            self.norm.parameters()
        ):
            parameter.requires_grad = False

    def unfreeze_backbone(
        self,
    ) -> None:
        for parameter in (
            self.features.parameters()
        ):
            parameter.requires_grad = True

        for parameter in (
            self.norm.parameters()
        ):
            parameter.requires_grad = True


def create_mta_swin_model(
    num_classes: int,
    pretrained: bool,
    dropout: float,
    attention_hidden_dim: int,
) -> MTASwinClassifier:
    return MTASwinClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        attention_hidden_dim=(
            attention_hidden_dim
        ),
    )