import torch
import torch.nn as nn

from torchvision.models import Swin_T_Weights, swin_t


class TokenAttentionPooling(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        feature_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, height, width, channels = feature_map.shape

        tokens = feature_map.reshape(
            batch_size,
            height * width,
            channels,
        )

        weights = torch.softmax(
            self.attention(tokens),
            dim=1,
        )

        pooled = torch.sum(
            tokens * weights,
            dim=1,
        )

        return pooled, weights


class MaskEncoder(nn.Module):
    def __init__(self, output_dim: int = 128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                64,
                3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                output_dim,
                3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(mask).flatten(1)


class MaskAwareMTASwinClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = True,
        dropout: float = 0.30,
        attention_hidden_dim: int = 256,
        mask_feature_dim: int = 128,
    ):
        super().__init__()

        weights = (
            Swin_T_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )

        backbone = swin_t(weights=weights)

        self.rgb_features = backbone.features
        self.rgb_norm = backbone.norm

        rgb_feature_dim = backbone.head.in_features

        self.token_attention = TokenAttentionPooling(
            feature_dim=rgb_feature_dim,
            hidden_dim=attention_hidden_dim,
        )

        self.mask_encoder = MaskEncoder(
            output_dim=mask_feature_dim,
        )

        fused_dim = rgb_feature_dim + mask_feature_dim

        self.fusion_classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, num_classes),
        )

    def forward(
        self,
        rgb_images: torch.Tensor,
        masks: torch.Tensor,
        return_attention: bool = False,
    ):
        rgb_feature_map = self.rgb_features(rgb_images)
        rgb_feature_map = self.rgb_norm(rgb_feature_map)

        rgb_features, attention_weights = (
            self.token_attention(rgb_feature_map)
        )

        mask_features = self.mask_encoder(masks)

        fused_features = torch.cat(
            [rgb_features, mask_features],
            dim=1,
        )

        logits = self.fusion_classifier(fused_features)

        if return_attention:
            return {
                "logits": logits,
                "attention_weights": attention_weights,
                "rgb_features": rgb_features,
                "mask_features": mask_features,
                "fused_features": fused_features,
            }

        return logits

    def freeze_rgb_backbone(self) -> None:
        for parameter in self.rgb_features.parameters():
            parameter.requires_grad = False
        for parameter in self.rgb_norm.parameters():
            parameter.requires_grad = False

    def unfreeze_rgb_backbone(self) -> None:
        for parameter in self.rgb_features.parameters():
            parameter.requires_grad = True
        for parameter in self.rgb_norm.parameters():
            parameter.requires_grad = True


def create_maskaware_mta_swin(
    num_classes: int,
    pretrained: bool,
    dropout: float,
    attention_hidden_dim: int,
    mask_feature_dim: int,
) -> MaskAwareMTASwinClassifier:
    return MaskAwareMTASwinClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        attention_hidden_dim=attention_hidden_dim,
        mask_feature_dim=mask_feature_dim,
    )
