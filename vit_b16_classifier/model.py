from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


class ViTB16Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = True,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = vit_b_16(weights=weights)
        features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(features, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)


def build_vit_b16(
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.30,
) -> ViTB16Classifier:
    return ViTB16Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )
