from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import Swin_T_Weights, swin_t


class SwinTClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = True,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = swin_t(weights=weights)
        features = self.backbone.head.in_features
        self.backbone.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(features, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)


def build_swin_t(
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.30,
) -> SwinTClassifier:
    return SwinTClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )
