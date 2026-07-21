from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3


class EfficientNetB3Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = True,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b3(weights=weights)
        features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(features, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)


def build_efficientnet_b3(
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.30,
) -> EfficientNetB3Classifier:
    return EfficientNetB3Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )
