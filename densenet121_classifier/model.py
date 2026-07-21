from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121


class DenseNet121Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = True,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = densenet121(weights=weights)
        features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(features, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)


def build_densenet121(
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.30,
) -> DenseNet121Classifier:
    return DenseNet121Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )
