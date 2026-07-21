from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = True,
        dropout: float = 0.30,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        weights = (
            ResNet50_Weights.IMAGENET1K_V2
            if pretrained
            else None
        )

        self.backbone = resnet50(weights=weights)
        input_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_features, num_classes),
        )

        if freeze_backbone:
            self.freeze_feature_extractor()

    def freeze_feature_extractor(self) -> None:
        for name, parameter in self.backbone.named_parameters():
            if not name.startswith("fc."):
                parameter.requires_grad = False

    def unfreeze_feature_extractor(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)


def build_resnet50(
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.30,
    freeze_backbone: bool = False,
) -> ResNet50Classifier:
    return ResNet50Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )
