import torch
import torch.nn as nn
from torchvision.models import Swin_T_Weights, swin_t


class TokenAttentionPooling(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feature_map):
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
        pooled = torch.sum(tokens * weights, dim=1)
        return pooled, weights


class MaskEncoder(nn.Module):
    def __init__(self, output_dim=128):
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

    def forward(self, mask):
        return self.encoder(mask).flatten(1)


class ROIMaskAwareMTASwin(nn.Module):
    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        dropout=0.30,
        attention_hidden_dim=256,
        mask_feature_dim=128,
    ):
        super().__init__()

        swin_weights = (
            Swin_T_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )
        roi_backbone = swin_t(weights=swin_weights)

        self.roi_features = roi_backbone.features
        self.roi_norm = roi_backbone.norm
        roi_feature_dim = roi_backbone.head.in_features

        self.roi_attention = TokenAttentionPooling(
            roi_feature_dim,
            attention_hidden_dim,
        )

        self.mask_encoder = MaskEncoder(mask_feature_dim)

        fused_dim = roi_feature_dim + mask_feature_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, num_classes),
        )

    def forward(
        self,
        roi_images,
        masks,
        return_features=False,
    ):
        roi_feature_map = self.roi_features(roi_images)
        roi_feature_map = self.roi_norm(roi_feature_map)
        roi_features, attention_weights = self.roi_attention(
            roi_feature_map
        )

        mask_features = self.mask_encoder(masks)

        fused_features = torch.cat(
            [
                roi_features,
                mask_features,
            ],
            dim=1,
        )

        logits = self.classifier(fused_features)

        if return_features:
            return {
                "logits": logits,
                "attention_weights": attention_weights,
                "roi_features": roi_features,
                "mask_features": mask_features,
                "fused_features": fused_features,
            }

        return logits

    def freeze_backbones(self):
        for parameter in self.roi_features.parameters():
            parameter.requires_grad = False
        for parameter in self.roi_norm.parameters():
            parameter.requires_grad = False

    def unfreeze_backbones(self):
        for parameter in self.roi_features.parameters():
            parameter.requires_grad = True
        for parameter in self.roi_norm.parameters():
            parameter.requires_grad = True


def create_roi_mask_model(
    num_classes,
    pretrained,
    dropout,
    attention_hidden_dim,
    mask_feature_dim,
):
    return ROIMaskAwareMTASwin(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        attention_hidden_dim=attention_hidden_dim,
        mask_feature_dim=mask_feature_dim,
    )
