import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


class PretrainedSegFormer(nn.Module):
    def __init__(self, checkpoint: str = "nvidia/mit-b0"):
        super().__init__()
        self.network = SegformerForSemanticSegmentation.from_pretrained(
            checkpoint, num_labels=1, ignore_mismatched_sizes=True
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.network(pixel_values=images).logits
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)


def build_model(pretrained: bool = True):
    if not pretrained:
        raise ValueError("This folder is specifically for the pretrained NVIDIA MiT-B0 checkpoint.")
    return PretrainedSegFormer()


def build_optimizer(model, encoder_lr: float, decoder_lr: float, weight_decay: float):
    return torch.optim.AdamW([
        {"params": model.network.segformer.parameters(), "lr": encoder_lr},
        {"params": model.network.decode_head.parameters(), "lr": decoder_lr},
    ], weight_decay=weight_decay)
