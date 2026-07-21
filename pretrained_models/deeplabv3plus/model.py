import segmentation_models_pytorch as smp


def build_model(pretrained: bool = True):
    return smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=1,
        activation=None,
        encoder_output_stride=16,
    )


def build_optimizer(model, encoder_lr: float, decoder_lr: float, weight_decay: float):
    return __import__("torch").optim.AdamW([
        {"params": model.encoder.parameters(), "lr": encoder_lr},
        {"params": model.decoder.parameters(), "lr": decoder_lr},
        {"params": model.segmentation_head.parameters(), "lr": decoder_lr},
    ], weight_decay=weight_decay)
