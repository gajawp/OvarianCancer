# Pretrained segmentation baselines

This folder adds publication-oriented pretrained baselines without overwriting the existing scratch-trained models.

Implemented and runnable:
- DeepLabV3+ with an ImageNet-pretrained ResNet-50 encoder (`segmentation-models-pytorch`)
- U-Net++ with an ImageNet-pretrained ResNet-50 encoder (`segmentation-models-pytorch`)
- SegFormer with NVIDIA MiT-B0 pretrained weights (Hugging Face Transformers)

Expected project placement:

```text
Ovarian_Cancer/
├── datasets/OTU_2d/images/
├── datasets/OTU_2d/annotations/
└── pretrained_models/
```

Masks are expected as `<image_stem>_binary.PNG`, matching the existing repository convention.

## Install

```bash
cd Ovarian_Cancer
source venv/bin/activate
pip install -r pretrained_models/requirements.txt
```

The first run downloads pretrained weights and therefore requires internet access.

## Run

```bash
cd pretrained_models
python run.py deeplabv3plus all
python run.py unetplusplus all
python run.py segformer all
```

To evaluate an existing checkpoint without retraining:

```bash
python run.py deeplabv3plus evaluate
```

Results are written under `pretrained_models/results/<model>/metrics.csv` and best checkpoints under `<model>/weights/best_model.pth`.

## Fair-comparison notes

The loader uses the same deterministic 80/20 split with seed 42, 256×256 resolution, threshold 0.5, BCE+Dice loss, and the same evaluation metrics for all three models. For the final paper, create a separate patient-level test set if multiple images belong to the same patient, and report mean ± standard deviation over multiple seeds.

## Official integrations

TransUNet, DSNet, and DS2Net are not included as fabricated drop-in models. Their folders contain integration notes because the exact official repository/checkpoint and architecture variant must be selected before claiming an official baseline.
