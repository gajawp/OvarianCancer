# MTA-Swin: A Multi-Token Attention Swin Transformer for Brain Tumor Classification with Leakage-Free MRI Benchmarking

This repository focuses on the **training and experiment code** used to reproduce selected parts of the MTA-Swin paper workflow. In particular, it currently covers **ImageNet-1K pretraining**, **brain tumor benchmark comparison**, and **ImageNet-100 ablation studies**, together with the core MTA-Swin model implementation. The manuscript is still under review, so the repository is intended primarily as a practical codebase for running the experiments rather than as a full archival release of every research asset. The cleaned brain tumor benchmark dataset used in the comparison experiments is already public on Kaggle.

## What This Repository Provides

- `experiments/pretrain`: ImageNet-1K pretraining for MTA-Swin
- `experiments/comparison`: brain tumor benchmark comparison scripts
- `experiments/ablation`: component and hyperparameter ablation scripts
- `src/models`: MTA-Swin implementation
- `src/data`: dataloaders used by the experiments

This repository focuses on the training and evaluation code used in the paper. Large datasets and pretrained checkpoints are intentionally **not tracked in Git**.

## Quick Start

Install a PyTorch build that matches your machine first, then install the remaining experiment dependencies.

For the comparison experiments:

```bash
pip install -r experiments/comparison/requirements.txt
```

For the ablation experiments:

```bash
pip install -r experiments/ablation/requirements.txt
```

Set the required environment variables as needed:

```bash
export MTA_COMPARISON_IMAGE_PATH=/path/to/cleaned-brain-tumour-dataset
export MTA_COMPARISON_PRETRAINED_WEIGHTS=/path/to/best_model.pth
export MTA_ABLATION_IMAGENET100_PATH=/path/to/imagenet-100
```

For detailed setup notes, environment expectations, and output directories, see:

- [`experiments/comparison/README.md`](./experiments/comparison/README.md)
- [`experiments/ablation/README.md`](./experiments/ablation/README.md)

## Experiment Entry Points

### Brain tumor comparison

Run one seed:

```bash
python experiments/comparison/run_seed_comparison.py --seed 0
```

Aggregate multiple seeds:

```bash
python experiments/comparison/aggregate_comparison.py
```

### Ablation studies

Run the component ablation group:

```bash
python experiments/ablation/run_component_ablation.py --gpu 0
```

Run the hyperparameter ablation group:

```bash
python experiments/ablation/run_hyperparameter_ablation.py --gpu 0
```

Aggregate ablation results:

```bash
python experiments/ablation/aggregate_results.py
```

### Pretraining

The ImageNet-1K pretraining entry point is:

```bash
python experiments/pretrain/pretrain_mta_ddp.py --config configs/pretrain/mta_swin_ddp_v2.yaml
```

The pretraining configs are stored under `configs/pretrain/`. The provided shell script in `experiments/pretrain/` is an example launcher for distributed training environments.

## Data and External Assets

### Cleaned brain tumor benchmark

The cleaned benchmark dataset used for the brain tumor comparison experiments is publicly available on Kaggle:

- https://www.kaggle.com/datasets/joeyz66/cleaned-brain-tumour-dataset

### What each experiment requires

- **Comparison experiments** require:
  - the cleaned brain tumor dataset
  - an MTA-Swin pretrained checkpoint
- **Ablation experiments** require:
  - ImageNet-100
- **Pretraining** requires:
  - ImageNet-1K

### Deduplication pipeline

The duplicate-detection / deduplication pipeline will be released separately. Its public link will be added to this README later.

## Paper Context

This repository accompanies the **MTA-Swin** paper on brain tumor MRI classification with leakage-free benchmarking. The paper addresses two connected problems: duplicate-induced data leakage in widely used public brain tumor MRI datasets, and the design of a stronger Swin-based classifier for this task. Our proposed model, **MTA-Swin**, augments Swin Transformer with a stage-aware multi-token attention design that improves local token interactions in early stages and cross-head communication in deeper stages.

## Highlights

- `3,522` unique MRI scans in the cleaned benchmark dataset
- `4` diagnostic classes: glioma, meningioma, pituitary tumor, and no tumor
- `98.57%` average accuracy over three random seeds
- Outperforms `13` representative baselines on the leakage-free benchmark
- Covers both **ImageNet-1K pretraining** and **ImageNet-100 ablation** in addition to the brain tumor comparison experiments

## Paper Status

- The manuscript is currently **under review**.
- Citation information will be added after publication.

## Notes on Reproducibility

- The main comparison results are reported over three random seeds: `0`, `1`, and `2`.
- MTA-Swin is pre-trained on ImageNet-1K before fine-tuning on the cleaned brain tumor benchmark.
- The ablation studies are run on ImageNet-100.
- Exact environment details for the comparison and ablation scripts are documented in their subdirectory READMEs.

## Citation

The manuscript is currently under review, so citation information is not yet available. A citation block will be added after acceptance or publication.

For now, please cite the repository URL or contact the authors directly if you need a temporary reference.

## Acknowledgement

We thank the contributors of the public datasets used in this study and the open-source software community for making this work possible.
