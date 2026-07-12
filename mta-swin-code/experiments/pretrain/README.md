# Pretraining (MTA-Swin)

This directory contains the multi-GPU pretraining pipeline for MTA-Swin using Distributed Data Parallel (DDP).

## Overview

- All model and training configurations are defined in `configs/`
- The main training logic is implemented in `pretrain_mta_ddp.py`
- Training is launched via Slurm using `pretrain_mta_ddp.sh`
- Designed for multi-GPU training on HPC clusters (e.g., H100 / H200)

## Environment Setup

We use a multi-GPU setup with **4× NVIDIA H200 GPUs** for pretraining.

Create the environment:

```bash
conda env create -f experiments/pretrain/environment.yml
conda activate <env_name>
```

Make sure PyTorch is installed with the correct CUDA version for your cluster.

## Quick Start

```bash
# modify config if needed
vim experiments/pretrain/configs/mta_swin_ddp_v2.yaml

# launch training
sbatch experiments/pretrain/pretrain_mta_ddp.sh
```