# Comparison Experiments

This directory contains the script version of the brain-tumor comparison notebooks.

## Files

- `config.py`: shared experiment configuration
- `run_seed_comparison.py`: run one seed and save seed-level artifacts
- `aggregate_comparison.py`: aggregate multiple seed runs into the final summary table
- `requirements.txt`: Python dependencies for these scripts

## Data And Weights

Large assets are intentionally not tracked in Git. Instead of hardcoding personal machine paths, the scripts read these two environment variables:

- `MTA_COMPARISON_IMAGE_PATH`: dataset root
- `MTA_COMPARISON_PRETRAINED_WEIGHTS`: MTA-Swin pretrained checkpoint

If the variables are not set, `config.py` falls back to these placeholder paths:

- `experiments/comparison/external/cleaned-brain-tumour-dataset`
- `experiments/comparison/external/best_model.pth`

This keeps the repository portable. When you later provide a download URL, users can place the files anywhere and point the scripts at the real locations with environment variables, without editing the repo.

Expected dataset layout:

```text
<dataset_root>/
  Training/
    glioma/
    meningioma/
    notumor/
    pituitary/
  Testing/
    glioma/
    meningioma/
    notumor/
    pituitary/
```

## Setup

Install a suitable PyTorch build for your machine, then install the remaining packages:

```bash
pip install -r experiments/comparison/requirements.txt
```

These comparison scripts were validated in an environment where `torch.__version__` was `2.9.0+cu128` and `torchvision.__version__` was `0.24.0+cu128`.
If you need CUDA-specific wheels, install `torch` and `torchvision` from the official PyTorch index first, then install the rest of the file.

## Usage

Set paths for the current shell:

```bash
export MTA_COMPARISON_IMAGE_PATH=/path/to/cleaned-brain-tumour-dataset
export MTA_COMPARISON_PRETRAINED_WEIGHTS=/path/to/best_model.pth
```

Run one seed at a time:

```bash
python experiments/comparison/run_seed_comparison.py --seed 0
python experiments/comparison/run_seed_comparison.py --seed 1
python experiments/comparison/run_seed_comparison.py --seed 2
```

Aggregate the saved seed outputs:

```bash
python experiments/comparison/aggregate_comparison.py
```

## Outputs

- `mta_seed_runs/`: seed-level metric CSVs and prediction PKLs
- `checkpoints/`: per-seed model checkpoints
- `plots/`: optional saved plots from `run_seed_comparison.py --save-plots`
- `final_tables/`: aggregated final summary tables
