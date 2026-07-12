# Ablation Experiments

This directory contains the public script version of the MTA-Swin ablation experiments.

## Files

- `core.py`: shared defaults, experiment manifests, training loop, output writing, and aggregation helpers
- `run_component_ablation.py`: run the 11 component ablation experiments
- `run_hyperparameter_ablation.py`: run the 9 hyperparameter ablation experiments
- `aggregate_results.py`: collect the latest successful run for each experiment and write summary CSV files
- `requirements.txt`: Python dependencies for these scripts

## Data

The scripts expect an ImageNet-100 dataset root with this layout:

```text
<dataset_root>/
  train/
    n01440764/
    ...
  val/
    ILSVRC2012_val_00000001.JPEG
    ...
  dataset_info.json
  ILSVRC2012_devkit_t12/
    data/
      ILSVRC2012_validation_ground_truth.txt
```

By default the scripts look for the dataset at:

```text
datasets/imagenet-100
```

Override it with:

```bash
export MTA_ABLATION_IMAGENET100_PATH=/path/to/imagenet-100
```

## Outputs

By default runs are written to:

```text
experiments/ablation/runs
```

Override it with:

```bash
export MTA_ABLATION_OUTPUT_ROOT=/path/to/output-root
```

Each run is stored under:

```text
<output_root>/<group>/<experiment_id>/<timestamp>/
```

Each run directory contains:

- `resolved_config.yaml`
- `experiment_spec.json`
- `metrics.json`
- `training_history.json`
- `train.log`
- `tensorboard/`
- `best_model.pth`

## Setup

Install a suitable PyTorch build for your machine first, then install the remaining Python dependencies:

```bash
pip install -r experiments/ablation/requirements.txt
```

These ablation scripts were validated in an environment where `torch.__version__` was `2.9.0+cu128`, `torchvision.__version__` was `0.24.0+cu128`, and `timm.__version__` was `1.0.19`.
If you need CUDA-specific wheels, install `torch` and `torchvision` from the official PyTorch index first, then install the rest of the file.

The scripts assume a GPU-capable environment, but will fall back to CPU if CUDA is not available.

## Usage

List the component experiments:

```bash
python experiments/ablation/run_component_ablation.py --list-experiments
```

List the hyperparameter experiments:

```bash
python experiments/ablation/run_hyperparameter_ablation.py --list-experiments
```

Run the full component ablation group:

```bash
python experiments/ablation/run_component_ablation.py --gpu 0
```

Run the full hyperparameter ablation group:

```bash
python experiments/ablation/run_hyperparameter_ablation.py --gpu 0
```

Run a subset of experiments:

```bash
python experiments/ablation/run_component_ablation.py --gpu 0 --experiment-id component_02 component_09
python experiments/ablation/run_hyperparameter_ablation.py --gpu 0 --experiment-id hyperparam_05_cq5_ck11
```

Override the output root for a run:

```bash
python experiments/ablation/run_component_ablation.py --gpu 0 --output-root /tmp/mta-ablation-runs
```

## Aggregate Results

After runs finish, aggregate the latest successful run for each experiment:

```bash
python experiments/ablation/aggregate_results.py
```

This writes:

- `experiments/ablation/results/component_ablation_summary.csv`
- `experiments/ablation/results/hyperparameter_ablation_summary.csv`
- `experiments/ablation/results/all_ablation_summary.csv`
