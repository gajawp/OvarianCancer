import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.dataset import (
    OvarianDataset,
    get_train_transform,
)


# ==========================================================
# Random Seed
# ==========================================================

def set_random_seed(seed):
    """
    Set random seeds for reproducible experiments.

    This affects:
    - Python random operations
    - NumPy operations
    - PyTorch model initialization
    - CUDA operations
    - DataLoader shuffling

    Parameters
    ----------
    seed : int
        Random seed value.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==========================================================
# Official Split Verification
# ==========================================================

def verify_no_split_overlap(
    train_dataset,
    test_dataset,
):
    """
    Verify that no image filename appears in both the
    official training and testing datasets.

    Parameters
    ----------
    train_dataset : OvarianDataset
        Dataset created from train_cls.txt.

    test_dataset : OvarianDataset
        Dataset created from val_cls.txt.

    Raises
    ------
    RuntimeError
        If one or more filenames occur in both partitions.
    """

    train_images = set(
        train_dataset.images
    )

    test_images = set(
        test_dataset.images
    )

    duplicate_train_entries = (
        len(train_dataset.images)
        - len(train_images)
    )

    duplicate_test_entries = (
        len(test_dataset.images)
        - len(test_images)
    )

    if duplicate_train_entries > 0:
        raise RuntimeError(
            "Duplicate filenames were found inside "
            f"the training split: "
            f"{duplicate_train_entries} duplicate entries."
        )

    if duplicate_test_entries > 0:
        raise RuntimeError(
            "Duplicate filenames were found inside "
            f"the testing split: "
            f"{duplicate_test_entries} duplicate entries."
        )

    overlap = train_images.intersection(
        test_images
    )

    if overlap:
        overlap_preview = sorted(
            overlap
        )[:10]

        raise RuntimeError(
            "Data leakage detected. "
            f"{len(overlap)} image filenames appear in both "
            "the training and testing partitions. "
            f"Examples: {overlap_preview}"
        )

    print(
        "Official split verification passed."
    )

    print(
        "Training images:",
        len(train_images),
    )

    print(
        "Testing images:",
        len(test_images),
    )

    print(
        "Overlapping filenames: 0"
    )


# ==========================================================
# Dataset File Verification
# ==========================================================

def verify_dataset_files(dataset):
    """
    Verify that every image and corresponding mask listed
    in a dataset exists.

    The expected mask naming format is:

        image:  658.JPG
        mask:   658_binary.PNG

    Parameters
    ----------
    dataset : OvarianDataset
        Segmentation dataset to verify.

    Raises
    ------
    FileNotFoundError
        If an image or mask does not exist.
    """

    missing_images = []
    missing_masks = []

    for image_name in dataset.images:
        image_path = (
            Path(dataset.image_dir)
            / image_name
        )

        base_name = Path(
            image_name
        ).stem

        mask_name = (
            base_name
            + "_binary.PNG"
        )

        mask_path = (
            Path(dataset.mask_dir)
            / mask_name
        )

        if not image_path.exists():
            missing_images.append(
                str(image_path)
            )

        if not mask_path.exists():
            missing_masks.append(
                str(mask_path)
            )

    if missing_images or missing_masks:
        error_messages = []

        if missing_images:
            error_messages.append(
                f"{len(missing_images)} images are missing. "
                f"Examples: {missing_images[:5]}"
            )

        if missing_masks:
            error_messages.append(
                f"{len(missing_masks)} masks are missing. "
                f"Examples: {missing_masks[:5]}"
            )

        raise FileNotFoundError(
            "\n".join(error_messages)
        )


# ==========================================================
# Train and Test DataLoaders
# ==========================================================

def create_data_loaders(config):
    """
    Create segmentation training and testing DataLoaders
    from the official dataset split.

    train_cls.txt is used for training.
    val_cls.txt is used as the official held-out test set.

    Training images use data augmentation.
    Test images do not use augmentation.

    Required configuration values
    -----------------------------
    config.IMAGE_DIR
    config.MASK_DIR
    config.TRAIN_LIST_PATH
    config.TEST_LIST_PATH
    config.IMAGE_SIZE
    config.BATCH_SIZE
    config.NUM_WORKERS
    config.DEVICE

    Returns
    -------
    tuple
        train_loader, test_loader
    """

    train_dataset = OvarianDataset(
        image_dir=config.IMAGE_DIR,
        mask_dir=config.MASK_DIR,
        split_file=config.TRAIN_LIST_PATH,
        image_size=config.IMAGE_SIZE,
        transform=get_train_transform(),
    )

    test_dataset = OvarianDataset(
        image_dir=config.IMAGE_DIR,
        mask_dir=config.MASK_DIR,
        split_file=config.TEST_LIST_PATH,
        image_size=config.IMAGE_SIZE,
        transform=None,
    )

    verify_dataset_files(
        train_dataset
    )

    verify_dataset_files(
        test_dataset
    )

    verify_no_split_overlap(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )

    pin_memory = (
        str(config.DEVICE) == "cuda"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=(
            config.NUM_WORKERS > 0
        ),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=(
            config.NUM_WORKERS > 0
        ),
    )

    return (
        train_loader,
        test_loader,
    )


# ==========================================================
# Official Test DataLoader
# ==========================================================

def create_test_loader(
    config,
    batch_size=1,
):
    """
    Create a DataLoader for the official test partition.

    This function is intended for evaluate.py.

    The test partition is read from val_cls.txt and does
    not use random augmentation.

    Parameters
    ----------
    config : module
        The common.config module.

    batch_size : int
        Number of test images per batch.

    Returns
    -------
    DataLoader
        Official test DataLoader.
    """

    test_dataset = OvarianDataset(
        image_dir=config.IMAGE_DIR,
        mask_dir=config.MASK_DIR,
        split_file=config.TEST_LIST_PATH,
        image_size=config.IMAGE_SIZE,
        transform=None,
    )

    verify_dataset_files(
        test_dataset
    )

    pin_memory = (
        str(config.DEVICE) == "cuda"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=(
            config.NUM_WORKERS > 0
        ),
    )

    return test_loader


# ==========================================================
# Backward-Compatible Evaluation Loader
# ==========================================================

def create_validation_loader(
    config,
    batch_size=1,
):
    """
    Backward-compatible wrapper for older evaluation files.

    The returned data now comes from the official test
    partition specified by config.TEST_LIST_PATH.

    New evaluation scripts should use create_test_loader().
    """

    return create_test_loader(
        config=config,
        batch_size=batch_size,
    )


# ==========================================================
# Model Comparison CSV
# ==========================================================

def append_result(
    csv_path,
    model_name,
    metrics,
):
    """
    Add or update one model's segmentation results.

    If the model already exists in the CSV, its previous
    result is replaced.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Model-comparison CSV path.

    model_name : str
        Name of the segmentation model.

    metrics : dict
        Expected keys:
        - dice
        - iou
        - precision
        - recall
        - specificity
        - hausdorff_distance
    """

    csv_path = Path(
        csv_path
    )

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_metrics = [
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
        "hausdorff_distance",
    ]

    missing_metrics = [
        metric_name
        for metric_name in required_metrics
        if metric_name not in metrics
    ]

    if missing_metrics:
        raise ValueError(
            "Missing metrics: "
            + ", ".join(
                missing_metrics
            )
        )

    field_names = [
        "model",
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
        "hausdorff_distance",
    ]

    existing_rows = []

    if csv_path.exists():
        with csv_path.open(
            mode="r",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            reader = csv.DictReader(
                csv_file
            )

            existing_rows = list(
                reader
            )

    existing_rows = [
        row
        for row in existing_rows
        if row.get("model") != model_name
    ]

    new_row = {
        "model": model_name,
        "dice": (
            f"{metrics['dice']:.6f}"
        ),
        "iou": (
            f"{metrics['iou']:.6f}"
        ),
        "precision": (
            f"{metrics['precision']:.6f}"
        ),
        "recall": (
            f"{metrics['recall']:.6f}"
        ),
        "specificity": (
            f"{metrics['specificity']:.6f}"
        ),
        "hausdorff_distance": (
            f"{metrics['hausdorff_distance']:.6f}"
        ),
    }

    existing_rows.append(
        new_row
    )

    with csv_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=field_names,
        )

        writer.writeheader()

        writer.writerows(
            existing_rows
        )


# ==========================================================
# Checkpoint Loading
# ==========================================================

def load_model_checkpoint(
    model,
    checkpoint_path,
    device,
):
    """
    Load a saved model checkpoint.

    Supports:
    1. A plain model state dictionary.
    2. A checkpoint dictionary containing
       'model_state_dict'.
    3. A checkpoint dictionary containing 'state_dict'.

    Parameters
    ----------
    model : torch.nn.Module
        Initialized segmentation model.

    checkpoint_path : str or pathlib.Path
        Saved checkpoint path.

    device : torch.device
        Device used to load the checkpoint.

    Returns
    -------
    torch.nn.Module
        Model containing the loaded parameters.
    """

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Model checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "state_dict"
        ]

    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    return model

# ==========================================================
# Evaluation Metric Utilities
# ==========================================================

def initialize_metric_storage():
    """
    Create storage for per-image segmentation metrics.
    """

    return {
        "dice": [],
        "iou": [],
        "precision": [],
        "recall": [],
        "specificity": [],
    }


def evaluate_batch_per_image(
    outputs,
    masks,
    metric_storage,
    result_rows,
    threshold=0.5,
):
    """
    Calculate segmentation metrics independently for every
    image in a batch.

    Parameters
    ----------
    outputs : torch.Tensor
        Raw model logits with shape [B, 1, H, W].

    masks : torch.Tensor
        Ground-truth masks with shape [B, 1, H, W].

    metric_storage : dict
        Dictionary returned by initialize_metric_storage().

    result_rows : list
        List where one dictionary is added for each image.

    threshold : float
        Probability threshold used to create binary masks.
    """

    from common.metrics import calculate_numpy_metrics

    probabilities = torch.sigmoid(
        outputs
    )

    predictions = (
        probabilities >= threshold
    ).float()

    batch_size = outputs.shape[0]

    for sample_index in range(batch_size):
        prediction_numpy = (
            predictions[sample_index]
            .detach()
            .cpu()
            .squeeze()
            .numpy()
        )

        mask_numpy = (
            masks[sample_index]
            .detach()
            .cpu()
            .squeeze()
            .numpy()
        )

        metrics = calculate_numpy_metrics(
            prediction=prediction_numpy,
            target=mask_numpy,
        )

        result_row = {
            "sample_index": len(result_rows),
        }

        for metric_name, metric_value in metrics.items():
            metric_storage[
                metric_name
            ].append(
                float(metric_value)
            )

            result_row[
                metric_name
            ] = float(
                metric_value
            )

        result_rows.append(
            result_row
        )


def summarize_metrics(
    metric_storage,
    sample_standard_deviation=False,
):
    """
    Calculate summary statistics for each segmentation metric.

    Returns mean, standard deviation, minimum, maximum,
    and median.
    """

    summary = {}

    degrees_of_freedom = (
        1
        if sample_standard_deviation
        else 0
    )

    for metric_name, values in metric_storage.items():
        values_array = np.asarray(
            values,
            dtype=np.float64,
        )

        valid_values = values_array[
            np.isfinite(
                values_array
            )
        ]

        if len(valid_values) == 0:
            summary[metric_name] = {
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "median": np.nan,
                "count": 0,
            }

            continue

        if (
            sample_standard_deviation
            and len(valid_values) < 2
        ):
            standard_deviation = np.nan
        else:
            standard_deviation = float(
                np.std(
                    valid_values,
                    ddof=degrees_of_freedom,
                )
            )

        summary[metric_name] = {
            "mean": float(
                np.mean(
                    valid_values
                )
            ),
            "std": standard_deviation,
            "min": float(
                np.min(
                    valid_values
                )
            ),
            "max": float(
                np.max(
                    valid_values
                )
            ),
            "median": float(
                np.median(
                    valid_values
                )
            ),
            "count": int(
                len(valid_values)
            ),
        }

    return summary


def print_metric_summary(summary):
    """
    Print metric summary statistics.
    """

    print("\nEvaluation Results")
    print("-" * 72)

    for metric_name, statistics in summary.items():
        mean_value = statistics["mean"]
        std_value = statistics["std"]
        minimum_value = statistics["min"]
        maximum_value = statistics["max"]
        median_value = statistics["median"]

        print(
            f"{metric_name:<18}: "
            f"{mean_value:.4f} ± {std_value:.4f} "
            f"| min={minimum_value:.4f} "
            f"| max={maximum_value:.4f} "
            f"| median={median_value:.4f}"
        )


def get_mean_metrics(summary):
    """
    Extract only the mean value for each metric.
    """

    return {
        metric_name: statistics["mean"]
        for metric_name, statistics in summary.items()
    }


def save_evaluation_results(
    model_name,
    result_rows,
    summary,
    results_root,
):
    """
    Save per-image metrics and summary statistics as CSV files.

    Output structure:

        results/
        └── model_name/
            ├── per_image_metrics.csv
            └── summary_metrics.csv
    """

    results_directory = (
        Path(results_root)
        / model_name
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # Per-image metrics
    # ------------------------------------------------------

    per_image_path = (
        results_directory
        / "per_image_metrics.csv"
    )

    if result_rows:
        field_names = []

        for row in result_rows:
            for key in row.keys():
                if key not in field_names:
                    field_names.append(
                        key
                    )

        with per_image_path.open(
            mode="w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=field_names,
            )

            writer.writeheader()

            writer.writerows(
                result_rows
            )

    # ------------------------------------------------------
    # Summary metrics
    # ------------------------------------------------------

    summary_path = (
        results_directory
        / "summary_metrics.csv"
    )

    summary_field_names = [
        "metric",
        "mean",
        "std",
        "min",
        "max",
        "median",
        "count",
    ]

    summary_rows = []

    for metric_name, statistics in summary.items():
        summary_rows.append({
            "metric": metric_name,
            "mean": statistics["mean"],
            "std": statistics["std"],
            "min": statistics["min"],
            "max": statistics["max"],
            "median": statistics["median"],
            "count": statistics["count"],
        })

    with summary_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=summary_field_names,
        )

        writer.writeheader()

        writer.writerows(
            summary_rows
        )

    print(
        "Per-image metrics saved to:",
        per_image_path,
    )

    print(
        "Summary metrics saved to:",
        summary_path,
    )