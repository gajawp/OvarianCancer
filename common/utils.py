
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from common.dataset import (
    OvarianDataset,
    get_train_transform
)


# ==========================================================
# Random Seed
# ==========================================================

def set_random_seed(seed):
    """
    Sets random seeds for reproducible experiments.

    This affects:

    - Python random operations
    - NumPy operations
    - PyTorch model initialization
    - CUDA operations
    - Dataset train-validation splitting

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

    # These settings improve reproducibility on CUDA.
    # They may slightly reduce training speed.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==========================================================
# Dataset Split
# ==========================================================

def create_split_indices(
    dataset_size,
    train_split,
    random_seed
):
    """
    Creates reproducible training and validation indices.

    The same split must be used for all models so that model
    comparisons remain fair.

    Parameters
    ----------
    dataset_size : int
        Total number of samples.

    train_split : float
        Fraction of data used for training.

        Example:
        0.8 means 80% training and 20% validation.

    random_seed : int
        Seed used for shuffling dataset indices.

    Returns
    -------
    tuple
        train_indices, validation_indices
    """

    if dataset_size <= 1:
        raise ValueError(
            "The dataset must contain at least two samples."
        )

    if not 0.0 < train_split < 1.0:
        raise ValueError(
            "TRAIN_SPLIT must be between 0 and 1."
        )

    generator = torch.Generator().manual_seed(
        random_seed
    )

    indices = torch.randperm(
        dataset_size,
        generator=generator
    ).tolist()

    train_size = int(
        train_split * dataset_size
    )

    # Ensure both splits contain at least one image.
    train_size = max(
        1,
        min(
            train_size,
            dataset_size - 1
        )
    )

    train_indices = indices[:train_size]

    validation_indices = indices[train_size:]

    return (
        train_indices,
        validation_indices
    )


# ==========================================================
# Train and Validation DataLoaders
# ==========================================================

def create_data_loaders(config):
    """
    Creates training and validation DataLoaders.

    Training data uses augmentation.

    Validation data does not use augmentation.

    Both datasets use the same reproducible split.

    Parameters
    ----------
    config : module
        The common.config module.

    Returns
    -------
    tuple
        train_loader, validation_loader
    """

    train_full_dataset = OvarianDataset(
        image_dir=config.IMAGE_DIR,
        mask_dir=config.MASK_DIR,
        image_size=config.IMAGE_SIZE,
        transform=get_train_transform()
    )

    validation_full_dataset = OvarianDataset(
        image_dir=config.IMAGE_DIR,
        mask_dir=config.MASK_DIR,
        image_size=config.IMAGE_SIZE,
        transform=None
    )

    if (
        len(train_full_dataset)
        != len(validation_full_dataset)
    ):
        raise RuntimeError(
            "Training and validation dataset sizes do not match."
        )

    (
        train_indices,
        validation_indices
    ) = create_split_indices(
        dataset_size=len(train_full_dataset),
        train_split=config.TRAIN_SPLIT,
        random_seed=config.RANDOM_SEED
    )

    train_dataset = Subset(
        train_full_dataset,
        train_indices
    )

    validation_dataset = Subset(
        validation_full_dataset,
        validation_indices
    )

    pin_memory = (
        config.DEVICE == "cuda"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory
    )

    return (
        train_loader,
        validation_loader
    )


# ==========================================================
# Evaluation DataLoader
# ==========================================================

def create_validation_loader(
    config,
    batch_size=1
):
    """
    Creates only the validation DataLoader.

    This function uses exactly the same split as training.

    It is normally used by evaluate.py.

    Parameters
    ----------
    config : module
        The common.config module.

    batch_size : int
        Evaluation batch size.

        A batch size of 1 is useful when saving one
        qualitative result for each image.

    Returns
    -------
    DataLoader
        Validation DataLoader.
    """

    validation_full_dataset = OvarianDataset(
        image_dir=config.IMAGE_DIR,
        mask_dir=config.MASK_DIR,
        image_size=config.IMAGE_SIZE,
        transform=None
    )

    (
        _,
        validation_indices
    ) = create_split_indices(
        dataset_size=len(
            validation_full_dataset
        ),
        train_split=config.TRAIN_SPLIT,
        random_seed=config.RANDOM_SEED
    )

    validation_dataset = Subset(
        validation_full_dataset,
        validation_indices
    )

    pin_memory = (
        config.DEVICE == "cuda"
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory
    )

    return validation_loader


# ==========================================================
# Model Comparison CSV
# ==========================================================

def append_result(
    csv_path,
    model_name,
    metrics
):
    """
    Adds or updates a model's evaluation result in a CSV file.

    If the model already exists in the CSV, its previous row
    is replaced instead of creating a duplicate row.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to the model comparison CSV file.

    model_name : str
        Name of the evaluated model.

    metrics : dict
        Dictionary containing:

        - dice
        - iou
        - precision
        - recall
        - specificity
        - hausdorff_distance
    """

    csv_path = Path(csv_path)

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    required_metrics = [
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
        "hausdorff_distance"
    ]

    missing_metrics = [
        metric_name
        for metric_name in required_metrics
        if metric_name not in metrics
    ]

    if missing_metrics:
        raise ValueError(
            "Missing metrics: "
            + ", ".join(missing_metrics)
        )

    field_names = [
        "model",
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
        "hausdorff_distance"
    ]

    existing_rows = []

    if csv_path.exists():
        with csv_path.open(
            mode="r",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            reader = csv.DictReader(
                csv_file
            )

            existing_rows = list(
                reader
            )

    # Remove an existing result for this model.
    existing_rows = [
        row
        for row in existing_rows
        if row.get("model") != model_name
    ]

    new_row = {
        "model": model_name,
        "dice": f"{metrics['dice']:.6f}",
        "iou": f"{metrics['iou']:.6f}",
        "precision": f"{metrics['precision']:.6f}",
        "recall": f"{metrics['recall']:.6f}",
        "specificity": f"{metrics['specificity']:.6f}",
        "hausdorff_distance": (
            f"{metrics['hausdorff_distance']:.6f}"
        )
    }

    existing_rows.append(
        new_row
    )

    with csv_path.open(
        mode="w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=field_names
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
    device
):
    """
    Loads a saved model checkpoint safely.

    Parameters
    ----------
    model : torch.nn.Module
        Initialized segmentation model.

    checkpoint_path : str or pathlib.Path
        Path to the model checkpoint.

    device : torch.device
        Device used for loading the checkpoint.

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
            f"Model checkpoint not found: "
            f"{checkpoint_path}"
        )

    state_dict = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        state_dict
    )

    return model

