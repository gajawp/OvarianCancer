# integration_ds2net_MtaSwinn/config.py

from pathlib import Path

import torch


# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_DATASET_ROOT = PROJECT_ROOT / "datasets" / "OTU_2d"
ORIGINAL_IMAGE_DIR = ORIGINAL_DATASET_ROOT / "images"

TRAIN_CLASSIFICATION_LIST = ORIGINAL_DATASET_ROOT / "train_cls.txt"
VAL_CLASSIFICATION_LIST = ORIGINAL_DATASET_ROOT / "val_cls.txt"


# ==========================================================
# Generated DS2Net Masked ROI Dataset
# ==========================================================

ROI_DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "OTU_2d_ds2net_masked_roi_padding20"
)

TRAIN_ROI_DIR = ROI_DATASET_ROOT / "train"
VAL_ROI_DIR = ROI_DATASET_ROOT / "val"

TRAIN_MASK_DIR = ROI_DATASET_ROOT / "predicted_masks" / "train"
VAL_MASK_DIR = ROI_DATASET_ROOT / "predicted_masks" / "val"

TRAIN_ROI_LIST = ROI_DATASET_ROOT / "train_roi_cls.txt"
VAL_ROI_LIST = ROI_DATASET_ROOT / "val_roi_cls.txt"


# ==========================================================
# DS2Net ROI Settings
# ==========================================================

ROI_MODE = "masked"
ROI_PADDING = 20
MINIMUM_COMPONENT_AREA = 50


# ==========================================================
# MTA-Swin Classification Settings
# ==========================================================

NUM_CLASSES = 8

IMAGE_SIZE = 224
BATCH_SIZE = 4
NUM_EPOCHS = 50

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 12

LABEL_SMOOTHING = 0.1
DROPOUT = 0.30
ATTENTION_HIDDEN_DIM = 256

RANDOM_SEED = 42

NUM_WORKERS = 0
PIN_MEMORY = False

USE_PRETRAINED_WEIGHTS = True
FREEZE_BACKBONE_EPOCHS = 3


# ==========================================================
# Output Paths
# ==========================================================

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "integration_ds2net_MtaSwinn"
    / "checkpoints_masked_padding20"
)

BEST_MODEL_PATH = (
    CHECKPOINT_DIR
    / "best_ds2net_mta_swin_masked_padding20.pth"
)

LAST_MODEL_PATH = (
    CHECKPOINT_DIR
    / "last_ds2net_mta_swin_masked_padding20.pth"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "classification_results"
    / "ds2net_mta_swin_masked_padding20"
)

TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.csv"
TRAINING_CHART_PATH = RESULTS_DIR / "training_curves.png"
EVALUATION_RESULTS_PATH = RESULTS_DIR / "evaluation_results.csv"
METRIC_SUMMARY_PATH = RESULTS_DIR / "metric_summary.csv"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
NORMALIZED_CONFUSION_MATRIX_PATH = (
    RESULTS_DIR
    / "confusion_matrix_normalized.png"
)
PREDICTIONS_PATH = RESULTS_DIR / "predictions.csv"


# ==========================================================
# Class Names
# ==========================================================

CLASS_NAMES = {
    0: "Class 0",
    1: "Class 1",
    2: "Class 2",
    3: "Class 3",
    4: "Class 4",
    5: "Class 5",
    6: "Class 6",
    7: "Class 7",
}


# ==========================================================
# Device
# ==========================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")