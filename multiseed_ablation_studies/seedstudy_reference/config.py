from pathlib import Path
import os
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ORIGINAL_DATASET_ROOT = PROJECT_ROOT / "datasets" / "OTU_2d"
MASK_AWARE_DATASET_ROOT = (
    PROJECT_ROOT / "datasets" / "OTU_2d_ds2net_rgb_mask_padding20"
)

TRAIN_RGB_DIR = MASK_AWARE_DATASET_ROOT / "train" / "images"
TRAIN_MASK_DIR = MASK_AWARE_DATASET_ROOT / "train" / "masks"
VAL_RGB_DIR = MASK_AWARE_DATASET_ROOT / "val" / "images"
VAL_MASK_DIR = MASK_AWARE_DATASET_ROOT / "val" / "masks"

TRAIN_LIST = MASK_AWARE_DATASET_ROOT / "train_rgb_mask_cls.txt"
VAL_LIST = MASK_AWARE_DATASET_ROOT / "val_rgb_mask_cls.txt"

BINARY_LABEL_MAP = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 1,
}

NUM_CLASSES = 2
CLASS_NAMES = {
    0: "Non-malignant / other",
    1: "High-grade serous carcinoma",
}
POSITIVE_CLASS_INDEX = 1

IMAGE_SIZE = 224
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 50

LEARNING_RATE = 1e-4
ROI_BACKBONE_LEARNING_RATE = 2.5e-5
WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 12
FREEZE_BACKBONES_EPOCHS = 3

FOCAL_GAMMA = 2.0
FOCAL_USE_CLASS_ALPHA = True

DROPOUT = 0.30
ATTENTION_HIDDEN_DIM = 256
MASK_FEATURE_DIM = 128

RANDOM_SEED = int(os.environ.get("EXPERIMENT_SEED", "42"))
NUM_WORKERS = 0
PIN_MEMORY = False
USE_PRETRAINED_WEIGHTS = True

USE_AUGMENTATION = True
HORIZONTAL_FLIP_PROBABILITY = 0.50
AFFINE_PROBABILITY = 0.50
ROTATION_LIMIT_DEGREES = 10.0
TRANSLATION_LIMIT_FRACTION = 0.05
SCALE_MIN = 0.95
SCALE_MAX = 1.05
BRIGHTNESS_PROBABILITY = 0.40
BRIGHTNESS_MIN = 0.90
BRIGHTNESS_MAX = 1.10
CONTRAST_PROBABILITY = 0.40
CONTRAST_MIN = 0.90
CONTRAST_MAX = 1.10

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "multiseed_checkpoints"
    / "reference"
    / f"seed_{RANDOM_SEED}"
    / "checkpoints"
)

BEST_MODEL_PATH = (
    CHECKPOINT_DIR / "best_model.pth"
)

LAST_MODEL_PATH = (
    CHECKPOINT_DIR / "last_model.pth"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "classification_results"
    / "multiseed_ablation"
    / "reference"
    / f"seed_{RANDOM_SEED}"
)
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.csv"
TRAINING_CURVES_PATH = RESULTS_DIR / "training_curves.png"
MALIGNANT_F1_CURVE_PATH = RESULTS_DIR / "malignant_f1_curve.png"
EVALUATION_RESULTS_PATH = RESULTS_DIR / "evaluation_results.csv"
METRIC_SUMMARY_PATH = RESULTS_DIR / "metric_summary.csv"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
NORMALIZED_CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix_normalized.png"
ROC_CURVE_PATH = RESULTS_DIR / "roc_curve.png"
PR_CURVE_PATH = RESULTS_DIR / "precision_recall_curve.png"

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
