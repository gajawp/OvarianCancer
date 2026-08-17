from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_DATASET_ROOT = PROJECT_ROOT / "datasets" / "OTU_2d"
ORIGINAL_IMAGE_DIR = ORIGINAL_DATASET_ROOT / "images"
TRAIN_CLASSIFICATION_LIST = ORIGINAL_DATASET_ROOT / "train_cls.txt"
VAL_CLASSIFICATION_LIST = ORIGINAL_DATASET_ROOT / "val_cls.txt"

MASK_AWARE_DATASET_ROOT = (
    PROJECT_ROOT / "datasets" / "OTU_2d_ds2net_rgb_mask_padding20"
)

TRAIN_RGB_DIR = MASK_AWARE_DATASET_ROOT / "train" / "images"
TRAIN_MASK_DIR = MASK_AWARE_DATASET_ROOT / "train" / "masks"
VAL_RGB_DIR = MASK_AWARE_DATASET_ROOT / "val" / "images"
VAL_MASK_DIR = MASK_AWARE_DATASET_ROOT / "val" / "masks"

TRAIN_LIST = MASK_AWARE_DATASET_ROOT / "train_rgb_mask_cls.txt"
VAL_LIST = MASK_AWARE_DATASET_ROOT / "val_rgb_mask_cls.txt"

ROI_PADDING = 20
MINIMUM_COMPONENT_AREA = 50

NUM_CLASSES = 8
IMAGE_SIZE = 224
BATCH_SIZE = 4
NUM_EPOCHS = 50

LEARNING_RATE = 1e-4
BACKBONE_LEARNING_RATE = 2.5e-5
WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 12
FREEZE_BACKBONE_EPOCHS = 3

LABEL_SMOOTHING = 0.0

# Multiclass focal-loss settings. No balanced sampler is used.
FOCAL_GAMMA = 2.0
FOCAL_USE_CLASS_ALPHA = False
DROPOUT = 0.30
ATTENTION_HIDDEN_DIM = 256
MASK_FEATURE_DIM = 128

RANDOM_SEED = 42
NUM_WORKERS = 0
PIN_MEMORY = False
USE_PRETRAINED_WEIGHTS = True

# Training-only augmentation. Validation and evaluation remain deterministic.
USE_AUGMENTATION = True
HORIZONTAL_FLIP_PROBABILITY = 0.50
AFFINE_PROBABILITY = 0.50
ROTATION_LIMIT_DEGREES = 15.0
TRANSLATION_LIMIT_FRACTION = 0.05
SCALE_MIN = 0.90
SCALE_MAX = 1.10
BRIGHTNESS_PROBABILITY = 0.40
BRIGHTNESS_MIN = 0.85
BRIGHTNESS_MAX = 1.15
CONTRAST_PROBABILITY = 0.40
CONTRAST_MIN = 0.85
CONTRAST_MAX = 1.15

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "integration_ds2net_maskaware_mtaswin_aug_focal"
    / "checkpoints"
)
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_ds2net_maskaware_mtaswin_aug_focal.pth"
LAST_MODEL_PATH = CHECKPOINT_DIR / "last_ds2net_maskaware_mtaswin_aug_focal.pth"

RESULTS_DIR = (
    PROJECT_ROOT
    / "classification_results"
    / "ds2net_maskaware_mtaswin_aug_focal"
)
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.csv"
TRAINING_LOSS_CHART_PATH = RESULTS_DIR / "training_curves.png"
MACRO_F1_CHART_PATH = RESULTS_DIR / "macro_f1_curve.png"
EVALUATION_RESULTS_PATH = RESULTS_DIR / "evaluation_results.csv"
METRIC_SUMMARY_PATH = RESULTS_DIR / "metric_summary.csv"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
NORMALIZED_CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix_normalized.png"

CLASS_NAMES = {i: f"Class {i}" for i in range(NUM_CLASSES)}

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
