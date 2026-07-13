from pathlib import Path
import torch


# ==========================================================
# Project Directories
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "OTU_2d"
)

IMAGE_DIR = (
    DATASET_ROOT
    / "images"
)

MASK_DIR = (
    DATASET_ROOT
    / "annotations"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

MODEL_COMPARISON_PATH = (
    RESULTS_DIR
    / "model_comparison.csv"
)


# ==========================================================
# Image Settings
# ==========================================================

IMAGE_SIZE = 256

BATCH_SIZE = 4


# ==========================================================
# Training Settings
# ==========================================================

EPOCHS = 50

LEARNING_RATE = 1e-4

TRAIN_SPLIT = 0.80

RANDOM_SEED = 42

NUM_WORKERS = 0


# ==========================================================
# Prediction Settings
# ==========================================================

PREDICTION_THRESHOLD = 0.5


# ==========================================================
# Early Stopping
# ==========================================================

EARLY_STOPPING_PATIENCE = 12


# ==========================================================
# Device
# ==========================================================

if torch.backends.mps.is_available():
    DEVICE = "mps"

elif torch.cuda.is_available():
    DEVICE = "cuda"

else:
    DEVICE = "cpu"


# ==========================================================
# DSNet
# ==========================================================

DSNET_MODEL_PATH = (
    PROJECT_ROOT
    / "dsnet"
    / "weights"
    / "best_model.pth"
)

DSNET_RESULTS_DIR = (
    PROJECT_ROOT
    / "dsnet"
    / "qualitative_results"
)


# ==========================================================
# Attention U-Net
# ==========================================================

ATTENTION_UNET_MODEL_PATH = (
    PROJECT_ROOT
    / "attention_unet"
    / "weights"
    / "best_model.pth"
)

ATTENTION_UNET_RESULTS_DIR = (
    PROJECT_ROOT
    / "attention_unet"
    / "qualitative_results"
)


# ==========================================================
# DS2Net
# ==========================================================

DS2NET_MODEL_PATH = (
    PROJECT_ROOT
    / "ds2net"
    / "weights"
    / "best_model.pth"
)

DS2NET_RESULTS_DIR = (
    PROJECT_ROOT
    / "ds2net"
    / "qualitative_results"
)


# ==========================================================
# DeepLabV3+
# ==========================================================

DEEPLABV3PLUS_MODEL_PATH = (
    PROJECT_ROOT
    / "deeplabv3plus"
    / "weights"
    / "best_model.pth"
)

DEEPLABV3PLUS_RESULTS_DIR = (
    PROJECT_ROOT
    / "deeplabv3plus"
    / "qualitative_results"
)


# ==========================================================
# Create Directories
# ==========================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DSNET_MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

DSNET_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ATTENTION_UNET_MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

ATTENTION_UNET_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DS2NET_MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

DS2NET_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DEEPLABV3PLUS_MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

DEEPLABV3PLUS_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ==========================================================
# SegFormer
# ==========================================================

SEGFORMER_MODEL_PATH = (
    PROJECT_ROOT
    / "segformer"
    / "weights"
    / "best_model.pth"
)

SEGFORMER_RESULTS_DIR = (
    PROJECT_ROOT
    / "segformer"
    / "qualitative_results"
)

SEGFORMER_MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

SEGFORMER_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ==========================================================
# TransUNet
# ==========================================================

TRANSUNET_MODEL_PATH = (
    PROJECT_ROOT
    / "transunet"
    / "weights"
    / "best_model.pth"
)

TRANSUNET_RESULTS_DIR = (
    PROJECT_ROOT
    / "transunet"
    / "qualitative_results"
)

TRANSUNET_MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

TRANSUNET_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ==========================================================
# U-Net++
# ==========================================================

UNETPLUSPLUS_MODEL_PATH = (
    PROJECT_ROOT
    / "unetplusplus"
    / "weights"
    / "best_model.pth"
)

UNETPLUSPLUS_RESULTS_DIR = (
    PROJECT_ROOT
    / "unetplusplus"
    / "qualitative_results"
)

UNETPLUSPLUS_MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

UNETPLUSPLUS_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)