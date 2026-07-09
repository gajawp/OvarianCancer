import torch
IMAGE_DIR = "OTU_2d/images"
MASK_DIR = "OTU_2d/annotations"

IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-4

#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
#MODEL_SAVE_PATH = "dsnet_ovarian.pth"
MODEL_SAVE_PATH = "ds2net_single_domain_ovarian.pth"