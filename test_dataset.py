from dataset import OvarianDataset

dataset = OvarianDataset(
    image_dir="OTU_2d/images",
    mask_dir="OTU_2d/annotations"
)

print("Total Images:", len(dataset))

image, mask = dataset[0]

print(image.shape)
print(mask.shape)

import torch

print(torch.backends.mps.is_available())