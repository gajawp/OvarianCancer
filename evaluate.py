import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm

import config
from dataset import OvarianDataset
from model import DSNet


device = torch.device(config.DEVICE)


def calculate_metrics(pred, target):
    pred = pred.astype(np.uint8)
    target = target.astype(np.uint8)

    tp = np.logical_and(pred == 1, target == 1).sum()
    tn = np.logical_and(pred == 0, target == 0).sum()
    fp = np.logical_and(pred == 1, target == 0).sum()
    fn = np.logical_and(pred == 0, target == 1).sum()

    smooth = 1e-6

    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)

    return dice, iou, precision, recall, specificity


def hausdorff_distance(pred, target):
    pred_points = np.argwhere(pred > 0)
    target_points = np.argwhere(target > 0)

    if len(pred_points) == 0 or len(target_points) == 0:
        return np.nan

    hd1 = directed_hausdorff(pred_points, target_points)[0]
    hd2 = directed_hausdorff(target_points, pred_points)[0]

    return max(hd1, hd2)


def save_qualitative_result(image, mask, pred, index, save_dir):
    image = image.permute(1, 2, 0).cpu().numpy()
    mask = mask.squeeze().cpu().numpy()
    pred = pred.squeeze()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(pred, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"result_{index}.png"))
    plt.close()


def main():
    os.makedirs("qualitative_results", exist_ok=True)

    dataset = OvarianDataset(
        image_dir=config.IMAGE_DIR,
        mask_dir=config.MASK_DIR,
        image_size=config.IMAGE_SIZE,
        transform=None
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    _, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = DSNet().to(device)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    dice_list = []
    iou_list = []
    precision_list = []
    recall_list = []
    specificity_list = []
    hausdorff_list = []

    with torch.no_grad():
        for idx, (image, mask) in enumerate(tqdm(val_loader)):
            image = image.to(device)
            mask = mask.to(device)

            output, _, _, _ = model(image)

            pred = torch.sigmoid(output)
            pred = (pred > 0.5).float()

            pred_np = pred.cpu().numpy().squeeze()
            mask_np = mask.cpu().numpy().squeeze()

            dice, iou, precision, recall, specificity = calculate_metrics(
                pred_np,
                mask_np
            )

            hd = hausdorff_distance(pred_np, mask_np)

            dice_list.append(dice)
            iou_list.append(iou)
            precision_list.append(precision)
            recall_list.append(recall)
            specificity_list.append(specificity)
            hausdorff_list.append(hd)

            if idx < 30:
                save_qualitative_result(
                    image=image.cpu().squeeze(0),
                    mask=mask.cpu().squeeze(0),
                    pred=pred_np,
                    index=idx,
                    save_dir="qualitative_results"
                )

    print("\nEvaluation Results")
    print("------------------")
    print(f"Dice Score      : {np.nanmean(dice_list):.4f}")
    print(f"IoU Score       : {np.nanmean(iou_list):.4f}")
    print(f"Precision       : {np.nanmean(precision_list):.4f}")
    print(f"Recall          : {np.nanmean(recall_list):.4f}")
    print(f"Specificity     : {np.nanmean(specificity_list):.4f}")
    print(f"Hausdorff Dist. : {np.nanmean(hausdorff_list):.4f}")

    print("\nQualitative results saved in:")
    print("qualitative_results/")


if __name__ == "__main__":
    main()