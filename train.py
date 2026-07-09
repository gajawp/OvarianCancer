import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import OvarianDataset, get_train_transform
from model import DSNet
from loss import dsnet_loss
from metrics import dice_score, iou_score
import config


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(config.DEVICE)
dataset = OvarianDataset(
    image_dir=config.IMAGE_DIR,
    mask_dir=config.MASK_DIR,
    image_size=config.IMAGE_SIZE,
    transform=get_train_transform()
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

model = DSNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

best_dice = 0

for epoch in range(config.EPOCHS):
    model.train()
    train_loss = 0

    for images, masks in tqdm(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = dsnet_loss(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_dice = 0
    val_iou = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            final_pred = outputs[0]

            val_dice += dice_score(final_pred, masks).item()
            val_iou += iou_score(final_pred, masks).item()

    avg_dice = val_dice / len(val_loader)
    avg_iou = val_iou / len(val_loader)

    print(
        f"Epoch [{epoch+1}/{config.EPOCHS}] "
        f"Loss: {train_loss / len(train_loader):.4f} "
        f"Dice: {avg_dice:.4f} "
        f"IoU: {avg_iou:.4f}"
    )

    if avg_dice > best_dice:
        best_dice = avg_dice
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        print("Model saved.")