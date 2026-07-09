import cv2
import torch
import matplotlib.pyplot as plt

from model import DSNet
import config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DSNet().to(device)
model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
model.eval()


image_path = "OTU_2d/images/1.JPG"

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

original = image.copy()

image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
image = image / 255.0
image = image.transpose(2, 0, 1)

image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    pred, _, _, _ = model(image_tensor)
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

pred_mask = pred.cpu().squeeze().numpy()

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.title("Input Ultrasound")
plt.imshow(original)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Predicted Mask")
plt.imshow(pred_mask, cmap="gray")
plt.axis("off")

plt.show()