import argparse
from pathlib import Path
import cv2, matplotlib.pyplot as plt, numpy as np, torch
from common import config
from common.utils import load_model_checkpoint
from unetplusplus.model import UNetPlusPlus

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("image_path"); parser.add_argument("--output",default="unetplusplus_prediction.png"); args=parser.parse_args()
    image=cv2.cvtColor(cv2.imread(args.image_path),cv2.COLOR_BGR2RGB); resized=cv2.resize(image,(config.IMAGE_SIZE,config.IMAGE_SIZE))
    tensor=torch.from_numpy((resized.astype(np.float32)/255.0).transpose(2,0,1)).float().unsqueeze(0)
    device=torch.device(config.DEVICE); model=load_model_checkpoint(UNetPlusPlus().to(device),config.UNETPLUSPLUS_MODEL_PATH,device); model.eval()
    with torch.no_grad(): pred=(torch.sigmoid(model(tensor.to(device))[-1])>config.PREDICTION_THRESHOLD).float().cpu().squeeze().numpy()
    fig=plt.figure(figsize=(10,4)); plt.subplot(1,2,1); plt.imshow(resized); plt.axis("off"); plt.title("Input"); plt.subplot(1,2,2); plt.imshow(pred,cmap="gray"); plt.axis("off"); plt.title("Predicted Mask"); plt.tight_layout(); fig.savefig(Path(args.output),dpi=200,bbox_inches="tight"); plt.close(fig)
if __name__=="__main__": main()
