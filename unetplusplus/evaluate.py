import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from common import config
from common.metrics import calculate_numpy_metrics, hausdorff_distance
from common.utils import append_result, create_validation_loader, load_model_checkpoint
from unetplusplus.model import UNetPlusPlus

def main():
    device=torch.device(config.DEVICE); loader=create_validation_loader(config,batch_size=1)
    model=load_model_checkpoint(UNetPlusPlus().to(device),config.UNETPLUSPLUS_MODEL_PATH,device); model.eval()
    out=config.UNETPLUSPLUS_RESULTS_DIR; out.mkdir(parents=True,exist_ok=True)
    values={k:[] for k in ["dice","iou","precision","recall","specificity","hausdorff_distance"]}
    with torch.no_grad():
        for idx,(image,mask) in enumerate(tqdm(loader,desc="Evaluating U-Net++")):
            image,mask=image.to(device),mask.to(device); logits=model(image)[-1]
            pred=(torch.sigmoid(logits)>config.PREDICTION_THRESHOLD).float().cpu().numpy().squeeze()
            truth=mask.cpu().numpy().squeeze(); metrics=calculate_numpy_metrics(pred,truth)
            for k,v in metrics.items(): values[k].append(v)
            values["hausdorff_distance"].append(hausdorff_distance(pred,truth))
            if idx<30:
                fig=plt.figure(figsize=(15,4))
                img=image.cpu().squeeze(0).permute(1,2,0).numpy()
                for p,(title,data,cmap) in enumerate([("Input Ultrasound",img,None),("Ground Truth",truth,"gray"),("Prediction",pred,"gray")],1):
                    plt.subplot(1,4,p); plt.title(title); plt.imshow(data,cmap=cmap); plt.axis("off")
                plt.subplot(1,4,4); plt.title("Prediction Overlay"); plt.imshow(img); plt.imshow(pred,cmap="jet",alpha=.35); plt.axis("off")
                plt.tight_layout(); fig.savefig(out/f"unetplusplus_result_{idx:03d}.png",dpi=200,bbox_inches="tight"); plt.close(fig)
    avg={k:float(np.nanmean(v)) for k,v in values.items()}
    print("\nU-Net++ Evaluation Results\n--------------------------")
    for label,key in [("Dice Score","dice"),("IoU Score","iou"),("Precision","precision"),("Recall","recall"),("Specificity","specificity"),("Hausdorff Dist.","hausdorff_distance")]: print(f"{label:<16}: {avg[key]:.4f}")
    append_result(csv_path=config.MODEL_COMPARISON_PATH,model_name="U-Net++",metrics=avg)
if __name__=="__main__": main()
