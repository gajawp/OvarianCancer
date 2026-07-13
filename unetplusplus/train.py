import time, torch
from tqdm import tqdm
from common import config
from common.loss import deep_supervision_loss
from common.metrics import dice_score, iou_score
from common.utils import create_data_loaders, set_random_seed
from unetplusplus.model import UNetPlusPlus

def final_output(x): return x[-1] if isinstance(x,(tuple,list)) else x
def validate(model,loader,device):
    model.eval(); d=i=0.0
    with torch.no_grad():
        for images,masks in loader:
            images,masks=images.to(device),masks.to(device); logits=final_output(model(images))
            d+=dice_score(logits,masks,threshold=config.PREDICTION_THRESHOLD).item()
            i+=iou_score(logits,masks,threshold=config.PREDICTION_THRESHOLD).item()
    return d/len(loader),i/len(loader)

def main():
    set_random_seed(config.RANDOM_SEED); device=torch.device(config.DEVICE)
    train_loader,val_loader=create_data_loaders(config)
    model=UNetPlusPlus().to(device); optimizer=torch.optim.Adam(model.parameters(),lr=config.LEARNING_RATE)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=.5,patience=5,min_lr=1e-7)
    best=-1.0; stale=0; path=config.UNETPLUSPLUS_MODEL_PATH; path.parent.mkdir(parents=True,exist_ok=True)
    print("Training U-Net++\n----------------\nDevice:",device)
    for epoch in range(config.EPOCHS):
        start=time.time(); model.train(); total=0.0
        bar=tqdm(train_loader,desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        for images,masks in bar:
            images,masks=images.to(device),masks.to(device); optimizer.zero_grad()
            loss=deep_supervision_loss(model(images),masks); loss.backward(); optimizer.step()
            total+=loss.item(); bar.set_postfix(loss=f"{loss.item():.4f}")
        dice,iou=validate(model,val_loader,device); scheduler.step(dice)
        print(f"Epoch [{epoch+1}/{config.EPOCHS}] Loss: {total/len(train_loader):.4f} Dice: {dice:.4f} IoU: {iou:.4f} Time: {time.time()-start:.1f}s")
        if dice>best: best=dice; stale=0; torch.save(model.state_dict(),path); print(f"Model saved: {path} (best Dice: {best:.4f})")
        else: stale+=1
        if stale>=config.EARLY_STOPPING_PATIENCE: print("Early stopping triggered."); break
if __name__=="__main__": main()
