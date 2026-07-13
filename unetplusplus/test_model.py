import torch
from unetplusplus.model import UNetPlusPlus
def main():
    device=torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model=UNetPlusPlus().to(device).eval(); x=torch.randn(2,3,256,256,device=device)
    with torch.no_grad(): outputs=model(x)
    assert len(outputs)==4
    for i,o in enumerate(outputs,1): assert tuple(o.shape)==(2,1,256,256); print(f"Output {i}: {tuple(o.shape)}")
    print("U-Net++ test passed.\nDevice:",device)
if __name__=="__main__": main()
