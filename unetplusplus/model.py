import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(i,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
    def forward(self,x): return self.block(x)

class UNetPlusPlus(nn.Module):
    def __init__(self,in_channels=3,out_channels=1,deep_supervision=True):
        super().__init__()
        f=[32,64,128,256,512]; self.deep_supervision=deep_supervision
        self.pool=nn.MaxPool2d(2); self.up=nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False)
        self.c00=ConvBlock(in_channels,f[0]); self.c10=ConvBlock(f[0],f[1]); self.c20=ConvBlock(f[1],f[2])
        self.c30=ConvBlock(f[2],f[3]); self.c40=ConvBlock(f[3],f[4])
        self.c01=ConvBlock(f[0]+f[1],f[0]); self.c11=ConvBlock(f[1]+f[2],f[1])
        self.c21=ConvBlock(f[2]+f[3],f[2]); self.c31=ConvBlock(f[3]+f[4],f[3])
        self.c02=ConvBlock(f[0]*2+f[1],f[0]); self.c12=ConvBlock(f[1]*2+f[2],f[1]); self.c22=ConvBlock(f[2]*2+f[3],f[2])
        self.c03=ConvBlock(f[0]*3+f[1],f[0]); self.c13=ConvBlock(f[1]*3+f[2],f[1])
        self.c04=ConvBlock(f[0]*4+f[1],f[0])
        self.o1=nn.Conv2d(f[0],out_channels,1); self.o2=nn.Conv2d(f[0],out_channels,1)
        self.o3=nn.Conv2d(f[0],out_channels,1); self.o4=nn.Conv2d(f[0],out_channels,1)
    def forward(self,x):
        x00=self.c00(x); x10=self.c10(self.pool(x00)); x01=self.c01(torch.cat([x00,self.up(x10)],1))
        x20=self.c20(self.pool(x10)); x11=self.c11(torch.cat([x10,self.up(x20)],1)); x02=self.c02(torch.cat([x00,x01,self.up(x11)],1))
        x30=self.c30(self.pool(x20)); x21=self.c21(torch.cat([x20,self.up(x30)],1)); x12=self.c12(torch.cat([x10,x11,self.up(x21)],1)); x03=self.c03(torch.cat([x00,x01,x02,self.up(x12)],1))
        x40=self.c40(self.pool(x30)); x31=self.c31(torch.cat([x30,self.up(x40)],1)); x22=self.c22(torch.cat([x20,x21,self.up(x31)],1)); x13=self.c13(torch.cat([x10,x11,x12,self.up(x22)],1)); x04=self.c04(torch.cat([x00,x01,x02,x03,self.up(x13)],1))
        outputs=(self.o1(x01),self.o2(x02),self.o3(x03),self.o4(x04))
        return outputs if self.deep_supervision else outputs[-1]
