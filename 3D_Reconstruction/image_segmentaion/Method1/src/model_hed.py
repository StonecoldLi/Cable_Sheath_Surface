# src/model_hed.py
import torch, torch.nn as nn
import torchvision.models as tv

def conv_1x1(in_ch): return nn.Conv2d(in_ch, 1, kernel_size=1, bias=True)

class HED(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        vgg = tv.vgg16_bn(weights=tv.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None).features
        # 按VGG16-BN分段
        self.stage1 = vgg[:6]    # 64
        self.stage2 = vgg[6:13]  # 128
        self.stage3 = vgg[13:23] # 256
        self.stage4 = vgg[23:33] # 512
        self.stage5 = vgg[33:43] # 512

        self.s1 = conv_1x1(64)
        self.s2 = conv_1x1(128)
        self.s3 = conv_1x1(256)
        self.s4 = conv_1x1(512)
        self.s5 = conv_1x1(512)
        self.fuse = nn.Conv2d(5, 1, kernel_size=1, bias=True)

    def forward(self, x):
        h, w = x.shape[2:]
        o1 = self.stage1(x)  # 1/2
        o2 = self.stage2(o1) # 1/4
        o3 = self.stage3(o2) # 1/8
        o4 = self.stage4(o3) # 1/16
        o5 = self.stage5(o4) # 1/32

        s1 = torch.sigmoid(nn.functional.interpolate(self.s1(o1), size=(h,w), mode='bilinear', align_corners=False))
        s2 = torch.sigmoid(nn.functional.interpolate(self.s2(o2), size=(h,w), mode='bilinear', align_corners=False))
        s3 = torch.sigmoid(nn.functional.interpolate(self.s3(o3), size=(h,w), mode='bilinear', align_corners=False))
        s4 = torch.sigmoid(nn.functional.interpolate(self.s4(o4), size=(h,w), mode='bilinear', align_corners=False))
        s5 = torch.sigmoid(nn.functional.interpolate(self.s5(o5), size=(h,w), mode='bilinear', align_corners=False))
        fuse = torch.sigmoid(self.fuse(torch.cat([s1,s2,s3,s4,s5], dim=1)))
        return [s1,s2,s3,s4,s5,fuse]
