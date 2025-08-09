import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), conv3x3(in_ch, out_ch))
    def forward(self, x): return self.block(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = conv3x3(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        # pad if needed
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        x = F.pad(x, [dw//2, dw - dw//2, dh//2, dh - dh//2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNetEdge(nn.Module):
    """
    两个头：
      - seg_head: 语义mask（1通道）
      - edge_head: 边界概率（1通道），来自浅层+融合特征
    """
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        c = base_ch
        self.inc = conv3x3(in_ch, c)
        self.d1 = Down(c, c*2)
        self.d2 = Down(c*2, c*4)
        self.d3 = Down(c*4, c*8)
        self.d4 = Down(c*8, c*8)

        self.u1 = Up(c*16, c*4)
        self.u2 = Up(c*8,  c*2)
        self.u3 = Up(c*4,  c)
        self.u4 = Up(c*2,  c)

        self.seg_head = nn.Conv2d(c, 1, kernel_size=1)

        # 边界头：融合浅层与解码高分辨率特征
        self.edge_conv1 = nn.Conv2d(c, c//2, 1)
        self.edge_conv2 = nn.Conv2d(c*2, c//2, 1)  # 融合 inc 与 u3
        self.edge_out   = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        h, w = x.shape[2:]
        x1 = self.inc(x)      # c, H, W
        x2 = self.d1(x1)      # 2c, H/2
        x3 = self.d2(x2)      # 4c, H/4
        x4 = self.d3(x3)      # 8c, H/8
        x5 = self.d4(x4)      # 8c, H/16

        y = self.u1(x5, x4)   # 4c
        y = self.u2(y, x3)    # 2c
        y = self.u3(y, x2)    # c
        y = self.u4(y, x1)    # c  (高分辨率)

        seg = torch.sigmoid(self.seg_head(y))

        # 边界头：把浅层(x1)与高分辨率(y)融合
        e1 = self.edge_conv1(y)             # c/2
        x1d = self.edge_conv1(x1)           # c/2
        e = torch.cat([e1, x1d], dim=1)     # c
        edge = torch.sigmoid(self.edge_out(e))
        return seg, edge
