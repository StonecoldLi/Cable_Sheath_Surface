import torch, torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6): super().__init__(); self.eps=eps
    def forward(self, pred, target):
        inter = (pred*target).sum(dim=(2,3))
        denom = pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + self.eps
        dice = 2*inter/denom
        return 1 - dice.mean()

class CBBCE(nn.Module):
    def __init__(self, pos_weight=0.9):
        super().__init__()
        self.pos_weight = pos_weight
        self.bce = nn.BCELoss(reduction='none')
    def forward(self, pred, target):
        loss = self.bce(pred, target)
        w = torch.where(target>0.5, torch.full_like(loss, self.pos_weight),
                        torch.full_like(loss, 1-self.pos_weight))
        return (loss*w).mean()

def signed_distance_transform(mask: torch.Tensor) -> torch.Tensor:
    """
    mask: (N,1,H,W) in {0,1}
    返回归一化到 [-1,1] 的有符号距离（内部为正，外部为负）
    """
    # 用2D欧氏距离近似（CPU实现简单但慢；训练时可提前CPU计算或小分辨率）
    # 这里给一个可用的近似（在小batch上没问题）
    import numpy as np
    import cv2
    outs = []
    for m in mask.detach().cpu().numpy():
        m = (m[0]>0.5).astype('uint8')
        dist_in  = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        dist_out = cv2.distanceTransform(1-m, cv2.DIST_L2, 3)
        sd = dist_in - dist_out
        if sd.max() > 0:
            sd = sd / (sd.max() + 1e-6)
        sd = np.clip(sd, -1, 1)
        outs.append(sd[None, ...])
    return torch.from_numpy(np.stack(outs,0)).to(mask.device).float()

class BoundaryLoss(nn.Module):
    """
    让分割预测靠近真值边界（signed distance supervision）
    """
    def __init__(self, weight=0.2):
        super().__init__()
        self.weight = weight
    def forward(self, seg_pred, seg_gt):
        # seg_pred in [0,1], seg_gt in {0,1}
        sd = signed_distance_transform(seg_gt)
        # 预测到边界的距离，用 L1 更稳
        loss = (torch.abs(seg_pred - seg_gt) * torch.abs(sd)).mean()
        return self.weight * loss

class MultiTaskLoss(nn.Module):
    """
    seg_loss = Dice + BCE
    edge_loss = CB-BCE
    boundary_loss = BoundaryLoss
    """
    def __init__(self, cb_edge=0.9, lam_dice=0.5, w_edge=0.5, w_boundary=0.2):
        super().__init__()
        self.dice = DiceLoss()
        self.bce  = nn.BCELoss()
        self.edge = CBBCE(cb_edge)
        self.boundary = BoundaryLoss(weight=w_boundary)
        self.lam_dice = lam_dice
        self.w_edge = w_edge

    def forward(self, seg_pred, seg_gt, edge_pred, edge_gt):
        seg = self.bce(seg_pred, seg_gt) + self.lam_dice*self.dice(seg_pred, seg_gt)
        edg = self.edge(edge_pred, edge_gt) * self.w_edge
        bdl = self.boundary(seg_pred, seg_gt)
        return seg + edg + bdl, {'seg': seg.item(), 'edge': edg.item(), 'boundary': bdl.item()}
