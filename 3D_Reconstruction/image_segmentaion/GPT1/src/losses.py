# src/losses.py
import torch, torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6): super().__init__(); self.eps=eps
    def forward(self, pred, target):
        # pred,target: (N,1,H,W) in [0,1]
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
        w = torch.where(target>0.5, torch.full_like(loss, self.pos_weight), torch.full_like(loss, 1-self.pos_weight))
        return (loss*w).mean()

class EdgeLoss(nn.Module):
    def __init__(self, cb=0.9, lam_dice=0.5):
        super().__init__()
        self.cb = CBBCE(pos_weight=cb)
        self.dice = DiceLoss()
        self.lam = lam_dice
    def forward(self, preds, target):
        # preds: list of side outputs + fuse
        losses = []
        for p in preds:
            losses.append(self.cb(p, target) + self.lam*self.dice(p, target))
        return sum(losses)/len(losses)
