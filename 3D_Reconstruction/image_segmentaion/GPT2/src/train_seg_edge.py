import torch, numpy as np
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
from models.unet_edge import UNetEdge
from datasets.seg_edge_dataset import SegEdgeDataset
from losses import MultiTaskLoss

def train(img_dir, mask_dir, save_dir='ckpts', epochs=60, bs=6, lr=3e-4, img_size=896, val_ratio=0.2, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    ds = SegEdgeDataset(img_dir, mask_dir, img_size=img_size, training=True, edge_width=3)
    n_val = int(len(ds)*val_ratio); n_tr = len(ds)-n_val
    ds_tr, ds_va = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(seed))
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    dl_va = DataLoader(SegEdgeDataset(img_dir, mask_dir, img_size=img_size, training=False, edge_width=3),
                       batch_size=2, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNetEdge(in_ch=3, base_ch=64).to(device)
    crit = MultiTaskLoss(cb_edge=0.9, lam_dice=0.5, w_edge=0.5, w_boundary=0.2)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    best = 1e9
    for ep in range(1, epochs+1):
        net.train(); tr_loss=0
        for img, seg_gt, edge_gt in tqdm(dl_tr, desc=f"Train ep{ep}"):
            img, seg_gt, edge_gt = img.to(device), seg_gt.to(device), edge_gt.to(device)
            seg_pred, edge_pred = net(img)
            loss, parts = crit(seg_pred, seg_gt, edge_pred, edge_gt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()*img.size(0)
        tr_loss /= len(dl_tr.dataset)

        # val
        net.eval(); va_loss=0
        with torch.no_grad():
            for img, seg_gt, edge_gt in dl_va:
                img, seg_gt, edge_gt = img.to(device), seg_gt.to(device), edge_gt.to(device)
                seg_pred, edge_pred = net(img)
                loss, _ = crit(seg_pred, seg_gt, edge_pred, edge_gt)
                va_loss += loss.item()*img.size(0)
        va_loss /= len(dl_va.dataset)
        sch.step()

        print(f"[ep{ep}] train {tr_loss:.4f}  val {va_loss:.4f}")
        if va_loss < best:
            best = va_loss
            torch.save({'ep':ep,'state_dict':net.state_dict()}, f"{save_dir}/seg_edge_best.pth")
            print("✓ saved best")

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_dir', required=True)
    ap.add_argument('--mask_dir', required=True)
    ap.add_argument('--save_dir', default='ckpts')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--bs', type=int, default=6)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--img_size', type=int, default=896)
    args = ap.parse_args()
    train(**vars(args))
