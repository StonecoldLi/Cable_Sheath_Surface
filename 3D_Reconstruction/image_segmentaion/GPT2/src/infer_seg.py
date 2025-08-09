import cv2, torch, numpy as np
from pathlib import Path
from models.unet_edge import UNetEdge

def load_model(ckpt, device):
    net = UNetEdge(in_ch=3, base_ch=64).to(device)
    state = torch.load(ckpt, map_location=device)['state_dict']
    net.load_state_dict(state, strict=True)
    net.eval()
    return net

def preprocess(img, size=1024):
    H,W = img.shape[:2]
    scale = size / max(H,W)
    newH,newW = int(H*scale), int(W*scale)
    im = cv2.resize(img, (newW,newH), interpolation=cv2.INTER_LINEAR)
    padH = size - newH; padW = size - newW
    im = cv2.copyMakeBorder(im, 0, padH, 0, padW, cv2.BORDER_CONSTANT, value=0)
    x = np.transpose(im[:,:,::-1].astype(np.float32)/255., (2,0,1))[None]
    return x, (H,W), (newH,newW, padH, padW)

def postprocess(prob, shape_info):
    H,W = shape_info
    prob = cv2.resize(prob, (W,H), interpolation=cv2.INTER_LINEAR)
    return prob

def refine_with_edge(seg_prob, edge_prob, k_edge=3, alpha=0.3, th=0.5):
    # 用边界图锐化分割概率（简单稳）
    edge = cv2.GaussianBlur(edge_prob, (k_edge|1, k_edge|1), 0)
    seg_ref = np.clip(seg_prob + alpha*(edge - 0.5), 0, 1)
    return (seg_ref > th).astype(np.uint8)

def crop_by_components(mask, margin=16, min_area=80, img=None):
    num, labels = cv2.connectedComponents(mask.astype(np.uint8))
    rois = []
    for i in range(1, num):
        ys, xs = np.where(labels==i)
        if xs.size < min_area: continue
        x1,x2 = xs.min(), xs.max()
        y1,y2 = ys.min(), ys.max()
        x1 = max(0, x1-margin); y1 = max(0, y1-margin)
        x2 = min(mask.shape[1]-1, x2+margin); y2 = min(mask.shape[0]-1, y2+margin)
        if img is not None:
            rois.append(img[y1:y2+1, x1:x2+1].copy())
        else:
            rois.append((x1,y1,x2,y2))
    return rois

def infer_folder(ckpt, img_dir, out_dir, size=1024, th=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = load_model(ckpt, device)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted([p for p in Path(img_dir).iterdir() if p.suffix.lower() in {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}])
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        x, (H,W), (newH,newW, padH, padW) = preprocess(img, size)
        with torch.no_grad():
            xx = torch.from_numpy(x).to(device)
            seg, edge = net(xx)
            seg = seg[0,0].cpu().numpy()
            edge = edge[0,0].cpu().numpy()
        # 去padding
        seg = seg[:newH, :newW]; edge = edge[:newH, :newW]
        # 还原到原尺寸
        seg = postprocess(seg, (H,W))
        edge = postprocess(edge, (H,W))
        # 边界引导细化 + 二值化
        mask = refine_with_edge(seg, edge, k_edge=3, alpha=0.3, th=th)
        cv2.imwrite(str(out_dir/f"{p.stem}_seg.png"), (seg*255).astype(np.uint8))
        cv2.imwrite(str(out_dir/f"{p.stem}_edge.png"), (edge*255).astype(np.uint8))
        cv2.imwrite(str(out_dir/f"{p.stem}_mask.png"), (mask*255).astype(np.uint8))

        rois = crop_by_components(mask, margin=16, min_area=80, img=img)
        for i, r in enumerate(rois):
            cv2.imwrite(str(out_dir/f"{p.stem}_crop_{i:02d}.png"), r)
        print(f"done: {p.name}, crops={len(rois)}")

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--img_dir', required=True)
    ap.add_argument('--out_dir', default='out')
    ap.add_argument('--size', type=int, default=1024)
    ap.add_argument('--th', type=float, default=0.5)
    args = ap.parse_args()
    infer_folder(args.ckpt, args.img_dir, args.out_dir, size=args.size, th=args.th)
