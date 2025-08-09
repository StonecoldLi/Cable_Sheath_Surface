# src/infer_and_crop.py
import torch, cv2, numpy as np
from pathlib import Path
from model_hed import HED
from postproc import edge_nms, hysteresis, trace_and_fit, buffer_polyline, estimate_width_by_gradient, crop_by_components

def load_model(ckpt_path, device):
    net = HED(pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location=device)['state_dict']
    net.load_state_dict(state, strict=True)
    net.eval()
    return net

def infer_prob(net, img, device, scales=(1.0, 0.75, 1.25)):
    H,W = img.shape[:2]
    probs = []
    with torch.no_grad():
        for s in scales:
            new = (int(H*s), int(W*s))
            im = cv2.resize(img, (new[1], new[0]), interpolation=cv2.INTER_LINEAR)
            x = torch.from_numpy(np.transpose(im[:,:,::-1].astype(np.float32)/255., (2,0,1))[None]).to(device)
            outs = net(x)[-1]  # fuse
            p = outs[0,0].cpu().numpy()
            p = cv2.resize(p, (W,H), interpolation=cv2.INTER_LINEAR)
            probs.append(p)
    prob = np.clip(np.mean(probs, axis=0), 0, 1)
    return prob

def pipeline_image(net, img_bgr, th_h=0.5, th_l=0.2, r_min=2, r_max=6):
    device = next(net.parameters()).device
    prob = infer_prob(net, img_bgr, device)
    nms = edge_nms(prob, k=3)
    bin_hys = hysteresis(nms, th_l, th_h)
    curves = trace_and_fit(bin_hys, min_len=80, smooth=1.0, num_pts=300)
    H,W = img_bgr.shape[:2]
    masks=[]
    for c in curves:
        r = estimate_width_by_gradient(img_bgr, c, r_min, r_max)
        m = buffer_polyline(c, r=r, shape=(H,W,3))
        masks.append(m)
    mask = np.zeros((H,W), np.uint8)
    for m in masks: mask |= m
    rois = crop_by_components(mask, margin=16, min_area=80, img=img_bgr)
    return prob, nms, bin_hys, mask, rois

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--img', required=True)        # 单张或文件夹
    ap.add_argument('--out_dir', default='out')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = load_model(args.ckpt, device)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(args.img)] if Path(args.img).is_file() else sorted(list(Path(args.img).glob('*')))
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        prob,nms,bin_hys,mask,rois = pipeline_image(net, img)
        name = p.stem
        cv2.imwrite(str(out_dir/f"{name}_prob.png"), (prob*255).astype(np.uint8))
        cv2.imwrite(str(out_dir/f"{name}_nms.png"), (nms*255).astype(np.uint8))
        cv2.imwrite(str(out_dir/f"{name}_mask.png"), (mask*255).astype(np.uint8))
        for i, r in enumerate(rois):
            cv2.imwrite(str(out_dir/f"{name}_crop_{i:02d}.png"), r)
        print(f"done: {p.name}, crops={len(rois)}")
