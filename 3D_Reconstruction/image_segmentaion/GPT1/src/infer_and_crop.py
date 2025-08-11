# src/infer_and_crop.py
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from model_hed import HED
from postproc import (
    edge_nms,
    hysteresis,
    trace_and_fit,
    # 新增的贯穿两线选择与中间带生成
    select_two_spanning_paths,
    middle_strip_mask_from_paths,
    # 兜底与裁切
    estimate_width_by_gradient,
    buffer_polyline,
    crop_by_components,
)


def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    net = HED(pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location=device)["state_dict"]
    net.load_state_dict(state, strict=True)
    net.eval()
    return net


def infer_prob(
    net: torch.nn.Module,
    img_bgr: np.ndarray,
    device: torch.device,
    scales=(1.0, 0.75, 1.25),
) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    probs = []
    with torch.no_grad():
        for s in scales:
            newH, newW = int(H * s), int(W * s)
            im = cv2.resize(img_bgr, (newW, newH), interpolation=cv2.INTER_LINEAR)
            x = np.transpose(im[:, :, ::-1].astype(np.float32) / 255.0, (2, 0, 1))[None]
            xx = torch.from_numpy(x).to(device)
            out = net(xx)[-1]  # fuse
            p = out[0, 0].cpu().numpy().astype(np.float32)
            p = cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR)
            probs.append(p)
    prob = np.clip(np.mean(probs, axis=0), 0.0, 1.0)
    return prob


def pipeline_image(
    net: torch.nn.Module,
    img_bgr: np.ndarray,
    th_h: float = 0.5,
    th_l: float = 0.2,
    # 贯穿线筛选与外推参数
    edge_margin: int = 10,          # 与左右边缘的最小距离
    y_gap_max: int = 60,            # 段间最大 y 间隙（用于拼接）
    x_gap_max: int = 60,            # 段间最大 x 间隙（用于拼接）
    dmin: float = 8.0, dmax: float = 200.0,  # 两线的合理间距
    ang_tol_deg: float = 25.0,      # 两线近平行的角度容忍
    alpha_len: float = 0.6, beta_int: float = 0.4,  # 评分权重
    # 兜底参数
    r_min: int = 2, r_max: int = 6,
):
    """
    返回：
      prob: 边缘概率
      nms:  NMS 后概率
      bin_hys: 双阈值连通后的二值边缘
      mask: 仅保留“两条贯穿白线之间”的中间带区域
      rois: 该 mask 对应的裁切图像列表
    """
    device = next(net.parameters()).device

    # 1) 概率
    prob = infer_prob(net, img_bgr, device)

    # 2) NMS + 双阈值
    nms = edge_nms(prob, k=3)
    bin_hys = hysteresis(nms, th_l, th_h)

    # 3) 曲线提取
    curves = trace_and_fit(bin_hys, min_len=80, smooth=1.0, num_pts=300)
    H, W = img_bgr.shape[:2]

    # 4) 选择两条“贯穿路径”（自动拼接 + 外推到 y=0/H-1，并远离左右边缘）
    pair = select_two_spanning_paths(
        curves, prob,
        H=H, W=W,
        edge_margin=edge_margin,
        y_gap_max=y_gap_max, x_gap_max=x_gap_max,
        dmin=dmin, dmax=dmax,
        ang_tol_deg=ang_tol_deg,
        alpha_len=alpha_len, beta_int=beta_int
    )

    if pair is None:
        # 兜底：找不到合格两线，退回“所有线 buffer 合并”
        masks = []
        for c in curves:
            r = estimate_width_by_gradient(img_bgr, c, r_min, r_max)
            m = buffer_polyline(c, r=r, shape=(H, W, 3))
            masks.append(m)
        mask = np.zeros((H, W), np.uint8)
        for m in masks:
            mask |= m
    else:
        p1, p2 = pair
        mask = middle_strip_mask_from_paths(p1, p2, (H, W, 3))

    # 5) 裁切
    rois = crop_by_components(mask, margin=16, min_area=80, img=img_bgr)
    return prob, nms, bin_hys, mask, rois


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="HED 权重路径")
    ap.add_argument("--img", required=True, help="单张图片或目录")
    ap.add_argument("--out_dir", default="out", help="输出目录")

    # 阈值
    ap.add_argument("--th_high", type=float, default=0.55)
    ap.add_argument("--th_low", type=float, default=0.25)

    # 贯穿线与外推参数
    ap.add_argument("--edge_margin", type=int, default=10)
    ap.add_argument("--y_gap_max", type=int, default=60)
    ap.add_argument("--x_gap_max", type=int, default=60)
    ap.add_argument("--dmin", type=float, default=8.0)
    ap.add_argument("--dmax", type=float, default=200.0)
    ap.add_argument("--ang_tol", type=float, default=25.0)
    ap.add_argument("--alpha_len", type=float, default=0.6)
    ap.add_argument("--beta_int", type=float, default=0.4)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_model(args.ckpt, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p = Path(args.img)
    paths = [p] if p.is_file() else sorted(
        [x for x in p.iterdir() if x.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}]
    )

    for img_path in paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] fail to read: {img_path}")
            continue

        prob, nms, bin_hys, mask, rois = pipeline_image(
            net,
            img,
            th_h=args.th_high,
            th_l=args.th_low,
            edge_margin=args.edge_margin,
            y_gap_max=args.y_gap_max,
            x_gap_max=args.x_gap_max,
            dmin=args.dmin,
            dmax=args.dmax,
            ang_tol_deg=args.ang_tol,
            alpha_len=args.alpha_len,
            beta_int=args.beta_int,
        )

        stem = img_path.stem
        cv2.imwrite(str(out_dir / f"{stem}_prob.png"), (prob * 255).astype(np.uint8))
        cv2.imwrite(str(out_dir / f"{stem}_nms.png"), (nms * 255).astype(np.uint8))
        cv2.imwrite(str(out_dir / f"{stem}_bin.png"), (bin_hys.astype(np.uint8) * 255))
        cv2.imwrite(str(out_dir / f"{stem}_mask.png"), (mask.astype(np.uint8) * 255))
        for i, r in enumerate(rois):
            cv2.imwrite(str(out_dir / f"{stem}_crop_{i:02d}.png"), r)

        print(f"done: {img_path.name}  |  crops={len(rois)}")


if __name__ == "__main__":
    main()
