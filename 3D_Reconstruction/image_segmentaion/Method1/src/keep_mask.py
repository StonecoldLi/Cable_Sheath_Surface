#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 *_between.png 掩膜保留原图白色区域：
- 默认把 mask=黑 的区域置黑，保留白色区域
- 可选导出透明背景（黑区变透明）
- 可选紧致裁剪（裁成白区最小外接矩形）
- 结果保存到 ./results

用法示例：
python keep_by_mask.py --img_dir ./images --mask_dir ./out_pics --out_dir ./results \
    --mask_suffix _between.png --alpha --tight_crop
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import sys

IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]

def find_image_for_mask(base_stem: str, img_dir: Path):
    """在 img_dir 下按多种扩展名寻找与 base_stem 同名的原图。"""
    for ext in IMG_EXTS:
        p = img_dir / f"{base_stem}{ext}"
        if p.exists():
            return p
    return None

def ensure_3ch(img):
    """把单通道或带Alpha的图像转换为3通道BGR（不丢颜色）。"""
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    # 意外情况兜底
    return img[..., :3].copy()

def apply_mask_keep_white(img, mask_bin, alpha=False):
    """
    根据二值mask保留白色区域：
    - alpha=False: 黑区置黑，返回BGR
    - alpha=True : 输出BGRA，黑区alpha=0
    """
    h, w = img.shape[:2]
    if mask_bin.shape[:2] != (h, w):
        # 最稳妥是按最近邻缩放到原图尺寸
        mask_bin = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)

    if alpha:
        # 输出BGRA，透明度来自mask
        if img.ndim == 2:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            bgr = img
        a = (mask_bin.astype(np.uint8) * 255)
        a = np.clip(a, 0, 255)
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        bgra[..., 3] = a
        return bgra
    else:
        bgr = ensure_3ch(img).copy()
        # 黑区置黑
        inv = (mask_bin == 0)
        bgr[inv] = 0
        return bgr

def tight_crop(arr, mask_bin):
    """根据mask的白区做最小外接矩形裁剪（对图像或数组均可）。"""
    ys, xs = np.where(mask_bin > 0)
    if xs.size == 0:
        return arr, False  # 无白区，不裁剪
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    # +1 因为切片右开区间
    return arr[y0:y1+1, x0:x1+1].copy(), True

def main():
    ap = argparse.ArgumentParser(description="根据 *_between.png 掩膜保留原图白色区域并导出。")
    ap.add_argument("--img_dir", required=True, help="原图目录")
    ap.add_argument("--mask_dir", required=True, help="mask目录（包含 *_between.png）")
    ap.add_argument("--out_dir", default="./results", help="输出目录（默认 ./results）")
    ap.add_argument("--mask_suffix", default="_between.png", help="mask文件名后缀（默认 _between.png）")
    ap.add_argument("--alpha", action="store_true", help="导出透明背景（黑区透明），输出PNG")
    ap.add_argument("--tight_crop", action="store_true", help="对结果做最小外接矩形裁剪")
    ap.add_argument("--mask_white_thr", type=int, default=127, help="阈值>thr视为白（默认127）")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    mask_dir = Path(args.mask_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    masks = sorted(mask_dir.glob(f"*{args.mask_suffix}"))
    if not masks:
        print(f"[ERR] 在 {mask_dir} 下未找到 *{args.mask_suffix}。", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 待处理mask数量：{len(masks)}")
    ok, skip, err = 0, 0, 0

    for mpath in masks:
        name = mpath.name
        stem = mpath.stem  # e.g. foo_between
        if not stem.endswith(args.mask_suffix.replace(".png", "").replace(".jpg","")):
            # 更稳妥：去掉固定后缀 _between
            if stem.endswith("_between"):
                base_stem = stem[:-len("_between")]
            else:
                # 保守处理：尝试去掉指定后缀（不含扩展名）
                suf = Path(args.mask_suffix).stem  # '_between'
                base_stem = stem[:-len(suf)] if stem.endswith(suf) else stem
        else:
            # 常规去掉 '_between'
            base_stem = stem.replace(Path(args.mask_suffix).stem, "")

        base_stem = base_stem.rstrip("_")  # 去掉可能多余的下划线

        img_path = find_image_for_mask(base_stem, img_dir)
        if img_path is None:
            print(f"[SKIP] 未找到原图: {base_stem}.*")
            skip += 1
            continue

        # 读mask与原图
        mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[ERR] 读取mask失败: {mpath.name}")
            err += 1
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[ERR] 读取原图失败: {img_path.name}")
            err += 1
            continue

        # 二值化（白=保留）
        mask_bin = (mask > args.mask_white_thr).astype(np.uint8)

        # 应用mask
        out_img = apply_mask_keep_white(img, mask_bin, alpha=args.alpha)

        # 可选紧致裁剪（基于mask）
        if args.tight_crop:
            # 注意：裁剪时 mask 要与 out_img 同尺寸
            h, w = out_img.shape[:2]
            if mask_bin.shape[:2] != (h, w):
                mask_bin = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)
            out_img, cropped = tight_crop(out_img, mask_bin)
        else:
            cropped = False

        # 保存：alpha输出强制PNG；否则沿用原扩展名（但建议统一为PNG）
        if args.alpha:
            out_ext = ".png"
        else:
            # 若原始是有损格式，建议也换成png以避免反复压缩损失
            out_ext = ".png"

        suffix = "_kept_cropped" if (args.tight_crop and cropped) else "_kept"
        out_path = out_dir / f"{base_stem}{suffix}{out_ext}"
        ok_flag = cv2.imwrite(str(out_path), out_img)
        if not ok_flag:
            print(f"[ERR] 写文件失败: {out_path.name}")
            err += 1
            continue

        print(f"[OK] {name} -> {out_path.name}")
        ok += 1

    print(f"\n[DONE] 成功:{ok}, 跳过(缺原图):{skip}, 失败:{err}")
    print(f"输出目录: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
