# src/build_edge_labels.py
import os
import cv2
import sys
import glob
import argparse
import numpy as np
from pathlib import Path
from typing import Optional
from skimage.morphology import dilation, disk
from skimage.segmentation import find_boundaries


def make_edge_from_mask(mask_bin: np.ndarray, edge_width: int = 3, mode: str = "outer") -> np.ndarray:
    """
    从二值mask生成边缘标签。
    mask_bin: uint8 {0,1}
    edge_width: 边缘加粗像素（建议2~3）
    mode: 'outer' | 'inner' | 'thick'
      - outer: 仅外边界
      - inner: 仅内边界
      - thick: skimage的thick模式（对细目标更连续）
    """
    if mode not in {"outer", "inner", "thick"}:
        mode = "outer"
    bnd = find_boundaries(mask_bin, mode=mode).astype(np.uint8)
    if edge_width > 1:
        bnd = dilation(bnd, disk(max(1, edge_width // 2))).astype(np.uint8)
    return bnd  # {0,1}


def find_mask_for_image(masks_dir: Path, stem: str) -> Optional[Path]:
    """
    在 masks_dir 中根据图像的 stem 匹配掩膜文件，后缀不限。
    返回匹配到的第一个文件路径，未找到则返回 None。
    """
    patterns = [
        str(masks_dir / f"{stem}.*"),
        str(masks_dir / f"{stem}*.*"),  # 兼容可能带后缀标记的命名
    ]
    for pat in patterns:
        candidates = sorted(glob.glob(pat))
        if candidates:
            return Path(candidates[0])
    return None


def run(images_dir: str, masks_dir: str, out_dir: str, edge_width: int = 3, mode: str = "outer",
        overwrite: bool = True, verbose: bool = True):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 常见图像后缀
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    img_paths = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not img_paths:
        print(f"[WARN] No images found in {images_dir}", file=sys.stderr)

    skipped_no_mask = 0
    written = 0

    for ip in sorted(img_paths):
        stem = ip.stem
        op = out_dir / ip.name  # 保持与原图完全相同的“文件名+后缀”

        if op.exists() and not overwrite:
            if verbose:
                print(f"[SKIP] exists: {op.name}")
            continue

        mp = find_mask_for_image(masks_dir, stem)
        if mp is None or (not mp.exists()):
            skipped_no_mask += 1
            if verbose:
                print(f"[MISS] mask not found for image: {ip.name}")
            continue

        # 读mask，宽容任何后缀
        mask = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if mask is None:
            skipped_no_mask += 1
            if verbose:
                print(f"[MISS] failed to read mask file: {mp}")
            continue

        # 转为二值 {0,1}
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_bin = (mask > 127).astype(np.uint8)

        edge = make_edge_from_mask(mask_bin, edge_width=edge_width, mode=mode)
        cv2.imwrite(str(op), (edge * 255).astype(np.uint8))
        written += 1
        if verbose:
            print(f"[OK] {ip.name}  <=  {mp.name}  ->  {op.name}")

    print(f"\nDone. Written: {written}, Missing masks: {skipped_no_mask}, Out: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build edge labels from masks, saved with the SAME names as images.")
    ap.add_argument("--images_dir", required=True, help="原始训练图像目录（以此目录的文件名与后缀为准）")
    ap.add_argument("--masks_dir", required=True, help="对应的mask目录（按同stem匹配，后缀不限）")
    ap.add_argument("--out_dir", required=True, help="输出edge标签目录（文件名与images_dir一致）")
    ap.add_argument("--edge_width", type=int, default=3, help="边缘加粗像素，建议2~3")
    ap.add_argument("--mode", type=str, default="outer", choices=["outer", "inner", "thick"], help="边界类型")
    ap.add_argument("--no_overwrite", action="store_true", help="存在同名文件时不覆盖")
    ap.add_argument("--quiet", action="store_true", help="静默输出")
    args = ap.parse_args()

    run(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        out_dir=args.out_dir,
        edge_width=args.edge_width,
        mode=args.mode,
        overwrite=(not args.no_overwrite),
        verbose=(not args.quiet),
    )
