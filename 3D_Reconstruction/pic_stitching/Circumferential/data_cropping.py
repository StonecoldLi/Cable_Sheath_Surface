# step1_cut_by_mask_and_alpha.py
# 用法示例：
#   python step1_cut_by_mask_and_alpha.py --indir ./data/ring1
# 产物：
#   ./temp/*_mask_with_lines.png      # 切分线画在 mask 上的可视化
#   ./out_step1/*_crop.png            # 裁剪后的 PNG（背景透明）

import argparse
from pathlib import Path
import re
import cv2
import numpy as np

# ---------- 工具 ----------
def load_image_any(path: Path) -> np.ndarray:
    """按原通道数读取图像（支持 alpha）。"""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img

def load_mask01(path: Path, thresh: int = 128) -> np.ndarray:
    """
    读取 mask，返回 0/1 的 uint8 二值图。
    约定：白=前景，黑=背景。
    """
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    mask = (m >= thresh).astype(np.uint8)
    # 小闭运算，去孔洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask

def first_foreground_span(mask_row: np.ndarray, min_span: int = 8):
    """返回该行前景的最左、最右索引及宽度；若无/太窄，返回 None。"""
    xs = np.where(mask_row > 0)[0]
    if xs.size == 0:
        return None
    left, right = int(xs[0]), int(xs[-1])
    width = right - left + 1
    if width < min_span:
        return None
    return left, right, width

def top_bottom_spans(mask01: np.ndarray, min_span: int = 8):
    """
    从上到下、从下到上分别找到第一条有效前景行及其左右边界。
    返回 (top_y, (L,R,W)), (bot_y, (L,R,W))；找不到则为 None。
    """
    H, _ = mask01.shape
    top = None
    for y in range(H):
        span = first_foreground_span(mask01[y], min_span=min_span)
        if span is not None:
            top = (y, span)
            break
    bot = None
    for y in range(H - 1, -1, -1):
        span = first_foreground_span(mask01[y], min_span=min_span)
        if span is not None:
            bot = (y, span)
            break
    return top, bot

def draw_lines_on_mask(mask01: np.ndarray, xL: int, xR: int) -> np.ndarray:
    """在 mask 上画两条垂直切分线，返回可视化 BGR。"""
    H, W = mask01.shape
    vis = cv2.cvtColor((mask01 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    x0, x1 = max(0, min(xL, xR)), min(W - 1, max(xL, xR))
    cv2.line(vis, (x0, 0), (x0, H - 1), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(vis, (x1, 0), (x1, H - 1), (0, 255, 255), 2, cv2.LINE_AA)
    return vis

def ensure_rgba(img: np.ndarray) -> np.ndarray:
    """把任意图像转换为 RGBA（保持原 RGB/BGR 颜色不变）。"""
    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        a = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
        return np.concatenate([rgb, a], axis=2)
    if img.shape[2] == 4:
        return img.copy()
    # BGR -> BGRA，alpha=255
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return bgra

def apply_alpha_by_mask(crop_img: np.ndarray, crop_mask01: np.ndarray) -> np.ndarray:
    """
    以 crop_mask01（0/1）控制透明度：前景 alpha=255，背景 alpha=0。
    输入 crop_img 可以是任意通道图，输出为 BGRA。
    """
    out = ensure_rgba(crop_img)
    # 若是 BGRA（OpenCV），通道顺序是 B,G,R,A
    out[:, :, 3] = (crop_mask01 * 255).astype(np.uint8)
    return out

# ---------- 配对与命名 ----------
IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

def base_from_mask_name(mask_path: Path) -> str:
    """
    从 mask 文件名推断基名：
    cam0_mask.png -> cam0
    cam270mask.png -> cam270
    """
    stem = mask_path.stem.lower()
    # 去掉尾部的 '_mask' / '-mask' / 'mask'
    stem = re.sub(r'[_\-]?mask$', '', stem, flags=re.IGNORECASE)
    return stem

def find_matching_image(indir: Path, base: str) -> Path:
    """
    寻找与 base 对应的“原图”（优先 camX.*；若找不到，最后兜底 camX_fg.*）。
    """
    # 优先：严格同名（不带后缀）
    for ext in IMG_EXTS:
        p = indir / f"{base}{ext}"
        if p.exists():
            return p
    # 容错：常见前/后缀
    patterns = [f"{base}.*", f"{base}_orig.*", f"{base}-orig.*"]
    for pat in patterns:
        for p in indir.glob(pat):
            if p.suffix.lower() in IMG_EXTS and ("mask" not in p.stem.lower()):
                return p
    # 兜底：不推荐，但若只有 *_fg.*，也用它
    for ext in IMG_EXTS:
        p = indir / f"{base}_fg{ext}"
        if p.exists():
            print(f"[WARN] 未找到原图，使用前景图代替：{p.name}")
            return p
    # 再兜底：任意以 base 开头但不是 mask 的图片
    for p in indir.glob(f"{base}*"):
        if p.suffix.lower() in IMG_EXTS and ("mask" not in p.stem.lower()):
            print(f"[WARN] 未严格匹配到原图，使用：{p.name}")
            return p
    raise FileNotFoundError(f"找不到与 {base} 匹配的原图。")

# ---------- 主流程（单张） ----------
def process_pair(mask_path: Path, indir: Path, tempdir: Path, outdir: Path,
                 min_span: int = 8, edge_pad: int = 0):
    """
    1) 读取 mask -> 二值
    2) 在 mask 中找上/下边界第一条有效行；比较宽度，取更短侧的左右边界作为切分线
    3) 在 mask 上画出切分线（保存到 ./temp）
    4) 在“原图”上按这两条垂直线裁剪；裁剪区域内的背景（mask=0）透明化，保存到 ./out_step1
    """
    mask01 = load_mask01(mask_path)
    H, W = mask01.shape

    top, bot = top_bottom_spans(mask01, min_span=min_span)
    if top is None and bot is None:
        print(f"[WARN] {mask_path.name}: 未找到有效前景行，跳过。")
        return

    # 选择更短的一侧
    if top is not None and bot is not None:
        _, (_, _, w_top) = top
        _, (_, _, w_bot) = bot
        chosen = top if w_top <= w_bot else bot
        chosen_side = "TOP" if chosen is top else "BOTTOM"
    elif top is not None:
        chosen, chosen_side = top, "TOP"
    else:
        chosen, chosen_side = bot, "BOTTOM"

    y_row, (xL, xR, w) = chosen

    # 可选扩展/收缩
    if edge_pad != 0:
        xL = max(0, xL - edge_pad)
        xR = min(W - 1, xR + edge_pad)

    # 可视化：切分线画在 mask 上
    vis = draw_lines_on_mask(mask01, xL, xR)
    tempdir.mkdir(parents=True, exist_ok=True)
    vis_path = tempdir / f"{mask_path.stem}_mask_with_lines.png"
    cv2.imwrite(str(vis_path), vis)
    print(f"[TEMP] {vis_path.name}: side={chosen_side}, row={y_row}, lines=({xL},{xR})")

    # 在原图上裁剪，并按 mask 透明化
    base = base_from_mask_name(mask_path)
    img_path = find_matching_image(indir, base)
    img = load_image_any(img_path)

    # 尺寸对齐（若原图与 mask 尺寸不同，按 mask 尺寸对齐原图）
    if img.shape[0] != H or img.shape[1] != W:
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        print(f"[INFO] {img_path.name}: 尺寸与 mask 不同，已重采样到 {W}x{H}")

    # 裁剪
    crop_img = img[:, xL:xR + 1]
    crop_mask = mask01[:, xL:xR + 1]

    # 背景透明化
    crop_rgba = apply_alpha_by_mask(crop_img, crop_mask)

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{base}_crop.png"
    cv2.imwrite(str(out_path), crop_rgba)
    print(f"[SAVE] {out_path.name}: size={crop_rgba.shape[1]}x{crop_rgba.shape[0]} (BG透明)")

# ---------- 入口 ----------
def main():
    ap = argparse.ArgumentParser(description="基于 mask 的垂直切分并在原图裁剪，背景透明化")
    ap.add_argument("--indir", type=str, default=".", help="输入目录（原图与 mask 同放）")
    ap.add_argument("--tempdir", type=str, default="./temp", help="切分线可视化输出目录")
    ap.add_argument("--outdir", type=str, default="./out_step1", help="裁剪结果输出目录")
    ap.add_argument("--min-span", type=int, default=8, help="视为有效前景行的最小宽度（像素）")
    ap.add_argument("--edge-pad", type=int, default=0, help="切分线左右各扩展的像素（负值表示收缩）")
    ap.add_argument("--mask-pattern", type=str, default="*mask*.png",
                    help="mask 文件匹配模式（默认 *mask*.png）")
    args = ap.parse_args()

    indir = Path(args.indir)
    tempdir = Path(args.tempdir)
    outdir = Path(args.outdir)

    mask_files = sorted(indir.glob(args.mask_pattern))
    if not mask_files:
        print(f"[ERR] 在 {indir} 下未找到匹配 {args.mask_pattern} 的 mask。")
        return

    for m in mask_files:
        try:
            process_pair(m, indir, tempdir, outdir, min_span=args.min_span, edge_pad=args.edge_pad)
        except Exception as e:
            print(f"[ERR] 处理 {m.name} 失败：{e}")

if __name__ == "__main__":
    main()
