# stitch_sequential_overlay.py
# 依次拼接：
#   1) cam0_crop(左侧) 覆盖 cam90_crop(右侧) —— 覆盖系数=theta1
#   2) [上一步结果](左侧) 覆盖 cam180_crop(右侧) —— 覆盖系数=theta2
#   3) [上一步结果](左侧) 覆盖 cam270_crop(右侧) —— 覆盖系数=theta3
#
# 用法示例：
#   python stitch_sequential_overlay.py \
#       --indir ./out_step1 \
#       --theta1 0.20 --theta2 0.20 --theta3 0.20 \
#       --outdir ./out_step2
#
# 产生文件：
#   out_step2/step1_cam0_over_cam90.png
#   out_step2/step2_prev_over_cam180.png
#   out_step2/step3_prev_over_cam270.png
#   out_step2/final_panorama.png

import argparse
from pathlib import Path
import cv2
import numpy as np

# ---------- 基础工具 ----------
def imread_any(path):
    """按原通道数读取（保留 alpha）。"""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img

def ensure_rgba(img):
    """转成 BGRA（OpenCV 通道顺序），alpha=255 表示不透明。"""
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        a = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
        return np.concatenate([bgr, a], axis=2)
    if img.shape[2] == 4:
        return img.copy()
    return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

def pad_to_height(img_rgba, H_out, valign="center"):
    """把 RGBA 图按 valign 垂直对齐放到高为 H_out 的透明画布上。"""
    h, w = img_rgba.shape[:2]
    canvas = np.zeros((H_out, w, 4), dtype=np.uint8)
    if valign == "top":
        y0 = 0
    elif valign == "bottom":
        y0 = H_out - h
    else:
        y0 = (H_out - h) // 2
    canvas[y0:y0+h, :w] = img_rgba
    return canvas, y0

def cosine_ramp(n, reverse=False):
    """长度为 n 的半余弦权重 0→1（reverse=True 为 1→0）。"""
    if n <= 1:
        w = np.array([1.0], dtype=np.float32)
    else:
        x = np.linspace(0.0, 1.0, n, dtype=np.float32)
        w = 0.5 - 0.5 * np.cos(np.pi * x)
    if reverse:
        w = w[::-1].copy()
    return w

def blend_overlap_patch(A_patch, B_patch, wA_cols):
    """
    在“已经对齐高度”的重叠区域内进行列方向加权融合。
    A_patch、B_patch: (H, Wovl, 4) 预先居中放置好的 BGRA
    wA_cols: (Wovl,)  每列 A 的权重，B 的权重 = 1 - wA_cols
    返回: (H, Wovl, 4) BGRA
    """
    H, Wovl = A_patch.shape[:2]
    # 归一化到 float，预乘 alpha
    A = A_patch.astype(np.float32) / 255.0
    B = B_patch.astype(np.float32) / 255.0
    aA = A[..., 3:4]
    aB = B[..., 3:4]
    cA = A[..., :3] * aA
    cB = B[..., :3] * aB

    wA = wA_cols.reshape(1, Wovl, 1).astype(np.float32)
    wB = 1.0 - wA

    c = wA * cA + wB * cB
    a = wA * aA + wB * aB
    # 防止除零
    out_rgb = np.where(a > 1e-6, c / a, 0.0)
    out_a = np.clip(a, 0.0, 1.0)

    out = np.concatenate([out_rgb, out_a], axis=2)
    out = np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)
    return out

def stitch_left_over_right(prev_rgba, next_rgba, overlap_ratio=0.2,
                           valign="center", feather="cosine"):
    """
    将 prev 的“左侧”覆盖 next 的“右侧”，按 overlap_ratio 控制重叠比例。
    返回拼接结果 RGBA。
    """
    A = ensure_rgba(prev_rgba)
    B = ensure_rgba(next_rgba)
    Ha, Wa = A.shape[:2]
    Hb, Wb = B.shape[:2]

    H = max(Ha, Hb)
    A_pad, ya = pad_to_height(A, H, valign)
    B_pad, yb = pad_to_height(B, H, valign)

    # 计算重叠宽度
    ovl = int(round(float(overlap_ratio) * min(Wa, Wb)))
    ovl = max(0, min(ovl, min(Wa, Wb)))  # 保证在范围内

    if ovl == 0:
        # 无重叠：直接拼边
        out_w = Wb + Wa
        out = np.zeros((H, out_w, 4), dtype=np.uint8)
        # 先放 next（左边）
        out[:, 0:Wb] = B_pad
        # 再放 prev（右边）
        out[:, Wb:Wb+Wa] = A_pad
        return out

    # 有重叠：next 放左边，prev 左侧 ovl 列覆盖到 next 的右侧 ovl 列
    xA = Wb - ovl                 # prev 在输出中的起始列
    out_w = xA + Wa               # B 左非重叠 + 重叠 + A 右非重叠

    out = np.zeros((H, out_w, 4), dtype=np.uint8)

    # 1) B 的左非重叠部分 [0, xA)
    if xA > 0:
        out[:, 0:xA] = B_pad[:, 0:xA]

    # 2) 重叠区 [xA, xA+ovl)
    A_ovl = A_pad[:, 0:ovl]                  # A 的左侧 ovl 列
    B_ovl = B_pad[:, Wb-ovl:Wb]              # B 的右侧 ovl 列

    if feather == "linear":
        wA_cols = np.linspace(1.0, 0.0, ovl, dtype=np.float32)
    else:  # "cosine"（默认）
        wA_cols = cosine_ramp(ovl, reverse=False)  # 左侧=1 → 右侧=0

    blended = blend_overlap_patch(A_ovl, B_ovl, wA_cols)
    out[:, xA:xA+ovl] = blended

    # 3) A 的右非重叠部分 [xA+ovl, end)
    right_w = Wa - ovl
    if right_w > 0:
        out[:, xA+ovl:xA+ovl+right_w] = A_pad[:, ovl:ovl+right_w]

    return out

# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser(description="顺序拼接：左覆盖右，三次覆盖比例分别为 theta1/2/3")
    ap.add_argument("--indir", type=str, default="./out_step1", help="输入目录（四张 *_crop.png 所在处）")
    ap.add_argument("--cam0", type=str, default="cam0_crop.png")
    ap.add_argument("--cam90", type=str, default="cam90_crop.png")
    ap.add_argument("--cam180", type=str, default="cam180_crop.png")
    ap.add_argument("--cam270", type=str, default="cam270_crop.png")
    ap.add_argument("--theta1", type=float, default=0.20, help="cam0 覆盖 cam90 的重叠比例 [0,1]")
    ap.add_argument("--theta2", type=float, default=0.20, help="(cam0⊕cam90) 覆盖 cam180 的重叠比例 [0,1]")
    ap.add_argument("--theta3", type=float, default=0.20, help="(前两步结果) 覆盖 cam270 的重叠比例 [0,1]")
    ap.add_argument("--valign", choices=["center", "top", "bottom"], default="center", help="垂直对齐方式")
    ap.add_argument("--feather", choices=["cosine", "linear"], default="cosine", help="重叠区的列权重类型")
    ap.add_argument("--outdir", type=str, default="./out_step2", help="输出目录")
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 读取四张裁切图（建议为 BG 透明的 PNG）
    cam0 = imread_any(indir / args.cam0)
    cam90 = imread_any(indir / args.cam90)
    cam180 = imread_any(indir / args.cam180)
    cam270 = imread_any(indir / args.cam270)

    # Step 1: cam0 ⊕ cam90
    step1 = stitch_left_over_right(cam0, cam90, overlap_ratio=args.theta1,
                                   valign=args.valign, feather=args.feather)
    p1 = outdir / "step1_cam0_over_cam90.png"
    cv2.imwrite(str(p1), step1)
    print(f"[STEP1] -> {p1.name}  (theta1={args.theta1})")

    # Step 2: (step1) ⊕ cam180
    step2 = stitch_left_over_right(step1, cam180, overlap_ratio=args.theta2,
                                   valign=args.valign, feather=args.feather)
    p2 = outdir / "step2_prev_over_cam180.png"
    cv2.imwrite(str(p2), step2)
    print(f"[STEP2] -> {p2.name}  (theta2={args.theta2})")

    # Step 3: (step2) ⊕ cam270
    step3 = stitch_left_over_right(step2, cam270, overlap_ratio=args.theta3,
                                   valign=args.valign, feather=args.feather)
    p3 = outdir / "step3_prev_over_cam270.png"
    cv2.imwrite(str(p3), step3)
    print(f"[STEP3] -> {p3.name}  (theta3={args.theta3})")

    # 最终结果
    pf = outdir / "final_panorama.png"
    cv2.imwrite(str(pf), step3)
    print(f"[DONE] Final -> {pf}")

if __name__ == "__main__":
    main()
