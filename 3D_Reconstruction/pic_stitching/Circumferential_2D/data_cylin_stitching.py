# stitch_sequential_nofeather.py
# 用法：
#   python stitch_sequential_nofeather.py \
#       --indir ./out_step1 \
#       --theta1 0.20 --theta2 0.20 --theta3 0.20 \
#       --outdir ./out_step2_no_feather
#
# 生成：
#   out_step2_no_feather/step1_cam0_over_cam90.png
#   out_step2_no_feather/step2_prev_over_cam180.png
#   out_step2_no_feather/step3_prev_over_cam270.png
#   out_step2_no_feather/final_panorama.png

import argparse
from pathlib import Path
import cv2
import numpy as np

# ---------- 基础工具 ----------
def imread_any(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img

def ensure_rgba(img: np.ndarray) -> np.ndarray:
    """转成 BGRA（OpenCV 通道顺序），alpha=255 表示不透明。"""
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        a = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
        return np.concatenate([bgr, a], axis=2)
    if img.shape[2] == 4:
        return img.copy()
    return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

def pad_to_height(img_rgba: np.ndarray, H_out: int, valign: str = "center"):
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

def alpha_over(dst_rgba: np.ndarray, src_rgba: np.ndarray) -> np.ndarray:
    """
    Porter-Duff 'over'：src 覆盖在 dst 上。
    输入输出：BGRA, uint8
    """
    dst = dst_rgba.astype(np.float32) / 255.0
    src = src_rgba.astype(np.float32) / 255.0

    ad = dst[..., 3:4]
    as_ = src[..., 3:4]
    cd = dst[..., :3]
    cs = src[..., :3]

    out_a = as_ + ad * (1.0 - as_)
    out_c = cs * as_ + cd * ad * (1.0 - as_)

    out = np.concatenate([out_c, out_a], axis=2)
    out = np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)
    return out

# ---------- 无羽化拼接：左覆盖右 ----------
def stitch_left_over_right(prev_rgba: np.ndarray, next_rgba: np.ndarray,
                           overlap_ratio: float = 0.2,
                           valign: str = "center") -> np.ndarray:
    """
    将 prev 的“左侧”按 overlap_ratio（相对两图较小宽度）覆盖 next 的“右侧”，无羽化。
    返回拼接后的 RGBA。
    """
    A = ensure_rgba(prev_rgba)   # 上一张
    B = ensure_rgba(next_rgba)   # 后一张
    Ha, Wa = A.shape[:2]
    Hb, Wb = B.shape[:2]

    # 垂直对齐
    H = max(Ha, Hb)
    A_pad, _ = pad_to_height(A, H, valign)
    B_pad, _ = pad_to_height(B, H, valign)

    # 计算重叠宽度
    overlap_ratio = float(max(0.0, min(1.0, overlap_ratio)))
    ovl = int(round(overlap_ratio * min(Wa, Wb)))
    ovl = max(0, min(ovl, min(Wa, Wb)))

    if ovl == 0:
        # 无重叠：直接拼边（next 在左，prev 在右）
        out_w = Wb + Wa
        out = np.zeros((H, out_w, 4), dtype=np.uint8)
        out[:, 0:Wb] = B_pad
        out[:, Wb:Wb+Wa] = A_pad
        return out

    # 有重叠：
    # 输出宽度 = next 全部 + prev 去掉其左边重叠那部分（因为叠到 next 上）
    out_w = Wb + Wa - ovl
    out = np.zeros((H, out_w, 4), dtype=np.uint8)

    # 1) 先放 next（完整）
    out[:, 0:Wb] = B_pad

    # 2) 在重叠区：把 prev 的左侧 ovl 列“覆盖”到 next 的右侧 ovl 列
    #    next 的右侧 ovl 区域位于 out 的列 [Wb-ovl, Wb)
    dst_region = out[:, Wb-ovl:Wb]
    src_region = A_pad[:, 0:ovl]
    out[:, Wb-ovl:Wb] = alpha_over(dst_region, src_region)

    # 3) prev 的剩余右侧非重叠部分，紧接着放到 out 右边
    right_w = Wa - ovl
    if right_w > 0:
        out[:, Wb:Wb+right_w] = A_pad[:, ovl:ovl+right_w]

    return out

# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser(description="顺序拼接（无羽化）：左覆盖右，三次覆盖比例 theta1/2/3")
    ap.add_argument("--indir", type=str, default="./out_step1", help="输入目录（四张 *_crop.png 所在处）")
    ap.add_argument("--cam0", type=str, default="cam0_crop.png")
    ap.add_argument("--cam90", type=str, default="cam90_crop.png")
    ap.add_argument("--cam180", type=str, default="cam180_crop.png")
    ap.add_argument("--cam270", type=str, default="cam270_crop.png")

    ap.add_argument("--theta1", type=float, default=0.20, help="cam0 左侧覆盖 cam90 右侧 的重叠比例 [0,1]")
    ap.add_argument("--theta2", type=float, default=0.20, help="(cam0⊕cam90) 左侧覆盖 cam180 右侧 的重叠比例 [0,1]")
    ap.add_argument("--theta3", type=float, default=0.20, help="(前两步结果) 左侧覆盖 cam270 右侧 的重叠比例 [0,1]")

    ap.add_argument("--valign", choices=["center", "top", "bottom"], default="center", help="垂直对齐方式")
    ap.add_argument("--outdir", type=str, default="./out_step2_no_feather", help="输出目录")
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 读取四张裁切图（建议背景透明 PNG）
    cam0 = imread_any(indir / args.cam0)
    cam90 = imread_any(indir / args.cam90)
    cam180 = imread_any(indir / args.cam180)
    cam270 = imread_any(indir / args.cam270)

    # Step 1: cam0 ⊕ cam90（cam0 的左侧覆盖 cam90 的右侧）
    step1 = stitch_left_over_right(cam0, cam90, overlap_ratio=args.theta1, valign=args.valign)
    p1 = outdir / "step1_cam0_over_cam90.png"
    cv2.imwrite(str(p1), step1)
    print(f"[STEP1] -> {p1.name}  (theta1={args.theta1})")

    # Step 2: (step1) ⊕ cam180（整体左侧覆盖 cam180 右侧）
    step2 = stitch_left_over_right(step1, cam180, overlap_ratio=args.theta2, valign=args.valign)
    p2 = outdir / "step2_prev_over_cam180.png"
    cv2.imwrite(str(p2), step2)
    print(f"[STEP2] -> {p2.name}  (theta2={args.theta2})")

    # Step 3: (step2) ⊕ cam270（整体左侧覆盖 cam270 右侧）
    step3 = stitch_left_over_right(step2, cam270, overlap_ratio=args.theta3, valign=args.valign)
    p3 = outdir / "step3_prev_over_cam270.png"
    cv2.imwrite(str(p3), step3)
    print(f"[STEP3] -> {p3.name}  (theta3={args.theta3})")

    # 最终结果
    pf = outdir / "final_panorama.png"
    cv2.imwrite(str(pf), step3)
    print(f"[DONE] Final -> {pf}")

if __name__ == "__main__":
    main()
