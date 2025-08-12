#改代码主要目的是确定边界由哪些边缘点进行拟合
import argparse
import cv2
import numpy as np
from pathlib import Path

def imread_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return img

def binarize_nms_range(nms, low_th=0.2, high_th=1.0):
    """返回二值mask：保留亮度在 [low_th, high_th] 之间的像素"""
    bw = ((nms >= low_th) & (nms <= high_th)).astype(np.uint8)
    return bw

def visualize_selected_points(nms, mask, point_color=(0, 0, 255), radius=1):
    """
    nms: 原始灰度图 (0-1)
    mask: 二值mask (0或1)
    在原图上标出mask==1的点
    """
    vis = (np.clip(nms, 0, 1) * 255).astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    ys, xs = np.where(mask > 0)
    for (x, y) in zip(xs, ys):
        cv2.circle(vis, (int(x), int(y)), radius, point_color, -1)
    return vis

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nms_path", required=True, help="NMS 图像路径（灰度/彩色均可）")
    ap.add_argument("--save_path", required=True, help="保存可视化结果的路径")
    ap.add_argument("--low_th", type=float, default=0.2, help="下阈值 (0~1)")
    ap.add_argument("--high_th", type=float, default=1.0, help="上阈值 (0~1)")
    ap.add_argument("--point_color", type=int, nargs=3, default=[0, 0, 255], help="标记点颜色 BGR")
    ap.add_argument("--radius", type=int, default=1, help="标记点半径")
    args = ap.parse_args()

    nms = imread_gray(args.nms_path)
    mask = binarize_nms_range(nms, args.low_th, args.high_th)
    vis = visualize_selected_points(nms, mask, tuple(args.point_color), radius=args.radius)

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.save_path, vis)
    print(f"已保存可视化结果到 {args.save_path}")
    print(f"选中点总数: {mask.sum()}")
