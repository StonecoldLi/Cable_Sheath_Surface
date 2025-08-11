import argparse
import json
from pathlib import Path
import cv2
import numpy as np


# ========== I/O 与点提取 ==========

def imread_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def imread_color(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

def coords_from_inlier_mask(mask_img):
    """mask中非零像素 -> (N,2) 点集 (x,y)"""
    ys, xs = np.where(mask_img > 0)
    if xs.size == 0:
        return np.zeros((0,2), dtype=np.float32)
    return np.stack([xs, ys], axis=1).astype(np.float32)

def coords_from_green(clean_vis_bgr, g_min=150, g_margin=40):
    """
    从可视化图中抽取绿色点：G高且 G - max(R,B) 足够大
    """
    b,g,r = cv2.split(clean_vis_bgr)
    max_rb = cv2.max(r, b)
    mask = (g >= g_min) & ((g.astype(np.int16) - max_rb.astype(np.int16)) >= g_margin)
    ys, xs = np.where(mask)
    if xs.size == 0:
        return np.zeros((0,2), dtype=np.float32)
    return np.stack([xs, ys], axis=1).astype(np.float32)

# ========== 直线工具 ==========

def line_from_two_points(p1, p2):
    """两点式 -> 一般式 ax+by+c=0，|[a,b]|=1"""
    x1,y1 = p1; x2,y2 = p2
    dx, dy = x2-x1, y2-y1
    a, b = dy, -dx
    n = np.hypot(a,b) + 1e-12
    a, b = a/n, b/n
    c = -(a*x1 + b*y1)
    return (a,b,c)

def point_line_distance(pts, line_abc):
    a,b,c = line_abc
    return np.abs(a*pts[:,0] + b*pts[:,1] + c)

def tls_refit(pts):
    """总最小二乘：PCA主方向 -> 一般式 ax+by+c=0，|[a,b]|=1"""
    mu = pts.mean(axis=0)
    X = pts - mu
    _,_,Vt = np.linalg.svd(X, full_matrices=False)
    v = Vt[0]
    a,b = v[1], -v[0]
    n = np.hypot(a,b) + 1e-12
    a,b = a/n, b/n
    c = -(a*mu[0] + b*mu[1])
    return (a,b,c)

def ransac_line(pts, tol=3.0, iters=1000, min_inliers=100, seed=42):
    if pts.shape[0] < 2:
        return None, np.array([], dtype=np.int32)
    rng = np.random.default_rng(seed)
    best_idx = np.array([], dtype=np.int32)
    best_line = None
    idx_all = np.arange(pts.shape[0])

    for _ in range(iters):
        i1, i2 = rng.choice(pts.shape[0], size=2, replace=False)
        if np.allclose(pts[i1], pts[i2]):
            continue
        line = line_from_two_points(pts[i1], pts[i2])
        d = point_line_distance(pts, line)
        idx = idx_all[d <= tol]
        if idx.size > best_idx.size:
            best_idx = idx
            best_line = line

    if best_line is None or best_idx.size < min_inliers:
        return None, np.array([], dtype=np.int32)

    line_refined = tls_refit(pts[best_idx])
    return line_refined, best_idx

def extend_line_to_image(line_abc, w, h):
    """把 ax+by+c=0 延长到图像边框，返回两个端点"""
    a,b,c = line_abc
    pts = []

    def valid(x,y):
        return -1e-6 <= x <= w-1+1e-6 and -1e-6 <= y <= h-1+1e-6

    # 与 x=0, x=w-1
    for x in [0, w-1]:
        if abs(b) > 1e-9:
            y = -(a*x + c)/b
            if valid(x,y): pts.append((float(x), float(y)))
    # 与 y=0, y=h-1
    for y in [0, h-1]:
        if abs(a) > 1e-9:
            x = -(b*y + c)/a
            if valid(x,y): pts.append((float(x), float(y)))

    if len(pts) < 2:
        # 兜底：几乎水平或竖直
        if abs(a) < 1e-9:
            y = -c/b
            pts = [(0.0, y), (w-1.0, y)]
        elif abs(b) < 1e-9:
            x = -c/a
            pts = [(x, 0.0), (x, h-1.0)]

    if len(pts) > 2:
        P = np.array(pts, dtype=np.float32)
        D = np.sum((P[None,...]-P[:,None,:])**2, axis=-1)
        i,j = np.unravel_index(np.argmax(D), D.shape)
        pts = [tuple(P[i]), tuple(P[j])]
    return pts[0], pts[1]

def angle_between_lines(l1, l2):
    """两条线的夹角（弧度，0~pi/2），用方向向量 (b,-a)"""
    a1,b1,_ = l1; a2,b2,_ = l2
    v1 = np.array([b1, -a1], dtype=np.float64)
    v2 = np.array([b2, -a2], dtype=np.float64)
    v1/=np.linalg.norm(v1)+1e-12; v2/=np.linalg.norm(v2)+1e-12
    cos = np.clip(np.abs(np.dot(v1,v2)), 0.0, 1.0)
    return float(np.arccos(cos))  # [0, pi/2]

def median_distance_between_lines(pts, line):
    """点集到某条线的中位距离"""
    return float(np.median(point_line_distance(pts, line)))


# ========== 主流程 ==========

def fit_two_main_lines(pts, w, h,
                       tol1=3.0, tol2=None,
                       iters=1000,
                       min_inliers_ratio=0.0005,
                       min_angle_sep_deg=3.0,
                       min_gap_px=5.0):
    """
    在点集上拟合两条主线。
    - 先拟合第一条线；剔除其内点；再拟合第二条
    - 对第二条施加：与第一条夹角 >= min_angle_sep_deg 或两线之间的“距离间隔”足够
    """
    if pts.shape[0] < 20:
        raise RuntimeError("点太少，无法拟合。")

    min_inliers = max(50, int(min_inliers_ratio * w * h))
    if tol2 is None: tol2 = tol1

    # 第1条
    line1, in1 = ransac_line(pts, tol=tol1, iters=iters, min_inliers=min_inliers, seed=123)
    if line1 is None:
        raise RuntimeError("第一条直线拟合失败。")
    p1a, p1b = extend_line_to_image(line1, w, h)

    # 剩余点
    mask = np.ones(pts.shape[0], dtype=bool)
    mask[in1] = False
    pts2 = pts[mask]
    if pts2.shape[0] < 20:
        raise RuntimeError("剔除第1条内点后点过少。")

    # 第2条（带分离约束的“最佳”）
    best = None
    best_idx = None
    idx_all = np.arange(pts2.shape[0])
    rng = np.random.default_rng(456)
    for _ in range(iters):
        # 随机采样两点
        i1, i2 = rng.choice(pts2.shape[0], size=2, replace=False)
        if np.allclose(pts2[i1], pts2[i2]): continue
        cand = line_from_two_points(pts2[i1], pts2[i2])
        # 与第1条的夹角或“间隔”检查
        ang = angle_between_lines(line1, cand) * 180/np.pi
        sep_ok = (ang >= min_angle_sep_deg)
        if not sep_ok:
            # 方向近似平行时，要求与第1条有足够“间隔”
            # 用第2批点到line1的中位距离近似两线间隔
            gap = median_distance_between_lines(pts2, line1)
            sep_ok = (gap >= min_gap_px)
        if not sep_ok:
            continue

        d = point_line_distance(pts2, cand)
        idx = idx_all[d <= tol2]
        if best is None or idx.size > best_idx.size:
            best = cand
            best_idx = idx

    if best is None or best_idx.size < max(30, min_inliers//2):
        # 回退：直接RANSAC不加分离约束
        line2, in2 = ransac_line(pts2, tol=tol2, iters=iters, min_inliers=max(30, min_inliers//2), seed=789)
    else:
        # 用内点细化
        line2 = tls_refit(pts2[best_idx])
        in2 = best_idx

    if line2 is None:
        raise RuntimeError("第二条直线拟合失败。")
    p2a, p2b = extend_line_to_image(line2, w, h)

    return (line1, p1a, p1b, in1), (line2, p2a, p2b, in2), mask


def main():
    ap = argparse.ArgumentParser(description="Fit two main lines from green inlier points.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--inlier_mask", help="上一步导出的内点二值图（白=内点）")
    g.add_argument("--clean_vis", help="上一步导出的可视化图（绿=内点，红=外点）")
    ap.add_argument("--save_vis", required=True, help="拟合结果可视化输出路径")
    ap.add_argument("--save_json", required=True, help="拟合参数JSON输出路径")
    ap.add_argument("--bg", help="可选：用于可视化的背景图（如原NMS或原图）；不提供则用黑底")

    # 从绿色图提点时的阈值
    ap.add_argument("--g_min", type=int, default=150)
    ap.add_argument("--g_margin", type=int, default=40)

    # RANSAC & 约束
    ap.add_argument("--ransac_tol", type=float, default=3.0)
    ap.add_argument("--ransac_iter", type=int, default=1000)
    ap.add_argument("--min_inliers_ratio", type=float, default=0.0005)
    ap.add_argument("--min_angle_sep_deg", type=float, default=3.0)
    ap.add_argument("--min_gap_px", type=float, default=5.0)

    args = ap.parse_args()

    # 读点
    if args.inlier_mask:
        m = imread_gray(args.inlier_mask)
        h, w = m.shape[:2]
        pts = coords_from_inlier_mask(m)
        bg = imread_color(args.bg) if args.bg else np.zeros((h, w, 3), np.uint8)
    else:
        vis = imread_color(args.clean_vis)
        h, w = vis.shape[:2]
        pts = coords_from_green(vis, g_min=args.g_min, g_margin=args.g_margin)
        bg = imread_color(args.bg) if args.bg else np.zeros((h, w, 3), np.uint8)

    if pts.shape[0] < 20:
        raise RuntimeError("绿色内点太少，请检查上一步阈值/过滤参数。")

    # 拟合两条主线
    (l1, p1a, p1b, in1), (l2, p2a, p2b, in2), mask = fit_two_main_lines(
        pts, w, h,
        tol1=args.ransac_tol, tol2=args.ransac_tol,
        iters=args.ransac_iter,
        min_inliers_ratio=args.min_inliers_ratio,
        min_angle_sep_deg=args.min_angle_sep_deg,
        min_gap_px=args.min_gap_px
    )

    # 可视化：背景 + 所有绿色点(浅绿) + 内点(深绿) + 两条线
    vis = bg.copy()
    # 所有点
    for x,y in pts:
        cv2.circle(vis, (int(x), int(y)), 1, (80, 200, 80), -1)
    # 第1条内点
    for x,y in pts[in1]:
        cv2.circle(vis, (int(x), int(y)), 1, (0, 255, 0), -1)
    # 第2条内点（剔除第1条后的索引）
    rest = np.where(mask)[0]
    for x,y in pts[rest][in2]:
        cv2.circle(vis, (int(x), int(y)), 1, (0, 255, 160), -1)

    # 画线
    cv2.line(vis, (int(p1a[0]), int(p1a[1])), (int(p1b[0]), int(p1b[1])), (0, 0, 255), 3)
    cv2.line(vis, (int(p2a[0]), int(p2a[1])), (int(p2b[0]), int(p2b[1])), (255, 0, 0), 3)

    # 保存
    Path(args.save_vis).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.save_vis, vis)

    out = {
        "image_size": {"w": int(w), "h": int(h)},
        "line1": {
            "abc": [float(l1[0]), float(l1[1]), float(l1[2])],
            "p1": [float(p1a[0]), float(p1a[1])],
            "p2": [float(p1b[0]), float(p1b[1])],
            "inliers": int(len(in1))
        },
        "line2": {
            "abc": [float(l2[0]), float(l2[1]), float(l2[2])],
            "p1": [float(p2a[0]), float(p2a[1])],
            "p2": [float(p2b[0]), float(p2b[1])],
            "inliers": int(len(in2))
        }
    }
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved: {args.save_vis}")
    print(f"Saved: {args.save_json}")
    print("Line1 (a,b,c) =", out["line1"]["abc"])
    print("Line2 (a,b,c) =", out["line2"]["abc"])


if __name__ == "__main__":
    main()
