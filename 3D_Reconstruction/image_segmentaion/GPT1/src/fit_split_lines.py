import argparse
import json
from pathlib import Path
import cv2
import numpy as np


# ---------- I/O 与基础 ----------

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


def save_points_vis(bg_gray01, pts_xy, save_path, color=(0, 0, 255), radius=1, line=None, color_line=(0, 255, 255)):
    vis = (np.clip(bg_gray01, 0, 1) * 255).astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    if pts_xy is not None and len(pts_xy) > 0:
        for x, y in pts_xy.astype(int):
            cv2.circle(vis, (int(x), int(y)), radius, color, -1)
    if line is not None:
        (x1, y1), (x2, y2) = line
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), color_line, 2)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, vis)


# ---------- 选点（亮度区间 + 空间过滤） ----------

def binarize_nms_range(nms, low_th=0.2, high_th=1.0):
    return ((nms >= low_th) & (nms <= high_th)).astype(np.uint8)


def remove_small_components(bw, min_size=40):
    num, lab = cv2.connectedComponents(bw, connectivity=8)
    if num <= 1:
        return bw
    out = np.zeros_like(bw)
    for i in range(1, num):
        if (lab == i).sum() >= min_size:
            out[lab == i] = 1
    return out


def neighbor_filter(bw, k=2):
    """3x3 邻域（含自身）至少 k 个前景像素"""
    kernel = np.ones((3, 3), np.uint8)
    count = cv2.filter2D(bw.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return (count >= k).astype(np.uint8)


def select_points_from_nms(nms, low_th, high_th, min_cc, nbr_k):
    bw = binarize_nms_range(nms, low_th, high_th)
    bw = remove_small_components(bw, min_cc)
    bw = neighbor_filter(bw, nbr_k)
    ys, xs = np.where(bw > 0)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    return pts, bw


# ---------- 直线拟合与奇异点剔除 ----------

def line_from_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    a, b = dy, -dx
    norm = np.hypot(a, b) + 1e-12
    a, b = a / norm, b / norm
    c = -(a * x1 + b * y1)
    return a, b, c


def point_line_distance(pts, line_abc):
    a, b, c = line_abc
    return np.abs(a * pts[:, 0] + b * pts[:, 1] + c)


def tls_refit(pts):
    mu = pts.mean(axis=0)
    X = pts - mu
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    v = Vt[0]  # 方向向量
    # 方向 -> 一般式
    a, b = v[1], -v[0]
    norm = np.hypot(a, b) + 1e-12
    a, b = a / norm, b / norm
    c = -(a * mu[0] + b * mu[1])
    return (mu, v), (a, b, c)


def ransac_line(pts, tol=2.5, iters=800, min_inliers=80, seed=123):
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
        line = line_from_points(pts[i1], pts[i2])
        d = point_line_distance(pts, line)
        idx = idx_all[d <= tol]
        if idx.size > best_idx.size:
            best_idx = idx
            best_line = line
    if best_line is None or best_idx.size < min_inliers:
        return None, np.array([], dtype=np.int32)
    # 用内点 TLS 细化
    _, line_refined = tls_refit(pts[best_idx])
    return line_refined, best_idx


def extend_line_to_image(line_abc, w, h):
    a, b, c = line_abc
    pts = []

    def ok(x, y): return -1e-6 <= x <= w - 1 + 1e-6 and -1e-6 <= y <= h - 1 + 1e-6

    for x in [0, w - 1]:
        if abs(b) > 1e-9:
            y = -(a * x + c) / b
            if ok(x, y): pts.append((float(x), float(y)))
    for y in [0, h - 1]:
        if abs(a) > 1e-9:
            x = -(b * y + c) / a
            if ok(x, y): pts.append((float(x), float(y)))

    if len(pts) < 2:
        if abs(a) < 1e-9:  # 水平
            y = -c / b
            pts = [(0.0, y), (w - 1.0, y)]
        elif abs(b) < 1e-9:  # 垂直
            x = -c / a
            pts = [(x, 0.0), (x, h - 1.0)]

    if len(pts) > 2:
        P = np.array(pts, dtype=np.float32)
        d = np.sum((P[None, ...] - P[:, None, :]) ** 2, axis=-1)
        i, j = np.unravel_index(np.argmax(d), d.shape)
        pts = [tuple(P[i]), tuple(P[j])]
    return pts[0], pts[1]


def robust_clean_inliers(pts, line_abc, k_sigma=3.0, keep_longest=True, gap_q=0.98, min_span_px=30):
    """
    基于残差的鲁棒清洗 + 一维连续性约束：
      - 残差 r = 点到线距离；用 MAD 估计尺度 s = 1.4826 * median(|r - median(r)|)
      - 保留 |r - median(r)| <= k_sigma * s 的点
      - 将内点投影到直线方向上，按大间隔分段，只保留最长段（可关掉）
    """
    a, b, c = line_abc
    # 残差
    r = point_line_distance(pts, line_abc)
    med = np.median(r)
    mad = np.median(np.abs(r - med)) + 1e-12
    s = 1.4826 * mad
    # 初步内点
    mask = np.abs(r - med) <= (k_sigma * s + 1e-6)
    inliers = pts[mask]
    if inliers.shape[0] < 2:
        return inliers

    if not keep_longest:
        return inliers

    # 沿直线方向的一维投影，清除尾巴/零星簇：只保留最长连续段
    # 直线的方向向量（单位）
    # 注意：一般式法向量 n=(a,b)，方向向量 v 与其垂直，可取 v=(b,-a)
    v = np.array([b, -a], dtype=np.float64)
    v /= (np.linalg.norm(v) + 1e-12)

    t = inliers @ v  # 投影坐标
    order = np.argsort(t)
    t_sorted = t[order]
    # 用分位数估计“正常间隔”的上界，过大的间隔视为裂缝
    diffs = np.diff(t_sorted)
    if diffs.size == 0:
        return inliers
    thr_gap = np.quantile(diffs, gap_q)
    # 分段
    segs = []
    start = 0
    for i, d in enumerate(diffs, start=0):
        if d > thr_gap:
            segs.append((start, i))
            start = i + 1
    segs.append((start, len(t_sorted) - 1))
    # 选择跨度最长的段
    best = max(segs, key=lambda ij: (t_sorted[ij[1]] - t_sorted[ij[0]]))
    if (t_sorted[best[1]] - t_sorted[best[0]]) < min_span_px:
        # 若最长段也太短，则不做段筛
        return inliers
    keep_idx = order[best[0]:best[1] + 1]
    return inliers[keep_idx.argsort()]  # 恢复大致顺序


def fit_and_clean_one_line(pts_all, img_w, img_h,
                           ransac_tol=2.5, ransac_iter=800,
                           min_inliers=80, k_sigma=3.0,
                           keep_longest=True, gap_q=0.98, min_span_px=30,
                           seed=123):
    line, idx = ransac_line(pts_all, tol=ransac_tol, iters=ransac_iter,
                            min_inliers=min_inliers, seed=seed)
    if line is None:
        return None, np.zeros((0, 2), dtype=np.float32), None
    # RANSAC 内点做鲁棒清洗
    inliers_raw = pts_all[idx]
    inliers = robust_clean_inliers(inliers_raw, line, k_sigma, keep_longest, gap_q, min_span_px)
    # 再次 TLS 细化（可选）
    if inliers.shape[0] >= 2:
        _, line = tls_refit(inliers)
    p1, p2 = extend_line_to_image(line, img_w, img_h)
    return line, inliers, (p1, p2)


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser(description="Select points from NMS by brightness, remove outliers, fit 1~2 lines, and save results.")
    ap.add_argument("--nms_path", required=True)
    ap.add_argument("--out_prefix", required=True, help="输出前缀，如 out/sample，会生成多个文件")
    # 选点
    ap.add_argument("--low_th", type=float, default=0.05, help="亮度下阈值(0~1)")
    ap.add_argument("--high_th", type=float, default=0.40, help="亮度上阈值(0~1)")
    ap.add_argument("--min_cc", type=int, default=40, help="小连通域阈值")
    ap.add_argument("--nbr_k", type=int, default=2, help="3x3邻域最少邻居数")
    # RANSAC & 清洗
    ap.add_argument("--ransac_tol", type=float, default=2.5)
    ap.add_argument("--ransac_iter", type=int, default=800)
    ap.add_argument("--min_inliers_ratio", type=float, default=0.001, help="最小内点数占总像素比例")
    ap.add_argument("--k_sigma", type=float, default=3.0, help="MAD σ倍数阈")
    ap.add_argument("--keep_longest", type=int, default=1, help="是否保留投影最长连续段(1/0)")
    ap.add_argument("--gap_q", type=float, default=0.98, help="投影间隔分位数阈(0~1)")
    ap.add_argument("--min_span_px", type=float, default=30, help="连续段最小跨度(像素)")
    ap.add_argument("--two_lines", type=int, default=1, help="是否拟合两条线(1/0)")
    args = ap.parse_args()

    nms = imread_gray(args.nms_path)
    h, w = nms.shape

    # 1) 亮度+空间过滤 -> 候选点
    pts_all, mask_bw = select_points_from_nms(
        nms,
        low_th=args.low_th,
        high_th=args.high_th,
        min_cc=args.min_cc,
        nbr_k=args.nbr_k
    )
    if pts_all.shape[0] < 50:
        raise RuntimeError(f"有效候选点过少：{pts_all.shape[0]}。请调整阈值/过滤参数。")

    # 可视化候选点
    save_points_vis(nms, pts_all, args.out_prefix + "_selected.png", color=(0, 0, 255), radius=1)

    min_inliers = max(80, int(args.min_inliers_ratio * w * h))

    # 2) 第1条线：拟合 + MAD清洗 + 最长段筛选
    line1, inliers1, seg1 = fit_and_clean_one_line(
        pts_all, w, h,
        ransac_tol=args.ransac_tol, ransac_iter=args.ransac_iter,
        min_inliers=min_inliers, k_sigma=args.k_sigma,
        keep_longest=bool(args.keep_longest),
        gap_q=args.gap_q, min_span_px=args.min_span_px,
        seed=123
    )
    if line1 is None or inliers1.shape[0] < 20:
        raise RuntimeError("第1条线拟合失败或有效内点过少。")

    # 3) 第2条线（可选）：从剩余点中再拟合
    result = {
        "image_size": {"w": int(w), "h": int(h)},
        "line1": {
            "abc": [float(line1[0]), float(line1[1]), float(line1[2])],
            "p1": [float(seg1[0][0]), float(seg1[0][1])],
            "p2": [float(seg1[1][0]), float(seg1[1][1])],
            "clean_inliers": int(inliers1.shape[0])
        }
    }
    save_points_vis(nms, inliers1, args.out_prefix + "_cleaned_line1.png",
                    color=(0, 255, 0), radius=1, line=seg1, color_line=(0, 0, 255))

    if args.two_lines:
        # 从所有点里剔除第1条线的最终清洗内点，再拟合第2条
        # 用距离阈把“接近第1条”的点去掉，避免第二条被第一条抢点
        d1 = point_line_distance(pts_all, line1)
        rem_mask = d1 > max(2.0, args.ransac_tol)  # 稍严格点
        pts_remain = pts_all[rem_mask]
        if pts_remain.shape[0] >= 50:
            line2, inliers2, seg2 = fit_and_clean_one_line(
                pts_remain, w, h,
                ransac_tol=args.ransac_tol, ransac_iter=args.ransac_iter,
                min_inliers=max(60, min_inliers // 2), k_sigma=args.k_sigma,
                keep_longest=bool(args.keep_longest),
                gap_q=args.gap_q, min_span_px=args.min_span_px,
                seed=456
            )
            if line2 is not None and inliers2.shape[0] >= 20:
                result["line2"] = {
                    "abc": [float(line2[0]), float(line2[1]), float(line2[2])],
                    "p1": [float(seg2[0][0]), float(seg2[0][1])],
                    "p2": [float(seg2[1][0]), float(seg2[1][1])],
                    "clean_inliers": int(inliers2.shape[0])
                }
                save_points_vis(nms, inliers2, args.out_prefix + "_cleaned_line2.png",
                                color=(255, 255, 0), radius=1, line=seg2, color_line=(255, 0, 0))

    # 4) 保存参数
    with open(args.out_prefix + "_lines.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("== Done ==")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
