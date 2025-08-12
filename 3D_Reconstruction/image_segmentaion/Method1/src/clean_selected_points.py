import argparse
from pathlib import Path
import cv2
import numpy as np


# ---------- 1) 读入与“红点”提取 ----------

def imread_bgr(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

def extract_red_points(bgr, min_r=120, min_diff=60):
    """
    从 selected_points.png 中提取标红的点。
    判定条件：R >= min_r 且 R - max(G,B) >= min_diff
    返回：二值mask，1表示红点像素
    """
    B,G,R = cv2.split(bgr)
    max_gb = cv2.max(G, B)
    red_mask = (R >= min_r) & ((R.astype(np.int16) - max_gb.astype(np.int16)) >= min_diff)
    return red_mask.astype(np.uint8)

def coords_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return np.zeros((0,2), dtype=np.float32)
    return np.stack([xs, ys], axis=1).astype(np.float32)


# ---------- 2) 稀疏点剔除（邻域密度过滤） ----------

def density_filter(mask_pts, radius=3, min_neighbors=3):
    """
    对像素平面上的红点mask做局部计数：
      - 用 (2*radius+1) 的方框核卷积得到每个像素邻域内的点数
      - 保留邻居计数 >= min_neighbors 的位置
    """
    k = 2*radius + 1
    kernel = np.ones((k, k), np.uint8)
    cnt = cv2.filter2D(mask_pts.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    kept = (cnt >= min_neighbors).astype(np.uint8)
    return kept


# ---------- 3) 直线拟合（RANSAC + TLS细化） ----------

def line_from_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    a, b = dy, -dx
    n = np.hypot(a, b) + 1e-12
    a, b = a/n, b/n
    c = -(a*x1 + b*y1)
    return (a, b, c)

def point_line_distance(pts, line_abc):
    a, b, c = line_abc
    return np.abs(a*pts[:,0] + b*pts[:,1] + c)

def tls_refit(pts):
    """
    主方向PCA：返回一般式 ax+by+c=0，a^2+b^2=1
    """
    mu = pts.mean(axis=0)
    X = pts - mu
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    v = Vt[0]
    a, b = v[1], -v[0]
    n = np.hypot(a,b) + 1e-12
    a, b = a/n, b/n
    c = -(a*mu[0] + b*mu[1])
    return (a, b, c)

def ransac_line(pts, tol=2.5, iters=800, min_inliers=100, seed=123):
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
    # 用内点再细化
    line_refined = tls_refit(pts[best_idx])
    return line_refined, best_idx


# ---------- 4) 主流程：提取→密度过滤→RANSAC一/二条线→合并内点→可视化 ----------

def clean_selected_points(sel_png_path,
                          out_vis_path,
                          out_mask_path,
                          min_r=120, min_diff=60,
                          dens_radius=3, dens_min_neighbors=3,
                          ransac_tol=2.5, ransac_iter=800,
                          min_inliers_ratio=0.0005,
                          fit_two_lines=True):
    """
    sel_png_path: selected_points.png（背景灰度+红点）
    out_vis_path: 输出可视化（绿=保留；红=剔除）
    out_mask_path: 输出内点二值mask
    """

    bgr = imread_bgr(sel_png_path)
    h, w = bgr.shape[:2]

    # 1) 取红点
    red_mask = extract_red_points(bgr, min_r=min_r, min_diff=min_diff)

    # 2) 密度过滤（先去孤立点/小散团）
    red_mask_dense = density_filter(red_mask, radius=dens_radius, min_neighbors=dens_min_neighbors)

    # 3) 取坐标
    pts_all = coords_from_mask(red_mask_dense)
    if pts_all.shape[0] < 20:
        raise RuntimeError("有效红点太少，请放宽红色阈值或降低密度过滤要求。")

    # 4) RANSAC 第1条线
    min_inliers = max(50, int(min_inliers_ratio * w * h))
    line1, in1 = ransac_line(pts_all, tol=ransac_tol, iters=ransac_iter, min_inliers=min_inliers, seed=123)

    inliers_mask = np.zeros(pts_all.shape[0], dtype=bool)
    if line1 is not None:
        inliers_mask[in1] = True

    # 5) RANSAC 第2条线（可选，剔除第1条内点后再拟合）
    if fit_two_lines:
        remain_idx = np.where(~inliers_mask)[0]
        if remain_idx.size >= 20:
            line2, in2_local = ransac_line(pts_all[remain_idx], tol=ransac_tol, iters=ransac_iter,
                                           min_inliers=max(30, min_inliers//2), seed=456)
            if line2 is not None:
                inliers_mask[remain_idx[in2_local]] = True

    # 6) 生成“内点二值图”
    inlier_mask_img = np.zeros((h, w), np.uint8)
    kept_pts = pts_all[inliers_mask]
    for x, y in kept_pts:
        inlier_mask_img[int(y), int(x)] = 255

    # 7) 可视化（绿=保留，红=剔除）
    vis = bgr.copy()
    # 先把所有红点画为淡红（可选）
    # 再覆盖绘制：内点为绿色，小半径；外点保持红色
    # 为便于看清，统一画半径=1或2
    radius = 1
    # 外点
    outlier_pts = pts_all[~inliers_mask]
    for x, y in outlier_pts:
        cv2.circle(vis, (int(x), int(y)), radius, (0, 0, 255), -1)
    # 内点
    for x, y in kept_pts:
        cv2.circle(vis, (int(x), int(y)), radius, (0, 255, 0), -1)

    # 8) 保存
    Path(out_vis_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_mask_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_vis_path, vis)
    cv2.imwrite(out_mask_path, inlier_mask_img)

    print(f"[INFO] total points: {pts_all.shape[0]}, kept: {kept_pts.shape[0]}, removed: {outlier_pts.shape[0]}")
    return inlier_mask_img, vis


# ---------- 5) CLI ----------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Remove outliers from selected (red) points and save cleaned result.")
    ap.add_argument("--selected_png", required=True, help="selected_points.png（背景+红点）")
    ap.add_argument("--save_vis", required=True, help="清洗后可视化（绿=保留，红=剔除）")
    ap.add_argument("--save_mask", required=True, help="内点二值mask输出路径")

    # 红点提取阈值
    ap.add_argument("--min_r", type=int, default=120, help="R通道最低值阈")
    ap.add_argument("--min_diff", type=int, default=60, help="R - max(G,B) 的最小差值")

    # 密度过滤
    ap.add_argument("--dens_radius", type=int, default=3, help="邻域半径（像素）")
    ap.add_argument("--dens_min_neighbors", type=int, default=3, help="最少邻居数")

    # RANSAC参数
    ap.add_argument("--ransac_tol", type=float, default=2.5, help="点到直线的最大距离（像素）")
    ap.add_argument("--ransac_iter", type=int, default=800, help="RANSAC迭代次数")
    ap.add_argument("--min_inliers_ratio", type=float, default=0.0005, help="最小内点数占原图像素比例")

    ap.add_argument("--fit_two_lines", action="store_true", help="启用后会拟合第二条主直线并取并集")

    args = ap.parse_args()

    clean_selected_points(
        sel_png_path=args.selected_png,
        out_vis_path=args.save_vis,
        out_mask_path=args.save_mask,
        min_r=args.min_r, min_diff=args.min_diff,
        dens_radius=args.dens_radius, dens_min_neighbors=args.dens_min_neighbors,
        ransac_tol=args.ransac_tol, ransac_iter=args.ransac_iter,
        min_inliers_ratio=args.min_inliers_ratio,
        fit_two_lines=args.fit_two_lines
    )
