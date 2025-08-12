#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline (no cropping version):
  1) show_selected_points: 从 NMS 灰度图按亮度区间选点并可视化（红点）
  2) clean_selected_points: 从选点图中提取红点 -> 密度过滤 -> RANSAC 清除离群（得到绿色内点）
  3) fit_two_lines_from_inliers: 基于绿色内点拟合两条主直线并保存 png/json
  4) make_between_mask: 依据两条线把“中间区域”填充为白色（背景黑），保存到 ./out_pics

输入：--nms_dir 中所有 *_nms.png
输出：./out_lines/*.json/*.png（拟合线）；./out_pics/*_between.png（两线之间的白色mask）
"""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np


# =========================
# Part 1: show_selected_points.py
# =========================

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
    在原图上标出mask==1的点（默认红色）
    """
    vis = (np.clip(nms, 0, 1) * 255).astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    ys, xs = np.where(mask > 0)
    for (x, y) in zip(xs, ys):
        cv2.circle(vis, (int(x), int(y)), radius, point_color, -1)
    return vis


# =========================
# Part 2: clean_selected_points.py
# =========================

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
    """主方向PCA：返回一般式 ax+by+c=0，a^2+b^2=1"""
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
    out_vis_path: 输出可视化（绿=保留，红=剔除）
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

    # 5) RANSAC 第2条线（可选）
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
    radius = 1
    outlier_pts = pts_all[~inliers_mask]
    for x, y in outlier_pts:
        cv2.circle(vis, (int(x), int(y)), radius, (0, 0, 255), -1)
    for x, y in kept_pts:
        cv2.circle(vis, (int(x), int(y)), radius, (0, 255, 0), -1)

    # 8) 保存
    Path(out_vis_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_mask_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_vis_path, vis)
    cv2.imwrite(out_mask_path, inlier_mask_img)

    print(f"[CLEAN] total={pts_all.shape[0]}, kept={kept_pts.shape[0]}, removed={outlier_pts.shape[0]}")
    return inlier_mask_img, vis


# =========================
# Part 3: fit_two_lines_from_inliers.py
# =========================

def coords_from_inlier_mask(mask_img):
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

def line_from_two_points(p1, p2):
    x1,y1 = p1; x2,y2 = p2
    dx, dy = x2-x1, y2-y1
    a, b = dy, -dx
    n = np.hypot(a,b) + 1e-12
    a, b = a/n, b/n
    c = -(a*x1 + b*y1)
    return (a,b,c)

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

def point_line_value_fullimg(w, h, line_abc):
    """返回整幅网格上 ax+by+c 的 signed 值（float32）"""
    a,b,c = line_abc
    X = np.arange(w, dtype=np.float32)[None, :].repeat(h, 0)
    Y = np.arange(h, dtype=np.float32)[:, None].repeat(w, 1)
    return a*X + b*Y + c  # shape (H,W)

def point_line_distance(pts, line_abc):
    a,b,c = line_abc
    return np.abs(a*pts[:,0] + b*pts[:,1] + c)

def tls_refit_pts(pts):
    mu = pts.mean(axis=0)
    X = pts - mu
    _,_,Vt = np.linalg.svd(X, full_matrices=False)
    v = Vt[0]
    a,b = v[1], -v[0]
    n = np.hypot(a,b) + 1e-12
    a,b = a/n, b/n
    c = -(a*mu[0] + b*mu[1])
    return (a,b,c)

def ransac_line_pts(pts, tol=3.0, iters=1000, min_inliers=100, seed=42):
    if pts.shape[0] < 2:
        return None, np.array([], dtype=np.int32)
    rng = np.random.default_rng(seed)
    best_idx = np.array([], dtype=np.int32)
    best_line = None
    idx_all = np.arange(pts.shape[0])
    for _ in range(iters):
        i1, i2 = rng.choice(pts.shape[0], size=2, replace=False)
        if np.allclose(pts[i1], pts[i2]): continue
        line = line_from_two_points(pts[i1], pts[i2])
        d = point_line_distance(pts, line)
        idx = idx_all[d <= tol]
        if idx.size > best_idx.size:
            best_idx = idx
            best_line = line
    if best_line is None or best_idx.size < min_inliers:
        return None, np.array([], dtype=np.int32)
    line_refined = tls_refit_pts(pts[best_idx])
    return line_refined, best_idx

def angle_between_lines(l1, l2):
    """两条线的夹角（弧度，0~pi/2），用方向向量 (b,-a)"""
    a1,b1,_ = l1; a2,b2,_ = l2
    v1 = np.array([b1, -a1], dtype=np.float64)
    v2 = np.array([b2, -a2], dtype=np.float64)
    v1/=np.linalg.norm(v1)+1e-12; v2/=np.linalg.norm(v2)+1e-12
    cos = np.clip(np.abs(np.dot(v1,v2)), 0.0, 1.0)
    return float(np.arccos(cos))  # [0, pi/2]

def median_distance_between_lines(pts, line):
    return float(np.median(point_line_distance(pts, line)))

def fit_two_main_lines(pts, w, h,
                       tol1=3.0, tol2=None,
                       iters=1000,
                       min_inliers_ratio=0.0005,
                       min_angle_sep_deg=3.0,
                       min_gap_px=5.0):
    """
    返回：
      (l1, p1a, p1b, in1, c1), (l2, p2a, p2b, in2, c2), mask
      其中 c1/c2 为各自内点质心（用于后续“朝向内侧”的判定）
    """
    if pts.shape[0] < 20:
        raise RuntimeError("点太少，无法拟合。")
    min_inliers = max(50, int(min_inliers_ratio * w * h))
    if tol2 is None: tol2 = tol1

    # 第1条
    line1, in1 = ransac_line_pts(pts, tol=tol1, iters=iters, min_inliers=min_inliers, seed=123)
    if line1 is None:
        raise RuntimeError("第一条直线拟合失败。")
    p1a, p1b = extend_line_to_image(line1, w, h)
    c1 = pts[in1].mean(axis=0)  # line1 的内点质心

    # 剩余点
    mask = np.ones(pts.shape[0], dtype=bool); mask[in1] = False
    pts2 = pts[mask]
    if pts2.shape[0] < 20:
        raise RuntimeError("剔除第1条内点后点过少。")

    # 第2条（带分离约束）
    best = None; best_idx = None
    idx_all = np.arange(pts2.shape[0])
    rng = np.random.default_rng(456)
    for _ in range(iters):
        i1, i2 = rng.choice(pts2.shape[0], size=2, replace=False)
        if np.allclose(pts2[i1], pts2[i2]): continue
        cand = line_from_two_points(pts2[i1], pts2[i2])
        ang = angle_between_lines(line1, cand) * 180/np.pi
        sep_ok = (ang >= min_angle_sep_deg)
        if not sep_ok:
            gap = median_distance_between_lines(pts2, line1)
            sep_ok = (gap >= min_gap_px)
        if not sep_ok:
            continue
        d = point_line_distance(pts2, cand)
        idx = idx_all[d <= tol2]
        if best is None or idx.size > (0 if best_idx is None else best_idx.size):
            best = cand; best_idx = idx
    if best is None or best_idx.size < max(30, min_inliers//2):
        line2, in2 = ransac_line_pts(pts2, tol=tol2, iters=iters, min_inliers=max(30, min_inliers//2), seed=789)
    else:
        line2 = tls_refit_pts(pts2[best_idx]); in2 = best_idx
    if line2 is None:
        raise RuntimeError("第二条直线拟合失败。")
    p2a, p2b = extend_line_to_image(line2, w, h)
    c2 = pts2[in2].mean(axis=0)  # line2 的内点质心（在 pts2 坐标，但与原图一致）

    return (line1, p1a, p1b, in1, c1), (line2, p2a, p2b, in2, c2), mask


# =========================
# 只输出两线之间的白色mask —— 方向无歧义的“内侧∧内侧”
# =========================

def _robust_side_sign(line_abc, ref_pt, fallback_pts=None):
    """
    计算 ax+by+c 在 ref_pt 的符号；若接近0，用 fallback_pts 的众数符号兜底。
    """
    a,b,c = line_abc
    s = a*ref_pt[0] + b*ref_pt[1] + c
    if np.isfinite(s) and abs(s) > 1e-6:
        return np.sign(s)
    if fallback_pts is not None and fallback_pts.shape[0] > 0:
        vals = a*fallback_pts[:,0] + b*fallback_pts[:,1] + c
        pos = np.sum(vals > 0); neg = np.sum(vals < 0)
        if pos >= neg: return 1.0
        else: return -1.0
    return 1.0  # 保守

def mask_between_two_lines_oriented(h, w, line1, line2, c1, c2, pts_all=None):
    """
    使用“内侧半平面交集”定义两线之间区域：
      - 对于 line1，内侧定义为包含 line2 内点质心 c2 的那一侧；
      - 对于 line2，内侧定义为包含 line1 内点质心 c1 的那一侧；
      - 最终 mask = inner1 ∧ inner2
    """
    S1 = point_line_value_fullimg(w, h, line1)
    S2 = point_line_value_fullimg(w, h, line2)

    s1o = _robust_side_sign(line1, c2, fallback_pts=pts_all)
    s2o = _robust_side_sign(line2, c1, fallback_pts=pts_all)

    inner1 = (S1 * s1o) >= 0.0
    inner2 = (S2 * s2o) >= 0.0
    between = inner1 & inner2

    out = np.zeros((h, w), np.uint8)
    out[between] = 255
    return out


# =========================
# 辅助：剔除边缘一圈选点
# =========================

def exclude_border(mask, border=5):
    """
    将 mask 的边缘（四边各 border 像素宽）清零，用于在选点阶段剔除边缘点
    """
    if border <= 0: return mask
    h, w = mask.shape[:2]
    mask2 = mask.copy()
    mask2[:border, :] = 0
    mask2[-border:, :] = 0
    mask2[:, :border] = 0
    mask2[:, -border:] = 0
    return mask2


# =========================
# 批处理主流程
# =========================

def process_one_nms(nms_path: Path,
                    out_lines: Path,
                    out_pics: Path,
                    # show_selected_points params
                    low_th=0.05, high_th=0.4, sel_radius=1, edge_margin=5,
                    # clean_selected_points params
                    min_r=120, min_diff=60, dens_radius=3, dens_min_neighbors=3,
                    ransac_tol_clean=2.5, ransac_iter_clean=800, min_inliers_ratio_clean=0.0005,
                    # fit_two_lines params
                    ransac_tol_fit=3.0, ransac_iter_fit=1000, min_inliers_ratio_fit=0.0005,
                    min_angle_sep_deg=3.0, min_gap_px=5.0,
                    save_intermediate=True):
    stem = nms_path.stem
    assert stem.endswith("_nms"), f"NMS文件名应为 <stem>_nms.png, 当前: {nms_path.name}"
    base_stem = stem[:-4]  # 去掉 "_nms"

    # 读取 NMS 并选点（红） + 剔除边缘点
    nms = imread_gray(nms_path)
    sel_mask = binarize_nms_range(nms, low_th=low_th, high_th=high_th)
    sel_mask = exclude_border(sel_mask, border=edge_margin)  # 剔除边缘选点
    sel_vis = visualize_selected_points(nms, sel_mask, point_color=(0,0,255), radius=sel_radius)
    if save_intermediate:
        out_lines.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_lines / f"{base_stem}_selected_points.png"), sel_vis)

    # 清洗红点 -> 绿色内点 & inlier mask
    clean_vis_path = out_lines / f"{base_stem}_clean_vis.png"
    inlier_mask_path = out_lines / f"{base_stem}_inliers_mask.png"
    try:
        inlier_mask_img, _ = clean_selected_points(
            sel_png_path=str(out_lines / f"{base_stem}_selected_points.png"),
            out_vis_path=str(clean_vis_path),
            out_mask_path=str(inlier_mask_path),
            min_r=min_r, min_diff=min_diff,
            dens_radius=dens_radius, dens_min_neighbors=dens_min_neighbors,
            ransac_tol=ransac_tol_clean, ransac_iter=ransac_iter_clean,
            min_inliers_ratio=min_inliers_ratio_clean,
            fit_two_lines=True
        )
    except Exception as e:
        print(f"[WARN] clean_selected_points 失败: {nms_path.name} -> {e}")
        return

    # 从内点拟合两条线
    m = cv2.imread(str(inlier_mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        print(f"[WARN] 无法读取 inlier mask: {inlier_mask_path}")
        return
    h, w = m.shape[:2]
    pts = coords_from_inlier_mask(m)
    if pts.shape[0] < 20:
        print(f"[WARN] 内点过少，跳过: {nms_path.name}")
        return

    try:
        (l1, p1a, p1b, in1, c1), (l2, p2a, p2b, in2, c2), mask = fit_two_main_lines(
            pts, w, h,
            tol1=ransac_tol_fit, tol2=ransac_tol_fit,
            iters=ransac_iter_fit,
            min_inliers_ratio=min_inliers_ratio_fit,
            min_angle_sep_deg=min_angle_sep_deg,
            min_gap_px=min_gap_px
        )
    except Exception as e:
        print(f"[WARN] 拟合两条线失败: {nms_path.name} -> {e}")
        return

    # 可视化两条线（叠加在 NMS 背景上）
    bg = (np.clip(nms, 0, 1) * 255).astype(np.uint8)
    bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    for x,y in pts:
        cv2.circle(bg, (int(x), int(y)), 1, (80,200,80), -1)
    cv2.line(bg, (int(p1a[0]), int(p1a[1])), (int(p1b[0]), int(p1b[1])), (0,0,255), 3)
    cv2.line(bg, (int(p2a[0]), int(p2a[1])), (int(p2b[0]), int(p2b[1])), (255,0,0), 3)

    # 保存线参数 & 可视化
    out_lines.mkdir(parents=True, exist_ok=True)
    fit_png = out_lines / f"{base_stem}_fit.png"
    fit_json = out_lines / f"{base_stem}_fit.json"
    cv2.imwrite(str(fit_png), bg)
    out_dict = {
        "image_size": {"w": int(w), "h": int(h)},
        "line1": {"abc": [float(l1[0]), float(l1[1]), float(l1[2])],
                  "p1": [float(p1a[0]), float(p1a[1])],
                  "p2": [float(p1b[0]), float(p1b[1])],
                  "inliers": int(len(in1)),
                  "centroid": [float(c1[0]), float(c1[1])]},
        "line2": {"abc": [float(l2[0]), float(l2[1]), float(l2[2])],
                  "p1": [float(p2a[0]), float(p2a[1])],
                  "p2": [float(p2b[0]), float(p2b[1])],
                  "inliers": int(len(in2)),
                  "centroid": [float(c2[0]), float(c2[1])]},
    }
    with open(fit_json, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=2)
    print(f"[OK] lines saved -> {fit_png.name}, {fit_json.name}")

    # 生成两线之间的白色mask（背景黑），使用“内侧∧内侧”而非符号相反
    mask_between = mask_between_two_lines_oriented(h, w, l1, l2, c1, c2, pts_all=pts)
    out_pics.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_pics / f"{base_stem}_between.png"), mask_between)
    print(f"[OK] between mask saved -> {base_stem}_between.png")


def main():
    ap = argparse.ArgumentParser(description="Batch: NMS -> select points -> clean -> fit two lines -> output white mask between lines.")
    ap.add_argument("--nms_dir", required=True, help="包含所有 *_nms.png 的文件夹")
    ap.add_argument("--out_lines", default="./out_lines", help="拟合线输出目录（png/json/中间图）")
    ap.add_argument("--out_pics", default="./out_pics", help="两线中间白色mask输出目录")

    # show_selected_points 参数
    ap.add_argument("--low_th", type=float, default=0.05, help="NMS亮度下阈（0~1）")
    ap.add_argument("--high_th", type=float, default=0.4, help="NMS亮度上阈（0~1）")
    ap.add_argument("--sel_radius", type=int, default=1, help="选点显示半径")
    ap.add_argument("--edge_margin", type=int, default=5, help="选点阶段剔除边缘像素宽度(px)")

    # clean_selected_points 参数
    ap.add_argument("--min_r", type=int, default=120, help="红点R阈值")
    ap.add_argument("--min_diff", type=int, default=60, help="R - max(G,B) 最小差")
    ap.add_argument("--dens_radius", type=int, default=3, help="密度过滤半径")
    ap.add_argument("--dens_min_neighbors", type=int, default=3, help="最少邻居数")
    ap.add_argument("--ransac_tol_clean", type=float, default=2.5, help="清洗阶段 RANSAC 容差(px)")
    ap.add_argument("--ransac_iter_clean", type=int, default=800, help="清洗阶段 RANSAC 迭代")
    ap.add_argument("--min_inliers_ratio_clean", type=float, default=0.0005, help="清洗阶段内点数比例")

    # fit_two_lines 参数
    ap.add_argument("--ransac_tol_fit", type=float, default=3.0, help="拟合阶段 RANSAC 容差(px)")
    ap.add_argument("--ransac_iter_fit", type=int, default=1000, help="拟合阶段 RANSAC 迭代")
    ap.add_argument("--min_inliers_ratio_fit", type=float, default=0.0005, help="拟合阶段内点数比例")
    ap.add_argument("--min_angle_sep_deg", type=float, default=3.0, help="两线最小夹角(度)")
    ap.add_argument("--min_gap_px", type=float, default=5.0, help="平行时两线最小间隔(px)")

    ap.add_argument("--no_intermediate", action="store_true", help="不保存中间可视化（选点/清洗/内点mask）")

    args = ap.parse_args()

    nms_dir = Path(args.nms_dir)
    out_lines = Path(args.out_lines)
    out_pics = Path(args.out_pics)
    save_intermediate = not args.no_intermediate

    nms_paths = sorted(nms_dir.glob("*_nms.png"))
    if not nms_paths:
        print(f"[ERR] {nms_dir} 下未找到 *_nms.png")
        return

    print(f"[INFO] 待处理 NMS 数量：{len(nms_paths)}")
    for p in nms_paths:
        print(f"\n>>> Processing: {p.name}")
        try:
            process_one_nms(
                p, out_lines, out_pics,
                low_th=args.low_th, high_th=args.high_th, sel_radius=args.sel_radius, edge_margin=args.edge_margin,
                min_r=args.min_r, min_diff=args.min_diff,
                dens_radius=args.dens_radius, dens_min_neighbors=args.dens_min_neighbors,
                ransac_tol_clean=args.ransac_tol_clean, ransac_iter_clean=args.ransac_iter_clean,
                min_inliers_ratio_clean=args.min_inliers_ratio_clean,
                ransac_tol_fit=args.ransac_tol_fit, ransac_iter_fit=args.ransac_iter_fit,
                min_inliers_ratio_fit=args.min_inliers_ratio_fit,
                min_angle_sep_deg=args.min_angle_sep_deg, min_gap_px=args.min_gap_px,
                save_intermediate=save_intermediate
            )
        except AssertionError as e:
            print(f"[SKIP] {p.name}: {e}")
        except Exception as e:
            print(f"[ERR] {p.name}: {e}")

    print("\n[DONE] 全部处理完成。输出：")
    print(f"  拟合线: {out_lines.resolve()}")
    print(f"  中间mask: {out_pics.resolve()}")


if __name__ == "__main__":
    main()
