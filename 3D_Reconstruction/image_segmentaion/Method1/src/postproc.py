# src/postproc.py
from __future__ import annotations
# src/postproc.py
import math
import numpy as np
import cv2
from scipy.ndimage import maximum_filter
from skimage.filters import apply_hysteresis_threshold
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev


# ---------------------------
# 基础：NMS / 双阈值 / 骨架 → 曲线
# ---------------------------
def edge_nms(prob: np.ndarray, k: int = 3) -> np.ndarray:
    m = maximum_filter(prob, size=k)
    keep = (prob == m)
    out = prob.copy()
    out[~keep] = 0
    return out

def hysteresis(prob_or_bin: np.ndarray, th_low: float, th_high: float) -> np.ndarray:
    return apply_hysteresis_threshold(prob_or_bin, th_low, th_high).astype(np.uint8)

def trace_and_fit(binary: np.ndarray, min_len: int = 50, smooth: float = 0.001, num_pts: int = 200):
    """
    将二值细线/边缘转成多条曲线（每条为 (N,2) 的 (x,y)）
    1) skeletonize 细化
    2) 连通域提取坐标
    3) B-spline 拟合（失败则返回原坐标）
    """
    sk = skeletonize(binary > 0).astype(np.uint8)
    comps = label(sk, connectivity=2)
    curves = []
    for r in regionprops(comps):
        coords = r.coords[:, ::-1]  # (x,y)
        if coords.shape[0] < min_len:
            continue
        pts = coords.astype(np.float64)
        # 以首尾连线方向排序，减少回折
        order = np.argsort(pts[:, 1])  # 先按 y 排序（便于后续单调化）
        pts = pts[order]
        try:
            tck, _ = splprep([pts[:, 0], pts[:, 1]], s=smooth)
            u = np.linspace(0, 1, num_pts)
            xnew, ynew = splev(u, tck)
            curves.append(np.stack([xnew, ynew], 1))
        except Exception:
            curves.append(pts)
    return curves


# ---------------------------
# 辅助：拼接 / 单调化 / 评分
# ---------------------------
def polyline_length(poly: np.ndarray) -> float:
    d = np.diff(poly, axis=0)
    return float(np.sqrt((d ** 2).sum(1)).sum())

def resample_polyline_by_arclength(poly: np.ndarray, N: int = 300) -> np.ndarray:
    pts = poly.astype(np.float32)
    seg = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(1))
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return np.repeat(pts[:1], N, axis=0)
    u = np.linspace(0, s[-1], N)
    x = np.interp(u, s, pts[:, 0])
    y = np.interp(u, s, pts[:, 1])
    return np.stack([x, y], 1)

def sample_prob_along_poly(prob: np.ndarray, poly: np.ndarray, step: int = 1) -> float:
    h, w = prob.shape
    pts = poly[::max(1, step)]
    xs = np.clip(pts[:, 0], 0, w - 1)
    ys = np.clip(pts[:, 1], 0, h - 1)
    return float(prob[ys.astype(np.int32), xs.astype(np.int32)].mean())

def main_direction(poly: np.ndarray) -> np.ndarray:
    v = poly[-1] - poly[0]
    n = np.linalg.norm(v) + 1e-8
    return v / n

def angle_between(u, v) -> float:
    d = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.degrees(np.arccos(d)))


# ---------------------------
# 关键1：按 y 单调表达 x(y)，方便“贯穿上下边界”
# ---------------------------
def to_monotonic_x_of_y(poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    将 poly (x,y) 转为按 y 单调的 (ys, xs)，ys 严格递增（去重）
    """
    p = poly.copy()
    order = np.argsort(p[:, 1])
    p = p[order]
    ys = p[:, 1]
    xs = p[:, 0]
    # 去除重复 y（取平均 x）
    y_unique, idx = np.unique(ys.round().astype(int), return_inverse=True)
    x_acc = np.zeros_like(y_unique, dtype=np.float64)
    c_acc = np.zeros_like(y_unique, dtype=np.int32)
    for i, j in enumerate(idx):
        x_acc[j] += xs[i]
        c_acc[j] += 1
    xs_avg = x_acc / np.maximum(c_acc, 1)
    return y_unique.astype(np.float64), xs_avg.astype(np.float64)

def stitch_segments_by_y(curves, y_gap_max: int = 40, x_gap_max: int = 40):
    """
    将多段上下相邻、x 距离不大的曲线按 y 方向拼接，得到更长的“上→下”轨迹。
    简单贪心：按起始 y 排序，能链到就链。
    """
    items = []
    for c in curves:
        ys, xs = to_monotonic_x_of_y(c)
        if len(ys) < 5:
            continue
        items.append({'ys': ys, 'xs': xs})
    items.sort(key=lambda d: d['ys'][0])

    stitched = []
    used = [False] * len(items)

    for i in range(len(items)):
        if used[i]:
            continue
        ys = items[i]['ys'].copy()
        xs = items[i]['xs'].copy()

        for j in range(i + 1, len(items)):
            if used[j]:
                continue
            ys2 = items[j]['ys']; xs2 = items[j]['xs']
            # 是否可拼接：y 间隙小 & x 差距小（用边界端点比较）
            gap_y = ys2[0] - ys[-1]
            if 0 < gap_y <= y_gap_max:
                # 对齐到 ys[-1] 附近估计 x2 的起点
                x1 = xs[-1]
                x2 = xs2[0]
                if abs(x1 - x2) <= x_gap_max:
                    # 拼接
                    ys = np.concatenate([ys, ys2])
                    xs = np.concatenate([xs, xs2])
                    used[j] = True
        used[i] = True
        stitched.append({'ys': ys, 'xs': xs})

    # 转回 (x,y) polyline 形式
    out = []
    for it in stitched:
        out.append(np.stack([it['xs'], it['ys']], 1))
    return out


# ---------------------------
# 关键2：将轨迹外推到上下边界（y=0 与 y=H-1）
# ---------------------------
def extend_to_top_bottom(poly: np.ndarray, H: int, W: int, edge_margin: int = 10, fit_tail: int = 25) -> np.ndarray:
    """
    以 y 为自变量，线性拟合“顶端/底端”各 fit_tail 个点的 x(y)，向 y=0 和 y=H-1 外推。
    并将 x 限制在 [edge_margin, W-1-edge_margin]，保证不贴左右边缘。
    返回：完整贯穿的 (x,y) 折线（按 y 递增）
    """
    ys, xs = to_monotonic_x_of_y(poly)
    if len(ys) < 4:
        return poly

    # 顶端拟合
    k = min(fit_tail, len(ys)//2)
    coef_top = np.polyfit(ys[:k], xs[:k], deg=1)
    x_top = np.polyval(coef_top, 0.0)

    # 底端拟合
    coef_bot = np.polyfit(ys[-k:], xs[-k:], deg=1)
    x_bot = np.polyval(coef_bot, float(H - 1))

    # 组合完整 y 轴网格（含 0 和 H-1）
    ys_full = np.arange(0, H, 1.0)
    xs_full = np.interp(ys_full, ys, xs)
    xs_full[0] = x_top
    xs_full[-1] = x_bot

    # 限制左右边缘外 margin
    xs_full = np.clip(xs_full, edge_margin, W - 1 - edge_margin)

    return np.stack([xs_full, ys_full], 1)


# ---------------------------
# 关键3：从一堆曲线中，挑“两条最优贯穿线”
# ---------------------------
def median_pair_distance_from_paths(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    p1, p2 已经是按 y=0..H-1 对齐的 (x,y)，同长度
    取对应 y 处的横向距离的中位数
    """
    d = np.sqrt(((p1[:, 0] - p2[:, 0]) ** 2))
    return float(np.median(d))

def sample_prob_along_path(prob: np.ndarray, path: np.ndarray, step: int = 2) -> float:
    h, w = prob.shape
    ys = np.clip(path[::step, 1], 0, h - 1).astype(np.int32)
    xs = np.clip(path[::step, 0], 0, w - 1).astype(np.int32)
    return float(prob[ys, xs].mean())

def select_two_spanning_paths(curves, prob,
                              H: int, W: int,
                              edge_margin: int = 10,
                              y_gap_max: int = 60, x_gap_max: int = 60,
                              dmin: float = 8.0, dmax: float = 200.0,
                              ang_tol_deg: float = 25.0,
                              alpha_len: float = 0.6, beta_int: float = 0.4):
    """
    1) 先把曲线按 y 方向拼接成长轨迹
    2) 将每条轨迹外推到 y=0 与 y=H-1，且限制左右 margin
    3) 对每条打分：长度(这里可用覆盖高度≈H) + 概率白度
    4) 在候选里选出一对：近平行 & 间距落在 [dmin, dmax]
    """
    if len(curves) < 2:
        return None

    # 拼接
    stitched = stitch_segments_by_y(curves, y_gap_max=y_gap_max, x_gap_max=x_gap_max)
    if len(stitched) < 2:
        return None

    # 延申到上下边界
    spanning = []
    for c in stitched:
        p_ext = extend_to_top_bottom(c, H, W, edge_margin=edge_margin, fit_tail=25)
        # 计算方向（用整体首尾）
        dir_vec = main_direction(p_ext)
        spanning.append({'path': p_ext, 'dir': dir_vec})

    # 单条评分
    singles = []
    for s in spanning:
        path = s['path']
        L = H  # 贯穿高度基本固定为 H，可加上横向变化的惩罚/奖励
        I = sample_prob_along_path(prob, path, step=3)
        score = alpha_len * L + beta_int * I * 1000.0
        singles.append((path, s['dir'], L, I, score))
    singles.sort(key=lambda x: x[-1], reverse=True)
    cand = [x for x in singles[:min(8, len(singles))]]

    # 选择最佳一对
    best = (None, None, -1e18)
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            p1, d1, _, _, _ = cand[i]
            p2, d2, _, _, _ = cand[j]
            ang = angle_between(d1, d2)
            if ang > ang_tol_deg:
                continue
            md = median_pair_distance_from_paths(p1, p2)
            if not (dmin <= md <= dmax):
                continue
            # 组合分
            Isum = sample_prob_along_path(prob, p1, 3) + sample_prob_along_path(prob, p2, 3)
            Lsum = 2 * H
            score = alpha_len * Lsum + beta_int * Isum * 1000.0
            if score > best[2]:
                best = (p1, p2, score)

    return None if best[0] is None else (best[0], best[1])


# ---------------------------
# 关键4：两条贯穿线 → 中间带多边形
# ---------------------------
def middle_strip_mask_from_paths(p1: np.ndarray, p2: np.ndarray, shape) -> np.ndarray:
    """
    p1, p2: 已经对齐到 y=0..H-1 的两条 (x,y) 轨迹
    直接拼一个闭合多边形：p1 正向 + p2 反向
    """
    H, W = shape[:2]
    a = np.round(p1).astype(np.int32)
    b = np.round(p2).astype(np.int32)
    poly = np.vstack([a, b[::-1]])
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return (mask > 0).astype(np.uint8)


# ---------------------------
# 兜底：原有的 buffer/裁切
# ---------------------------
def buffer_polyline(poly: np.ndarray, r: int = 3, shape=None):
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    pts = np.round(poly).astype(np.int32)
    for i in range(len(pts) - 1):
        cv2.line(mask, tuple(pts[i]), tuple(pts[i + 1]), 255, thickness=int(max(1, 2 * r)))
    return (mask > 0).astype(np.uint8)

def estimate_width_by_gradient(img, poly, r_min=2, r_max=6):
    return int(0.5 * (r_min + r_max))

def crop_by_components(mask, margin=16, min_area=80, img=None):
    num, labels = cv2.connectedComponents(mask.astype(np.uint8))
    rois = []
    for i in range(1, num):
        ys, xs = np.where(labels == i)
        if xs.size < min_area:
            continue
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(mask.shape[1] - 1, x2 + margin)
        y2 = min(mask.shape[0] - 1, y2 + margin)
        if img is not None:
            rois.append(img[y1 : y2 + 1, x1 : x2 + 1].copy())
        else:
            rois.append((x1, y1, x2, y2))
    return rois
