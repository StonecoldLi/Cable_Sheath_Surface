# stitch_routeB_subjectfill.py
# 圆柱化 + “仅主体列”参与 + 主体θ范围自适应拉伸到扇区（消黑带）+ 接缝羽化
import argparse, json, math
from pathlib import Path
import numpy as np
import cv2

# ---------- I/O ----------
def imread_color(p):
    if not p: return None
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(p)
    return img

def imread_mask01(p):
    if not p or not Path(p).exists(): return None
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None: return None
    m = (m.astype(np.float32) / 255.0)
    return np.clip(m, 0.0, 1.0)

def ensure_mask_size(mask01, H, W):
    if mask01 is None:
        return np.ones((H, W), dtype=np.float32)
    if mask01.shape[:2] != (H, W):
        mask01 = cv2.resize(mask01, (W, H), interpolation=cv2.INTER_NEAREST)
    return np.clip(mask01.astype(np.float32), 0.0, 1.0)

# ---------- 几何 ----------
def f_from_fov_deg(img_width_px, fov_h_deg):
    return 0.5 * float(img_width_px) / math.tan(math.radians(fov_h_deg) * 0.5)

def cylindrical_maps(H, W, f_pix):
    """构建圆柱化逆映射、θ网格与有效性掩膜(valid)。"""
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    x_min = (0 - cx) / f_pix
    x_max = ((W - 1) - cx) / f_pix
    theta_min = math.atan(x_min)
    theta_max = math.atan(x_max)
    theta_center = 0.5 * (theta_min + theta_max)

    out_w = int(math.ceil(f_pix * (theta_max - theta_min)))
    out_h = H

    u_idx = np.arange(out_w, dtype=np.float32)
    v_idx = np.arange(out_h, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(u_idx, v_idx)  # (H', W')

    theta = theta_min + u_grid / f_pix
    h = (v_grid - (out_h - 1) / 2.0) / f_pix

    x = np.tan(theta)
    den = np.sqrt(1.0 + x * x)
    y = h * den

    map_x = (f_pix * x + cx).astype(np.float32).copy()
    map_y = (f_pix * y + cy).astype(np.float32).copy()

    valid = (map_x >= 0.0) & (map_x <= (W - 1)) & (map_y >= 0.0) & (map_y <= (H - 1))
    valid = valid.astype(np.float32)

    return map_x, map_y, theta.astype(np.float32), float(theta_center), valid

def cylindrical_warp_strict(img_bgr, mask01, f_pix, mask_thresh=0.5, erode_px=1):
    """圆柱化 + 严格主体掩膜（去除黑边/插值灰边）+ 视角权重。"""
    H, W = img_bgr.shape[:2]
    mask01 = ensure_mask_size(mask01, H, W)

    map_x, map_y, theta, theta_center, valid = cylindrical_maps(H, W, f_pix)

    warped = cv2.remap(img_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask_remap = cv2.remap(mask01, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    warped_mask = (mask_remap >= float(mask_thresh)).astype(np.uint8)
    warped_mask = (warped_mask.astype(np.float32) * valid).astype(np.uint8)

    if erode_px > 0:
        k = max(1, int(erode_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        warped_mask = cv2.erode(warped_mask, kernel, iterations=1)

    warped_mask_f = warped_mask.astype(np.float32)

    # 视角余弦权重（列越靠本地中心越大）
    w_view = np.cos(theta - theta_center)
    w_view = np.clip(w_view, 0.0, 1.0).astype(np.float32)
    weight_base = warped_mask_f * w_view

    info = {
        "theta_min": float(theta[0, 0]),
        "theta_max": float(theta[0, -1]),
        "theta_center_local": float(theta_center),
        "out_h": int(warped.shape[0]),
        "out_w": int(warped.shape[1]),
    }
    return warped, weight_base, theta, info, warped_mask

# ---------- 主体θ范围检测 & 边缘软化 ----------
def subject_theta_bounds(warped_mask, theta_local, col_cov_thresh=0.05, margin_frac=0.02):
    """基于掩膜的列覆盖率找到主体列范围，并预留 margin 收紧边界。"""
    Hc, Wc = warped_mask.shape[:2]
    col_cov = warped_mask.mean(axis=0)  # 每列主体占比
    cols = np.where(col_cov > col_cov_thresh)[0]
    if cols.size == 0:
        return None  # 退化：无主体
    c0, c1 = int(cols[0]), int(cols[-1])
    # 收紧两端，去掉最边缘列
    m = int(round(max(0.0, min(0.45, margin_frac)) * (c1 - c0 + 1)))
    c0 = min(max(0, c0 + m), Wc - 1)
    c1 = max(min(Wc - 1, c1 - m), 0)
    if c1 <= c0:
        c0 = max(0, (Wc // 2) - 1)
        c1 = min(Wc - 1, (Wc // 2) + 1)
    th0 = float(theta_local[0, c0])
    th1 = float(theta_local[0, c1])
    return {"col0": c0, "col1": c1, "theta0": th0, "theta1": th1}

def edge_soften_columns(weight_base, theta_local, col0, col1, edge_soft_deg):
    """对主体左右边界附近做半余弦降低（按角度换算像素宽度）。"""
    Hc, Wc = weight_base.shape[:2]
    if edge_soft_deg <= 0:
        return weight_base
    # θ 每像素（近似常数）
    dtheta = float(theta_local[0, -1] - theta_local[0, 0]) / max(1, (Wc - 1))
    soft_px = int(round(math.radians(edge_soft_deg) / max(1e-9, abs(dtheta))))
    soft_px = max(1, min(soft_px, (col1 - col0 + 1) // 2))
    wcol = np.ones(Wc, dtype=np.float32)
    # 左边界 [col0, col0+soft_px)
    L = min(soft_px, col1 - col0 + 1)
    if L > 0:
        j = np.arange(L, dtype=np.float32)
        ramp = 0.5 - 0.5 * np.cos(np.pi * (j + 1) / (L + 1))  # 0→1
        wcol[col0:col0 + L] *= ramp
        # 右边界 (col1-L+1..col1]
        ramp_r = ramp[::-1]
        wcol[col1 - L + 1:col1 + 1] *= ramp_r
    return (weight_base * wcol[None, :]).astype(np.float32)

# ---------- 角度/扇区 ----------
def wrap_0_2pi(a): return np.mod(a, 2.0 * math.pi)

def angle_diff_signed(a, b):
    d = a - b
    return (d + math.pi) % (2.0 * math.pi) - math.pi

def circ_mid(a, b):
    d = angle_diff_signed(b, a) * 0.5
    return (a + d + 2.0 * math.pi) % (2.0 * math.pi)

def build_sectors(theta_centers):
    centers = np.array(sorted([wrap_0_2pi(t) for t in theta_centers]), dtype=np.float64)
    n = len(centers)
    sectors = []
    for i in range(n):
        c = centers[i]
        c_prev = centers[(i - 1) % n]
        c_next = centers[(i + 1) % n]
        left_bound = circ_mid(c_prev, c)
        right_bound = circ_mid(c, c_next)
        half_left = abs(angle_diff_signed(c, left_bound))
        half_right = abs(angle_diff_signed(right_bound, c))
        sectors.append((c, half_left, half_right))
    return sectors, centers

def edge_feather_weights(theta_global, center, half_left, half_right, blend_rad):
    d = angle_diff_signed(theta_global, center)  # (-π, π]
    w = np.ones_like(d, dtype=np.float32)
    left_edge, right_edge = -half_left, +half_right

    eps = 1e-6
    blend_L = max(eps, min(blend_rad, max(half_left - eps, eps)))
    blend_R = max(eps, min(blend_rad, max(half_right - eps, eps)))

    w = np.where(d <= left_edge, 0.0, w)
    w = np.where(d >= right_edge, 0.0, w)

    if blend_L > 0:
        tL = (d - left_edge) / blend_L
        mL = (tL >= 0) & (tL <= 1)
        wL = 0.5 - 0.5 * np.cos(np.pi * np.clip(tL, 0.0, 1.0))
        w = np.where(mL, w * wL.astype(np.float32), w)

    if blend_R > 0:
        tR = (right_edge - d) / blend_R
        mR = (tR >= 0) & (tR <= 1)
        wR = 0.5 - 0.5 * np.cos(np.pi * np.clip(tR, 0.0, 1.0))
        w = np.where(mR, w * wR.astype(np.float32), w)

    return w

# ---------- 累加 ----------
def place_on_panorama(accum_rgb, accum_w,
                      warped_bgr, weight_base, theta_local,
                      theta_offset_rad, f_pix,
                      sector_tuple, blend_rad, v_offset=0,
                      theta_map_mode="stretch-subject",
                      subject_theta0=None, subject_theta1=None):
    """
    theta_map_mode:
      - 'physical'        : 直接 θ_local + offset
      - 'stretch'         : 把整张 [θ_min, θ_max] 映射到扇区
      - 'stretch-subject' : 仅把主体 [θ0, θ1] 映射到扇区（推荐）
    """
    Hc, Wc = warped_bgr.shape[:2]
    Hpan, Wpan = accum_w.shape
    rows_local = np.repeat(np.arange(Hc, dtype=np.int32), Wc)
    rows_global = rows_local + int(v_offset)
    ok_y = (rows_global >= 0) & (rows_global < Hpan)
    if not np.any(ok_y):
        return

    center, half_left, half_right = sector_tuple
    if theta_map_mode == "physical":
        theta_global = wrap_0_2pi(theta_local + theta_offset_rad)
    else:
        # 目标扇区范围
        tgt_min = center - half_left
        tgt_max = center + half_right
        if theta_map_mode == "stretch-subject" and subject_theta0 is not None and subject_theta1 is not None:
            src_min, src_max = float(subject_theta0), float(subject_theta1)
        else:
            src_min, src_max = float(theta_local[0, 0]), float(theta_local[0, -1])
        # 线性映射到扇区；主体外像素权重本来≈0，不影响
        s = (theta_local - src_min) / max(1e-8, (src_max - src_min))
        s = np.clip(s, 0.0, 1.0)
        theta_global = wrap_0_2pi(tgt_min + s * (tgt_max - tgt_min))

    # 扇区窗 + 接缝羽化
    w_window = edge_feather_weights(theta_global, center, half_left, half_right, blend_rad)
    weight = (weight_base * w_window).astype(np.float32).reshape(-1)

    # 等角度采样 → 列坐标；左右线性分配
    u_pan_f = theta_global * f_pix
    u0 = np.floor(u_pan_f).astype(np.int32) % Wpan
    a = (u_pan_f - np.floor(u_pan_f)).astype(np.float32)
    u1 = (u0 + 1) % Wpan

    a_flat = a.reshape(-1)
    u0_flat = u0.reshape(-1)
    u1_flat = u1.reshape(-1)
    rows_global_flat = rows_global

    ok = (weight > 1e-6) & ok_y
    if not np.any(ok):
        return
    w_flat = weight[ok]
    a_flat = a_flat[ok]
    u0_flat = u0_flat[ok]
    u1_flat = u1_flat[ok]
    rows_global_flat = rows_global_flat[ok]
    bgr = warped_bgr.reshape(-1, 3).astype(np.float32)[ok]

    wl = w_flat * (1.0 - a_flat)
    wr = w_flat * a_flat

    np.add.at(accum_w, (rows_global_flat, u0_flat), wl)
    np.add.at(accum_rgb, (rows_global_flat, u0_flat, 0), wl * bgr[:, 0])
    np.add.at(accum_rgb, (rows_global_flat, u0_flat, 1), wl * bgr[:, 1])
    np.add.at(accum_rgb, (rows_global_flat, u0_flat, 2), wl * bgr[:, 2])

    np.add.at(accum_w, (rows_global_flat, u1_flat), wr)
    np.add.at(accum_rgb, (rows_global_flat, u1_flat, 0), wr * bgr[:, 0])
    np.add.at(accum_rgb, (rows_global_flat, u1_flat, 1), wr * bgr[:, 1])
    np.add.at(accum_rgb, (rows_global_flat, u1_flat, 2), wr * bgr[:, 2])

# ---------- 主 ----------
def main():
    ap = argparse.ArgumentParser(description="Cylindrical stitch using only subject; stretch subject θ to sectors.")
    ap.add_argument("--config", required=True, help="JSON 列表：img/mask/可选fg/theta_deg/可选z_shift_px")
    ap.add_argument("--outdir", default="out/ring1", help="输出目录")
    ap.add_argument("--use-fg", action="store_true", help="若提供 fg，则用其做颜色（掩膜仍以 mask 判定主体）")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--f-pix", type=float, default=None, help="像素焦距（优先）")
    g.add_argument("--fov-h-deg", type=float, default=73.74, help="水平 FOV（度），默认≈24mm 等效")
    ap.add_argument("--orig-width", type=int, default=1920, help="由 FOV 推算 f_pix 时的原始宽度")

    # 主体检测与边缘控制
    ap.add_argument("--mask-thresh", type=float, default=0.5, help="圆柱化后掩膜阈值（0~1）")
    ap.add_argument("--erode-px", type=int, default=1, help="腐蚀像素，收边；0=关闭")
    ap.add_argument("--col-cov-thresh", type=float, default=0.05, help="一列被视为主体列的最低覆盖率")
    ap.add_argument("--subject-margin-frac", type=float, default=0.03, help="收紧主体θ范围的相对边距(0~0.45)")
    ap.add_argument("--edge-soft-deg", type=float, default=2.0, help="主体边缘的半余弦软化角宽（度）")

    # 扇区与接缝
    ap.add_argument("--theta-blend-deg", type=float, default=6.0, help="相邻扇区边界羽化半宽（度）")
    ap.add_argument("--theta-map", choices=["physical", "stretch", "stretch-subject"], default="stretch-subject",
                    help="θ映射模式：physical=真实；stretch=整张拉伸；stretch-subject=仅主体范围拉伸（推荐）")

    # 输出
    ap.add_argument("--save-warped", action="store_true", help="保存中间结果（warped/掩膜/权重）")
    ap.add_argument("--pan-h", type=int, default=None, help="全景高度；缺省=各图圆柱化高度最大值")
    ap.add_argument("--theta-wrap", action="store_true", help="末列附加首列，便于环视预览")
    args = ap.parse_args()

    items = json.loads(Path(args.config).read_text(encoding="utf-8"))
    assert len(items) >= 2, "至少需要两张图像"

    # 统一 f_pix
    f_pix = float(args.f_pix) if args.f_pix is not None else f_from_fov_deg(args.orig_width, args.fov_h_deg)

    outdir = Path(args.outdir)
    (outdir / "warped").mkdir(parents=True, exist_ok=True)

    warped_list, weight_base_list, theta_list = [], [], []
    info_list, name_list, zshift_list = [], [], []
    theta_center_deg_list, h_list = [], []
    subj_bounds_list = []

    # 逐张：圆柱化 + 主体θ范围提取 + 边缘降权
    for idx, it in enumerate(items):
        name = it.get("name", f"cam{idx}")
        img = imread_color(it["img"])
        H, W = img.shape[:2]
        mask = imread_mask01(it.get("mask", None))
        mask = ensure_mask_size(mask, H, W)

        if args.use_fg and it.get("fg", None) and Path(it["fg"]).exists():
            color_img = imread_color(it["fg"])
        else:
            color_img = (img.astype(np.float32) * mask[..., None]).astype(np.uint8)

        warped, w_base, theta_local, info, warped_mask = cylindrical_warp_strict(
            color_img, mask, f_pix, mask_thresh=args.mask_thresh, erode_px=args.erode_px
        )

        # 主体θ范围（基于列覆盖率）
        bounds = subject_theta_bounds(
            warped_mask, theta_local,
            col_cov_thresh=args.col_cov_thresh,
            margin_frac=args.subject_margin_frac
        )
        subj_bounds_list.append(bounds)

        # 边缘软化（只在主体内部软化）
        if bounds is not None:
            w_base = edge_soften_columns(w_base, theta_local, bounds["col0"], bounds["col1"], args.edge_soft_deg)

        if args.save_warped:
            cv2.imwrite(str(outdir / "warped" / f"{idx:02d}_{name}_warped.png"), warped)
            cv2.imwrite(str(outdir / "warped" / f"{idx:02d}_{name}_mask.png"), (warped_mask * 255).astype(np.uint8))
            w_vis = np.clip(w_base / (w_base.max() + 1e-6) * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(str(outdir / "warped" / f"{idx:02d}_{name}_weight.png"), w_vis)

        warped_list.append(warped)
        weight_base_list.append(w_base)
        theta_list.append(theta_local)
        info_list.append(info)
        name_list.append(name)
        zshift_list.append(int(it.get("z_shift_px", 0)))
        theta_center_deg_list.append(float(it["theta_deg"]))
        h_list.append(warped.shape[0])

    # 全景画布
    Hpan = int(max(h_list)) if (args.pan_h is None) else int(args.pan_h)
    Wpan = int(round(f_pix * 2.0 * math.pi))
    accum_rgb = np.zeros((Hpan, Wpan, 3), dtype=np.float32)
    accum_w = np.zeros((Hpan, Wpan), dtype=np.float32)

    # 扇区（根据中心角自动计算边界）
    centers_rad = [wrap_0_2pi(math.radians(t)) for t in theta_center_deg_list]
    sectors, centers_sorted = build_sectors(centers_rad)
    c2s = {c: s for c, s in zip(centers_sorted, sectors)}
    sector_per_img = []
    for tdeg in theta_center_deg_list:
        c = wrap_0_2pi(math.radians(tdeg))
        best_c = min(centers_sorted, key=lambda x: abs(angle_diff_signed(c, x)))
        sector_per_img.append(c2s[best_c])

    blend_rad = math.radians(max(0.0, args.theta_blend_deg))

    # 逐张放入（默认垂直居中 + 可选 z_shift）
    for warped, w_base, theta_local, info, name, zshift, tdeg, sector, bounds in zip(
        warped_list, weight_base_list, theta_list, info_list, name_list,
        zshift_list, theta_center_deg_list, sector_per_img, subj_bounds_list
    ):
        v0 = (Hpan - warped.shape[0]) // 2 + zshift
        theta_off = wrap_0_2pi(math.radians(tdeg))
        subject_theta0 = bounds["theta0"] if (bounds is not None) else None
        subject_theta1 = bounds["theta1"] if (bounds is not None) else None
        place_on_panorama(
            accum_rgb, accum_w,
            warped, w_base, theta_local,
            theta_off, f_pix,
            sector, blend_rad, v_offset=v0,
            theta_map_mode=args.theta_map,
            subject_theta0=subject_theta0, subject_theta1=subject_theta1
        )

    # 输出
    eps = 1e-6
    out = (accum_rgb / (accum_w[..., None] + eps)).astype(np.uint8)
    if args.theta_wrap:
        out = np.concatenate([out, out[:, :1, :]], axis=1)

    # 覆盖率报告
    col_has = (accum_w > 1e-6).any(axis=0)
    coverage = float(col_has.sum()) / float(col_has.size) * 100.0
    print(f"[Coverage] non-empty columns = {coverage:.1f}% (mode={args.theta_map})")

    outdir = Path(args.outdir)
    cv2.imwrite(str(outdir / "panorama.png"), out)
    w_vis = np.clip(accum_w / (accum_w.max() + eps) * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(outdir / "weight.png"), w_vis)

    print(f"[Done] {outdir/'panorama.png'}  f_pix={f_pix:.1f}, Wpan={Wpan}, Hpan={Hpan}, theta_blend_deg={args.theta_blend_deg}, mode={args.theta_map}")

if __name__ == "__main__":
    main()
