import argparse
from pathlib import Path
import json
import math
import numpy as np
import cv2

# ----------------------------- I/O -----------------------------

def imread_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def imread_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 127).astype(np.uint8)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

# ------------------------- 相机与展开（保留原匹配做法） --------------------------

def compute_fx_pixels(img_w_px, f_mm, f_mode="equivalent", sensor_w_mm=7.6):
    """
    - 'equivalent': 把 f_mm 当作 35mm 等效焦距（36mm 为宽） → fx ≈ f_mm/36 * W
    - 'physical'  : 把 f_mm 当作物理焦距，结合传感器宽度 → fx ≈ f_mm/sensor_w_mm * W
    """
    if f_mode == "equivalent":
        return float(f_mm / 36.0 * img_w_px)
    elif f_mode == "physical":
        return float(f_mm / sensor_w_mm * img_w_px)
    else:
        raise ValueError("f_mode must be 'equivalent' or 'physical'")

def compute_fy_pixels(img_h_px, f_mm, f_mode="equivalent", sensor_h_mm=5.7):
    """
    纵向像素焦距：
    - 'equivalent': fy ≈ f_mm/24 * H      （35mm 胶片高=24mm）
    - 'physical'  : fy ≈ f_mm/sensor_h_mm * H  （1/1.7'' 典型高≈5.7mm，若已知更精确值可传）
    """
    if f_mode == "equivalent":
        return float(f_mm / 24.0 * img_h_px)
    elif f_mode == "physical":
        return float(f_mm / sensor_h_mm * img_h_px)
    else:
        raise ValueError("f_mode must be 'equivalent' or 'physical'")

def estimate_cx_from_mask(mask, use_mid_fraction=0.6):
    """基于 mask 左右边界中点的中位数估计像面中线 cx（仅统计中间若干行）"""
    h, w = mask.shape
    y0 = int((1 - use_mid_fraction) / 2 * h)
    y1 = h - y0
    centers = []
    for y in range(y0, y1):
        xs = np.where(mask[y] > 0)[0]
        if xs.size >= 2:
            centers.append(0.5 * (xs[0] + xs[-1]))
    return float(np.median(centers)) if centers else w / 2.0

def theta_from_x(x, cx, fx):
    return np.arctan((x - cx) / fx)

def build_cylindrical_unwrap(img, mask, fx, cx, theta_cols=2048, use_mid_fraction=0.9):
    """
    把原图按圆柱模型展开到 (θ, z) 网格：
      - 横向：均匀采样 θ
      - 纵向：保持 y 不变
    自动基于 mask 估计可用的 θ 范围（取中间若干行的左右边界）
    """
    h, w = img.shape
    # 统计可用水平范围
    y0 = int((1 - use_mid_fraction) / 2 * h)
    y1 = h - y0
    lefts, rights = [], []
    for y in range(y0, y1):
        xs = np.where(mask[y] > 0)[0]
        if xs.size >= 2:
            lefts.append(xs[0]); rights.append(xs[-1])
    if not lefts:
        x_min, x_max = 0, w - 1
    else:
        x_min = int(np.percentile(lefts, 5))
        x_max = int(np.percentile(rights, 95))

    theta_min = theta_from_x(x_min, cx, fx)
    theta_max = theta_from_x(x_max, cx, fx)
    if theta_min > theta_max:
        theta_min, theta_max = theta_max, theta_min

    thetas = np.linspace(theta_min, theta_max, theta_cols, dtype=np.float32)
    xs = fx * np.tan(thetas) + cx

    map_x = np.tile(xs.reshape(1, -1), (h, 1)).astype(np.float32)
    map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, theta_cols))

    unwrapped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    m_u8 = (mask * 255).astype(np.uint8)
    mask_unwrap = cv2.remap(m_u8, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_unwrap = (mask_unwrap > 127).astype(np.uint8)

    meta = {
        "theta_min": float(theta_min),
        "theta_max": float(theta_max),
        "rad_per_pix": float((theta_max - theta_min) / theta_cols),
        "x_min": int(x_min),
        "x_max": int(x_max)
    }
    return unwrapped, mask_unwrap, meta

# ------------------------- ROI & 特征提取（保持原有方法） ------------------------

def make_roi_mask(base_mask, roi_rect=None, midband_fraction=0.5):
    """
    roi_rect: (xmin, xmax, ymin, ymax) 像素坐标（闭开区间：xmin<=x<xmax）
    若为空，则取中间竖带：宽度比例 midband_fraction，高度全幅
    """
    h, w = base_mask.shape
    roi = np.zeros_like(base_mask, dtype=np.uint8)

    if roi_rect is None:
        band_w = int(w * midband_fraction)
        x0 = (w - band_w) // 2
        x1 = x0 + band_w
        y0, y1 = 0, h
    else:
        x0, x1, y0, y1 = roi_rect
        x0 = max(0, min(w, int(x0))); x1 = max(0, min(w, int(x1)))
        y0 = max(0, min(h, int(y0))); y1 = max(0, min(h, int(y1)))
        if x0 >= x1 or y0 >= y1:
            x0, x1, y0, y1 = 0, w, 0, h

    roi[y0:y1, x0:x1] = 1
    return (roi & (base_mask > 0)).astype(np.uint8)

def detect_features(img, valid_mask, method='SIFT', grid_nx=8, grid_ny=6, per_cell=15, edge_margin=6):
    """
    在 valid_mask==1 的区域内检测特征点并做网格均匀化
    返回：keypoints(list), descriptors(np.ndarray)
    """
    h, w = img.shape
    # 边界收缩，避免靠近 mask 边缘的点
    if edge_margin > 0:
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_margin*2+1, edge_margin*2+1))
        inner_mask = cv2.erode(valid_mask.astype(np.uint8), kern, iterations=1)
    else:
        inner_mask = valid_mask

    # 选择检测器
    detector_name = method.upper()
    sift_ok = False
    if detector_name == 'SIFT':
        try:
            sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02, edgeThreshold=10, sigma=1.2)
            kps, des = sift.detectAndCompute(img, mask=inner_mask)
            if kps is not None and len(kps) > 0:
                sift_ok = True
                kps_all, des_all = kps, des
        except Exception:
            sift_ok = False

    if not sift_ok:
        # 退化为 ORB
        orb = cv2.ORB_create(nfeatures=6000, scaleFactor=1.2, nlevels=8, edgeThreshold=15, fastThreshold=7)
        kps_all, des_all = orb.detectAndCompute(img, mask=inner_mask)
        detector_name = 'ORB'

    if kps_all is None or len(kps_all) == 0:
        return [], None, detector_name

    # 网格均匀化
    cell_w = w / grid_nx
    cell_h = h / grid_ny
    # 将 kp 按响应排序（大的优先）
    idxs = list(range(len(kps_all)))
    idxs.sort(key=lambda i: kps_all[i].response, reverse=True)

    selected = [[] for _ in range(grid_nx * grid_ny)]
    keep_idx = []

    for i in idxs:
        x, y = kps_all[i].pt
        cx = int(min(grid_nx - 1, max(0, x // cell_w)))
        cy = int(min(grid_ny - 1, max(0, y // cell_h)))
        cid = cy * grid_nx + cx
        if len(selected[cid]) < per_cell:
            selected[cid].append(i)
            keep_idx.append(i)

    kps = [kps_all[i] for i in keep_idx]
    des = des_all[keep_idx] if des_all is not None and len(keep_idx) > 0 else None
    return kps, des, detector_name

def match_features(des1, des2, method='SIFT', ratio=0.75):
    """
    返回筛选后的 DMatch 列表（Lowe 比率测试）
    """
    if method.upper() == 'SIFT':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in raw:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

# ---------------------- RANSAC 平移估计(θ,z)（保留原像素域估计） ---------------------

def wrap_to_pi(angle):
    """把角度包裹到 (-pi, pi]"""
    return (angle + np.pi) % (2*np.pi) - np.pi

def ransac_translation_theta_z(theta1, z1, theta2, z2,
                               thr_theta_rad=0.02, thr_z_px=6.0,
                               iters=2000, random_state=0):
    """
    输入：成对数组 (theta1, z1) ↔ (theta2, z2)
    返回：dtheta, dz, inlier_mask(bool), stats(dict)
    """
    assert len(theta1) == len(theta2) == len(z1) == len(z2)
    n = len(theta1)
    if n == 0:
        return None, None, np.zeros(0, dtype=bool), {"num_inliers": 0, "num_matches": 0}

    rng = np.random.RandomState(random_state)
    best_inliers = None
    best_count = -1

    dtheta_all = wrap_to_pi(theta2 - theta1)
    dz_all = (z2 - z1)

    for _ in range(iters):
        i = int(rng.randint(0, n))
        dth = dtheta_all[i]
        dz = dz_all[i]
        # 评估内点
        inliers = (np.abs(wrap_to_pi(dtheta_all - dth)) < thr_theta_rad) & (np.abs(dz_all - dz) < thr_z_px)
        cnt = int(inliers.sum())
        if cnt > best_count:
            best_count = cnt
            best_inliers = inliers

    if best_inliers is None or best_count <= 0:
        return None, None, np.zeros(n, dtype=bool), {"num_inliers": 0, "num_matches": n}

    # 在内点上用中位数细化
    dtheta_refined = float(np.median(dtheta_all[best_inliers]))
    dz_refined = float(np.median(dz_all[best_inliers]))

    # 统计量（MAD）
    mad_theta = float(np.median(np.abs(wrap_to_pi(dtheta_all[best_inliers] - dtheta_refined))) + 1e-12)
    mad_z = float(np.median(np.abs(dz_all[best_inliers] - dz_refined)) + 1e-12)

    stats = {
        "num_matches": n,
        "num_inliers": int(best_inliers.sum()),
        "inlier_ratio": float(best_inliers.mean()),
        "mad_theta_rad": mad_theta,
        "mad_theta_deg": mad_theta * 180.0 / math.pi,
        "mad_z_px": mad_z
    }
    return dtheta_refined, dz_refined, best_inliers, stats

# -------------------------- 可视化 -------------------------------

def draw_matches(img1, kps1, img2, kps2, matches, out_path, max_draw=300):
    """绘制匹配连线"""
    if len(matches) == 0:
        return
    draw = matches[:min(max_draw, len(matches))]
    vis = cv2.drawMatches(
        cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR), kps1,
        cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR), kps2,
        draw, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(str(out_path), vis)

# ------------------------ 相位相关校验（修正 Hanning 断言） ---------------------------

def phase_corr_shift(img1_u, img2_u, use_hanning=True):
    f1 = img1_u.astype(np.float32)
    f2 = img2_u.astype(np.float32)
    if use_hanning:
        h, w = f1.shape
        window = cv2.createHanningWindow((w, h), cv2.CV_32F)  # 2D窗，兼容 OpenCV 4.5.1
        f1 *= window
        f2 *= window
    (dx, dy), resp = cv2.phaseCorrelate(f1, f2)
    return float(dx), float(dy), float(resp)

# ------------------------ 物理几何求解（新增，仅用于匹配完成后） -------------------------

def fx_fy_pixels(f_mode, f_mm, W, H, sensor_w_mm=7.6, sensor_h_mm=5.7):
    """根据模式返回 (fx, fy) 像素焦距"""
    fx = compute_fx_pixels(W, f_mm, f_mode=f_mode, sensor_w_mm=sensor_w_mm)
    fy = compute_fy_pixels(H, f_mm, f_mode=f_mode, sensor_h_mm=sensor_h_mm)
    return fx, fy

def ray_cylinder_intersect_and_phi(xn, yn, R_mm, rho_mm):
    """
    相机坐标：光轴 +Z，成像面归一化 ray=[xn, yn, 1]。
    圆柱轴与 +Z 平行，轴线在 X 方向距相机中心 rho_mm，半径 R_mm。
    交点方程：A Z^2 + B Z + C = 0
      A = xn^2 + yn^2
      B = -2*rho*xn
      C = rho^2 - R^2
    取最近正根；返回 (Z_mm, phi=atan2(Y, X-rho))
    """
    A = xn * xn + yn * yn
    B = -2.0 * rho_mm * xn
    C = rho_mm * rho_mm - R_mm * R_mm
    disc = B * B - 4.0 * A * C
    if A <= 1e-12 or disc < 0:
        return None, None
    sqrt_disc = math.sqrt(disc)
    Z1 = (-B - sqrt_disc) / (2.0 * A)
    Z2 = (-B + sqrt_disc) / (2.0 * A)
    candidates = [z for z in (Z1, Z2) if z > 0]
    if not candidates:
        return None, None
    Z = min(candidates)
    X = xn * Z
    Y = yn * Z
    phi = math.atan2(Y, X - rho_mm)
    return Z, phi

def ransac_scalar(values, thresh, iters=1500, random_state=0):
    n = len(values)
    if n == 0:
        return None, np.zeros(0, dtype=bool)
    rng = np.random.RandomState(random_state)
    best_inl = None
    best_cnt = -1
    for _ in range(iters):
        i = int(rng.randint(0, n))
        h = values[i]
        inl = np.abs(values - h) < thresh
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_inl = inl
    if best_inl is None:
        return None, np.zeros(n, dtype=bool)
    med = float(np.median(values[best_inl]))
    return med, best_inl

def ransac_circular(angles, thr_rad=math.radians(2.0), iters=1500, random_state=0):
    n = len(angles)
    if n == 0:
        return None, np.zeros(0, dtype=bool)
    rng = np.random.RandomState(random_state)
    best_inl = None
    best_cnt = -1
    for _ in range(iters):
        i = int(rng.randint(0, n))
        h = angles[i]
        d = np.abs(wrap_to_pi(angles - h))
        inl = d < thr_rad
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_inl = inl
    if best_inl is None:
        return None, np.zeros(n, dtype=bool)
    # 圆均值
    mu = float(np.angle(np.mean(np.exp(1j * angles[best_inl]))))
    return mu, best_inl

# ---------------------------- 主流程 -----------------------------

def process_pair(img1_path, img2_path, mask1_path, mask2_path, args):
    outdir = ensure_dir(Path(args.out_dir) / f"pair_{Path(img1_path).stem}_{Path(img2_path).stem}")

    # 读取
    img1 = imread_gray(img1_path)
    img2 = imread_gray(img2_path)
    m1 = imread_mask(mask1_path)
    m2 = imread_mask(mask2_path)
    assert img1.shape == img2.shape == m1.shape == m2.shape, "两张图及其 mask 必须同尺寸"

    H, W = img1.shape
    # cx & fx
    cx1 = estimate_cx_from_mask(m1, use_mid_fraction=0.6)
    cx2 = estimate_cx_from_mask(m2, use_mid_fraction=0.6)
    cx = 0.5 * (cx1 + cx2)
    fx = compute_fx_pixels(W, args.f_mm, args.f_mode, args.sensor_w_mm)
    fy = compute_fy_pixels(H, args.f_mm, args.f_mode, args.sensor_h_mm)
    cy = H / 2.0  # 纵向主点：默认取图像中线（若有更精确标定可替换）

    # ROI ∩ mask
    roi_mask1 = make_roi_mask(m1, None, args.midband_fraction)
    roi_mask2 = make_roi_mask(m2, None, args.midband_fraction)

    # ——【保持原有】特征检测（网格均匀化）——
    kps1, des1, det1 = detect_features(img1, roi_mask1, method='SIFT',
                                       grid_nx=args.grid_nx, grid_ny=args.grid_ny,
                                       per_cell=args.per_cell, edge_margin=args.edge_margin)
    kps2, des2, det2 = detect_features(img2, roi_mask2, method='SIFT',
                                       grid_nx=args.grid_nx, grid_ny=args.grid_ny,
                                       per_cell=args.per_cell, edge_margin=args.edge_margin)
    if des1 is None or des2 is None or len(kps1) < 8 or len(kps2) < 8:
        raise RuntimeError("特征不足，无法估计。请检查纹理/ROI/掩膜。")

    # ——【保持原有】匹配 + Lowe 比率——
    matches = match_features(des1, des2, method=det1 if det1 == det2 else 'SIFT', ratio=args.ratio)

    # 取匹配点坐标
    pts1 = np.array([kps1[m.queryIdx].pt for m in matches], dtype=np.float64)
    pts2 = np.array([kps2[m.trainIdx].pt for m in matches], dtype=np.float64)

    # ========= A) 原像素域：映射到 (θ, z) ，做 RANSAC 平移 =========
    theta1 = np.arctan((pts1[:, 0] - cx) / fx)
    theta2 = np.arctan((pts2[:, 0] - cx) / fx)
    z1 = pts1[:, 1]
    z2 = pts2[:, 1]

    dtheta, dz, inlier_mask_pix, stats = ransac_translation_theta_z(
        theta1, z1, theta2, z2,
        thr_theta_rad=args.ransac_thr_theta_rad,
        thr_z_px=args.ransac_thr_z_px,
        iters=args.ransac_iters,
        random_state=0
    )
    if dtheta is None:
        raise RuntimeError("RANSAC 估计失败：有效内点为 0。")

    # 可视化：匹配（全部 & 像素域内点）
    draw_matches(img1, kps1, img2, kps2, matches, outdir / "matches_all.jpg", max_draw=400)
    inlier_matches_pix = [m for i, m in enumerate(matches) if inlier_mask_pix[i]]
    draw_matches(img1, kps1, img2, kps2, inlier_matches_pix, outdir / "matches_inliers.jpg", max_draw=400)

    # ========= B) 物理域：利用 fx/fy + 圆柱几何反演（新增，但不改匹配方法） =========
    # 归一化像平面坐标
    xn1 = (pts1[:, 0] - cx) / fx
    yn1 = (pts1[:, 1] - cy) / fy
    xn2 = (pts2[:, 0] - cx) / fx
    yn2 = (pts2[:, 1] - cy) / fy

    R_mm = args.radius_mm
    rho_mm = args.radius_mm + args.standoff_mm  # 相机到轴心距离

    Z1 = np.zeros(len(matches), dtype=np.float64)
    Z2 = np.zeros(len(matches), dtype=np.float64)
    phi1 = np.zeros(len(matches), dtype=np.float64)
    phi2 = np.zeros(len(matches), dtype=np.float64)
    ok = np.ones(len(matches), dtype=bool)

    for i in range(len(matches)):
        z1_mm, ph1 = ray_cylinder_intersect_and_phi(xn1[i], yn1[i], R_mm, rho_mm)
        z2_mm, ph2 = ray_cylinder_intersect_and_phi(xn2[i], yn2[i], R_mm, rho_mm)
        if (z1_mm is None) or (z2_mm is None):
            ok[i] = False
        else:
            Z1[i], phi1[i], Z2[i], phi2[i] = z1_mm, ph1, z2_mm, ph2

    if ok.sum() < 8:
        print("[Warn] 物理几何可解的匹配过少，跳过物理域稳健估计。")
        do_phys = False
    else:
        do_phys = True
        keep_idx = np.where(ok)[0]
        # 物理量：逐对
        dy_px = pts2[keep_idx, 1] - pts1[keep_idx, 1]
        dz_i_mm = - dy_px * (Z1[keep_idx] / fy)           # 竖向位移（mm）
        dtheta_i = wrap_to_pi(phi2[keep_idx] - phi1[keep_idx])  # 环向角（rad）

        # 阈值：若未指定，基于先验（fps & 速度）设置
        expected_dz = args.speed_mm_s / args.fps  # ~10 mm（100fps & 1m/s）
        thr_dz_mm = args.thr_dz_mm_phys if args.thr_dz_mm_phys is not None else max(15.0, 2.5 * expected_dz)
        thr_th_rad = math.radians(args.thr_theta_deg_phys) if args.thr_theta_deg_phys is not None else math.radians(2.0)

        dz_mm_med, inl_z = ransac_scalar(dz_i_mm, thresh=thr_dz_mm, iters=max(1500, args.ransac_iters))
        dth_rad_mu, inl_th = ransac_circular(dtheta_i, thr_rad=thr_th_rad, iters=max(1500, args.ransac_iters))

        # 质量指标
        mad_dz_mm = float(np.median(np.abs(dz_i_mm[inl_z] - dz_mm_med))) if inl_z.any() else float('nan')
        mad_th_rad = float(np.median(np.abs(wrap_to_pi(dtheta_i[inl_th] - dth_rad_mu)))) if inl_th.any() else float('nan')

    # ========= 展开域相位相关（保持原逻辑；修掉 Hanning 断言） =========
    I1u, M1u, meta1 = build_cylindrical_unwrap(img1, m1, fx, cx, theta_cols=args.theta_cols, use_mid_fraction=0.9)
    I2u, M2u, meta2 = build_cylindrical_unwrap(img2, m2, fx, cx, theta_cols=args.theta_cols, use_mid_fraction=0.9)
    cv2.imwrite(str(outdir/"unwrap_1.png"), I1u)
    cv2.imwrite(str(outdir/"unwrap_2.png"), I2u)

    def masked_fill(a, m):
        b = a.astype(np.float32).copy()
        if m.sum() > 0:
            meanv = float(a[m > 0].mean())
        else:
            meanv = float(a.mean())
        b[m == 0] = meanv
        b = (b - b.mean()) / (b.std() + 1e-6)
        return b

    A = masked_fill(I1u, M1u & M2u)
    B = masked_fill(I2u, M1u & M2u)
    dx, dy, resp = phase_corr_shift(A, B, use_hanning=True)
    rad_per_pix = meta1["rad_per_pix"]
    dtheta_pc = dx * rad_per_pix
    dz_pc = dy

    # 展开域对齐叠加图
    h, w = I1u.shape
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    I2_align = cv2.warpAffine(I2u, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    blend = np.clip(0.5*I1u.astype(np.float32) + 0.5*I2_align.astype(np.float32), 0, 255).astype(np.uint8)
    cv2.imwrite(str(outdir/"unwrap_blend_aligned.png"), blend)

    # 结果汇总
    result = {
        "image1": str(img1_path),
        "image2": str(img2_path),
        "mask1": str(mask1_path),
        "mask2": str(mask2_path),
        "W": W, "H": H,
        "fx_pixels": fx,
        "fy_pixels": fy,
        "principal_point": {"cx_used": cx, "cy_used": cy},
        "cx_estimates": {"img1": cx1, "img2": cx2, "used": cx},
        "detector": {"img1": det1, "img2": det2},
        "matches": {
            "num_raw": int(len(matches)),
            "num_inliers_pixel_model": int(stats["num_inliers"]),
            "inlier_ratio_pixel_model": stats["inlier_ratio"],
            "mad_theta_deg_pixel_model": stats["mad_theta_deg"],
            "mad_z_px_pixel_model": stats["mad_z_px"]
        },
        "ransac_pixel_model": {
            "dtheta_rad": dtheta,
            "dtheta_deg": dtheta * 180.0 / math.pi,
            "dz_px": dz,
            "dz_mm": (dz * args.mm_per_px) if (args.mm_per_px is not None and args.mm_per_px > 0) else None,
            "thr_theta_rad": args.ransac_thr_theta_rad,
            "thr_z_px": args.ransac_thr_z_px
        },
        "phase_corr_check": {
            "resp": resp,
            "dtheta_rad": dtheta_pc,
            "dtheta_deg": dtheta_pc * 180.0 / math.pi,
            "dz_px": dz_pc
        },
        "unwrap": {
            "theta_min": meta1["theta_min"],
            "theta_max": meta1["theta_max"],
            "rad_per_pix": meta1["rad_per_pix"]
        },
        "config": {
            "f_mode": args.f_mode,
            "f_mm": args.f_mm,
            "sensor_w_mm": args.sensor_w_mm,
            "sensor_h_mm": args.sensor_h_mm,
            "theta_cols": args.theta_cols,
            "midband_fraction": args.midband_fraction,
            "grid_nx": args.grid_nx, "grid_ny": args.grid_ny, "per_cell": args.per_cell,
            "edge_margin": args.edge_margin,
            "ratio": args.ratio
        }
    }

    if do_phys:
        result["physical_model"] = {
            "radius_mm": R_mm,
            "standoff_mm": args.standoff_mm,
            "rho_mm": rho_mm,
            "fps": args.fps,
            "speed_mm_s": args.speed_mm_s,
            "expected_dz_mm": args.speed_mm_s / args.fps,
            "dtheta_rad": dth_rad_mu,
            "dtheta_deg": float(dth_rad_mu * 180.0 / math.pi),
            "dz_mm": dz_mm_med,
            "thr_dz_mm": float(args.thr_dz_mm_phys if args.thr_dz_mm_phys is not None else max(15.0, 2.5 * (args.speed_mm_s/args.fps))),
            "thr_theta_deg": float(args.thr_theta_deg_phys if args.thr_theta_deg_phys is not None else 2.0),
            "inliers_z": int(inl_z.sum()),
            "inliers_theta": int(inl_th.sum()),
            "mad_dz_mm": float(np.median(np.abs(dz_i_mm[inl_z] - dz_mm_med))) if inl_z.any() else None,
            "mad_theta_deg": float(np.degrees(mad_th_rad)) if not np.isnan(mad_th_rad) else None
        }

    with open(outdir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 控制台友好输出
    print("==== RESULT ====")
    print(f"Detector: {det1} / {det2}")
    print(f"Raw matches: {len(matches)} | Pixel-Model Inliers: {stats['num_inliers']} ({stats['inlier_ratio']*100:.1f}%)")
    print(f"[Pixel-Model] Δθ: {result['ransac_pixel_model']['dtheta_deg']:.4f} deg "
          f"({result['ransac_pixel_model']['dtheta_rad']:.6f} rad), Δz: {result['ransac_pixel_model']['dz_px']:.3f} px")
    if do_phys:
        print(f"[Physical]    Δθ: {result['physical_model']['dtheta_deg']:.4f} deg "
              f"({result['physical_model']['dtheta_rad']:.6f} rad), Δz: {result['physical_model']['dz_mm']:.3f} mm "
              f"(inliers θ/z = {result['physical_model']['inliers_theta']}/{result['physical_model']['inliers_z']})")
    print(f"[PhaseCorr] Δθ≈{result['phase_corr_check']['dtheta_deg']:.4f} deg, Δz≈{result['phase_corr_check']['dz_px']:.3f} px, resp={resp:.3f}")
    print(f"Saved to: {outdir}")

def main():
    ap = argparse.ArgumentParser(description="Keep SIFT matching as-is; then compute true vertical shift (mm) and circumferential angle (rad/deg) using focal length and cylinder geometry.")
    ap.add_argument("--img1", required=True, type=str)
    ap.add_argument("--img2", required=True, type=str)
    ap.add_argument("--mask1", required=True, type=str)
    ap.add_argument("--mask2", required=True, type=str)
    ap.add_argument("--out-dir", type=str, default="./out")

    # 相机/展开参数（与原脚本兼容）
    ap.add_argument("--f-mode", choices=["equivalent", "physical"], default="equivalent")
    ap.add_argument("--f-mm", type=float, default=12.0)
    ap.add_argument("--sensor-w-mm", type=float, default=7.6)
    ap.add_argument("--sensor-h-mm", type=float, default=5.7)   # 新增：物理模式用于 fy 计算
    ap.add_argument("--theta-cols", type=int, default=2048)
    ap.add_argument("--midband-fraction", type=float, default=0.5, help="ROI宽度占整图宽度的比例（中间竖带）")

    # 特征与匹配（保持原参数）
    ap.add_argument("--grid-nx", type=int, default=8)
    ap.add_argument("--grid-ny", type=int, default=6)
    ap.add_argument("--per-cell", type=int, default=15, help="每个网格最多保留的特征数")
    ap.add_argument("--edge-margin", type=int, default=6, help="与mask边缘的安全距离(px)")
    ap.add_argument("--ratio", type=float, default=0.75, help="Lowe比率阈值")

    # 像素域 RANSAC（保持原参数）
    ap.add_argument("--ransac-thr-theta-rad", type=float, default=0.02, help="θ差阈值（弧度）≈1.15°")
    ap.add_argument("--ransac-thr-z-px", type=float, default=6.0, help="z差阈值（像素）")
    ap.add_argument("--ransac-iters", type=int, default=2000)
    ap.add_argument("--mm-per-px", type=float, default=None, help="若提供，则输出 Δz 的 mm 值（像素域结果附带）")

    # 物理域几何参数（新增但默认就是你给的数值）
    ap.add_argument("--radius-mm", type=float, default=150.0, help="圆柱体半径")
    ap.add_argument("--standoff-mm", type=float, default=50.0, help="相机到圆柱表面的距离（轴心距=R+standoff）")
    ap.add_argument("--fps", type=float, default=100.0)
    ap.add_argument("--speed-mm-s", type=float, default=1000.0)  # 1 m/s
    ap.add_argument("--thr-theta-deg-phys", type=float, default=None, help="物理域 RANSAC 的角度阈值（度），默认 2°")
    ap.add_argument("--thr-dz-mm-phys", type=float, default=None, help="物理域 RANSAC 的竖向阈值（mm），默认~max(15, 2.5*预期)")

    args = ap.parse_args()
    process_pair(args.img1, args.img2, args.mask1, args.mask2, args)

if __name__ == "__main__":
    main()
