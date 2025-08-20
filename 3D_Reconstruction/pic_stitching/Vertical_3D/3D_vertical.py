import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


# ------------------------- I/O -------------------------

def imread_color(p: Path):
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return img

def imread_mask01(p: Path, thresh: int = 128):
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(p)
    return (m >= thresh).astype(np.uint8)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------- 相机与几何 ----------------------

def compute_fx_pixels(img_w_px, f_mm, f_mode="equivalent", sensor_w_mm=7.6):
    """
    - equivalent: fx ≈ f_mm / 36 * W
    - physical  : fx ≈ f_mm / sensor_w_mm * W
    """
    if f_mode == "equivalent":
        return float(f_mm / 36.0 * img_w_px)
    elif f_mode == "physical":
        return float(f_mm / sensor_w_mm * img_w_px)
    else:
        raise ValueError("f_mode must be 'equivalent' or 'physical'")

def estimate_cx_from_mask(mask, use_mid_fraction=0.6):
    h, w = mask.shape
    y0 = int((1 - use_mid_fraction) / 2 * h); y1 = h - y0
    centers = []
    for y in range(y0, y1):
        xs = np.where(mask[y] > 0)[0]
        if xs.size >= 2:
            centers.append(0.5 * (xs[0] + xs[-1]))
    return float(np.median(centers)) if centers else w / 2.0

def theta_from_x(x, cx, fx):
    return np.arctan((x - cx) / fx)

def theta_span_from_mask(mask, fx, cx, use_mid_fraction=0.9, ql=5, qr=95):
    """根据 mask 统计可用 θ 范围"""
    h, w = mask.shape
    y0 = int((1 - use_mid_fraction) / 2 * h); y1 = h - y0
    lefts, rights = [], []
    for y in range(y0, y1):
        xs = np.where(mask[y] > 0)[0]
        if xs.size >= 2:
            lefts.append(xs[0]); rights.append(xs[-1])
    if not lefts:
        x_min, x_max = 0, w - 1
    else:
        x_min = int(np.percentile(lefts, ql))
        x_max = int(np.percentile(rights, qr))
    tmin = theta_from_x(x_min, cx, fx)
    tmax = theta_from_x(x_max, cx, fx)
    if tmin > tmax: tmin, tmax = tmax, tmin
    return float(tmin), float(tmax)

def unwrap_to_theta(img_bgr, mask01, fx, cx, theta_min, theta_max, theta_cols):
    """
    单次圆柱展开：横向按 θ 均匀采样，纵向保持像素 y，不再缩放。
    """
    h, w = img_bgr.shape[:2]
    thetas = np.linspace(theta_min, theta_max, theta_cols, dtype=np.float32)
    xs = fx * np.tan(thetas) + cx
    map_x = np.tile(xs.reshape(1, -1), (h, 1)).astype(np.float32)
    map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, theta_cols))
    U = cv2.remap(img_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    m_u8 = (mask01 * 255).astype(np.uint8)
    Mu = cv2.remap(m_u8, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return U, (Mu > 127).astype(np.uint8)

def unwrap_interval_to_match(ref_center, tmin, tmax):
    """
    把 [tmin,tmax] 通过加减 2π 移到最接近 ref_center 的连续区间，避免跨 ±π 造成错位。
    """
    center = 0.5 * (tmin + tmax)
    k = round((ref_center - center) / (2.0 * math.pi))
    shift = k * 2.0 * math.pi
    return tmin + shift, tmax + shift


# --------------------- OBJ/MTL 导出（单片圆柱） ---------------------

def build_cyl_patch(R, theta_start, theta_end, z_start, z_end, seg_theta=256, seg_z=256):
    thetas = np.linspace(theta_start, theta_end, seg_theta + 1, dtype=np.float64)
    zvals  = np.linspace(z_start,   z_end,   seg_z    + 1, dtype=np.float64)
    V, VT, F = [], [], []
    for j in range(seg_z + 1):
        v_param = j / seg_z
        z = zvals[j]
        v_tex = 1.0 - v_param
        for i in range(seg_theta + 1):
            u_param = i / seg_theta
            th = thetas[i]
            x = R * math.cos(th); y = R * math.sin(th)
            V.append((x, y, z))
            VT.append((u_param, v_tex))
    def vid(i, j): return j * (seg_theta + 1) + i
    for j in range(seg_z):
        for i in range(seg_theta):
            v00 = vid(i, j); v10 = vid(i+1, j); v01 = vid(i, j+1); v11 = vid(i+1, j+1)
            F.append((v00+1, v10+1, v11+1))
            F.append((v00+1, v11+1, v01+1))
    return V, VT, F

def write_obj_mtl(out_dir: Path, name: str, V, VT, F, tex_name: str):
    out_dir = ensure_dir(out_dir)
    obj_path = out_dir / f"{name}.obj"
    mtl_path = out_dir / f"{name}.mtl"
    with open(mtl_path, "w", encoding="utf-8") as f:
        f.write("newmtl mat\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 2\n")
        f.write(f"map_Kd {tex_name}\n")
    with open(obj_path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_path.name}\nusemtl mat\ns off\n")
        for x, y, z in V:   f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for u, v in VT:     f.write(f"vt {u:.6f} {v:.6f}\n")
        for a, b, c in F:   f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
    return obj_path, mtl_path


# --------------------- 主流程：Z向拼接（下盖上） ---------------------

def main():
    ap = argparse.ArgumentParser(description="Z-direction cylinder stitching: overlap resolved by LOWER patch covering UPPER patch (e.g., 0001 covers 0002 if dz_px<0).")
    ap.add_argument("--img1", required=True, type=str)
    ap.add_argument("--img2", required=True, type=str)
    ap.add_argument("--mask1", required=True, type=str)
    ap.add_argument("--mask2", required=True, type=str)
    ap.add_argument("--result-json", required=True, type=str, help="包含 dtheta_rad 与 dz_px 的 result.json")

    # 圆柱与相机
    ap.add_argument("--radius-mm", type=float, default=150.0)
    ap.add_argument("--f-mode", choices=["equivalent", "physical"], default="equivalent")
    ap.add_argument("--f-mm", type=float, default=12.0)
    ap.add_argument("--sensor-w-mm", type=float, default=7.6)

    # 全局纹理分辨率（θ 列数越大越细；z 高度按两图并集像素高度）
    ap.add_argument("--theta-cols", type=int, default=4096)

    # 导出
    ap.add_argument("--out-dir", type=str, default="./out_stitch_z")
    ap.add_argument("--name", type=str, default="cyl_z_lower_covers")
    ap.add_argument("--mm-per-px", type=float, default=None, help="若提供，则 OBJ 的 z 用 mm 表达；否则 z 单位为像素。")

    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))

    # --- 读图 ---
    img1 = imread_color(Path(args.img1))
    img2 = imread_color(Path(args.img2))
    m1 = imread_mask01(Path(args.mask1))
    m2 = imread_mask01(Path(args.mask2))
    H, W = img1.shape[:2]
    assert img2.shape[:2] == (H, W) and m1.shape == (H, W) and m2.shape == (H, W), "两图及掩膜尺寸需一致"

    # --- 相机横向像素焦距与主点 ---
    fx = compute_fx_pixels(W, args.f_mm, args.f_mode, args.sensor_w_mm)
    cx = 0.5 * (estimate_cx_from_mask(m1) + estimate_cx_from_mask(m2))

    # --- θ 范围（各自图） ---
    t1_min, t1_max = theta_span_from_mask(m1, fx, cx)
    t2_min, t2_max = theta_span_from_mask(m2, fx, cx)

    # --- 读位姿结果（dtheta, dz_px） ---
    with open(args.result_json, "r", encoding="utf-8") as f:
        R = json.load(f)

    def fetch_dtheta_rad(R):
        for key in ["physical_model", "ransac", "ransac_pixel_model", "phase_corr_check"]:
            if key in R and R[key].get("dtheta_rad", None) is not None:
                return float(R[key]["dtheta_rad"])
        raise RuntimeError("result.json 未找到 dtheta_rad")

    def fetch_dz_px(R):
        for key in ["ransac", "ransac_pixel_model", "phase_corr_check"]:
            if key in R and R[key].get("dz_px", None) is not None:
                return float(R[key]["dz_px"])
        # 退路：dz_mm + mm_per_px
        mp = R.get("config", {}).get("mm_per_px", None)
        if "physical_model" in R and R["physical_model"].get("dz_mm", None) is not None and mp:
            return float(R["physical_model"]["dz_mm"]) / float(mp)
        raise RuntimeError("result.json 未找到 dz_px；若仅有 dz_mm，请在 config 中提供 mm_per_px。")

    dtheta = fetch_dtheta_rad(R)
    dz_px = fetch_dz_px(R)

    # --- 把第二张的 θ 区间平移到最靠近第一张的连续域，避免跨 ±π 裂缝 ---
    t2_min_s = t2_min + dtheta
    t2_max_s = t2_max + dtheta
    t2_min_s, t2_max_s = unwrap_interval_to_match(0.5*(t1_min+t1_max), t2_min_s, t2_max_s)

    # --- 统一 θ 与 z 全局范围 ---
    tG_min = min(t1_min, t2_min_s)
    tG_max = max(t1_max, t2_max_s)
    Wglob  = int(args.theta_cols)
    rad_per_col = (tG_max - tG_min) / Wglob

    # z（像素坐标），图1在 [0,H)，图2在 [dz_px, dz_px+H)
    z1_min, z1_max = 0, H
    z2_min, z2_max = int(round(dz_px)), int(round(dz_px)) + H
    zG_min = min(z1_min, z2_min)
    zG_max = max(z1_max, z2_max)
    Hglob  = zG_max - zG_min

    # --- 计算两图在全局 θ 轴上的列区间 ---
    def theta_to_cols(tmin, tmax):
        c0 = int(round((tmin - tG_min) / rad_per_col))
        c1 = int(round((tmax - tG_min) / rad_per_col))
        c0 = max(0, min(Wglob, c0)); c1 = max(0, min(Wglob, c1))
        if c1 <= c0: c1 = min(Wglob, c0 + 1)
        return c0, c1, (c1 - c0)

    c1_0, c1_1, w1_cols = theta_to_cols(t1_min, t1_max)
    c2_0, c2_1, w2_cols = theta_to_cols(t2_min_s, t2_max_s)

    # --- 单次展开：直接以“各自的全局列宽”展开，避免二次缩放 ---
    U1, M1u = unwrap_to_theta(img1, m1, fx, cx, t1_min, t1_max, theta_cols=w1_cols)
    U2, M2u = unwrap_to_theta(img2, m2, fx, cx, t2_min, t2_max, theta_cols=w2_cols)  # 注意：这里未加 dtheta，放置时平移

    # --- 构建全局纹理画布 ---
    tex  = np.zeros((Hglob, Wglob, 3), np.uint8)
    mtex = np.zeros((Hglob, Wglob), np.uint8)

    # 图1 的放置（不移动 z）
    r1_0 = z1_min - zG_min  # 行起始
    r1_1 = r1_0 + H
    tex[r1_0:r1_1, c1_0:c1_1] = U1
    mtex[r1_0:r1_1, c1_0:c1_1] = np.maximum(mtex[r1_0:r1_1, c1_0:c1_1], (M1u*255).astype(np.uint8))

    # 图2 的放置（θ 平移后列区间是 [c2_0,c2_1]；z 平移 dz_px）
    r2_0 = z2_min - zG_min
    r2_1 = r2_0 + H

    # 裁剪到全局画布范围
    r2_0c = max(0, r2_0); r2_1c = min(Hglob, r2_1)
    c2_0c = max(0, c2_0); c2_1c = min(Wglob, c2_1)
    if r2_1c > r2_0c and c2_1c > c2_0c:
        # 若越界裁切，对应裁掉 U2/M2
        top_cut = r2_0c - r2_0
        bot_cut = (r2_1) - (r2_1c)
        left_cut = c2_0c - c2_0
        right_cut = (c2_1) - (c2_1c)

        U2_res = U2[max(0, top_cut):U2.shape[0]-max(0, bot_cut),
                    max(0, left_cut):U2.shape[1]-max(0, right_cut)]
        M2_res = M2u[max(0, top_cut):M2u.shape[0]-max(0, bot_cut),
                     max(0, left_cut):M2u.shape[1]-max(0, right_cut)]

        # 目标 ROI
        roi_tex = tex[r2_0c:r2_1c, c2_0c:c2_1c]
        roi_m   = mtex[r2_0c:r2_1c, c2_0c:c2_1c]

        # --- 关键规则：重合区由“更靠下”的贴片覆盖“更靠上”的贴片 ---
        # 计算重叠与非重叠
        m2 = (M2_res > 0)
        m1_in_roi = (roi_m > 0)

        # 判定谁在“下”
        # dz_px < 0  => 图2在“上”，图1在“下” => 下(图1)覆盖上(图2) => 图2只在图1空白处写入
        # dz_px > 0  => 图2在“下”，图1在“上” => 下(图2)覆盖上(图1) => 图2直接按掩膜覆盖
        if dz_px < 0:
            # 仅在原先空白处填入图2
            empty = (~m1_in_roi) & m2
            roi_tex[empty] = U2_res[empty]
            roi_m[empty] = 255
        else:
            # 图2为“下”，按掩膜直接覆盖（含重叠）
            roi_tex[m2] = U2_res[m2]
            roi_m[m2] = 255

        # 回写
        tex[r2_0c:r2_1c, c2_0c:c2_1c] = roi_tex
        mtex[r2_0c:r2_1c, c2_0c:c2_1c] = roi_m

    # --- 保存融合后的展开纹理 ---
    tex_name = "texture.png"
    cv2.imwrite(str(out_dir / tex_name), tex)

    # --- 生成 OBJ（z 单位：像素或毫米） ---
    if args.mm_per_px and args.mm_per_px > 0:
        z_start = 0.0
        z_end   = Hglob * float(args.mm_per_px)
    else:
        z_start = 0.0
        z_end   = float(Hglob)

    V, VT, F = build_cyl_patch(args.radius_mm, tG_min, tG_max, z_start, z_end, seg_theta=256, seg_z=256)
    obj_path, mtl_path = write_obj_mtl(out_dir, args.name, V, VT, F, tex_name)

    # --- 元数据记录 ---
    meta = {
        "rule": "Lower patch covers upper patch in overlap (Z-direction overwrite).",
        "dtheta_rad": dtheta, "dtheta_deg": float(math.degrees(dtheta)),
        "dz_px": dz_px,
        "theta_global": {"min": tG_min, "max": tG_max, "rad_per_col": rad_per_col, "Wglob": Wglob},
        "z_global_px": {"min": zG_min, "max": zG_max, "Hglob": Hglob},
        "mm_per_px": args.mm_per_px,
        "fx_pixels": fx, "cx_used": cx,
        "inputs": {"img1": args.img1, "img2": args.img2, "mask1": args.mask1, "mask2": args.mask2, "result_json": args.result_json},
        "outputs": {"obj": str(obj_path), "mtl": str(mtl_path), "texture": str(out_dir / tex_name)}
    }
    with open(out_dir / "stitch_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("=== STITCH Z (lower covers) DONE ===")
    print(f"Δθ = {math.degrees(dtheta):.4f} deg, Δz = {dz_px:.2f} px")
    print(f"θ-global: [{math.degrees(tG_min):.2f}°, {math.degrees(tG_max):.2f}°] -> {Wglob} cols")
    print(f"z-global(px): [{zG_min}, {zG_max}] -> H={Hglob}")
    if args.mm_per_px and args.mm_per_px > 0:
        print(f"z unit: mm  (mm_per_px={args.mm_per_px})")
    else:
        print("z unit: px")
    print(f"OBJ : {obj_path}\nMTL : {mtl_path}\nTEX : {out_dir/tex_name}")


if __name__ == "__main__":
    main()
