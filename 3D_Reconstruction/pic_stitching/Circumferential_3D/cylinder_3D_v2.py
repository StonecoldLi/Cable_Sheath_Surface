# bake_cylinder_seamcontrol.py
# Python 3.7+
# 功能：
# - 多相机投影到圆柱表面，离线烘焙成贴图(albedo.png)
# - 接缝融合：avg(默认)/max/softmax，可调温度
# - 相机扇区窗 + 边界羽化，稳定缝位置
# - 镜像：烘焙坐标镜像(--mirror-u) 与 贴图像素事后镜像(--final-mirror)
# - 导出 OBJ+MTL（支持外/内壁 & UV 镜像）
# 依赖：numpy, opencv-python

import argparse
import math
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2

# ---------------------------
# 基础 I/O
# ---------------------------

def imread_any(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def imread_mask01(path: Path, thresh: int = 128) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    mask = (m >= thresh).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask

# ---------------------------
# 相机与几何
# ---------------------------

def fx_from_fov(img_w: int, fov_h_deg: float) -> float:
    return 0.5 * float(img_w) / math.tan(math.radians(fov_h_deg) * 0.5)

def build_extrinsics(cam_center: np.ndarray,
                     up_world: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float32)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    C = cam_center.astype(np.float32)
    k = (-C)
    k = k / (np.linalg.norm(k) + 1e-9)  # 相机前向(指向原点)
    i = np.cross(up_world, k); i /= (np.linalg.norm(i) + 1e-9)  # 右
    up_cam = np.cross(k, i); up_cam /= (np.linalg.norm(up_cam) + 1e-9)
    j = -up_cam  # 下(使像素v向下为正)
    R_wc = np.stack([i, j, k], axis=0)
    t_wc = -R_wc @ C
    return R_wc.astype(np.float32), t_wc.astype(np.float32), k.astype(np.float32)

def project_points(Pw_xyz: Tuple[np.ndarray,np.ndarray,np.ndarray],
                   R: np.ndarray, t: np.ndarray,
                   fx: float, fy: float, cx: float, cy: float) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    Xw, Yw, Zw = Pw_xyz
    Xc = R[0,0]*Xw + R[0,1]*Yw + R[0,2]*Zw + t[0]
    Yc = R[1,0]*Xw + R[1,1]*Yw + R[1,2]*Zw + t[1]
    Zc = R[2,0]*Xw + R[2,1]*Yw + R[2,2]*Zw + t[2]
    eps = 1e-9
    u = fx * (Xc / (Zc + eps)) + cx
    v = fy * (Yc / (Zc + eps)) + cy
    return u.astype(np.float32), v.astype(np.float32), Zc.astype(np.float32)

# ---------------------------
# 角度工具 / 扇区窗
# ---------------------------

def angle_wrap_pi(x: np.ndarray) -> np.ndarray:
    # wrap to (-pi, pi]
    return (x + math.pi) % (2.0 * math.pi) - math.pi

def sector_window(theta_global: np.ndarray, theta_cam: float,
                  sector_deg: float, feather_deg: float) -> np.ndarray:
    """
    带羽化的矩形扇区窗：
      |d| <= half-feather     -> 1
      half-feather < |d| <= half+feather -> 半余弦降到0
      |d| > half+feather      -> 0
    """
    if not sector_deg or sector_deg <= 0:
        return np.ones_like(theta_global, dtype=np.float32)
    half = math.radians(sector_deg * 0.5)
    blend = max(1e-6, math.radians(max(0.0, feather_deg)))
    d = angle_wrap_pi(theta_global - float(theta_cam))
    ad = np.abs(d)
    w = np.ones_like(theta_global, dtype=np.float32)
    # 截断
    w[ad > (half + blend)] = 0.0
    # 羽化带
    m = (ad > (half - blend)) & (ad <= (half + blend))
    if np.any(m):
        t = (half + blend - ad[m]) / (2.0 * blend)  # 1→0
        w[m] = 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0.0, 1.0)).astype(np.float32)
    return w

# ---------------------------
# OBJ/MTL 导出（支持内/外壁与 UV 镜像）
# ---------------------------

def write_cylinder_obj(outdir: Path, tex_w: int, tex_h: int,
                       radius: float, height: float,
                       seg_u: int = 256, seg_v: int = 64,
                       obj_name: str = "cylinder.obj",
                       mtl_name: str = "cylinder.mtl",
                       tex_name: str = "albedo.png",
                       flip_side: str = "outside",
                       mirror_uv: bool = False) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    obj_path = outdir / obj_name
    mtl_path = outdir / mtl_name

    vs, vts, vns, faces = [], [], [], []

    for j in range(seg_v + 1):
        v = j / seg_v
        z = (v - 0.5) * height
        for i in range(seg_u + 1):
            u = i / seg_u
            th = 2.0 * math.pi * u
            x = radius * math.cos(th)
            y = radius * math.sin(th)

            nx, ny, nz = math.cos(th), math.sin(th), 0.0
            if flip_side == "inside":
                nx, ny, nz = -nx, -ny, -nz

            uu = (1.0 - u) if mirror_uv else u
            vv = 1.0 - v

            vs.append((x, y, z))
            vns.append((nx, ny, nz))
            vts.append((uu, vv))

    def vid(i, j):
        return j * (seg_u + 1) + i + 1

    for j in range(seg_v):
        for i in range(seg_u):
            v00 = vid(i,   j)
            v10 = vid(i+1, j)
            v01 = vid(i,   j+1)
            v11 = vid(i+1, j+1)
            if flip_side == "inside":
                faces.append((v00, v11, v10))
                faces.append((v00, v01, v11))
            else:
                faces.append((v00, v10, v11))
                faces.append((v00, v11, v01))

    with open(mtl_path, "w", encoding="utf-8") as f:
        f.write("newmtl material_0\n")
        f.write("Ka 1.000 1.000 1.000\n")
        f.write("Kd 1.000 1.000 1.000\n")
        f.write("Ks 0.000 0.000 0.000\n")
        f.write("d 1.0\n")
        f.write(f"map_Kd {tex_name}\n")

    with open(obj_path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_name}\n")
        for x,y,z in vs:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for u,v in vts:
            f.write(f"vt {u:.6f} {v:.6f}\n")
        for nx,ny,nz in vns:
            f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
        f.write("usemtl material_0\n")
        f.write("s off\n")
        for (a,b,c) in faces:
            f.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")

    print(f"[OBJ] {obj_path.name}  flip_side={flip_side}, mirror_uv={mirror_uv}")

# ---------------------------
# 主烘焙逻辑
# ---------------------------

class CamItem:
    def __init__(self, name: str, img_path: Path, mask_path: Path, theta_deg: float,
                 fov_h_deg: float = 73.74, fov_v_deg: float = None,
                 f_pix_x: float = None, f_pix_y: float = None):
        self.name = name
        self.img = imread_any(img_path)
        self.mask01 = imread_mask01(mask_path)
        H, W = self.img.shape[:2]
        if self.mask01.shape != (H, W):
            self.mask01 = cv2.resize(self.mask01, (W, H), interpolation=cv2.INTER_NEAREST)
            print(f"[{name}] mask resized to {W}x{H}")
        self.W, self.H = W, H

        # 内参（支持外部指定）
        if f_pix_x is not None:
            self.fx = float(f_pix_x)
        else:
            self.fx = fx_from_fov(W, fov_h_deg)

        if f_pix_y is not None:
            self.fy = float(f_pix_y)
        elif fov_v_deg is not None:
            self.fy = 0.5 * H / math.tan(math.radians(fov_v_deg) * 0.5)
        else:
            self.fy = self.fx  # 默认方像素，避免竖向拉伸

        self.cx = (W - 1) * 0.5
        self.cy = (H - 1) * 0.5
        self.theta_rad = math.radians(theta_deg)
        self.R = None; self.t = None; self.forward_k = None

    def place_around_cylinder(self, cam_radius: float):
        C = np.array([cam_radius * math.cos(self.theta_rad),
                      cam_radius * math.sin(self.theta_rad),
                      0.0], dtype=np.float32)
        R, t, k = build_extrinsics(C)
        self.R, self.t, self.forward_k = R, t, k

def bake_texture(cams: List[CamItem],
                 tex_w: int, tex_h: int,
                 cyl_radius: float, cyl_height: float,
                 cam_radius: float,
                 u_offset_deg: float = 0.0,
                 gamma: float = 2.0,
                 mirror_u: bool = False,
                 blend: str = "avg",
                 softmax_tau: float = 0.25,
                 sector_deg: float = 0.0,
                 sector_feather_deg: float = 6.0,
                 final_mirror: bool = False,
                 outdir: Path = Path("out_bake")) -> Path:

    outdir.mkdir(parents=True, exist_ok=True)

    for c in cams:
        c.place_around_cylinder(cam_radius)

    # 纹理坐标（可镜像）
    u_base = (np.arange(tex_w, dtype=np.float32) + 0.5) / float(tex_w)
    if mirror_u:
        u_base = 1.0 - u_base
    v_base = (np.arange(tex_h, dtype=np.float32) + 0.5) / float(tex_h)
    U, V = np.meshgrid(u_base, v_base)

    theta = 2.0 * math.pi * (U + (u_offset_deg / 360.0))
    z = (V - 0.5) * cyl_height

    X = cyl_radius * np.cos(theta)
    Y = cyl_radius * np.sin(theta)
    Z = z

    # 外法线（角度权重用 -N ⋅ k_cam）
    Nx = np.cos(theta); Ny = np.sin(theta); Nz = np.zeros_like(Nx, dtype=np.float32)
    n_minus = (-Nx, -Ny, -Nz)

    # 容器
    accum_rgb = np.zeros((tex_h, tex_w, 3), dtype=np.float32)
    accum_w   = np.zeros((tex_h, tex_w), dtype=np.float32)

    win_w  = np.full((tex_h, tex_w), -1.0, dtype=np.float32) if blend == "max" else None
    win_rgb = np.zeros((tex_h, tex_w, 3), dtype=np.float32)    if blend == "max" else None

    store_w = []   # for softmax
    store_rgb = [] # for softmax

    # 投影 + 权重
    for cam in cams:
        # 角度余弦权重
        dot = (n_minus[0] * cam.forward_k[0] +
               n_minus[1] * cam.forward_k[1] +
               n_minus[2] * cam.forward_k[2])
        w_angle = np.clip(dot, 0.0, 1.0) ** float(gamma)

        # 扇区窗（稳定接缝位置）
        w_sector = sector_window(theta, cam.theta_rad, sector_deg=sector_deg, feather_deg=sector_feather_deg)

        u_px, v_px, Zc = project_points((X, Y, Z), cam.R, cam.t, cam.fx, cam.fy, cam.cx, cam.cy)
        vis = (Zc > 1e-6).astype(np.float32)

        color = cv2.remap(cam.img.astype(np.float32)/255.0, u_px, v_px,
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask  = cv2.remap(cam.mask01, u_px, v_px,
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        w = vis * mask * w_angle * w_sector  # (H,W)

        if blend == "avg":
            accum_rgb += color * w[..., None]
            accum_w   += w

        elif blend == "max":
            better = w > win_w
            if np.any(better):
                win_w[better] = w[better]
                win_rgb[better] = color[better]

        elif blend == "softmax":
            store_w.append(w)
            store_rgb.append(color)

    # 归一化/输出
    eps = 1e-8
    if blend == "avg":
        valid = accum_w > eps
        out_rgb = np.zeros_like(accum_rgb, dtype=np.float32)
        out_rgb[valid] = accum_rgb[valid] / (accum_w[valid, None] + eps)
        alpha = np.clip(accum_w, 0.0, 1.0)

    elif blend == "max":
        out_rgb = win_rgb
        alpha = np.clip(np.maximum(win_w, 0.0), 0.0, 1.0)

    else:  # softmax
        if not store_w:
            out_rgb = np.zeros_like(accum_rgb, dtype=np.float32)
            alpha = np.zeros((tex_h, tex_w), dtype=np.float32)
        else:
            W = np.stack(store_w, axis=0)        # (C,H,W)
            Cimg = np.stack(store_rgb, axis=0)   # (C,H,W,3)
            # 温度 softmax
            S = np.exp(W / max(1e-6, float(softmax_tau)))
            S_sum = S.sum(axis=0) + eps          # (H,W)
            Wn = S / S_sum[:, :]                 # (C,H,W)
            out_rgb = (Wn[..., None] * Cimg).sum(axis=0)  # (H,W,3)
            # alpha 用总“原始权重”的截断（可感知覆盖强度）
            wsum_raw = np.sum(W, axis=0)
            alpha = np.clip(wsum_raw, 0.0, 1.0)

    # 组合 BGRA 写入
    out_rgba = np.dstack([
        np.clip(np.round(out_rgb[..., 2]*255), 0, 255).astype(np.uint8),  # B
        np.clip(np.round(out_rgb[..., 1]*255), 0, 255).astype(np.uint8),  # G
        np.clip(np.round(out_rgb[..., 0]*255), 0, 255).astype(np.uint8),  # R
        np.clip(np.round(alpha*255), 0, 255).astype(np.uint8)             # A
    ])

    if final_mirror:
        out_rgba = cv2.flip(out_rgba, 1)
        print("[INFO] final_mirror applied to albedo.")

    tex_path = outdir / "albedo.png"
    cv2.imwrite(str(tex_path), out_rgba)
    print(f"[TEX] {tex_path.name}  size={tex_w}x{tex_h}, blend={blend}")

    # 覆盖度可视化
    if blend == "avg":
        cov = (accum_w / (accum_w.max() + eps) * 255.0).astype(np.uint8) if accum_w.max() > 0 else np.zeros((tex_h, tex_w), np.uint8)
    elif blend == "max":
        ww = np.maximum(win_w, 0.0)
        cov = (ww / (ww.max() + eps) * 255.0).astype(np.uint8) if ww.max() > 0 else np.zeros((tex_h, tex_w), np.uint8)
    else:
        # softmax：用原始 w 的总和做覆盖度
        wsum_raw = np.sum(np.stack(store_w, axis=0), axis=0) if store_w else np.zeros((tex_h, tex_w), dtype=np.float32)
        cov = (wsum_raw / (wsum_raw.max() + eps) * 255.0).astype(np.uint8) if wsum_raw.max() > 0 else np.zeros((tex_h, tex_w), np.uint8)

    if final_mirror:
        cov = cv2.flip(cov, 1)
    cv2.imwrite(str(outdir / "coverage.png"), cov)

    return tex_path

# ---------------------------
# CLI
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Bake cylinder texture with seam control (avg/max/softmax), sector windowing, mirror & inside/outside OBJ."
    )
    # 多相机：--cam NAME IMG MASK THETA_DEG  可重复
    ap.add_argument("--cam", nargs=4, action="append", metavar=("NAME","IMG","MASK","THETA_DEG"),
                    required=True, help="Add a camera: NAME IMG MASK THETA_DEG (deg around cylinder).")

    # 内参 / FOV
    ap.add_argument("--fov-h-deg", type=float, default=73.74, help="水平 FOV（度）")
    ap.add_argument("--fov-v-deg", type=float, default=None, help="垂直 FOV（度，可选）")
    ap.add_argument("--f-pix-x", type=float, default=None, help="像素焦距 fx（可选，优先于 FOV_h）")
    ap.add_argument("--f-pix-y", type=float, default=None, help="像素焦距 fy（可选，优先于 FOV_v）")

    # 纹理与几何
    ap.add_argument("--tex-w", type=int, default=2048, help="贴图宽度")
    ap.add_argument("--tex-h", type=int, default=1024, help="贴图高度")
    ap.add_argument("--cyl-radius", type=float, default=1.0, help="圆柱半径")
    ap.add_argument("--cyl-height", type=float, default=2.0, help="圆柱高度")
    ap.add_argument("--cam-radius", type=float, default=2.5, help="相机到圆心距离")
    ap.add_argument("--u-offset-deg", type=float, default=0.0, help="整体水平偏移（度）")

    # 接缝控制
    ap.add_argument("--blend", choices=["avg","max","softmax"], default="avg",
                    help="avg=加权平均；max=硬接缝；softmax=软硬可调")
    ap.add_argument("--softmax-tau", type=float, default=0.25, help="softmax 温度（越小越接近硬接缝）")
    ap.add_argument("--gamma", type=float, default=2.0, help="角度余弦指数，越大越偏向正对相机")

    # 扇区窗
    ap.add_argument("--sector-deg", type=float, default=0.0, help="每台相机的扇区总宽度（度）；0=关闭")
    ap.add_argument("--sector-feather-deg", type=float, default=6.0, help="扇区边界羽化半宽（度）")

    # 镜像 / OBJ
    ap.add_argument("--mirror-u", action="store_true", help="烘焙时 U 方向镜像（坐标级）")
    ap.add_argument("--final-mirror", action="store_true", help="贴图像素完成后再水平镜像一次")
    ap.add_argument("--flip-side", choices=["outside","inside"], default="outside",
                    help="outside=外壁(法线向外)，inside=内壁(法线向内、反绕序)")
    ap.add_argument("--mirror-uv-obj", action="store_true", help="OBJ 的 UV 也做 U 镜像（通常不与 final-mirror 同时用）")

    ap.add_argument("--outdir", type=str, default="out_bake", help="输出目录")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 装载相机
    cams: List[CamItem] = []
    for name, imgp, maskp, tdeg in args.cam:
        cams.append(CamItem(
            name=name,
            img_path=Path(imgp),
            mask_path=Path(maskp),
            theta_deg=float(tdeg),
            fov_h_deg=args.fov_h_deg,
            fov_v_deg=args.fov_v_deg,
            f_pix_x=args.f_pix_x,
            f_pix_y=args.f_pix_y
        ))
    if len(cams) < 2:
        raise ValueError("需要至少两台相机。")

    tex_path = bake_texture(
        cams,
        tex_w=args.tex_w, tex_h=args.tex_h,
        cyl_radius=args.cyl_radius, cyl_height=args.cyl_height,
        cam_radius=args.cam_radius,
        u_offset_deg=args.u_offset_deg,
        gamma=args.gamma,
        mirror_u=args.mirror_u,
        blend=args.blend,
        softmax_tau=args.softmax_tau,
        sector_deg=args.sector_deg,
        sector_feather_deg=args.sector_feather_deg,
        final_mirror=args.final_mirror,
        outdir=outdir
    )

    write_cylinder_obj(
        outdir, args.tex_w, args.tex_h,
        radius=args.cyl_radius, height=args.cyl_height,
        seg_u=256, seg_v=64,
        obj_name="cylinder.obj",
        mtl_name="cylinder.mtl",
        tex_name=tex_path.name,
        flip_side=args.flip_side,
        mirror_uv=args.mirror_uv_obj
    )

    print(f"[DONE] Exported -> {outdir/'cylinder.obj'}  texture={tex_path.name}")

if __name__ == "__main__":
    main()
