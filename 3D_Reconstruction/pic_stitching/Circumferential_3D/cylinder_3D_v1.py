# bake_cylinder_from_cams_insideout_mirror.py
# 依赖：numpy, opencv-python
# Python: 3.7+

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
    k = (-C); k = k / (np.linalg.norm(k) + 1e-9)  # forward
    i = np.cross(up_world, k); i /= (np.linalg.norm(i) + 1e-9)  # right
    up_cam = np.cross(k, i); up_cam /= (np.linalg.norm(up_cam) + 1e-9)
    j = -up_cam  # down
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
# OBJ/MTL 导出（支持内外壁、UV 镜像）
# ---------------------------

def write_cylinder_obj(outdir: Path, tex_w: int, tex_h: int,
                       radius: float, height: float,
                       seg_u: int = 256, seg_v: int = 64,
                       obj_name: str = "cylinder.obj",
                       mtl_name: str = "cylinder.mtl",
                       tex_name: str = "albedo.png",
                       flip_side: str = "outside",
                       mirror_uv: bool = False) -> None:
    """
    flip_side: 'outside'（法线向外）或 'inside'（法线向内，反转面绕序）
    mirror_uv: True 则对 U 做 1-U 镜像（仅影响 OBJ 的 UV）
    """
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
                 fov_h_deg: float = 73.74):
        self.name = name
        self.img = imread_any(img_path)
        self.mask01 = imread_mask01(mask_path)
        H, W = self.img.shape[:2]
        if self.mask01.shape != (H, W):
            self.mask01 = cv2.resize(self.mask01, (W, H), interpolation=cv2.INTER_NEAREST)
            print(f"[{name}] mask resized to {W}x{H}")
        self.W, self.H = W, H
        self.fov_h_deg = fov_h_deg
        self.fx = fx_from_fov(W, fov_h_deg)
        self.fy = self.fx * (H / max(1, W))
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
                 final_mirror: bool = False,
                 outdir: Path = Path("out_bake")):

    outdir.mkdir(parents=True, exist_ok=True)

    for c in cams:
        c.place_around_cylinder(cam_radius)

    # 烘焙阶段的 U 基座：是否镜像
    u_base = (np.arange(tex_w, dtype=np.float32) + 0.5) / float(tex_w)
    if mirror_u:
        u_base = 1.0 - u_base
    v_base = (np.arange(tex_h, dtype=np.float32) + 0.5) / float(tex_h)
    U, V = np.meshgrid(u_base, v_base)  # (H,W)

    theta = 2.0 * math.pi * (U + (u_offset_deg / 360.0))
    z = (V - 0.5) * cyl_height

    X = cyl_radius * np.cos(theta)
    Y = cyl_radius * np.sin(theta)
    Z = z

    Nx = np.cos(theta); Ny = np.sin(theta); Nz = np.zeros_like(Nx, dtype=np.float32)
    n_minus = (-Nx, -Ny, -Nz)

    accum_rgb = np.zeros((tex_h, tex_w, 3), dtype=np.float32)
    accum_w   = np.zeros((tex_h, tex_w), dtype=np.float32)

    for cam in cams:
        dot = (n_minus[0] * cam.forward_k[0] +
               n_minus[1] * cam.forward_k[1] +
               n_minus[2] * cam.forward_k[2])
        w_angle = np.clip(dot, 0.0, 1.0) ** gamma

        u_px, v_px, Zc = project_points((X, Y, Z), cam.R, cam.t, cam.fx, cam.fy, cam.cx, cam.cy)
        vis = (Zc > 1e-6).astype(np.float32)

        color = cv2.remap(cam.img.astype(np.float32)/255.0, u_px, v_px,
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask  = cv2.remap(cam.mask01, u_px, v_px,
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        w = vis * mask * w_angle
        if np.max(w) <= 0: continue
        accum_rgb += color * w[..., None]
        accum_w   += w

    eps = 1e-8
    valid = accum_w > eps
    out_rgb = np.zeros_like(accum_rgb, dtype=np.float32)
    out_rgb[valid] = accum_rgb[valid] / (accum_w[valid, None] + eps)
    alpha = np.clip(accum_w, 0.0, 1.0).astype(np.float32)

    # 组合 BGRA
    out_rgba = np.dstack([
        np.clip(np.round(out_rgb[..., 2]*255), 0, 255).astype(np.uint8),
        np.clip(np.round(out_rgb[..., 1]*255), 0, 255).astype(np.uint8),
        np.clip(np.round(out_rgb[..., 0]*255), 0, 255).astype(np.uint8),
        np.clip(np.round(alpha*255), 0, 255).astype(np.uint8)
    ])

    # ★ 额外一次镜像：对贴图进行像素级左右翻转（外部映射再镜像）
    if final_mirror:
        out_rgba = cv2.flip(out_rgba, 1)  # 1=水平翻转
        print("[INFO] final_mirror applied to albedo.")

    tex_path = outdir / "albedo.png"
    cv2.imwrite(str(tex_path), out_rgba)
    print(f"[TEX] {tex_path.name}  size={tex_w}x{tex_h}, mirror_u={mirror_u}, final_mirror={final_mirror}")

    cov = (accum_w / (accum_w.max() + eps) * 255.0).astype(np.uint8)
    if final_mirror:
        cov = cv2.flip(cov, 1)
    cv2.imwrite(str(outdir / "coverage.png"), cov)

    return tex_path

def angle_diff_signed(a, b):
    """最小化环域差（-π, π]"""
    d = a - b
    return (d + math.pi) % (2.0 * math.pi) - math.pi

def sector_window(theta_global, theta_cam, sector_deg, feather_deg):
    """
    在圆柱周向上给相机一个扇区窗（带羽化的矩形窗）。
    theta_global: 纹理像素对应的全局角 θ (H,W)
    theta_cam:    该相机的朝向角（标量，弧度）
    sector_deg:   扇区总宽度（度）；None/0 表示关闭
    feather_deg:  扇区边界羽化半宽（度）
    返回 w_sector ∈ [0,1]，形状(H,W)
    """
    if not sector_deg or sector_deg <= 0:
        return np.ones_like(theta_global, dtype=np.float32)
    half = math.radians(sector_deg * 0.5)
    blend = max(1e-6, math.radians(max(0.0, feather_deg)))
    d = np.vectorize(angle_diff_signed)(theta_global, theta_cam).astype(np.float32)
    w = np.ones_like(theta_global, dtype=np.float32)
    # 硬截断
    w[np.abs(d) > (half + blend)] = 0.0
    # 羽化带：|d| ∈ [half-blend, half+blend]
    m = (np.abs(d) > (half - blend)) & (np.abs(d) <= (half + blend))
    t = (half + blend - np.abs(d[m])) / (2.0 * blend)  # 1→0 线性
    w[m] = 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0.0, 1.0)).astype(np.float32)
    return w

# ---------------------------
# CLI
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Bake cylinder texture with inside/outside & mirror options (+ final pixel mirror)."
    )
    ap.add_argument("--cam", nargs=4, action="append", metavar=("NAME","IMG","MASK","THETA_DEG"),
                    required=True)
    ap.add_argument("--fov-h-deg", type=float, default=73.74)
    ap.add_argument("--tex-w", type=int, default=2048)
    ap.add_argument("--tex-h", type=int, default=1024)
    ap.add_argument("--cyl-radius", type=float, default=1.0)
    ap.add_argument("--cyl-height", type=float, default=2.0)
    ap.add_argument("--cam-radius", type=float, default=2.5)
    ap.add_argument("--u-offset-deg", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--flip-side", choices=["outside","inside"], default="outside",
                    help="outside=法线向外（外壁），inside=法线向内（内壁为正面）")
    ap.add_argument("--mirror-u", action="store_true",
                    help="烘焙时对 U 方向做坐标镜像（坐标级）")
    ap.add_argument("--final-mirror", action="store_true",
                    help="烘焙完成后对贴图做像素级左右翻转（再镜一次）")
    ap.add_argument("--mirror-uv-obj", action="store_true",
                    help="OBJ 的 UV 也做 1-U 镜像（通常与 --final-mirror 互斥，默认 False）")
    ap.add_argument("--outdir", type=str, default="out_bake")
    # === 在 CLI 解析处，新增参数 ===
    ap.add_argument("--blend", choices=["avg","max","softmax"], default="avg",
                    help="接缝融合模式：avg=加权平均；max=硬接缝；softmax=软硬可调")
    ap.add_argument("--softmax-tau", type=float, default=0.25,
                    help="softmax 温度（越小越接近硬接缝）")
    ap.add_argument("--sector-deg", type=float, default=0.0,
                    help="每台相机的扇区总宽度（度）；0=关闭")
    ap.add_argument("--sector-feather-deg", type=float, default=6.0,
                    help="扇区边界羽化半宽（度）")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    cams: List[CamItem] = []
    for name, imgp, maskp, tdeg in args.cam:
        cams.append(CamItem(name=name, img_path=Path(imgp), mask_path=Path(maskp),
                            theta_deg=float(tdeg), fov_h_deg=args.fov_h_deg))
    if len(cams) < 2:
        raise ValueError("需要至少两台相机。")

    tex_path = bake_texture(cams,
                            tex_w=args.tex_w, tex_h=args.tex_h,
                            cyl_radius=args.cyl_radius, cyl_height=args.cyl_height,
                            cam_radius=args.cam_radius,
                            u_offset_deg=args.u_offset_deg,
                            gamma=args.gamma,
                            mirror_u=args.mirror_u,
                            final_mirror=args.final_mirror,
                            outdir=outdir)

    # 注意：若已 final_mirror（直接翻贴图像素），一般不需要再镜 OBJ 的 UV
    write_cylinder_obj(outdir, args.tex_w, args.tex_h,
                       radius=args.cyl_radius, height=args.cyl_height,
                       seg_u=256, seg_v=64,
                       obj_name="cylinder.obj",
                       mtl_name="cylinder.mtl",
                       tex_name=tex_path.name,
                       flip_side=args.flip_side,
                       mirror_uv=args.mirror_uv_obj)

    print(f"[DONE] Exported -> {outdir/'cylinder.obj'} with {tex_path.name}")

if __name__ == "__main__":
    main()
