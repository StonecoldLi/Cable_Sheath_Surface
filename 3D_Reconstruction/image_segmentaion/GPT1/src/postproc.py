# src/postproc.py
import numpy as np, cv2
from scipy.ndimage import maximum_filter
from skimage.filters import apply_hysteresis_threshold
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev

def edge_nms(prob, k=3):
    # prob: float32 [0..1]
    m = maximum_filter(prob, size=k)
    keep = (prob==m)
    out = prob.copy(); out[~keep]=0
    return out

def hysteresis(binary, th_low, th_high):
    # binary: [0..1] soft; use hysteresis on prob
    return apply_hysteresis_threshold(binary, th_low, th_high).astype(np.uint8)

def trace_and_fit(binary, min_len=50, smooth=0.001, num_pts=200):
    # skeletonize to 1-px
    sk = skeletonize(binary>0).astype(np.uint8)
    comps = label(sk, connectivity=2)
    curves = []
    for r in regionprops(comps):
        coords = r.coords[:, ::-1]  # (x,y)
        if coords.shape[0] < min_len: continue
        # simple order by path: sort by projection on first PCA dir
        pts = coords.astype(np.float64)
        # fit spline
        try:
            tck, u = splprep([pts[:,0], pts[:,1]], s=smooth)
            unew = np.linspace(0,1,num_pts)
            xnew, ynew = splev(unew, tck)
            curves.append(np.stack([xnew,ynew],1))
        except Exception:
            curves.append(pts)
    return curves  # list of (N,2) arrays in (x,y)

def buffer_polyline(poly, r=3, shape=None):
    # rasterize thick polyline into mask
    h,w = shape[:2]
    mask = np.zeros((h,w), np.uint8)
    pts = np.round(poly).astype(np.int32)
    for i in range(len(pts)-1):
        cv2.line(mask, tuple(pts[i]), tuple(pts[i+1]), 255, thickness=int(max(1,2*r)))
    return (mask>0).astype(np.uint8)

def estimate_width_by_gradient(img, poly, r_min=2, r_max=6):
    # 简化：固定r，或按梯度能量微调；这里先返回固定中值
    return int(0.5*(r_min+r_max))

def crop_by_components(mask, margin=16, min_area=80, img=None):
    num, labels = cv2.connectedComponents(mask.astype(np.uint8))
    rois = []
    for i in range(1, num):
        ys, xs = np.where(labels==i)
        if xs.size < 1: continue
        if xs.size < min_area: continue
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        x1 = max(0, x1 - margin); y1=max(0, y1 - margin)
        x2 = min(mask.shape[1]-1, x2 + margin); y2=min(mask.shape[0]-1, y2 + margin)
        if img is not None:
            rois.append(img[y1:y2+1, x1:x2+1].copy())
        else:
            rois.append((x1,y1,x2,y2))
    return rois
