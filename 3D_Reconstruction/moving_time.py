import cv2, sys, numpy as np, pathlib

# --- tunables -----------------------------------------------------------
N_STATIC       = 25        # baseline frames assumed motion‑free
MAX_CORNERS    = 800
QUALITY        = 0.01
MIN_DIST       = 6
VERT_THRESH    = 1.2       # px
CONSIST_RATIO  = 0.6       # ≥ 60 % points must move downward in frame
CONSEC_FRAMES  = 5
# ------------------------------------------------------------------------

def make_roi_mask(shape):
    h, w = shape
    y0, y1 = h // 6, (h * 5) // 6          # central 2/3 vertically
    x0, x1 = w // 6, (w * 5) // 6          # central 2/3 horizontally
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask, (x0, y0, x1, y1)

def in_roi(pts, box):
    x0, y0, x1, y1 = box
    return (pts[:, 0] >= x0) & (pts[:, 0] < x1) & \
           (pts[:, 1] >= y0) & (pts[:, 1] < y1)

def detect(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    ret, frame0 = cap.read()
    if not ret:
        raise IOError("Cannot open video")
    gray_prev = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

    roi_mask, roi_box = make_roi_mask(gray_prev.shape)
    pts_prev = cv2.goodFeaturesToTrack(gray_prev, MAX_CORNERS,
                                       QUALITY, MIN_DIST, mask=roi_mask)

    baseline_dy, motion_cnt, idx = [], 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        pts_next, st, _ = cv2.calcOpticalFlowPyrLK(gray_prev, gray,
                                                   pts_prev, None)
        if pts_next is None:
            pts_prev = cv2.goodFeaturesToTrack(gray_prev, MAX_CORNERS,
                                               QUALITY, MIN_DIST,
                                               mask=roi_mask)
            gray_prev = gray
            continue

        good_prev = pts_prev[st == 1]
        good_next = pts_next[st == 1]

        # keep only pairs staying inside ROI
        in_prev = in_roi(good_prev.reshape(-1, 2), roi_box)
        in_next = in_roi(good_next.reshape(-1, 2), roi_box)
        keep    = in_prev & in_next
        good_prev, good_next = good_prev[keep], good_next[keep]

        if len(good_prev) < 10:             # too sparse → reseed
            pts_prev = cv2.goodFeaturesToTrack(gray_prev, MAX_CORNERS,
                                               QUALITY, MIN_DIST,
                                               mask=roi_mask)
            gray_prev = gray
            continue

        dy         = (good_next[:, 1] - good_prev[:, 1]).astype(np.float32)
        median_dy  = float(np.median(dy))
        pos_ratio  = np.mean(dy > 0)

        if idx <= N_STATIC:                 # baseline learning
            baseline_dy.append(median_dy)
        else:
            base   = np.median(baseline_dy)
            noise  = np.median(np.abs(baseline_dy - base)) + 1e-6  # MAD
            thresh = max(VERT_THRESH, base + 3 * noise)            # robust :contentReference[oaicite:3]{index=3}

            if median_dy > thresh and pos_ratio >= CONSIST_RATIO:
                motion_cnt += 1
                if motion_cnt >= CONSEC_FRAMES:
                    f0 = idx - CONSEC_FRAMES + 1
                    print(f"first_motion_frame = {f0}")
                    print(f"first_motion_time  = {f0 / fps:.3f} s")
                    break
            else:
                motion_cnt = 0

        gray_prev, pts_prev = gray, good_next.reshape(-1, 1, 2)
        if idx % 200 == 0 or len(pts_prev) < 0.3 * MAX_CORNERS:
            pts_prev = cv2.goodFeaturesToTrack(gray_prev, MAX_CORNERS,
                                               QUALITY, MIN_DIST,
                                               mask=roi_mask)

    cap.release()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python detect_upward_start_roi.py <video_path>")
    else:
        detect(pathlib.Path(sys.argv[1]))
