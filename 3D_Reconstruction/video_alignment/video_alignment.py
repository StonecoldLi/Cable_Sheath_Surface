import cv2
import numpy as np

VIDEO_PATH       = "../../data/B01/T01.mp4"   # MP4 文件
MAX_FRAMES       = 12000               # 2 min @100 fps
INIT_CALIB_FRAMES = 1000              # 用前 10.00 s 估计噪声
CONSEC_FRAMES     = 50                # 连续满足阈值的帧数才触发

FLOW_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError("无法打开视频文件  %s" % VIDEO_PATH)

fps   = cap.get(cv2.CAP_PROP_FPS)
w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ----------- 定义 ROI: 横向 40 %-60 %，纵向全高 -----------
x1, x2 = int(w * 0.40), int(w * 0.60)

# ----------- 变量初始化 -----------
prev_gray_roi  = None
prev_points    = None
dy_series      = []                    # 保存 dy 序列
trigger_frame  = None
consec_counter = 0

for fidx in range(MAX_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break

    # 裁剪出 ROI 并转灰度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_roi = gray[:, x1:x2]          # 全高，40–60% 宽

    # --------- 第一帧：提取特征点 ---------
    if prev_gray_roi is None:
        prev_gray_roi = gray_roi
        prev_points = cv2.goodFeaturesToTrack(
            prev_gray_roi, maxCorners=400, qualityLevel=0.01, minDistance=7, blockSize=7
        )
        # 坐标加上 x1 偏移，方便后续光流调用
        if prev_points is not None:
            prev_points += np.array([[x1, 0]], dtype=np.float32)
        continue

    # --------- 计算光流 ---------
    if prev_points is not None and len(prev_points) > 0:
        next_points, st, err = cv2.calcOpticalFlowPyrLK(
            prev_gray_roi, gray_roi, prev_points - np.array([[x1, 0]], dtype=np.float32),
            None, **FLOW_PARAMS
        )
        if next_points is None:
            prev_points = None
        else:
            # 仅保留成功跟踪的点
            good_prev = prev_points[st.ravel() == 1]
            good_next = next_points[st.ravel() == 1] + np.array([[x1, 0]], dtype=np.float32)
            
            good_prev = good_prev.reshape(-1, 2)      # 或 good_prev = np.squeeze(good_prev, axis=1)
            good_next = good_next.reshape(-1, 2)

            # 计算每个点的 dy
            dy = good_next[:, 1] - good_prev[:, 1]
            if len(dy) > 0:
                median_dy = np.median(np.abs(dy))
                dy_series.append(median_dy)

                # -------------- 自适应阈值的形成与检测 --------------
                if fidx == INIT_CALIB_FRAMES:
                    # 统计初始静止噪声：均值 + 3σ
                    mu  = np.mean(dy_series)
                    std = np.std(dy_series)
                    thresh = mu + 3 * std
                elif fidx > INIT_CALIB_FRAMES:
                    if median_dy > thresh:
                        consec_counter += 1
                        if consec_counter == CONSEC_FRAMES:
                            trigger_frame = fidx - CONSEC_FRAMES + 1
                            break
                    else:
                        consec_counter = 0

            # 更新特征点 & 前一帧
            prev_points = good_next
            prev_gray_roi = gray_roi.copy()
    else:
        # 若点过少则重新检测
        prev_points = cv2.goodFeaturesToTrack(
            gray_roi, maxCorners=400, qualityLevel=0.01, minDistance=7, blockSize=7
        )
        if prev_points is not None:
            prev_points += np.array([[x1, 0]], dtype=np.float32)
        prev_gray_roi = gray_roi.copy()

cap.release()

if trigger_frame is not None:
    print(f"摄像机开始移动的帧号: {trigger_frame}")
    print(f"对应时间: {trigger_frame / fps:.3f} 秒")
else:
    print("在 2 分钟内未检测到明显由下至上的移动")
