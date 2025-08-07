import cv2, numpy as np, argparse, pathlib

def main(img1_path, img2_path, ratio=0.75, rth=3.0):
    # 1. 读图并提取 SIFT 描述子
    sift = cv2.SIFT_create()
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    kp1, des1 = sift.detectAndCompute(img1, None)   # 特征点+描述子:contentReference[oaicite:2]{index=2}
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 2. KNN+Lowe 比率检验
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)          # :contentReference[oaicite:3]{index=3}
    good = [m for m, n in matches if m.distance < ratio * n.distance]

    # 3. RANSAC 估计仿射（平移+旋转）
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                         ransacReprojThreshold=rth)  # :contentReference[oaicite:4]{index=4}
    dx, dy = M[0,2], M[1,2]
    theta = np.degrees(np.arctan2(M[1,0], M[0,0]))
    print(f"Δx={dx:.2f}px  Δy={dy:.2f}px  θ={theta:.2f}°  内点率={inl.mean()*100:.1f}%")

    # 4. 可视化匹配（可选）
    draw = cv2.drawMatches(img1, kp1, img2, kp2,
                           [good[i] for i in range(min(80,len(good)))],
                           None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite("matches.jpg", draw)                # :contentReference[oaicite:5]{index=5}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("img1"); ap.add_argument("img2")
    args = ap.parse_args()
    main(args.img1, args.img2)
