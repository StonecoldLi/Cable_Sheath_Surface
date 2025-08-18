# file: sift_ransac_match.py
import argparse, os, cv2, numpy as np

def imread_color(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

def load_mask(path, shape):
    if not path:
        return None
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    if m.shape[:2] != shape[:2]:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    _, m = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY)
    return m

def detect_describe(img, mask=None):
    sift = cv2.SIFT_create()
    kps, desc = sift.detectAndCompute(img, mask)
    if desc is None or len(kps) == 0:
        raise RuntimeError("No features found. Check image content or try removing mask.")
    return kps, desc

def ratio_test_knn(desc1, desc2, ratio=0.75):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for pair in raw:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
    good.sort(key=lambda m: m.distance)
    return good

def matches_to_points(matches, kps1, kps2):
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])
    return pts1, pts2

def estimate_model(pts1, pts2, model="fundamental", ransac_thresh=1.0, conf=0.999, K=None):
    model = model.lower()
    if model == "fundamental":
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC,
                                         ransacReprojThreshold=ransac_thresh, confidence=conf)
        return F, mask, "F"
    elif model == "homography":
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC,
                                     ransacReprojThreshold=ransac_thresh, confidence=conf)
        return H, mask, "H"
    elif model == "essential":
        if K is None:
            raise ValueError("Essential model requires K.")
        E, mask = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K, method=cv2.RANSAC,
                                       threshold=ransac_thresh, prob=conf)
        return E, mask, "E"
    else:
        raise ValueError("model must be fundamental/homography/essential")

def draw_matches(img1, img2, kps1, kps2, matches, mask=None, max_draw=200):
    if mask is None:
        draw = matches[:max_draw]
        return cv2.drawMatches(img1, kps1, img2, kps2, draw, None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    inlier_flags = mask.ravel().tolist()
    inliers = [m for m, f in zip(matches, inlier_flags) if f] [:max_draw]
    outliers = [m for m, f in zip(matches, inlier_flags) if not f] [:max_draw]

    h = max(img1.shape[0], img2.shape[0])
    w = img1.shape[1] + img2.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:img1.shape[0], :img1.shape[1]] = img1
    canvas[:img2.shape[0], img1.shape[1]:] = img2

    def p1(m): 
        x,y = kps1[m.queryIdx].pt
        return (int(round(x)), int(round(y)))
    def p2(m): 
        x,y = kps2[m.trainIdx].pt
        return (int(round(x))+img1.shape[1], int(round(y)))

    for m in inliers:
        cv2.circle(canvas, p1(m), 3, (0,255,0), -1)
        cv2.circle(canvas, p2(m), 3, (0,255,0), -1)
        cv2.line(canvas, p1(m), p2(m), (0,255,0), 1, cv2.LINE_AA)
    for m in outliers:
        cv2.circle(canvas, p1(m), 2, (0,0,255), -1)
        cv2.circle(canvas, p2(m), 2, (0,0,255), -1)
        cv2.line(canvas, p1(m), p2(m), (0,0,255), 1, cv2.LINE_AA)
    return canvas

def main():
    ap = argparse.ArgumentParser("SIFT + RANSAC matching (3D默认Fundamental)")
    ap.add_argument("--img1", required=True)
    ap.add_argument("--img2", required=True)
    ap.add_argument("--mask1", default=None, help="optional binary mask for img1")
    ap.add_argument("--mask2", default=None, help="optional binary mask for img2")
    ap.add_argument("--model", default="fundamental", choices=["fundamental","homography","essential"])
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--ransac_thresh", type=float, default=1.0, help="px for F/E; reproj px for H")
    ap.add_argument("--conf", type=float, default=0.999)
    ap.add_argument("--outdir", default="./out_match")
    ap.add_argument("--restrict_to_mask", action="store_true",
                   help="if set, detect features only inside mask (otherwise ignore masks for detection)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    img1, img2 = imread_color(args.img1), imread_color(args.img2)
    m1 = load_mask(args.mask1, img1) if args.mask1 else None
    m2 = load_mask(args.mask2, img2) if args.mask2 else None

    # 特征提取（默认不限制在mask内，更稳；如需只在索上取点，使用 --restrict_to_mask）
    kps1, desc1 = detect_describe(img1, m1 if args.restrict_to_mask else None)
    kps2, desc2 = detect_describe(img2, m2 if args.restrict_to_mask else None)
    print(f"[INFO] keypoints: img1={len(kps1)}, img2={len(kps2)}")

    # KNN + Lowe 比例
    matches = ratio_test_knn(desc1, desc2, ratio=args.ratio)
    if len(matches) < 8:
        raise RuntimeError(f"Too few matches after ratio test: {len(matches)}")
    print(f"[INFO] matches after ratio test: {len(matches)}")

    # RANSAC 模型
    pts1, pts2 = matches_to_points(matches, kps1, kps2)
    model_mat, inlier_mask, tag = estimate_model(
        pts1, pts2, model=args.model, ransac_thresh=args.ransac_thresh, conf=args.conf, K=None
    )
    if model_mat is None or inlier_mask is None:
        raise RuntimeError("RANSAC failed.")
    inliers = int(inlier_mask.sum())
    print(f"[INFO] model={args.model} ({tag}), inliers={inliers}/{len(inlier_mask)} = {inliers/len(inlier_mask):.2%}")

    # 可视化
    vis_all = draw_matches(img1, img2, kps1, kps2, matches, None)
    vis_in  = draw_matches(img1, img2, kps1, kps2, matches, inlier_mask)
    cv2.imwrite(os.path.join(args.outdir, "matches_all.jpg"), vis_all)
    cv2.imwrite(os.path.join(args.outdir, "matches_inliers.jpg"), vis_in)

    # 保存内点坐标，便于后续位姿/三角化
    inlier_pts1 = pts1[inlier_mask.ravel() == 1]
    inlier_pts2 = pts2[inlier_mask.ravel() == 1]
    np.save(os.path.join(args.outdir, "inlier_pts1.npy"), inlier_pts1)
    np.save(os.path.join(args.outdir, "inlier_pts2.npy"), inlier_pts2)
    if tag == "F":
        np.save(os.path.join(args.outdir, "F.npy"), model_mat)
    elif tag == "H":
        np.save(os.path.join(args.outdir, "H.npy"), model_mat)
    elif tag == "E":
        np.save(os.path.join(args.outdir, "E.npy"), model_mat)

    print(f"[DONE] results saved to {args.outdir}")

if __name__ == "__main__":
    # 示例（3D场景，默认 Fundamental）:
    # python ./pic_stitching/sift_ransac_point_matching.py --img1 ./pic_stitching/data/1/frame_000000_kept_cropped.png --img2 ./pic_stitching/data/2/frame_000000_kept_cropped.png --outdir out_match
    #
    # 只在掩膜内取特征（不一定更稳）:
    # python sift_ransac_match.py --img1 cam0.jpg --img2 cam90.jpg --mask1 m0.png --mask2 m90.png --restrict_to_mask
    #
    # 平面目标可试 Homography：
    # python sift_ransac_match.py --img1 a.jpg --img2 b.jpg --model homography
    main()
