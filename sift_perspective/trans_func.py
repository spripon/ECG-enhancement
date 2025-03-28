
import cv2
import numpy as np

def compute_transform(kp1, desc1, kp2, desc2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return M
    else:
        raise Exception("Pas assez de correspondances pour calculer l'homographie.")

def apply_transform(img, M):
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, M, (w, h))
