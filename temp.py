import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import getCameraParams
import os
from scipy.optimize import least_squares
DATA_DIR = './data/statue/images/dslr_images_undistorted/'


def detectAndCompute(images):
    keypoints, descriptors = [], []
    sift = cv.SIFT_create()
    for img in images:
        kp, des = sift.detectAndCompute(img, None)
        keypoints.append(kp)
        descriptors.append(des)
    return keypoints, descriptors


def match_features(des1, des2, ratio=0.7):
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < ratio * n.distance]
    return good


def estimate_motion_RANSAC(pts1, pts2, K, threshold):
    E, _ = cv.findEssentialMat(pts1, pts2, K, method=cv.RANSAC, threshold=threshold)
    num_inliers, R, t, _ = cv.recoverPose(E, pts1, pts2, K)
    return R, t, num_inliers


def triangulate(pts1, pts2, R, t, K):
    P1 = K @ np.hstack((np.identity(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))
    pts4d_hom = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = pts4d_hom[:3] / pts4d_hom[3]
    return pts3d.T


def reprojection_error(params, pt3d, p1, p2, K):
    R, _ = cv.Rodrigues(params[:3])
    t = params[3:].reshape(3, 1)

    p1_proj, _ = cv.projectPoints(pt3d, R, t, K, None)
    p2_proj, _ = cv.projectPoints(pt3d, R, -R.T @ t, K, None)

    error1 = np.linalg.norm(p1 - p1_proj.squeeze())
    error2 = np.linalg.norm(p2 - p2_proj.squeeze())

    return error1 + error2


def nonlinear_estimate_3d_point(pts3d, pts1, pts2, R, t, K):
    refined_pts3d = []

    for i, pt3d in enumerate(pts3d):
        pt3d = pt3d.reshape(3, 1)
        p1_proj = np.array([pts1[i].tolist()])
        p2_proj = np.array([pts2[i].tolist()])

        rvec, _ = cv.Rodrigues(R)
        params0 = np.hstack((rvec.squeeze(), t.squeeze()))

        res = least_squares(reprojection_error, params0, args=(pt3d, p1_proj, p2_proj, K))

        refined_R, _ = cv.Rodrigues(res.x[:3])
        refined_t = res.x[3:].reshape(3, 1)

        refined_pt3d = (refined_t + refined_R @ refined_t) / 2
        refined_pts3d.append(refined_pt3d.reshape(3))

    return np.array(refined_pts3d)


def main():
    # Load images and convert them to grayscale
    image_filenames = sorted(os.listdir(DATA_DIR))
    images = [cv.imread(os.path.join(DATA_DIR, fname), cv.IMREAD_GRAYSCALE) for fname in image_filenames]


    # Detect keypoints and compute descriptors for each image
    keypoints, descriptors = detectAndCompute(images)

    # Match features between consecutive image pairs
    matched_points = []
    for i in range(len(images) - 1):
        good_matches = match_features(descriptors[i], descriptors[i + 1])
        pts1 = np.array([keypoints[i][m.queryIdx].pt for m in good_matches], dtype=np.float32)
        pts2 = np.array([keypoints[i + 1][m.trainIdx].pt for m in good_matches], dtype=np.float32)
        matched_points.append((pts1, pts2))

    K = getCameraParams('./data/statue/dslr_calibration_undistorted/cameras.txt')
    threshold = 1
    all_3d_points = []

    for i, (pts1, pts2) in enumerate(matched_points):
        R, t, inliers = estimate_motion_RANSAC(pts1, pts2, K, threshold)
        pts3d = triangulate(pts1, pts2, R, t, K)
        pts3d_refined = nonlinear_estimate_3d_point(pts3d, pts1, pts2, R, t, K)
        all_3d_points.extend(pts3d_refined)
        print(f"Pair {i + 1}-{i + 2}: Number of inliers: {inliers}")

    all_3d_points = np.array(all_3d_points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(all_3d_points[:, 0], all_3d_points[:, 1], all_3d_points[:, 2], c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

    print("3D points:", all_3d_points)


if __name__ == '__main__':
    main()


