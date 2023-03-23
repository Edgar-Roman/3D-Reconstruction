import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os


DATA_DIR = './data/statue/images/dslr_images_undistorted/'


def estimate_motion_RANSAC(pts1, pts2, K, threshold):
    E, _ = cv.findEssentialMat(pts1, pts2, K, method=cv.RANSAC, threshold=threshold)

    # Decompose the essential matrix into rotation and translation
    num_inliers, R, t, _ = cv.recoverPose(E, pts1, pts2, K)

    return R, t, num_inliers


def detectAndCompute():
    img1 = cv.imread(os.path.join(DATA_DIR, "DSC_0490.JPG"), cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(os.path.join(DATA_DIR, "DSC_0491.JPG"), cv.IMREAD_GRAYSCALE)

    # akaze = cv.AKAZE_create()
    sift = cv.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # kp1, des1 = akaze.detectAndCompute(img1, None)
    # kp2, des2 = akaze.detectAndCompute(img2, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    query = np.array([kp1[m.queryIdx].pt for m in good], dtype=np.float32)
    train = np.array([kp2[m.trainIdx].pt for m in good], dtype=np.float32)

    np.save('query.npy', query)
    np.save('train.npy', train)

    img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3), plt.show()


def getCameraParams(filename):
    K = np.zeros((3, 3))
    with open(filename, "r") as f:
        for line in f:
            if '#' in line:
                continue
            line = line.split()
            fx, fy, cx, cy = float(line[4]), float(line[5]), float(line[6]), float(line[7])
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy
            K[2, 2] = 1

    return K


if __name__ == '__main__':
    detectAndCompute()

    # Load the matched points
    pts1 = np.load('query.npy')
    pts2 = np.load('train.npy')

    print(pts1.shape)

    # Assuming you have the camera intrinsic matrix K
    # ...

    # Set RANSAC parameters
    threshold = 1  # Reprojection error threshold (in pixels)
    max_iterations = 1000  # Maximum number of RANSAC iterations

    K = getCameraParams('./data/statue/dslr_calibration_undistorted/cameras.txt')
    print(K)
    # Estimate the camera motion using RANSAC
    R, t, inliers = estimate_motion_RANSAC(pts1, pts2, K, threshold)

    print("Rotation matrix:\n", R)
    print("Translation vector:\n", t)
    print("Number of inliers:", inliers)
