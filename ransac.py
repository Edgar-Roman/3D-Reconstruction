import numpy as np
import cv2
import utils


# Input:
#   - pts1: Nx2 array of feature points in the first image
#   - pts2: Nx2 array of feature points in the second image
#   - K: 3x3 camera intrinsic matrix
#   - threshold: threshold for RANSAC algorithm
#   - max_iterations: maximum number of iterations for RANSAC algorithm
# Output:
#   - R: 3x3 rotation matrix between the two cameras
#   - t: 3x1 translation vector between the two cameras
def estimate_motion_RANSAC(pts1, pts2, K, threshold, max_iterations):
    # Convert feature points to homogeneous coordinates
    pts1_homo = np.concatenate((pts1, np.ones((pts1.shape[0], 1))), axis=1)
    pts2_homo = np.concatenate((pts2, np.ones((pts2.shape[0], 1))), axis=1)

    best_R = None
    best_t = None
    best_num_inliers = 0

    for i in range(max_iterations):
        # Randomly select 8 points to estimate the essential matrix
        indices = np.random.choice(pts1.shape[0], 8, replace=False)
        E, _ = cv2.findEssentialMat(pts1[indices], pts2[indices], K)

        # Decompose essential matrix into rotation and translation
        _, R, t, _ = cv2.recoverPose(E, pts1[indices], pts2[indices], K)

        # Compute reprojection error for all feature point matches
        pts1_proj = (K @ pts1_homo.T).T
        pts2_proj = (K @ (R @ pts2_homo.T + t)).T

        # Normalize the homogeneous coordinates
        pts1_proj /= pts1_proj[:, 2].reshape(-1, 1)
        pts2_proj /= pts2_proj[:, 2].reshape(-1, 1)

        diff = pts2_proj - pts1_proj
        error = np.sum(diff * diff, axis=1)
        # print(error)

        # Count number of inliers
        num_inliers = np.sum(error < threshold)

        if num_inliers > best_num_inliers:
            # Update best R and t if we found more inliers
            best_num_inliers = num_inliers
            best_R = R
            best_t = t

    return best_R, best_t


if __name__ == '__main__':
    camera_path = "/Users/alina/Desktop/3D-Reconstruction/data/statue/dslr_calibration_undistorted/cameras.txt"
    K = utils.getCameraParams(camera_path)
    pts1 = np.load('query.npy')
    pts2 = np.load('train.npy')
    print(pts1.shape)
    print(pts2.shape)
    best_R, best_t = estimate_motion_RANSAC(pts1, pts2, K, 100, 100)
    
    print(best_R)
    print(best_t)