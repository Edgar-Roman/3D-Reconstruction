import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import getCameraParams
from sklearn.cluster import DBSCAN
import os
from scipy.spatial import Delaunay
from scipy.optimize import least_squares
from pyntcloud import PyntCloud
import pandas as pd

DATA_DIR = './data/statue/images/dslr_images_undistorted/'
# DATA_DIR = './data/statue/images/dslr_images_backgroundless/'


def detectAndCompute(images):
    keypoints, descriptors = [], []
    sift = cv.SIFT_create()
    for img in images:
        kp, des = sift.detectAndCompute(img, None)
        keypoints.append(kp)
        descriptors.append(des)
    return keypoints, descriptors


def visualize_keypoints(image, keypoints):
    img_with_keypoints = cv.drawKeypoints(image, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img_with_keypoints)
    plt.show()


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


def nonlinear_estimate_3d_point(pts3d, pts1, pts2, R, t, K):
    refined_pts3d = []

    for i, pt3d in enumerate(pts3d):
        pt3d = pt3d.reshape(3, 1)
        p1_proj = np.array([pts1[i].tolist()])
        p2_proj = np.array([pts2[i].tolist()])

        _, rvec1, tvec1 = cv.solvePnP(objectPoints=pt3d, imagePoints=p1_proj, cameraMatrix=K, distCoeffs=None)
        _, rvec2, tvec2 = cv.solvePnP(objectPoints=pt3d, imagePoints=p2_proj, cameraMatrix=K, distCoeffs=None, rvec=R, tvec=t)

        _, refined_rvec1, refined_tvec1 = cv.solvePnPRefineLM(objectPoints=pt3d, imagePoints=p1_proj, cameraMatrix=K, distCoeffs=None, rvec=rvec1, tvec=tvec1)
        _, refined_rvec2, refined_tvec2 = cv.solvePnPRefineLM(objectPoints=pt3d, imagePoints=p2_proj, cameraMatrix=K, distCoeffs=None, rvec=rvec2, tvec=tvec2)

        refined_pt3d = (refined_tvec1 + refined_tvec2) / 2
        refined_pts3d.append(refined_pt3d.reshape(3))

    return np.array(refined_pts3d)


def filter_outliers(points, eps, min_samples):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    filtered_points = points[labels != -1]
    return filtered_points


def main():
    # Load images and convert them to grayscale
    image_filenames = sorted(os.listdir(DATA_DIR))
    images = [cv.imread(os.path.join(DATA_DIR, fname), cv.IMREAD_GRAYSCALE) for fname in image_filenames]

    # Detect keypoints and compute descriptors for each image
    keypoints, descriptors = detectAndCompute(images)

    first_image = images[0]
    first_image_keypoints = keypoints[0]
    visualize_keypoints(first_image, first_image_keypoints)

    # Match features between consecutive image pairs
    matched_points = []
    for i in range(len(images) - 1):
        good_matches = match_features(descriptors[i], descriptors[i + 1])
        pts1 = np.array([keypoints[i][m.queryIdx].pt for m in good_matches], dtype=np.float32)
        pts2 = np.array([keypoints[i + 1][m.trainIdx].pt for m in good_matches], dtype=np.float32)
        matched_points.append((pts1, pts2))

        # Draw and display the matches
        img_matches = cv.drawMatches(images[i], keypoints[i], images[i + 1], keypoints[i + 1], good_matches, None,
                                     flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img_matches)
        plt.title(f'Matched keypoints between images {i + 1} and {i + 2}')
        plt.show()

    K = getCameraParams('./data/statue/dslr_calibration_undistorted/cameras.txt')
    threshold = 1
    all_3d_points = []

    for i, (pts1, pts2) in enumerate(matched_points):
        R, t, inliers = estimate_motion_RANSAC(pts1, pts2, K, threshold)
        pts3d = triangulate(pts1, pts2, R, t, K)
        # pts3d_refined = nonlinear_estimate_3d_point(pts3d, pts1, pts2, R, t, K)
        # all_3d_points.extend(pts3d_refined)
        all_3d_points.extend(pts3d)
        print(f"Pair {i + 1}-{i + 2}: Number of inliers: {inliers}")

    all_3d_points = np.array(all_3d_points)

    # Adjust eps and min_samples according to your dataset
    eps = 0.3
    min_samples = 15

    # filtered_3d_points = filter_outliers(all_3d_points, eps, min_samples)
    filtered_3d_points = all_3d_points
    print("Total 3D points plotted:", len(filtered_3d_points))

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(filtered_3d_points[:, 0], filtered_3d_points[:, 1], filtered_3d_points[:, 2], c='r', marker='o')
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()
    #
    # print("3D points:", all_3d_points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(filtered_3d_points[:, 0], filtered_3d_points[:, 1], filtered_3d_points[:, 2], c='r', marker='o',
               s=3)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Compute Delaunay triangulation and plot the resulting mesh
    # tri = Delaunay(filtered_3d_points)
    # ax.plot_trisurf(filtered_3d_points[:, 0], filtered_3d_points[:, 1], filtered_3d_points[:, 2],
    #                 triangles=tri.simplices, color='b',
    #                 alpha=0.1)  # Set alpha=0.1 (or another desired value) to control the mesh transparency
    # ax.grid(False)
    ax.set_axis_off()
    plt.show()

    print("3D points:", all_3d_points)

    # # Convert filtered_3d_points to a Pandas DataFrame
    # point_cloud_pd = pd.DataFrame(filtered_3d_points, columns=['x', 'y', 'z'])
    #
    # # Create a PyntCloud object
    # point_cloud_pynt = PyntCloud(point_cloud_pd)
    #
    # # Get k-neighbors and compute the scalar field
    # k_neighbors = point_cloud_pynt.get_neighbors(k=10)
    # ev = point_cloud_pynt.add_scalar_field("eigen_values", k_neighbors=k_neighbors)
    #
    # point_cloud_pynt.add_scalar_field("curvature")
    #
    # # Filter out points based on the curvature scalar field
    # curvature_values = point_cloud_pynt.points["curvature"]
    # point_cloud_pd_filtered = point_cloud_pynt.points[(curvature_values >= 0.001) & (curvature_values < 0.1)]
    #
    # # Create a new PyntCloud object with the filtered points
    # point_cloud_pynt_filtered = PyntCloud(point_cloud_pd_filtered)
    #
    # # Compute and visualize the mesh
    # mesh = point_cloud_pynt_filtered.get_mesh("ConvexHull")
    # mesh_points = point_cloud_pynt_filtered.points.loc[mesh.vertices]
    # triangles = mesh.simplices
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_trisurf(mesh_points["x"], mesh_points["y"], mesh_points["z"], triangles=triangles, color='b', alpha=0.1)
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()
    #
    # print("3D points:", all_3d_points)


"""
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

    # Initialize the rotation matrices list and translation vectors list
    Rs = [np.identity(3)]
    ts = [np.zeros((3, 1))]

    for i, (pts1, pts2) in enumerate(matched_points):
        R, t, inliers = estimate_motion_RANSAC(pts1, pts2, K, threshold)

        # Accumulate the transformations
        R_cumulative = Rs[-1] @ R
        t_cumulative = Rs[-1] @ t + ts[-1]

        # Store the accumulated transformations
        Rs.append(R_cumulative)
        ts.append(t_cumulative)

    all_3d_points = []

    for i, (pts1, pts2) in enumerate(matched_points):
        R1, t1 = Rs[i], ts[i]
        R2, t2 = Rs[i + 1], ts[i + 1]

        P1 = K @ np.hstack((R1, t1))
        P2 = K @ np.hstack((R2, t2))

        pts3d = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts3d = (pts3d[:3] / pts3d[3]).T

        all_3d_points.extend(pts3d)

    all_3d_points = np.array(all_3d_points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(all_3d_points[:, 0], all_3d_points[:, 1], all_3d_points[:, 2], c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

    print("3D points:", all_3d_points)
"""

if __name__ == '__main__':
    main()


