import numpy as np
import utils

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

def triangulatePoints(K, H, p1, p2):
    # Compute camera matrices from homography matrix
    K_inv = np.linalg.inv(K)
    H1 = np.eye(3, 4)
    H2 = np.dot(K_inv, H)
    H2 = np.divide(H2, np.linalg.norm(H2[:, 0]))

    # Stack image points into homogeneous coordinates
    p1_h = np.vstack((p1, np.ones((1, p1.shape[1]))))
    p2_h = np.vstack((p2, np.ones((1, p2.shape[1]))))

    # Triangulate image points
    A = np.zeros((4, 4))
    P = np.zeros((3, p1.shape[1]))
    for i in range(p1.shape[1]):
        A[0, :] = p1_h[:, i].T * H1[2, :] - H1[0, :]
        A[1, :] = p1_h[:, i].T * H1[2, :] - H1[1, :]
        A[2, :] = p2_h[:, i].T * H2[2, :] - H2[0, :]
        A[3, :] = p2_h[:, i].T * H2[2, :] - H2[1, :]
        _, _, v = np.linalg.svd(A)
        P[:, i] = v[-1, :3] / v[-1, 3]

    return P

if __name__ == "__main__":
    camera_path = "/Users/alina/Desktop/3D-Reconstruction/data/statue/dslr_calibration_undistorted/cameras.txt"
    K = utils.getCameraParams(camera_path)
    pts1 = np.load('query.npy')
    pts2 = np.load('train.npy')
    H = np.load('M.npy')

    P = triangulatePoints(K, H, pts1, pts2)
    print(P)
