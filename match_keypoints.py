import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os


DATA_DIR = './data/statue/images/dslr_images_undistorted/'

if __name__ == '__main__':
    img1 = cv.imread(os.path.join(DATA_DIR, "DSC_0490.JPG"), cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(os.path.join(DATA_DIR, "DSC_0490.JPG"), cv.IMREAD_GRAYSCALE)

    sift = cv.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = [[m] for m, n in matches if m.distance < 0.75 * n.distance]

    query = np.array([kp1[g[0].queryIdx].pt for g in good], dtype=np.float32)
    train = np.array([kp2[g[0].trainIdx].pt for g in good], dtype=np.float32)

    np.save('query.npy', query)
    np.save('train.npy', train)

    # cv.drawMatchesKnn expects list of lists as matches.
    # img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img3), plt.show()
