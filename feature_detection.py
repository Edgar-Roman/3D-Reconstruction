import numpy as np
import cv2 as cv
import os

# SURF, ORB, AKAZE

DATA_DIR = './data/statue/images/dslr_images_undistorted/'

# SIFT


def getFeaturesSIFT(filename):
    img = cv.imread(filename)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp = sift.detect(gray, None)
    img = cv.drawKeypoints(gray, kp, img)
    save_to = "SIFT_" + os.path.basename(filename)
    cv.imwrite(save_to, img)


if __name__ == '__main__':
    for image in os.scandir(DATA_DIR):
        getFeaturesSIFT(image.path)
