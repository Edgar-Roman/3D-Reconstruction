import numpy as np

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