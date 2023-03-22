import py7zr
import os

if __name__ == '__main__':
    with py7zr.SevenZipFile("data/statue_dslr_undistorted.7z", 'r') as archive:
        archive.extractall()


