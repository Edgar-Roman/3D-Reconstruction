import os
import requests
import py7zr
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from glob import glob

"""
Class for downloading datasets available at https://www.eth3d.net/datasets
"""

DATA_KEY_UNDISTORT = 'undistorted'
DATA_KEY_JPG = 'jpg'
DATA_KEY_RAW = 'raw'

DATA_STREAM_SIZE = 1024   # Kilobytes


class StatueDataset:
  
  _urls = {
    DATA_KEY_UNDISTORT: 'https://www.eth3d.net/data/statue_dslr_undistorted.7z',
    DATA_KEY_JPG: 'https://www.eth3d.net/data/statue_dslr_jpg.7z',
    DATA_KEY_RAW: 'https://www.eth3d.net/data/statue_dslr_raw.7z'
  }
  
  _name = 'statue'

  def __init__(self, root: str = os.getcwd(),
               datasetType: str = DATA_KEY_UNDISTORT) -> None:
    self._images = np.zeros(0)
    self._intrinsics = np.zeros(0)
    self._points3d = np.zeros(0)
    
    self._root = root
    self._dataDir = '{}/{}'.format(root, self._name)
    self._downloadFile = self._download(datasetType)
    self._unpack()
    self._extract()

  def _download(self, datasetType: str) -> str:
    url = self._urls[datasetType]               # Download URL
    filename = os.path.basename(url)            # Zipped file name
    downloadFile = '{}/{}'.format(self._root, filename)

    # Check if file has been downloaded
    if len(glob(downloadFile)):
      print('Downloaded dataset found on disk.')
    else:
      res = requests.get(url, stream=True)
      progress = tqdm(total=int(res.headers.get('content-length', 0)), 
                      unit='iB', unit_scale=True)
      if res.status_code == 200:
        with open(downloadFile, 'wb') as f:
          for data in res.iter_content(DATA_STREAM_SIZE):
            progress.update(len(data))
            f.write(data)
        progress.close()
      else:
        print('Request failed for url: {}'.format(url))
    return downloadFile

  def _unpack(self) -> None:
    # Check if unpacked
    if len(glob(self._dataDir)):
      print('Unpacked dataset found on disk.')
      print('Warning: If same dataset but different type is unpacked, contents' 
            'may not be the same')
    else:
      print('Unpacking dataset at: {}'.format(self._downloadFile))
      with py7zr.SevenZipFile(self._downloadFile, 'r') as f:
        f.extractall(self._root)

  def _extractImages(self) -> np.ndarray:
    imageDir = glob('{}/images/*'.format(self._dataDir))[0]
    imageFiles = glob('{}/*'.format(imageDir))
    imageFiles.sort()
    
    images = []
    for imageFile in tqdm(imageFiles):
      image = cv2.imread(imageFile)
      images.append(image)

    return np.array(images)

  def _extractCameraIntrinsics(self, calibDir: str) -> np.ndarray:
    cameraFile = glob('{}/cameras.txt'.format(calibDir))[0]

    intrinsics = []
    with open(cameraFile, 'r') as f:
      for line in tqdm(f.readlines()[3:]):
        _, _, W, H, fx, fy, cx, cy = line.strip().split()
        intrinsic = np.array([
          [fx, 0, cx],
          [0, fy, cy],
          [0, 0, 1]
        ]).astype(np.float32)
        intrinsics.append(intrinsic)
    return np.array(intrinsics)

  def _extract2DPoints(self, calibDir: str) -> None:
    imagesFile = glob('{}/images.txt'.format(calibDir))[0]
    with open(imagesFile, 'r') as f:
      data = f.readlines()[4:]
      for i in range(0, len(data) // 2, 2):
        imgId, qw, qx, qy, qz, tx, ty, tz, camId, name = data[i].strip().split()
        data2 = np.array(data[i+1].split(), dtype=np.float32).reshape((-1, 3))
        data2 = np.array([d for d in data2 if d[2] > -1])
        print(imgId)
        # pts2d = data2[:, :2]
        # pts3dId = data2[:, 2].astype(np.int32)
        # print(pts3dId)
      

  def _extract(self) -> None:
    # Extract image data
    print('Extracting image data...')
    self._images = self._extractImages()
    
    # Extract Calibration Parameters
    calibDir = glob('{}/*calibration*'.format(self._dataDir))[0]

    # Extract camera parameters
    self._intrinsics = self._extractCameraIntrinsics(calibDir)

    # Extract 2D points
    self._extract2DPoints(calibDir)

    # Extract 3D points


if __name__ == '__main__':
  # Extract Statue Undistorted Data
  root = str(Path(os.getcwd()).parent / 'data')
  statueset = StatueDataset(root)