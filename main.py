import cv2
import glob
import numpy as np
from dataclasses import dataclass

from stitching.stitching_error import StitchingError

from EniPy import imageUtils
from EniPy import colors

from stitching import Stitcher
from stitching import AffineStitcher
def process(path):
    imagesPath = glob.glob(f'{path}/*.JPG')

    #stitcher = Stitcher(detector="sift", confidence_threshold=0.2)
    stitcher = AffineStitcher(confidence_threshold=0.5, crop=False, nfeatures=10000)
    #stitcher = AffineStitcher(confidence_threshold=0.5, crop=False, nfeatures=5000, range_width=1, blender_type="no", finder="no")
    imagesList = []
    prev = None
    avalanche = None
    for imagePath in reversed(imagesPath):
        print(f'\nProcessed: {imagePath}')
        current = cv2.imread(imagePath)
        current = imageUtils.getScaledImage(current, 1080)
        imagesList.append(current)

        if prev is not None:
            cv2.imshow('prev', prev)
            cv2.imshow('current', current)
            try:
                panorama = stitcher.stitch([prev, current])
                cv2.imshow('panorama', panorama)
            except StitchingError:
                cv2.imshow('panorama', imageUtils.getBlankImage(10, 10))
            cv2.waitKey(100)
        prev = current
    panorama = stitcher.stitch(imagesList)
    cv2.imshow('FullPanorama', imageUtils.getScaledImage(panorama, 3840))
    cv2.imwrite('FullPanorama.png', panorama)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    process('images/MaxResolution/HighDensity/B/')