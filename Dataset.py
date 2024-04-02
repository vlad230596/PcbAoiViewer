import cv2
from EniPy import imageUtils
import time

from stitching.stitching_error import StitchingError
from PlaneStitcher import PlaneStitcher


class ImagePart:
    def __init__(self, path):
        self.fullImage = None
        self.path = path

    def load(self):
        self.fullImage = cv2.imread(self.path)
        
    def getImage(self):
        self.image = imageUtils.getScaledImage(self.fullImage, 1080)
        return self.image


class Dataset:

    def __init__(self):
        self.raw = {}
        enoughBigInt = 1000000
        self.minX = enoughBigInt
        self.minY = enoughBigInt

        self.maxX = -enoughBigInt
        self.maxY = -enoughBigInt

        self.stepX = enoughBigInt
        self.stepY = enoughBigInt

    def at(self, row, column):
        return self.raw[self.minX + row * self.stepX][self.maxY - column * self.stepY]

    def getStitchResult(self, left, right):
        l = self.at(left[0], left[1]).getImage()
        r = self.at(right[0], right[1]).getImage()
        try:
            start_time = time.time()
            result = PlaneStitcher.stitch([l, r])
            print(f"Elapsed: {time.time() - start_time}")
            return True
        except StitchingError:
            return False
    def append(self, x, y, path):
        if x not in self.raw:
            self.raw[x] = {}
        self.raw[x][y] = ImagePart(path)

    def rowsCount(self) -> int:
        return int((self.maxX - self.minX) / self.stepX + 1)
    def columnsCount(self) -> int:
        return int((self.maxY - self.minY) / self.stepY + 1)

    def calculateRanges(self):
        self.raw = dict(sorted(self.raw.items()))
        keys = list(self.raw.keys())
        if len(keys) == 0:
            pass
        self.minX = keys[0]
        self.maxX = keys[-1]

        for i in range(0, len(keys) - 1):
            diff = keys[i + 1] - keys[i]
            if diff < self.stepX:
                self.stepX = diff

        for rowIndex in self.raw:
            self.raw[rowIndex] = dict(sorted(self.raw[rowIndex].items()))
            row = self.raw[rowIndex]
            keys = list(row.keys())
            if len(keys) == 0:
                continue

            for i in range(0, len(keys) - 1):
                diff = keys[i + 1] - keys[i]
                if diff < self.stepY:
                    self.stepY = diff

            if keys[0] < self.minY:
                self.minY = keys[0]
            if keys[-1] > self.maxY:
                self.maxY = keys[-1]

    def loadAllImages(self):
        for row in self.raw:
            for col in self.raw[row]:
                self.raw[row][col].load()