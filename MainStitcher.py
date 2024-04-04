
from PlaneStitcher import PlaneStitcher
import time

class MainStitcher:
    def __init__(self, dataset):
        self.dataset = dataset
        self.successStitchCount = 0
        self.failStitchCount = 0
        self.substitchResult = [[None for _ in range(dataset.rowsCount() * dataset.columnsCount())] for _ in range(dataset.columnsCount() * dataset.columnsCount())]

    def posToIndex(self, pos):
        return pos[0] * self.dataset.rowsCount() + pos[1]
    def getCell(self, left, right):
        if self.substitchResult[self.posToIndex(left)] is None:
            self.substitchResult[self.posToIndex(left)] = []
        row = self.substitchResult[self.posToIndex(left)]
        return row[self.posToIndex(right)]

    def setCell(self, left, right, value):
        if self.substitchResult[self.posToIndex(left)] is None:
            self.substitchResult[self.posToIndex(left)] = []
        row = self.substitchResult[self.posToIndex(left)]
        row[self.posToIndex(right)] = value
    def getStitchResult(self, left, right):
        existingResult = self.getCell(left, right)
        if existingResult is not None:
            return existingResult.status
        l = self.dataset.at(left[0], left[1]).getImage()
        r = self.dataset.at(right[0], right[1]).getImage()

        start_time = time.time()
        result = PlaneStitcher.stitch([l, r])
        self.setCell(left, right, result)
        print(f"Elapsed: {time.time() - start_time}")
        if result.status:
            self.successStitchCount = self.successStitchCount + 1
        else:
            self.failStitchCount = self.failStitchCount + 1
        return result.status