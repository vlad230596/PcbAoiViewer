import csv
from dataclasses import dataclass
import cv2
from EniPy import colors
from EniPy import imageUtils
@dataclass
class ComponentInfo:
    designator: str
    x: float
    y: float
    rotation: float
    footprint: str
    comment: str

    def __init__(self, designator, x, y, rotation, footprint, comment):
        self.designator = designator
        self.x = float(x)
        self.y = float(y)
        self.rotation = rotation
        self.footprint = footprint
        self.comment = comment


topComponents = []
with open('PickPlace/test.csv', newline='') as csvfile:
    skipEmptyLines = 3
    for l in csvfile:
        if l == '\r\n':
            skipEmptyLines -= 1
        if skipEmptyLines == 0:
            break
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['Layer'] == 'TopLayer':
            topComponents.append(ComponentInfo(designator=row['Designator'], x=row['Center-X(mm)'], y=row['Center-Y(mm)'], rotation=row['Rotation'], footprint=row['Footprint'], comment=row['Comment']))

class PositionResolver:
    def __init__(self, pixelSize, worldSize, worldCenter = (0, 0)):
        self.k = pixelSize[0] / worldSize[0]
        self.pixelSize = pixelSize
        self.worldCenter = worldCenter

    def worldMmToPixel(self, world):
        x = (world[0] / 1000 + self.worldCenter[0]) * self.k
        y = self.pixelSize[1] - ((world[1] / 1000 + self.worldCenter[1]) * self.k)
        return (int(x), int(y))

initial = cv2.imread('PickPlace/test.png')


render = initial.copy()

positionResolver = PositionResolver((render.shape[1], render.shape[0]), (0.100, 0.060), worldCenter=(0.003, 0.003))

for comp in topComponents:
    print(comp.designator)
    cv2.circle(render, positionResolver.worldMmToPixel((comp.x, comp.y)), 10, color=colors.Red, thickness=-1)

cv2.imshow('render', imageUtils.getScaledImage(render, 1920))
cv2.waitKey()