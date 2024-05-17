import csv
from dataclasses import dataclass
import cv2
import numpy as np
from EniPy import colors
from EniPy import imageUtils
from EniPy import eniUtils
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
    def worldMmToPixelLength(self, world):
        x = (world[0] / 1000 ) * self.k
        y = (world[1] / 1000 ) * self.k
        return (int(x), int(y))


initial = cv2.imread('PickPlace/test.png')

footprints = eniUtils.readJson('PickPlace/AntilatencyFootprints.txt')

component_sizes = footprints['footprints']

# component_sizes["RES_0201"] = (0.6, 0.3)
# component_sizes["RES_0402"] = (1.0, 0.5)
# component_sizes["RES_0603"] = (1.6, 0.8)
# component_sizes["RES_0805"] = (2.0, 1.25)
# component_sizes["RES_1206"] = (3.2, 1.6)
# component_sizes["RES_1210"] = (3.2, 2.5)
# component_sizes["RES_1812"] = (4.5, 3.2)
# component_sizes["RES_2010"] = (5.0, 2.5)
# component_sizes["RES_2512"] = (6.35, 3.2)

render = initial.copy()

positionResolver = PositionResolver((render.shape[1], render.shape[0]), (0.100, 0.060), worldCenter=(0.003, 0.003))
undefinedComponents = {}
for comp in topComponents:
    if comp.footprint in component_sizes:
        comp_size = (component_sizes[comp.footprint]['width'], component_sizes[comp.footprint]['length'])
        l = (comp.x - comp_size[0] / 2, comp.y - comp_size[1] / 2)
        r = (comp.x + comp_size[0] / 2, comp.y + comp_size[1] / 2)
        rot_rectangle = (positionResolver.worldMmToPixel((comp.x, comp.y)), positionResolver.worldMmToPixelLength(comp_size), int(comp.rotation))
        box = cv2.boxPoints(rot_rectangle)
        box = np.intp(box)
        rectangle = cv2.drawContours(render, [box], 0, color=colors.Red, thickness=2)
        # cv2.rectangle(render, positionResolver.worldMmToPixel(l), positionResolver.worldMmToPixel(r), color=colors.Red)
    else:
        if comp.footprint not in undefinedComponents:
            undefinedComponents[comp.footprint] = 0
        undefinedComponents[comp.footprint] += 1
    cv2.circle(render, positionResolver.worldMmToPixel((comp.x, comp.y)), 5, color=colors.Violet, thickness=-1)

if len(undefinedComponents) != 0:
    print('Not defined footprints:')
    for k, v in undefinedComponents.items():
        print(f'{k} {v}')
cv2.imshow('render', imageUtils.getScaledImage(render, 1920))
cv2.imwrite('PickPlace/result.png', render)
cv2.waitKey()