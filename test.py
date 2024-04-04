import cv2
import glob

from pathlib import Path
from dataclasses import dataclass

import dearpygui.dearpygui as dpg

from stitching.stitching_error import StitchingError

from stitching.blender import Blender

from EniPy import imageUtils
from EniPy import colors

from Dataset import Dataset
from ViewDrawer import ViewDrawer

import time

from stitching.stitching_error import StitchingError
from PlaneStitcher import PlaneStitcher

def processLine(path):
    imagesPath = glob.glob(f'{path}/*.png')
    dataset = Dataset()

    for path in imagesPath:
        p = Path(path)
        xy = p.stem.split("_")
        x = int(xy[0])
        y = int(xy[1])
        dataset.append(x, y, path)
    dataset.calculateRanges()
    dataset.loadAllImages()

    for i in range(0, dataset.rowsCount()):
        finalResults = None
        prevX, prevY = 0, 0
        for j in range(1, dataset.columnsCount()):
            print(f'\nProcessed: {j - 1} - > {j}')
            prev = dataset.at(i, j - 1).getImage()
            current = dataset.at(i, j).getImage()

            cv2.imshow('prev', prev)
            cv2.imshow('current', current)

            try:
                start_time = time.time()
                result = PlaneStitcher.stitch([prev, current])
                print(f"Elapsed: {time.time() - start_time}")
                patchedResult = result
                patchedResult.corners = [((x + prevX), (y + prevY)) for x, y in patchedResult.corners]

                prevX, prevY = patchedResult.corners[-1]

                if finalResults is None:
                    finalResults = patchedResult
                else:
                    finalResults.masks.append(patchedResult.masks[-1])
                    finalResults.imgs.append(patchedResult.imgs[-1])
                    finalResults.corners.append(patchedResult.corners[-1])
                    finalResults.sizes.append(patchedResult.sizes[-1])

                cv2.imshow('panorama', result.panorama)

                blender = Blender(blender_type='multiband')

                blender.prepare(result.corners, result.sizes)
                for img, mask, corner in zip(result.imgs, result.masks, result.corners):
                    blender.feed(img, mask.get(), corner)
                blended = blender.blend()
                cv2.imshow('blended', blended[0])

                print(f'Sizes: {result.sizes} Corners: {result.corners}')
                for m in result.masks:
                    mask = m.get()
                    print(f'Mask sizes {mask.shape}')
                # width = 0
                # height = 0
                # for corner, size in zip(result.corners, result.sizes):
                #     xSize = corner[0] + size[0]
                #     ySize = corner[1] + size[1]
                #     if xSize > width:
                #         width = xSize
                #     if ySize > height:
                #         height = ySize
                # print(f'panorama w: {result.panorama.shape[1]} h: {result.panorama.shape[0]}')
                # print(f'my w: {width} h: {height}')
                # blank = imageUtils.getBlankImage(width, height)
                # cv2.imshow('blank', blank)
                # for image, corner, size, mask in zip(result.imgs, result.corners, result.sizes, result.masks):
                #     m = mask.get()
                #     mask3 = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
                #     withMask = cv2.bitwise_and(image, mask3)
                #
                #     roiPart = blank[corner[1]:corner[1]+size[1],corner[0]:corner[0] + size[0]]
                #     roiPartUpdated = cv2.bitwise_or(roiPart, withMask)
                #
                #     blank[corner[1]:corner[1]+size[1],corner[0]:corner[0] + size[0]] = roiPartUpdated
                # cv2.imshow('processed', blank)

            except StitchingError:
                cv2.imshow('panorama', imageUtils.getBlankImage(10, 10))
            cv2.waitKey(500)


        blender = Blender(blender_type='multiband')

        blender.prepare(finalResults.corners, finalResults.sizes)
        for img, mask, corner in zip(finalResults.imgs, finalResults.masks, finalResults.corners):
            blender.feed(img, mask.get(), corner)
        blended = blender.blend()
        cv2.imshow('finalBlended', imageUtils.getScaledImage(blended[0], 640))
        cv2.imwrite(f'finalBlended{i}.png', blended[0])

    # multi = PlaneStitcher.stitch(allImages)
    # cv2.imshow('multi', multi.panorama)

    cv2.waitKey()
    cv2.destroyAllWindows()

processLine(f'NewPlatform/captured_80.0')
