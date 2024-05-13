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

    images = [imageUtils.getScaledImage(cv2.imread(path), 1080) for path in sorted(imagesPath)]

    finalResults = None
    prevX, prevY = 0, 0
    for i in range(1, len(images)):
        print(f'\nProcessed: {i - 1} - > {i}')
        prev = images[i - 1]
        current = images[i]

        cv2.imshow('prev', prev)
        cv2.imshow('current', current)


        start_time = time.time()
        result = PlaneStitcher.stitch([prev, current])
        if result.status:
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
        else:
            cv2.imshow('panorama', imageUtils.getBlankImage(10, 10))
        cv2.waitKey()


    blender = Blender(blender_type='multiband')

    blender.prepare(finalResults.corners, finalResults.sizes)
    for img, mask, corner in zip(finalResults.imgs, finalResults.masks, finalResults.corners):
        blender.feed(img, mask.get(), corner)
    blended = blender.blend()
    cv2.imshow('lineBlended', imageUtils.getScaledImage(blended[0], 640))
    cv2.imwrite(f'lineBlended.png', blended[0])

    # multi = PlaneStitcher.stitch(allImages)
    # cv2.imshow('multi', multi.panorama)

    cv2.waitKey()
    cv2.destroyAllWindows()

def refStitch(path):
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

    for j in range(0, dataset.columnsCount()):
    # for j in range(0, 1):
        refLeftIndex = 4
        refRightIndex = refLeftIndex + 1

        refLeft = dataset.at(refLeftIndex, j).getImage()
        refRight = dataset.at(refRightIndex, j).getImage()

        cv2.imshow('refLeft', refLeft)
        cv2.imshow('refRight', refRight)
        result = PlaneStitcher.stitch([refLeft, refRight])
        if not result.status:
            continue
        cv2.imshow('panorama', result.panorama)

        targetWidth = refLeft.shape[1]
        targetHeight = refLeft.shape[0]
        targetSize = [targetWidth, targetHeight]

        finalSizes = [targetSize for _ in range(dataset.rowsCount())]
        finalImages = [dataset.at(i, j).getImage() for i in range(dataset.rowsCount())]

        refLeftMask = cv2.resize(result.masks[0].get(), targetSize, interpolation=cv2.INTER_LINEAR_EXACT)
        refRightMask = cv2.resize(result.masks[1].get(), targetSize, interpolation=cv2.INTER_LINEAR_EXACT)
        middleMask = cv2.bitwise_and(refLeftMask, refRightMask)

        finalMasks = []
        finalMasks.append(cv2.UMat(refLeftMask))
        finalMasks.extend([cv2.UMat(middleMask)] * (dataset.rowsCount() - 2))
        finalMasks.append(cv2.UMat(refRightMask))

        finalCorners = [None] * dataset.rowsCount()
        finalCorners[refLeftIndex] = result.corners[0]
        for i in range(refRightIndex, dataset.rowsCount()):
            x, y = finalCorners[i - 1]
            nextCorners = (x + result.corners[1][0], y + result.corners[1][1])
            finalCorners[i] = nextCorners
        for i in reversed(range(refLeftIndex)):
            x, y = finalCorners[i + 1]
            prevCorners = (x - result.corners[1][0], y - result.corners[1][1])
            finalCorners[i] = prevCorners

        blender = Blender(blender_type='multiband')

        blender.prepare(finalCorners, finalSizes)
        for img, mask, corner in zip(finalImages, finalMasks, finalCorners):
            blender.feed(img, mask.get(), corner)
        blended, _ = blender.blend()
        cv2.imshow('finalBlended', imageUtils.getScaledImage(blended, 1920))
        cv2.imwrite(f'finalBlended{j}.png', blended)
        cv2.waitKey()

    cv2.destroyAllWindows()

# refStitch(f'withLight/captured_80.0')
processLine(f'FinalBlended')
