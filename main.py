import cv2
import glob
import numpy as np
from pathlib import Path
from dataclasses import dataclass

import dearpygui.dearpygui as dpg

from stitching.stitching_error import StitchingError

from EniPy import imageUtils
from EniPy import colors

from stitching import Stitcher
from stitching import AffineStitcher
from stitching.images import Images

import time

def stitch(images, feature_masks=[]):
    #stitcher = AffineStitcher(confidence_threshold=0.5, crop=False, nfeatures=10000, blender_type="no", finder="no", compensator="no")
    stitcher = AffineStitcher(confidence_threshold=0.5, crop=False, nfeatures=10000)

    # return stitcher.stitch_verbose(images, feature_masks, f'verbose')
    images = Images.of(
        images, stitcher.medium_megapix, stitcher.low_megapix, stitcher.final_megapix
    )

    # Resize Images
    imgs = list(images.resize(Images.Resolution.MEDIUM))

    # Find Features
    finder = stitcher.detector
    features = stitcher.find_features(imgs, feature_masks)

    # Match Features
    matcher = stitcher.matcher
    matches = matcher.match_features(features)

    # Subset
    subsetter = stitcher.subsetter

    all_relevant_matches = list(
        matcher.draw_matches_matrix(
            imgs,
            features,
            matches,
            conf_thresh=subsetter.confidence_threshold,
            inliers=True,
            matchColor=(0, 255, 0),
        )
    )
    # for idx1, idx2, img in all_relevant_matches:
    #     cv2.imshow('featureMatch', img)

    # Subset
    subsetter = stitcher.subsetter
    indices = subsetter.get_indices_to_keep(features, matches)

    imgs = subsetter.subset_list(imgs, indices)
    features = subsetter.subset_list(features, indices)
    matches = subsetter.subset_matches(matches, indices)
    images.subset(indices)

    # Camera Estimation, Adjustion and Correction
    camera_estimator = stitcher.camera_estimator
    camera_adjuster = stitcher.camera_adjuster
    wave_corrector = stitcher.wave_corrector

    cameras = camera_estimator.estimate(features, matches)
    cameras = camera_adjuster.adjust(features, matches, cameras)
    cameras = wave_corrector.correct(cameras)

    # Warp Images
    low_imgs = list(images.resize(Images.Resolution.LOW, imgs))
    imgs = None  # free memory

    warper = stitcher.warper
    warper.set_scale(cameras)

    low_sizes = images.get_scaled_img_sizes(Images.Resolution.LOW)
    camera_aspect = images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.LOW)

    low_imgs = list(warper.warp_images(low_imgs, cameras, camera_aspect))
    low_masks = list(warper.create_and_warp_masks(low_sizes, cameras, camera_aspect))
    low_corners, low_sizes = warper.warp_rois(low_sizes, cameras, camera_aspect)

    final_sizes = images.get_scaled_img_sizes(Images.Resolution.FINAL)
    camera_aspect = images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.FINAL)

    final_imgs = list(images.resize(Images.Resolution.FINAL))
    final_imgs = list(warper.warp_images(final_imgs, cameras, camera_aspect))
    final_masks = list(
        warper.create_and_warp_masks(final_sizes, cameras, camera_aspect)
    )
    final_corners, final_sizes = warper.warp_rois(final_sizes, cameras, camera_aspect)


    # Crop
    cropper = stitcher.cropper

    if cropper.do_crop:
        mask = cropper.estimate_panorama_mask(
            low_imgs, low_masks, low_corners, low_sizes
        )


        low_corners = cropper.get_zero_center_corners(low_corners)
        cropper.prepare(low_imgs, low_masks, low_corners, low_sizes)

        low_masks = list(cropper.crop_images(low_masks))
        low_imgs = list(cropper.crop_images(low_imgs))
        low_corners, low_sizes = cropper.crop_rois(low_corners, low_sizes)

        lir_aspect = images.get_ratio(Images.Resolution.LOW, Images.Resolution.FINAL)
        final_masks = list(cropper.crop_images(final_masks, lir_aspect))
        final_imgs = list(cropper.crop_images(final_imgs, lir_aspect))
        final_corners, final_sizes = cropper.crop_rois(
            final_corners, final_sizes, lir_aspect
        )


    # Seam Masks
    seam_finder = stitcher.seam_finder

    seam_masks = seam_finder.find(low_imgs, low_corners, low_masks)
    seam_masks = [
        seam_finder.resize(seam_mask, mask)
        for seam_mask, mask in zip(seam_masks, final_masks)
    ]

    # Exposure Error Compensation
    compensator = stitcher.compensator

    compensator.feed(low_corners, low_imgs, low_masks)

    compensated_imgs = [
        compensator.apply(idx, corner, img, mask)
        for idx, (img, mask, corner) in enumerate(
            zip(final_imgs, final_masks, final_corners)
        )
    ]

    # Blending
    blender = stitcher.blender
    blender.prepare(final_corners, final_sizes)
    for img, mask, corner in zip(compensated_imgs, seam_masks, final_corners):
        blender.feed(img, mask, corner)
    panorama, _ = blender.blend()

    blended_seam_masks = seam_finder.blend_seam_masks(
        seam_masks, final_corners, final_sizes
    )

    with_seam_polygons = seam_finder.draw_seam_polygons(panorama, blended_seam_masks)
    # cv2.imshow('with_seam_polygons', with_seam_polygons)

    return panorama
def processLine(path):
    imagesPath = glob.glob(f'{path}/*.png')
    allImages = []
    prev = None
    avalanche = None
    for imagePath in reversed(imagesPath):
        print(f'\nProcessed: {imagePath}')
        current = cv2.imread(imagePath)
        current = imageUtils.getScaledImage(current, 1080)
        allImages.append(current)
        if prev is not None:
            cv2.imshow('prev', prev)
            cv2.imshow('current', current)

            try:
                start_time = time.time()
                panorama = stitch([prev, current])
                print(f"Elapsed: {time.time() - start_time}")
                cv2.imshow('panorama', panorama)
            except StitchingError:
                cv2.imshow('panorama', imageUtils.getBlankImage(10, 10))
            cv2.waitKey()
        prev = current

    multi = stitch(allImages)
    cv2.imshow('multi', multi)
    cv2.imwrite('multi.png', multi)

    cv2.waitKey()
    cv2.destroyAllWindows()

class ImagePart:
    def __init__(self, path):
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

def processSquare(path):
    imagesPath = glob.glob(f'{path}/*.png')
    dataset = Dataset()

    for path in imagesPath:
        p = Path(path)
        xy = p.stem.split("_")
        x = int(xy[0])
        y = int(xy[1])
        dataset.append(x, y, path)
    dataset.calculateRanges()

    for row in dataset.raw:
        for col in dataset.raw[row]:
            print(f'{row} {col}')


def testMulti(pathA):
    imagesAPath = glob.glob(f'{pathA}/*.png')
    imagesBPath = glob.glob(f'{pathA}/*.png')

    indexSuccess = 0
    for aPath in imagesAPath:
        a = cv2.imread(aPath)
        a = imageUtils.getScaledImage(a, 1080)
        for bPath in imagesBPath:
            print(f'\nProcessed: {aPath}->{bPath}')
            b = cv2.imread(bPath)
            b = imageUtils.getScaledImage(b, 1080)
            cv2.imshow('a', a)
            cv2.imshow('b', b)

            try:
                start_time = time.time()
                panorama = stitch([a, b])
                print(f"Elapsed: {time.time() - start_time}")
                cv2.imshow('panorama', panorama)
                cv2.imwrite(f'{indexSuccess}.jpg', panorama)
                indexSuccess = indexSuccess + 1
            except StitchingError:
                cv2.imshow('panorama', imageUtils.getBlankImage(10, 10))
            cv2.waitKey(100)

    cv2.waitKey()
    cv2.destroyAllWindows()

dpg.create_context()
dpg.create_viewport(title='Custom Title', width=1920 + 100, height=1080 + 100)
dpg.setup_dearpygui()

def loadNewDataset(path):
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

    dpg.delete_item(w, children_only=True)

    viewDrawer = ViewDrawer(w, dataset)
    viewDrawer.drawDatasetIamges()
    viewDrawer.drawHLinks()
    viewDrawer.drawVLinks()



def rectangeCenterTo2Points(center, width = 10, height = 20):
    return [center[0] - width / 2, center[1] - height / 2], [center[0] + width / 2, center[1] + height / 2]
def change_text(sender, app_data):
    print('click')
def addImage(name, frame, pos, parent):
    data = np.flip(frame, 2)  # because the camera data comes in as BGR and we need RGB
    data = data.ravel()  # flatten camera data to a 1 d stricture
    data = np.asfarray(data, dtype='f')  # change data type to 32bit floats
    texture_data = np.true_divide(data, 255.0)  # normalize image data to prepare for GPU

    try:
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(frame.shape[1], frame.shape[0], texture_data, tag=name, format=dpg.mvFormat_Float_rgb)
    except SystemError:
        dpg.set_value(name, texture_data)
    #dpg.add_image(name, pos=[pos[0], pos[1]], tag=f'{name}_')
    # with dpg.item_handler_registry(tag=f'{name}_widget handler') as handler:
    #     dpg.add_item_clicked_handler(callback=change_text)
    # dpg.bind_item_handler_registry(f'{name}_', f'{name}_widget handler')
    dpg.draw_image(name, [pos[0], pos[1]], [pos[0] + frame.shape[1], pos[1] + frame.shape[0]], parent=parent)

def callback(sender, app_data):
    print('OK was clicked.')
    print("Sender: ", sender)
    print("App Data: ", app_data)
    dpg.configure_item('inputPath', default_value=app_data['current_path'])


def cancel_callback(sender, app_data):
    print('Cancel was clicked.')
    print("Sender: ", sender)
    print("App Data: ", app_data)

class ViewDrawer:
    def __init__(self, parent, dataset):
        self.dataset = dataset
        self.imageWidth = 360
        self.imageHeight = self.imageWidth * 9 / 16
        self.vSpace = 50
        self.hSpace = 50
        self.drawlist = dpg.add_drawlist(width=((self.imageWidth + self.hSpace) * self.dataset.rowsCount()) - self.hSpace, height=((self.imageHeight + self.vSpace) * self.dataset.columnsCount()) - self.vSpace, parent=parent)



    def drawDatasetIamges(self):
        for i in range(0, self.dataset.rowsCount()):
            for j in range(0, self.dataset.columnsCount()):
                name = f'{i}:{j}'
                print(name)
                imagePart = self.dataset.at(i, j)
                image = imageUtils.getScaledImage(imagePart.fullImage, self.imageWidth)
                imagePosition = [i * (self.imageWidth + self.hSpace), j * (self.imageHeight + self.vSpace)]
                addImage(name, image, pos=imagePosition, parent=self.drawlist)

    def drawHLinks(self):
        for i in range(0, self.dataset.rowsCount() - 1):
            for j in range(0, self.dataset.columnsCount()):
                l = self.dataset.at(i, j).getImage()
                r = self.dataset.at(i + 1, j).getImage()
                color = (0, 255, 0)
                try:
                    start_time = time.time()
                    panorama = stitch([l, r])
                    print(f"Elapsed: {time.time() - start_time}")
                except StitchingError:
                    color = (255, 0, 0)

                center = [i * (self.imageWidth + self.hSpace) + self.imageWidth + self.hSpace / 2, j * (self.imageHeight + self.vSpace) + self.imageHeight / 2]
                pos = rectangeCenterTo2Points(center)
                dpg.draw_rectangle(pos[0], pos[1], fill=color, parent=self.drawlist)

    def drawVLinks(self):
        for i in range(0, self.dataset.rowsCount()):
            for j in range(0, self.dataset.columnsCount() - 1):
                l = self.dataset.at(i, j).getImage()
                r = self.dataset.at(i, j + 1).getImage()
                color = (0, 255, 0)
                try:
                    start_time = time.time()
                    panorama = stitch([l, r])
                    print(f"Elapsed: {time.time() - start_time}")
                except StitchingError:
                    color = (255, 0, 0)
                center = [i * (self.imageWidth + self.hSpace) + self.imageWidth / 2, j * (self.imageHeight + self.vSpace) + self.imageHeight + self.vSpace / 2]
                pos = rectangeCenterTo2Points(center)
                dpg.draw_rectangle(pos[0], pos[1], fill=color, parent=self.drawlist)


dpg.add_file_dialog(
    directory_selector=True, show=False, callback=callback, tag="file_dialog_id",
    cancel_callback=cancel_callback, width=700 ,height=400)


with dpg.window(label="Controls", width=200, height=1080):
    with dpg.group(horizontal=True):
        dpg.add_input_text(hint='Enter path here', tag='inputPath')
        dpg.add_button(label="...", callback=lambda: dpg.show_item("file_dialog_id"))
    dpg.add_button(label='Reload', callback=lambda:  loadNewDataset(dpg.get_value('inputPath')))


w = dpg.add_window(pos=[200, 0], label="BoardView", horizontal_scrollbar=True, width=1920, height=1080, tag='BoardViewWindow')
print(f'window BoardView {w} {dpg.last_root()}')
#dpg.show_metrics()
#dpg.show_style_editor()
dpg.show_viewport()
dpg.show_imgui_demo()


while dpg.is_dearpygui_running():
    dpg.render_dearpygui_frame()

dpg.destroy_context()


# processLine('./sq4')

#testMulti('multi')
# start_time = time.time()
# multi = stitch(glob.glob('line/*.png'))
# print(f"Elapsed: {time.time() - start_time}")
# cv2.imshow('multi', multi)
# cv2.imwrite('multi.png', multi)
cv2.waitKey()
cv2.destroyAllWindows()