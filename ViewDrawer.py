import dearpygui.dearpygui as dpg
from EniPy import imageUtils
import numpy as np
class ViewDrawer:
    def __init__(self, parent, mainStitcher):
        self.dataset = mainStitcher.dataset
        self.mainStitcher = mainStitcher
        self.updateImageWidth(360)
        self.vSpace = 50
        self.hSpace = 50
        self.parent = parent
        self.drawlist = dpg.add_drawlist(width=((self.imageWidth + self.hSpace) * self.dataset.rowsCount()) - self.hSpace, height=((self.imageHeight + self.vSpace) * self.dataset.columnsCount()) - self.vSpace, parent=parent)

    def updateView(self):
        dpg.delete_item(self.drawlist)
        self.drawlist = dpg.add_drawlist(
            width=((self.imageWidth + self.hSpace) * self.dataset.rowsCount()) - self.hSpace,
            height=((self.imageHeight + self.vSpace) * self.dataset.columnsCount()) - self.vSpace,
            parent=self.parent)

    def updateImageWidth(self, width):
        self.imageWidth = width
        self.imageHeight = self.imageWidth * 9 / 16
    @staticmethod
    def rectangeCenterTo2Points(center, width=10, height=20):
        return [center[0] - width / 2, center[1] - height / 2], [center[0] + width / 2, center[1] + height / 2]

    @staticmethod
    def addImage(name, frame, pos, parent):
        unique_tag = dpg.generate_uuid()
        data = np.flip(frame, 2)  # because the camera data comes in as BGR and we need RGB
        data = data.ravel()  # flatten camera data to a 1 d stricture
        data = np.asfarray(data, dtype='f')  # change data type to 32bit floats
        texture_data = np.true_divide(data, 255.0)  # normalize image data to prepare for GPU

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                frame.shape[1], frame.shape[0], texture_data, tag=unique_tag, format=dpg.mvFormat_Float_rgb)

        # dpg.add_image(name, pos=[pos[0], pos[1]], tag=f'{name}_')
        # with dpg.item_handler_registry(tag=f'{name}_widget handler') as handler:
        #     dpg.add_item_clicked_handler(callback=change_text)
        # dpg.bind_item_handler_registry(f'{name}_', f'{name}_widget handler')
        dpg.draw_image(unique_tag, [pos[0], pos[1]], [pos[0] + frame.shape[1], pos[1] + frame.shape[0]], parent=parent)

    def drawDatasetImages(self):
        for i in range(0, self.dataset.rowsCount()):
            for j in range(0, self.dataset.columnsCount()):
                name = f'{i}:{j}'
                print(name)
                imagePart = self.dataset.at(i, j)
                image = imageUtils.getScaledImage(imagePart.fullImage, self.imageWidth)
                imagePosition = [i * (self.imageWidth + self.hSpace), j * (self.imageHeight + self.vSpace)]
                self.addImage(name, image, pos=imagePosition, parent=self.drawlist)

    def drawHLinks(self):
        for i in range(0, self.dataset.rowsCount() - 1):
            for j in range(0, self.dataset.columnsCount()):
                if self.mainStitcher.getStitchResult([i, j], [i + 1, j]):
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)
                center = [i * (self.imageWidth + self.hSpace) + self.imageWidth + self.hSpace / 2, j * (self.imageHeight + self.vSpace) + self.imageHeight / 2]
                pos = self.rectangeCenterTo2Points(center)
                dpg.draw_rectangle(pos[0], pos[1], fill=color, parent=self.drawlist)

    def drawVLinks(self):
        for i in range(0, self.dataset.rowsCount()):
            for j in range(0, self.dataset.columnsCount() - 1):
                if self.mainStitcher.getStitchResult([i, j], [i, j + 1]):
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)
                center = [i * (self.imageWidth + self.hSpace) + self.imageWidth / 2, j * (self.imageHeight + self.vSpace) + self.imageHeight + self.vSpace / 2]
                pos = self.rectangeCenterTo2Points(center)
                dpg.draw_rectangle(pos[0], pos[1], fill=color, parent=self.drawlist)