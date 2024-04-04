import cv2
import glob
import time

from pathlib import Path
from dataclasses import dataclass

import dearpygui.dearpygui as dpg

from stitching.stitching_error import StitchingError

from EniPy import imageUtils
from EniPy import colors

from Dataset import Dataset
from ViewDrawer import ViewDrawer
from MainStitcher import MainStitcher


dpg.create_context()
dpg.create_viewport(title='Custom Title', width=1920 + 100, height=1080 + 100)
dpg.setup_dearpygui()
mainStitcher = None
viewDrawer = None
def loadNewDataset(path):
    global mainStitcher
    global viewDrawer
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
    mainStitcher = MainStitcher(dataset)

    dpg.delete_item(w, children_only=True)
    viewDrawer = ViewDrawer(w, mainStitcher)

    updateView()

def updateView():
    global viewDrawer
    if viewDrawer is not None:
        viewDrawer.updateImageWidth(dpg.get_value('imageWidthField'))
        viewDrawer.vSpace = dpg.get_value('vSpaceField')
        viewDrawer.hSpace = dpg.get_value('hSpaceField')
        viewDrawer.updateView()

        start_time = time.time()
        viewDrawer.drawDatasetImages()
        print(f"drawDatasetImages: {time.time() - start_time}")

        start_time = time.time()
        viewDrawer.drawHLinks()
        print(f"drawHLinks: {time.time() - start_time}")

        start_time = time.time()
        viewDrawer.drawVLinks()
        print(f"drawVLinks: {time.time() - start_time}")

def change_text(sender, app_data):
    print('click')

def callback(sender, app_data):
    print('OK was clicked.')
    print("Sender: ", sender)
    print("App Data: ", app_data)
    dpg.configure_item('inputPath', default_value=app_data['current_path'])


def cancel_callback(sender, app_data):
    print('Cancel was clicked.')
    print("Sender: ", sender)
    print("App Data: ", app_data)




dpg.add_file_dialog(
    directory_selector=True, show=False, callback=callback, tag="file_dialog_id",
    cancel_callback=cancel_callback, width=700 ,height=400)


with dpg.window(label="Controls", width=200, height=1080):
    with dpg.group(horizontal=True):
        dpg.add_input_text(hint='Enter path here', tag='inputPath')
        dpg.add_button(label="...", callback=lambda: dpg.show_item("file_dialog_id"))
    dpg.add_button(label='Reload', callback=lambda:  loadNewDataset(dpg.get_value('inputPath')))
    dpg.add_input_text(label='Stitch parts', width=50, enabled=False, tag='stitch_parts_text')
    dpg.add_input_int(label='ImageWidth', default_value=360, step=90, width=100, tag='imageWidthField')
    dpg.add_input_int(label='vSpace', default_value=50, width=100, tag='vSpaceField')
    dpg.add_input_int(label='hSpace', default_value=50, width=100, tag='hSpaceField')
    dpg.add_button(label='Repaint', callback=lambda: updateView())


w = dpg.add_window(pos=[200, 0], label="BoardView", horizontal_scrollbar=True, width=1920, height=1080, tag='BoardViewWindow')
print(f'window BoardView {w} {dpg.last_root()}')
#dpg.show_metrics()
#dpg.show_style_editor()

dpg.show_viewport()
dpg.show_imgui_demo()


while dpg.is_dearpygui_running():
    if mainStitcher is not None:
        dpg.set_value('stitch_parts_text', f'{mainStitcher.successStitchCount}/{mainStitcher.successStitchCount + mainStitcher.failStitchCount}')
    dpg.render_dearpygui_frame()

dpg.destroy_context()

cv2.waitKey()
cv2.destroyAllWindows()