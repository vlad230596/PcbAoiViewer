import cv2
import glob

from pathlib import Path
from dataclasses import dataclass

import dearpygui.dearpygui as dpg

from stitching.stitching_error import StitchingError

from EniPy import imageUtils
from EniPy import colors

from Dataset import Dataset
from ViewDrawer import ViewDrawer


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


w = dpg.add_window(pos=[200, 0], label="BoardView", horizontal_scrollbar=True, width=1920, height=1080, tag='BoardViewWindow')
print(f'window BoardView {w} {dpg.last_root()}')
#dpg.show_metrics()
#dpg.show_style_editor()
dpg.show_viewport()
dpg.show_imgui_demo()



while dpg.is_dearpygui_running():
    dpg.render_dearpygui_frame()

dpg.destroy_context()

cv2.waitKey()
cv2.destroyAllWindows()