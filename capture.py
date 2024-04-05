import serial
import serial.tools.list_ports
from threading import Thread, Lock
import time
import cv2
from datetime import datetime
from pathlib import Path

from EniPy import colors

import dearpygui.dearpygui as dpg

StartX = 40
Width = 60
StepX = 10

StartY = 20
Length = 110
StepY = 10

StartZ = 48000
Height = 10000
StepZ = 500

outputFolderPrefix = './captured'

class VideoStream(object):
    def __init__(self, capture):
        self.lock = Lock()
        self.requestedFrame = None
        self.isFrameRequested = False
        self.capture = capture
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
                with self.lock:
                    if self.isFrameRequested and self.requestedFrame is None:
                        self.requestedFrame = self.frame.copy()
                view = self.frame.copy()
                height, width, _ = view.shape
                cv2.line(view, [int(width / 2), 0], [int(width / 2), height], thickness=1, color=colors.Red)
                cv2.line(view, [0, int(height / 2)], [width, int(height / 2)], thickness=1, color=colors.Red)
                cv2.imshow('view', view)
                key = cv2.waitKey(1)
            else:
                time.sleep(.01)

    def waitNewFrame(self):
        with self.lock:
            self.requestedFrame = None
            self.isFrameRequested = True
        while True:
            with self.lock:
                if self.requestedFrame is not None:
                    return self.requestedFrame
            time.sleep(0.005)



def command(ser, command):
    print(f'Command = {command}')

    ser.write(str.encode(f'{command}\r\n'))

    while True:
        line = ser.readline()
        print(line)

        if line == b'ok\n':
            break

        if b'pwm_nrfx_set_cycles' in line:#Todo remove, debug only
            break




# videoCapture = cv2.VideoCapture(0)
# # videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# # videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# time.sleep(2)
# videoStream = VideoStream(videoCapture)
#
# while True:
#     time.sleep(2)




# ser = serial.Serial('COM5', 115200)
# time.sleep(2)
# command(ser, f'G28 X0 Y0')
# command(ser, 'M114')
#
# for z in range(StartZ, StartZ + Height + StepZ, StepZ):
#     z_mm = z / 1000
#     outputFolder = f'{outputFolderPrefix}_{z_mm}'
#     print(f'outputFolder = {outputFolder}')
#     Path(outputFolder).mkdir(parents=True, exist_ok=True)
#
#     command(ser, f'G1 Z{z_mm} F1000')
#     command(ser, 'M114')
#     time.sleep(5)
#     for y in range(StartY, StartY + Length + StepY, StepY):
#         for x in range(StartX, StartX + Width + StepX, StepX):
#             command(ser, f'G1 X{x} Y{y} F1000')
#             command(ser, 'M114')
#             time.sleep(1)
#             capturedFrame = videoStream.waitNewFrame()
#             cv2.imwrite(f'{outputFolder}/{x}_{y}.png', capturedFrame)
#             cv2.imshow('capturedFrame', capturedFrame)
#             cv2.waitKey(1)
#
# ser.close()

class App:
    def __init__(self):

        dpg.create_context()
        dpg.create_viewport(title='PcbInspection', width=1920, height=1080)
        dpg.setup_dearpygui()

        dpg.add_file_dialog(directory_selector=True, show=False, callback=lambda sender, app_data: dpg.configure_item('outputPath', default_value=app_data['current_path']), tag="file_dialog_id", width=700 ,height=400)

        with dpg.window(label="Controls", width=200, height=1080):
            with dpg.group(horizontal=True):
                dpg.add_input_text(hint='Enter output path', tag='outputPath')
                dpg.add_button(label="...", callback=lambda: dpg.show_item("file_dialog_id"))
            dpg.add_button(label='Update com port lists', callback=lambda:  self.fillComportLists())
            dpg.add_combo([], label='Com port', tag='ComPortList')
            self.fillComportLists()
            # dpg.add_input_text(label='Stitch parts', width=70, enabled=False, tag='stitch_parts_text')
            # dpg.add_input_int(label='ImageWidth', default_value=360, step=90, width=100, tag='imageWidthField')
            # dpg.add_input_int(label='vSpace', default_value=50, width=100, tag='vSpaceField')
            # dpg.add_input_int(label='hSpace', default_value=50, width=100, tag='hSpaceField')
            # dpg.add_button(label='Repaint', callback=lambda: updateView())

        dpg.show_viewport()

    def fillComportLists(self):
        comPorts = []
        for n, (portname, desc, hwid) in enumerate(sorted(serial.tools.list_ports.comports())):
            comPorts.append(portname)
            print(f'{portname} - {desc}')
        dpg.configure_item('ComPortList', items=comPorts)
        if len(comPorts):
            dpg.set_value('ComPortList', comPorts[0])
        else:
            dpg.set_value('ComPortList', '')
    def run(self):
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
        dpg.destroy_context()


app = App()
app.run()
