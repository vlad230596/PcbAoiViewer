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
Length = 100
StepY = 10

StartZ = 80000
Height = 00000
StepZ = 500

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







videoCapture = cv2.VideoCapture(0)
videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
videoStream = VideoStream(videoCapture)





class CncController:
    def __init__(self, serial):
        self.serial = serial

    def command(self, cmd):
        print(f'Command = {cmd}')

        self.serial.write(str.encode(f'{cmd}\r\n'))

        while True:
            line = self.serial.readline()
            print(line)

            if line == b'ok\n':
                break
    def home(self):
        self.command(f'G28 X0 Y0')
        self.command('M114')

    def move(self, x=None, y=None, z=None):
        coordinate = f''
        if x is not None:
            coordinate = coordinate + f'X{x} '
        if y is not None:
            coordinate = coordinate + f'Y{y} '
        if z is not None:
            coordinate = coordinate + f'Z{z} '

        self.command(f'G1 {coordinate} F1000')
        self.command('M114')


class App:
    def __init__(self):
        self.cncController = None

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
            dpg.add_button(label='Connect', callback=lambda: self.connectComPort(dpg.get_value('ComPortList')))
            dpg.add_button(label='Home', callback=lambda: self.cncController.home())
            dpg.add_button(label='Start', callback=lambda: self.start())
            dpg.add_button(label='End', callback=lambda: self.end())
            dpg.add_button(label='Run', callback=lambda: self.capture(dpg.get_value('outputPath')))

        dpg.show_viewport()

    def start(self):
        self.cncController.move(StartX, StartY)
        pass
    def end(self):
        self.cncController.move(StartX + Width, StartY + Length)
        pass
    def connectComPort(self, name):
        print(f'Connecting to {name}')
        self.cncController = CncController(serial.Serial(name, 115200))
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

    def capture(self, outputFolderPrefix):
        for z in range(StartZ, StartZ + Height + StepZ, StepZ):
            z_mm = z / 1000
            outputFolder = f'{outputFolderPrefix}_{z_mm}'
            print(f'outputFolder = {outputFolder}')
            Path(outputFolder).mkdir(parents=True, exist_ok=True)

            self.cncController.move(z=z_mm)

            for y in range(StartY, StartY + Length + StepY, StepY):
                for x in range(StartX, StartX + Width + StepX, StepX):
                    self.cncController.move(x, y)
                    time.sleep(1)
                    capturedFrame = videoStream.waitNewFrame()
                    cv2.imwrite(f'{outputFolder}/{x}_{y}.png', capturedFrame)
                    cv2.imshow('capturedFrame', capturedFrame)
                    cv2.waitKey(1)


app = App()
app.run()
