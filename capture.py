import serial
import serial.tools.list_ports
from threading import Thread, Lock
import time
import cv2
from datetime import datetime
from pathlib import Path

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

                cv2.imshow('frame', self.frame)
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


for n, (portname, desc, hwid) in enumerate(sorted(serial.tools.list_ports.comports())):
    print(f'{portname} - {desc}')

videoCapture = cv2.VideoCapture(1)
videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
time.sleep(2)
videoStream = VideoStream(videoCapture)

# while True:
#     time.sleep(2)




ser = serial.Serial('COM5', 115200)
time.sleep(2)
command(ser, f'G28 X0 Y0')
command(ser, 'M114')

for z in range(StartZ, StartZ + Height + StepZ, StepZ):
    z_mm = z / 1000
    outputFolder = f'{outputFolderPrefix}_{z_mm}'
    print(f'outputFolder = {outputFolder}')
    Path(outputFolder).mkdir(parents=True, exist_ok=True)

    command(ser, f'G1 Z{z_mm} F1000')
    command(ser, 'M114')
    time.sleep(5)
    for y in range(StartY, StartY + Length + StepY, StepY):
        for x in range(StartX, StartX + Width + StepX, StepX):
            command(ser, f'G1 X{x} Y{y} F1000')
            command(ser, 'M114')
            time.sleep(1)
            capturedFrame = videoStream.waitNewFrame()
            cv2.imwrite(f'{outputFolder}/{x}_{y}.png', capturedFrame)
            cv2.imshow('capturedFrame', capturedFrame)
            cv2.waitKey(1)

ser.close()
