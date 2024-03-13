import serial
import serial.tools.list_ports
from threading import Thread, Lock
import time
import cv2
from datetime import datetime

StartX = 50
Width = 60
StepX = 10

StartY = 20
Height = 20
StepY = 5

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
    time.sleep(1)

    while True:
        line = ser.readline()
        print(line)

        if line == b'ok\n':
            break

        if b'pwm_nrfx_set_cycles' in line:#Todo remove, debug only
            break


for n, (portname, desc, hwid) in enumerate(sorted(serial.tools.list_ports.comports())):
    print(f'{portname} - {desc}')

videoCapture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
videoStream = VideoStream(videoCapture)

ser = serial.Serial('COM3', 115200)
time.sleep(2)

for y in range(StartY, StartY + Height + StepY, StepY):
    for x in range(StartX, StartX + Width + StepX, StepX):
        command(ser, f'G28 X{x} Y{y}')
        time.sleep(1)
        capturedFrame = videoStream.waitNewFrame()
        cv2.imwrite(f'captured/{x}_{y}.png', capturedFrame)
        cv2.imshow('capturedFrame', capturedFrame)
        cv2.waitKey(1)

ser.close()
