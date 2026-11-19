"""Grid capture on a Marlin printer with a USB microscope on the head.

    uv run captureGrid.py --x 48:80:16 --y 118:143:25 --out captured_test --dry-run
    uv run captureGrid.py --x 48:80:16 --y 118:143:25 --out captured_test

Coordinates are head (nozzle) coordinates in mm as Marlin reports them; frames are saved as
`{x}_{y}.png`, which is what GridStitcher reads. The script never homes and never moves Z unless
--z is given: X and Y must already be homed (checked with M119), Z has no endstop.
The serial port is opened with DTR off so the board does not reset (a reset loses the homing).
"""
import argparse
import json
import os
import re
import sys
import threading
import time

import cv2
import serial


def parseRange(text):
    """'48:80:16' -> [48, 64, 80] (inclusive); '50' -> [50]."""
    parts = [float(p) for p in text.split(':')]
    if len(parts) == 1:
        return parts
    start, stop, step = parts
    values, v = [], start
    while v <= stop + 1e-9:
        values.append(round(v, 3))
        v += step
    return values


def fmt(v):
    return f'{v:g}'


class Printer:
    def __init__(self, port, baud=115200, log=print):
        self.log = log
        self.serial = serial.Serial()
        self.serial.port = port
        self.serial.baudrate = baud
        self.serial.timeout = 0.2
        self.serial.dtr = False   # DTR edge resets the board and loses the homing
        self.serial.rts = False
        self.serial.open()
        time.sleep(0.3)
        self.serial.reset_input_buffer()

    def close(self):
        self.serial.close()

    def command(self, cmd, timeout=60.0):
        self.serial.write(f'{cmd}\n'.encode())
        lines, deadline = [], time.time() + timeout
        while time.time() < deadline:
            line = self.serial.readline().decode(errors='replace').strip()
            if not line:
                continue
            if line == 'ok':
                return lines
            if line.startswith('echo:busy'):
                deadline = time.time() + timeout
                continue
            if line.lower().startswith('error'):
                raise RuntimeError(f'{cmd}: {line}')
            lines.append(line)
        raise TimeoutError(f'{cmd}: no "ok" in {timeout} s')

    def position(self):
        for line in self.command('M114'):
            m = re.match(r'X:(-?[\d.]+) Y:(-?[\d.]+) Z:(-?[\d.]+)', line)
            if m:
                return tuple(float(v) for v in m.groups())
        raise RuntimeError('M114: no position')

    def homedAxes(self):
        flags = {}
        for line in self.command('M119'):
            m = re.match(r'test_axis_known_(\w)_flag = (\w+)', line)
            if m:
                flags[m.group(1)] = m.group(2) == 'true'
        return flags

    def moveTo(self, x=None, y=None, z=None, feed=3000):
        axes = ''.join(f' {a}{fmt(v)}' for a, v in (('X', x), ('Y', y), ('Z', z)) if v is not None)
        self.command('G90')
        self.command(f'G0{axes} F{feed}')
        self.command('M400', timeout=120)   # wait until the move is finished


class Camera:
    """Reads frames in a thread so a frame taken after a move is fresh, not one buffered during it."""

    def __init__(self, index=0, width=1920, height=1080):
        self.capture = cv2.VideoCapture(index, cv2.CAP_MSMF)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.lock = threading.Lock()
        self.frame, self.stamp = None, 0.0
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        if self.frameAfter(time.time(), timeout=8) is None:
            self.close()
            raise RuntimeError('camera gives no frames (is another app, e.g. Windows Camera, holding it?)')

    def loop(self):
        while self.running:
            ok, frame = self.capture.read()
            if ok:
                with self.lock:
                    self.frame, self.stamp = frame, time.time()
            else:
                time.sleep(0.01)

    def frameAfter(self, t, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                if self.frame is not None and self.stamp > t:
                    return self.frame.copy()
            time.sleep(0.01)
        return None

    def close(self):
        self.running = False
        self.thread.join(timeout=2)
        self.capture.release()


def gridOrder(xs, ys):
    """Serpentine: rows along Y, the next row back, so the head never jumps across the board."""
    order = []
    for i, x in enumerate(xs):
        row = ys if i % 2 == 0 else ys[::-1]
        order.extend((x, y) for y in row)
    return order


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--x', required=True, help='head X: start:stop:step mm (inclusive) or one value')
    ap.add_argument('--y', required=True, help='head Y: start:stop:step mm (inclusive) or one value')
    ap.add_argument('--z', type=float, help='move Z here first (no Z endstop: relative to power-on height)')
    ap.add_argument('--out', required=True, help='output folder, frames saved as {x}_{y}.png')
    ap.add_argument('--port', default='COM3')
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--width', type=int, default=1920)
    ap.add_argument('--height', type=int, default=1080)
    ap.add_argument('--feed', type=float, default=3000, help='mm/min for G0')
    ap.add_argument('--settle', type=float, default=0.7, help='s after the move before the frame is taken')
    ap.add_argument('--dry-run', action='store_true', help='print the plan, touch nothing')
    args = ap.parse_args()

    xs, ys = parseRange(args.x), parseRange(args.y)
    order = gridOrder(xs, ys)
    print(f'{len(order)} frames: X {xs}, Y {ys}' + (f', Z {fmt(args.z)}' if args.z is not None else ', Z unchanged'))
    for n, (x, y) in enumerate(order, 1):
        print(f'  {n:3d}  X{fmt(x)} Y{fmt(y)}')
    if args.dry_run:
        return

    os.makedirs(args.out, exist_ok=True)
    printer = Printer(args.port)
    camera = None
    try:
        homed = printer.homedAxes()
        if not (homed.get('x') and homed.get('y')):
            sys.exit(f'X/Y not homed ({homed}); home them first (G28 X, G28 Y)')
        start = printer.position()
        print(f'start position X{fmt(start[0])} Y{fmt(start[1])} Z{fmt(start[2])}')
        camera = Camera(args.camera, args.width, args.height)
        if args.z is not None:
            printer.moveTo(z=args.z, feed=600)

        frames, t0 = [], time.time()
        for n, (x, y) in enumerate(order, 1):
            printer.moveTo(x, y, feed=args.feed)
            time.sleep(args.settle)
            frame = camera.frameAfter(time.time())
            if frame is None:
                raise RuntimeError('camera stopped giving frames')
            pos = printer.position()
            name = f'{fmt(x)}_{fmt(y)}.png'
            cv2.imwrite(os.path.join(args.out, name), frame)
            frames.append({'file': name, 'x': x, 'y': y, 'reported': pos})
            print(f'  {n:3d}/{len(order)}  {name}  mean {frame.mean():.1f}  ({time.time() - t0:.0f} s)')

        with open(os.path.join(args.out, 'capture.json'), 'w', encoding='utf-8') as f:
            json.dump({'port': args.port, 'camera': args.camera, 'size': [args.width, args.height],
                       'z': args.z if args.z is not None else start[2], 'settle': args.settle,
                       'feed': args.feed, 'frames': frames}, f, indent=1)
        print(f'done: {len(frames)} frames -> {args.out}')
    finally:
        if camera:
            camera.close()
        printer.close()


if __name__ == '__main__':
    main()
