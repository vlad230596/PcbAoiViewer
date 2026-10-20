"""Tesseract 5.4 LSTM (portable: the UB-Mannheim build extracted with 7-Zip into tools/tesseract, nothing installed).

variant: "<model>-psm<N>[-wl][-inv][-bin]" e.g. best-psm6-wl
  model  fast (the installer's eng) | best (tessdata_best eng)
  psm    6 block, 7 one line, 11 sparse text
  wl     whitelist A-Z 0-9 . - (no lowercase, no dictionary)
  inv    invert the grey image (markings are mostly light on a dark body; Tesseract is trained dark on light)
  bin    Otsu binarisation after a 3x3 closing (dots of dot-matrix codes merged into strokes)
Score: mean word confidence / 100.
"""
import os
import subprocess

import cv2
import numpy as np

import common as C

TOOLS = os.path.join(C.HERE, "tools", "tesseract")


class Engine:
    def __init__(self, variant):
        parts = (variant or "best-psm6").split("-")
        self.model, self.psm = parts[0], parts[1][3:]
        self.wl, self.inv, self.bin = "wl" in parts, "inv" in parts, "bin" in parts
        self.tessdata = os.path.join(TOOLS, "tessdata_best" if self.model == "best" else "tessdata")

    def prep(self, img):
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.bin:
            g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            _, g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if self.inv:
            g = 255 - g
        return cv2.copyMakeBorder(g, 20, 20, 20, 20, cv2.BORDER_REPLICATE)

    def read(self, img):
        ok, png = cv2.imencode(".png", self.prep(img))
        cmd = [os.path.join(TOOLS, "tesseract.exe"), "stdin", "stdout", "--tessdata-dir", self.tessdata, "-l", "eng",
               "--psm", self.psm, "--oem", "1"]
        if self.wl:
            cmd += ["-c", "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-", "-c", "load_system_dawg=0",
                    "-c", "load_freq_dawg=0"]
        cmd += ["-c", "tessedit_create_tsv=1"]
        out = subprocess.run(cmd, input=png.tobytes(), capture_output=True, creationflags=0x08000000).stdout
        words = {}
        for row in out.decode("utf-8", "replace").splitlines()[1:]:
            f = row.split("\t")
            if len(f) < 12 or f[0] != "5" or not f[11].strip() or float(f[10]) < 0:
                continue
            key = (int(f[2]), int(f[3]), int(f[4]))        # block, paragraph, line
            words.setdefault(key, []).append((int(f[6]), f[11].strip(), float(f[10])))
        lines, confs = [], []
        for key in sorted(words):
            ws = sorted(words[key])
            lines.append(" ".join(w[1] for w in ws))
            confs += [w[2] for w in ws]
        return lines, (float(np.mean(confs)) / 100 if confs else 0.0)


def make(variant):
    return Engine(variant)
