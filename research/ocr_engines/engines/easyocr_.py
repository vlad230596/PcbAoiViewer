"""EasyOCR 1.7 (CRAFT detector + CRNN 'english_g2' recogniser, PyTorch CPU); models downloaded into cache/easyocr.

variant: [wl][-mag2][-nodet]
  wl     allowlist A-Z 0-9 . -
  mag2   mag_ratio 2 (detector on a 2x image), low_text 0.3
  nodet  recognise the whole image as one box (body source)
"""
import os

import numpy as np

import common as C

WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-"


class Engine:
    def __init__(self, variant):
        import easyocr
        p = (variant or "").split("-")
        self.wl, self.mag, self.nodet = "wl" in p, "mag2" in p, "nodet" in p
        d = os.path.join(C.HERE, "cache", "easyocr"); os.makedirs(d, exist_ok=True)
        self.r = easyocr.Reader(["en"], gpu=False, model_storage_directory=d, verbose=False)

    def read(self, img):
        kw = {"allowlist": WL} if self.wl else {}
        if self.nodet:
            h, w = img.shape[:2]
            res = self.r.recognize(img, horizontal_list=[[0, w, 0, h]], free_list=[], **kw)
        elif self.mag:
            res = self.r.readtext(img, mag_ratio=2, low_text=0.3, **kw)
        else:
            res = self.r.readtext(img, **kw)
        boxes = []
        for box, text, score in res:
            p = np.array(box, float)
            boxes.append((p[:, 1].mean(), np.ptp(p[:, 1]), p[:, 0].min(), text, float(score)))
        return C.groupBoxes(boxes)


def make(variant):
    return Engine(variant)
