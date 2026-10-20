"""docTR 1.1 (PyTorch CPU): db_resnet50 / fast_base detector + crnn / parseq / vitstr recogniser.

variant: "<det>+<reco>" (default db_resnet50+parseq); "nodet+<reco>" recognises the whole image as one word.
"""
import os

import numpy as np

import common as C


class Engine:
    def __init__(self, variant):
        os.environ.setdefault("DOCTR_CACHE_DIR", os.path.join(C.HERE, "cache", "doctr"))
        from doctr.models import ocr_predictor, recognition_predictor
        det, reco = (variant or "db_resnet50+parseq").split("+")
        self.nodet = det == "nodet"
        if self.nodet:
            self.p = recognition_predictor(reco, pretrained=True)
        else:
            self.p = ocr_predictor(det, reco, pretrained=True, assume_straight_pages=True, preserve_aspect_ratio=True)

    def read(self, img):
        rgb = C.toBgr(img)[:, :, ::-1].copy()
        if self.nodet:
            (text, score), = self.p([rgb])
            return ([text], float(score)) if text.strip() else ([], 0.0)
        doc = self.p([rgb])
        boxes = []
        for blk in doc.pages[0].blocks:
            for line in blk.lines:
                for w in line.words:
                    (x0, y0), (x1, y1) = w.geometry
                    boxes.append(((y0 + y1) / 2, y1 - y0, x0, w.value, float(w.confidence)))
        return C.groupBoxes(boxes)


def make(variant):
    return Engine(variant)
