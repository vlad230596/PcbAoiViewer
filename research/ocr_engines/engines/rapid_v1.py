"""Baseline: rapidocr_onnxruntime 1.2.3 (PP-OCRv3 det/rec ONNX), exactly as ocrMarkings.py calls it (use_cls=False)."""
import numpy as np
import common as C


class Engine:
    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self.e = RapidOCR()

    def read(self, img):
        res, _ = self.e(C.toBgr(img), use_cls=False)
        boxes = []
        for box, text, score in res or []:
            p = np.array(box, float)
            boxes.append((p[:, 1].mean(), np.ptp(p[:, 1]), p[:, 0].min(), text, float(score)))
        return C.groupBoxes(boxes)


def make(variant):
    return Engine()
