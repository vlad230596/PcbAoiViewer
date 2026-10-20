"""PaddleOCR 3.7 full pipeline on paddlepaddle 3.3 CPU (PaddleX models, cached in cache/paddlex).

variant: "" = the pipeline's default det/rec models; "v5server" = PP-OCRv5_server_det + PP-OCRv5_server_rec;
"+tlo" adds the text-line orientation classifier. Document orientation / unwarping are off (not documents).
"""
import os

os.environ.setdefault("PADDLE_PDX_CACHE_HOME", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                            "cache", "paddlex"))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import numpy as np  # noqa: E402

import common as C  # noqa: E402


class Engine:
    def __init__(self, variant):
        from paddleocr import PaddleOCR
        kw = dict(use_doc_orientation_classify=False, use_doc_unwarping=False,
                  use_textline_orientation="tlo" in variant, enable_mkldnn=False, cpu_threads=int(os.environ.get("BENCH_THREADS", 4)))
        if "v5server" in variant:
            kw.update(text_detection_model_name="PP-OCRv5_server_det", text_recognition_model_name="PP-OCRv5_server_rec")
        self.o = PaddleOCR(**kw)

    def read(self, img):
        res = self.o.predict(C.toBgr(img))
        boxes = []
        for r in res:
            for poly, text, score in zip(r["rec_polys"], r["rec_texts"], r["rec_scores"]):
                p = np.array(poly, float)
                boxes.append((p[:, 1].mean(), np.ptp(p[:, 1]), p[:, 0].min(), text, float(score)))
        return C.groupBoxes(boxes)


def make(variant):
    return Engine(variant)
