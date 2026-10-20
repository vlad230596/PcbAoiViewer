"""RapidOCR 3.10 (PaddleOCR det+rec as ONNX on onnxruntime CPU): PP-OCRv4 / v5 / v6 models, downloaded on first use.

variant: "<ver>-<size>[-en][-nodet][-loose]"
  ver/size  v6-tiny|small|medium, v5-mobile|server, v4-mobile|server
  en        English recogniser (v4/v5 en_*_rec_mobile: digits + latin only)
  nodet     recognition only: the whole image is one text line (meant for the body source of 1-line parts)
  loose     detector box_thresh 0.3, thresh 0.2, text_score 0.2 (faint laser marking)
"""
import os

import numpy as np

import common as C


class Engine:
    def __init__(self, variant):
        from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR
        p = (variant or "v6-small").split("-")
        ver = {"v4": OCRVersion.PPOCRV4, "v5": OCRVersion.PPOCRV5, "v6": OCRVersion.PPOCRV6}[p[0]]
        size = ModelType(p[1])
        self.nodet = "nodet" in p
        params = {"Global.log_level": "error", "Global.model_root_dir": os.path.join(C.HERE, "cache", "rapidocr"),
                  "EngineConfig.onnxruntime.intra_op_num_threads": int(os.environ.get("BENCH_THREADS", 4)),
                  "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                  "Global.text_score": 0.2 if "loose" in p else 0.3,
                  "Det.ocr_version": ver, "Det.model_type": size if p[0] != "v4" or size.value in ("mobile", "server") else size,
                  "Rec.ocr_version": ver, "Rec.model_type": size}
        if "en" in p:
            params.update({"Rec.lang_type": LangRec.EN, "Rec.model_type": ModelType.MOBILE})
        if "loose" in p:
            params.update({"Det.box_thresh": 0.3, "Det.thresh": 0.2})
        os.makedirs(params["Global.model_root_dir"], exist_ok=True)
        self.e = RapidOCR(params=params)

    def read(self, img):
        bgr = C.toBgr(img)
        if self.nodet:
            r = self.e(bgr, use_det=False, use_cls=False)
            txt = [t for t in (r.txts or ()) if t.strip()]
            return (txt, float(np.mean(r.scores))) if txt else ([], 0.0)
        r = self.e(bgr, use_cls=False)
        if r.boxes is None:
            return [], 0.0
        boxes = []
        for box, text, score in zip(r.boxes, r.txts, r.scores):
            p = np.array(box, float)
            boxes.append((p[:, 1].mean(), np.ptp(p[:, 1]), p[:, 0].min(), text, float(score)))
        return C.groupBoxes(boxes)


def make(variant):
    return Engine(variant)
