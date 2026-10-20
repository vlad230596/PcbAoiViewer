"""Worker: reads part markings with a small local vision-language model (Qwen3-VL-2B-Instruct by default).

Runs INSIDE a separate venv with torch + transformers (the project venv stays light); vlmMarkings.py starts it as a
subprocess. Self-contained on purpose: imports nothing from the project.

    <vlm venv>/Scripts/python.exe vlmWorker.py jobs.json results.json [--model Qwen/Qwen3-VL-2B-Instruct] [--device cuda]

jobs.json: [{"ref": "R12", "image": "<abs path to parts/R12.jpg>", "turn": 180}, ...]
results.json: {"model": ..., "device": ..., "sec": ..., "parts": {ref: {"lines": [...], "score": 0.93, "raw": "..."}}}

Settings found best in research/ocr_engines/REPORT.md: the colour crop as cut, no enhancement (CLAHE costs 8 of 116),
turned by the known turn so the text stands upright, short side enlarged to >= 448 px, greedy decoding, a fixed
prompt that never mentions the expected value (the check would turn into an echo). score = mean probability of the
generated tokens. "NONE" = no marking.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

PROMPT = ("Read the marking printed or laser-etched on top of this electronic component (a part on a circuit board). "
          "Output only the marking text, one line per printed line, top to bottom. If there is no marking, output NONE.")
ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def pil(bgr, minSide=448):
    from PIL import Image
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    s = minSide / min(rgb.shape[:2])
    if s > 1:
        rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(rgb)


def toLines(text):
    lines = [l.strip() for l in text.replace("<br>", "\n").splitlines()]
    return [l for l in lines if l and l.upper().rstrip(".") not in ("NONE", "NO MARKING")]


class Reader:
    def __init__(self, model, device):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.dev = device
        self.dt = torch.bfloat16 if device == "cuda" else torch.float32
        self.proc = AutoProcessor.from_pretrained(model)
        self.model = AutoModelForImageTextToText.from_pretrained(model, torch_dtype=self.dt).to(device).eval()

    @torch.inference_mode()
    def read(self, bgr):
        msgs = [{"role": "user", "content": [{"type": "image", "image": pil(bgr)}, {"type": "text", "text": PROMPT}]}]
        inp = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                            return_tensors="pt").to(self.dev)
        if "pixel_values" in inp:
            inp["pixel_values"] = inp["pixel_values"].to(self.dt)
        start = inp["input_ids"].shape[1]
        out = self.model.generate(**inp, max_new_tokens=48, do_sample=False, num_beams=1, output_scores=True,
                                  return_dict_in_generate=True)
        ps = [torch.softmax(sc[0].float(), -1)[out.sequences[0, start + i]].item() for i, sc in enumerate(out.scores)]
        text = self.proc.decode(out.sequences[0, start:], skip_special_tokens=True)
        return text, float(np.mean(ps)) if ps else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs")
    ap.add_argument("results")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    jobs = json.load(open(a.jobs, encoding="utf-8"))
    t0 = time.time()
    reader = Reader(a.model, a.device)
    print(f"model {a.model} on {a.device}, loaded in {time.time() - t0:.1f} s", flush=True)
    parts, t1 = {}, time.time()
    for i, j in enumerate(jobs):
        img = cv2.imread(j["image"], cv2.IMREAD_COLOR)
        if img is None:
            parts[j["ref"]] = {"lines": [], "score": 0.0, "raw": "", "error": "cannot read image"}
            continue
        if j.get("turn", 0) % 360:
            img = cv2.rotate(img, ROT[j["turn"] % 360])
        text, score = reader.read(img)
        parts[j["ref"]] = {"lines": toLines(text), "score": round(score, 3), "raw": text}
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(jobs)}", flush=True)
    json.dump({"model": a.model, "device": a.device, "sec": round(time.time() - t1, 1), "parts": parts},
              open(a.results, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(jobs)} parts in {time.time() - t1:.1f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
