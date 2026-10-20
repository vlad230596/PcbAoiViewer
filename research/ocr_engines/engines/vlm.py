"""Transformer OCR models and small VLMs on the GPU (venvs/gpu: PyTorch 2.14 cu130, RTX 5080), weights in cache/hf.
BENCH_DEVICE=cpu runs them on the CPU in float32 (venvs/hf, CPU PyTorch) for timing.

variant:
  qwen3vl-2b | qwen3vl-4b | qwen25vl-3b   Qwen-VL chat models with a fixed reading prompt
  florence2-large                         Florence-2 <OCR> task (florence-community port, native transformers)
  got                                     GOT-OCR 2.0 (stepfun-ai/GOT-OCR-2.0-hf), plain OCR
  paddleocrvl                             PaddleOCR-VL 0.9B, "OCR:" prompt
  trocr-base | trocr-large                TrOCR printed (recogniser only: one text line, meant for the body source)
Score: mean probability of the generated tokens (greedy), so "turn unknown" can pick the most confident turn.
Small images are enlarged so the short side is >= 448 px (VLM patches are 14-16 px; a 0603 code is ~60 px high).
"""
import os

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "hf"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import common as C  # noqa: E402

PROMPT = ("Read the marking printed or laser-etched on top of this electronic component (a part on a circuit board). "
          "Output only the marking text, one line per printed line, top to bottom. If there is no marking, output NONE.")
IDS = {"qwen3vl-2b": "Qwen/Qwen3-VL-2B-Instruct", "qwen3vl-4b": "Qwen/Qwen3-VL-4B-Instruct",
       "qwen25vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct", "florence2-large": "florence-community/Florence-2-large",
       "got": "stepfun-ai/GOT-OCR-2.0-hf", "paddleocrvl": "PaddlePaddle/PaddleOCR-VL",
       "trocr-base": "microsoft/trocr-base-printed", "trocr-large": "microsoft/trocr-large-printed"}


def pil(img, minSide=448):
    from PIL import Image
    rgb = C.toBgr(img)[:, :, ::-1]
    s = minSide / min(rgb.shape[:2])
    if s > 1:
        rgb = cv2.resize(np.ascontiguousarray(rgb), None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(np.ascontiguousarray(rgb))


def meanProb(out, start):
    """Mean probability of the greedy tokens from generate(output_scores=True)."""
    if not out.scores:
        return 0.0
    ps = []
    for i, sc in enumerate(out.scores):
        tok = out.sequences[0, start + i]
        ps.append(torch.softmax(sc[0].float(), -1)[tok].item())
    return float(np.mean(ps))


def toLines(text):
    lines = [l.strip() for l in text.replace("<br>", "\n").splitlines()]
    lines = [l for l in lines if l and l.upper() not in ("NONE", "NONE.", "NO MARKING")]
    return lines


class Engine:
    def __init__(self, variant):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.v = variant
        mid = IDS[variant]
        self.dev = os.environ.get("BENCH_DEVICE", "cuda")       # BENCH_DEVICE=cpu: float32 on the CPU (venvs/hf)
        dt = self.dt = torch.bfloat16 if self.dev == "cuda" else torch.float32
        if variant.startswith("trocr"):
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            self.proc = TrOCRProcessor.from_pretrained(mid)
            self.model = VisionEncoderDecoderModel.from_pretrained(mid, torch_dtype=dt, use_safetensors=True).to(self.dev)
        elif variant == "florence2-large":
            from transformers import Florence2ForConditionalGeneration
            self.proc = AutoProcessor.from_pretrained(mid)
            self.model = Florence2ForConditionalGeneration.from_pretrained(mid, torch_dtype=dt).to(self.dev)
        else:
            self.proc = AutoProcessor.from_pretrained(mid, trust_remote_code=False)
            self.model = AutoModelForImageTextToText.from_pretrained(mid, torch_dtype=dt,
                                                                     trust_remote_code=False).to(self.dev)
        self.model.eval()

    @torch.inference_mode()
    def gen(self, inputs, start, maxNew=48, **kw):
        out = self.model.generate(**inputs, max_new_tokens=maxNew, do_sample=False, num_beams=1,
                                  output_scores=True, return_dict_in_generate=True, **kw)
        return out, meanProb(out, start)

    def read(self, img):
        v = self.v
        if v.startswith("trocr"):
            px = self.proc(images=pil(img, 64), return_tensors="pt").pixel_values.to(self.dev, self.dt)
            out = self.model.generate(px, max_new_tokens=24, output_scores=True, return_dict_in_generate=True)
            text = self.proc.batch_decode(out.sequences, skip_special_tokens=True)[0]
            return toLines(text), meanProb(out, 1)
        if v == "florence2-large":
            task = "<OCR>"
            inp = self.proc(text=task, images=pil(img, 384), return_tensors="pt").to(self.dev, self.dt)
            out = self.model.generate(**inp, max_new_tokens=48, do_sample=False, num_beams=1, output_scores=True,
                                      return_dict_in_generate=True)
            text = self.proc.batch_decode(out.sequences, skip_special_tokens=False)[0]
            parsed = self.proc.post_process_generation(text, task=task, image_size=(img.shape[1], img.shape[0]))
            return toLines(parsed[task] if isinstance(parsed, dict) else str(parsed)), meanProb(out, 1)
        if v == "got":
            inp = self.proc(pil(img), return_tensors="pt").to(self.dev, self.dt)
            start = inp["input_ids"].shape[1]
            out, p = self.gen(inp, start, stop_strings="<|im_end|>", tokenizer=self.proc.tokenizer)
            text = self.proc.decode(out.sequences[0, start:], skip_special_tokens=True)
            return toLines(text), p
        prompt = "OCR:" if v == "paddleocrvl" else PROMPT
        msgs = [{"role": "user", "content": [{"type": "image", "image": pil(img)}, {"type": "text", "text": prompt}]}]
        inp = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                            return_tensors="pt").to(self.dev)
        if "pixel_values" in inp:
            inp["pixel_values"] = inp["pixel_values"].to(self.dt)
        start = inp["input_ids"].shape[1]
        out, p = self.gen(inp, start)
        text = self.proc.decode(out.sequences[0, start:], skip_special_tokens=True)
        return toLines(text), p


def make(variant):
    return Engine(variant or "qwen3vl-2b")
