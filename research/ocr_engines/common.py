"""Shared benchmark data: the parts with a reference reading, their images, groups and scoring.

Imports fold / agree / editDistance / enhance straight from the project's ocrMarkings.py (read-only use), so every
engine is scored exactly like the current RapidOCR second reading. Not its turn(): there `not ROT[90]` is true
(cv2.ROTATE_90_CLOCKWISE == 0), so a 90 degree turn is a no-op; marking_img/ of turn-90 parts is unturned for the same reason. Needs only cv2 + numpy, so every engine venv runs it.

Sources (every image is returned upright, i.e. already turned by the reference reading's `turn`):
  raw   parts/<ref>.jpg in colour (outline + pads + 0.5 mm margin, ~77.5 px/mm)
  mimg  marking_img/<ref>.jpg (grey + CLAHE + enlarged to ~420 px; the image the reference reading used)
  body  only the part outline from the CAD (no pads, no board), grey + CLAHE, enlarged so the long side is ~480 px
  frame the same crop as raw but cut from the sharpest single source frame (frames.py; no multiband blending)
  fmed  per-pixel median of the part over all source frames that see its body (frames.py)
The benchmark turns each upright image by 0/90/180/270: rot 0 = "turn given", best score over the 4 = "turn unknown".
"""
import json
import math
import os
import re
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PANO = os.path.join(ROOT, "out", "captured_51.5")
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, ROOT)
from ocrMarkings import FOLD, agree, editDistance, enhance, fold  # noqa: E402,F401

ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def turn(img, deg):
    """Clockwise turn by a multiple of 90 degrees."""
    deg %= 360
    return img if deg == 0 else cv2.rotate(img, ROT[deg])

GROUPS = ("resistor", "ic", "diode/transistor", "cap/ind/xtal")


def group(ref):
    p = re.match(r"[A-Z]+", ref).group(0)
    return {"R": "resistor", "U": "ic", "D": "diode/transistor", "Q": "diode/transistor"}.get(p, "cap/ind/xtal")


def load():
    mk = json.load(open(os.path.join(PANO, "markings.json"), encoding="utf-8"))["parts"]
    board = json.load(open(os.path.join(PANO, "board.json"), encoding="utf-8"))
    return mk, board


def refsWithLines(mk):
    return [r for r, m in mk.items() if m.get("lines")]


def refsBlank(mk, n=40):
    """Present parts the reference reading found unmarked (for false readings / hallucinations), every k-th one."""
    blank = [r for r, m in mk.items() if not m.get("lines") and m.get("present", True)]
    k = max(1, len(blank) // n)
    return blank[::k][:n]


def flattenOutline(d):
    nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", d)]
    return np.array(nums, float).reshape(-1, 2)


def bodyBox(ref, board):
    """The CAD outline in crop px (same geometry as alignBoard.cropParts): x0, y0, x1, y1."""
    c = board["components"][ref]
    pts = [flattenOutline(c["outline"])] if c.get("outline") else []
    for p in c["pins"]:
        x0, y0, x1, y1 = p["bb"]
        pts.append(np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
    C = np.array([c["x"], c["y"]])
    a = math.radians(c["rot"])
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    allLocal = (np.vstack(pts) - C) @ R
    u0, v0 = allLocal.min(0) - 0.5
    _, v1 = allLocal.max(0) + 0.5
    k = board["align"]["crops"][ref]["pxPerMm"]
    body = (flattenOutline(c["outline"]) - C) @ R if c.get("outline") else allLocal
    bu0, bv0 = body.min(0)
    bu1, bv1 = body.max(0)
    return (bu0 - u0) * k, (v1 - bv1) * k, (bu1 - u0) * k, (v1 - bv0) * k


def image(ref, src, mk, board):
    """Upright image of a part for one source (BGR for raw, grey for mimg/body)."""
    t = mk[ref].get("turn", 0)
    crop = cv2.imread(os.path.join(PANO, board["align"]["crops"][ref]["file"]))
    if src == "mimg":
        m = cv2.imread(os.path.join(PANO, "marking_img", re.sub(r"[^\w.+-]", "_", ref) + ".jpg"), cv2.IMREAD_GRAYSCALE)
        # ocrMarkings.turn bug: a quarter-turned part whose marking_img keeps the crop's aspect was never turned
        if t % 180 == 90 and (m.shape[0] > m.shape[1]) == (crop.shape[0] > crop.shape[1]) and m.shape[0] != m.shape[1]:
            m = turn(m, t)
        return m
    if src == "raw":
        return turn(crop, t)
    if src in ("frame", "fmed"):                             # frames.py output; parts no frame sees fall back to raw
        d = os.path.join(HERE, "fusion", ref)
        files = sorted(f for f in os.listdir(d) if f != "median.jpg") if os.path.isdir(d) else []
        if not files:
            return turn(crop, t)
        if src == "fmed":
            return cv2.imread(os.path.join(d, "median.jpg"))
        x0, y0, x1, y1 = [int(v) for v in bodyBox(ref, board)]

        def sharp(f):                                       # Laplacian variance on the body (glare/blur -> lower)
            g = cv2.cvtColor(cv2.imread(os.path.join(d, f)), cv2.COLOR_BGR2GRAY)
            g = turn(g, -t)[max(0, y0):y1, max(0, x0):x1]
            return cv2.Laplacian(g, cv2.CV_64F).var()
        return cv2.imread(os.path.join(d, max(files, key=sharp)))
    if src == "body":
        x0, y0, x1, y1 = bodyBox(ref, board)
        m = 0.08 * board["align"]["crops"][ref]["pxPerMm"]    # 0.08 mm: keep the edge of the body
        h, w = crop.shape[:2]
        x0, y0 = max(0, int(x0 - m)), max(0, int(y0 - m))
        x1, y1 = min(w, int(math.ceil(x1 + m))), min(h, int(math.ceil(y1 + m)))
        return turn(enhance(crop[y0:y1, x0:x1], target=480), t)
    raise ValueError(src)


def toBgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def groupBoxes(boxes):
    """[(cy, h, x0, text, score)] -> ([line text], mean score); boxes grouped into lines by vertical centre."""
    boxes = [b for b in boxes if b[3].strip()]
    if not boxes:
        return [], 0.0
    boxes.sort()
    lines, cur = [], [boxes[0]]
    for b in boxes[1:]:
        if abs(b[0] - cur[-1][0]) < 0.5 * max(b[1], cur[-1][1]):
            cur.append(b)
        else:
            lines.append(cur); cur = [b]
    lines.append(cur)
    text = [" ".join(x[3].strip() for x in sorted(l, key=lambda x: x[2])) for l in lines]
    return text, float(np.mean([b[4] for b in boxes]))


def cer(refLines, lines):
    a, b = fold(refLines), fold(lines)
    return editDistance(a, b) / max(1, len(a))


def prep(img, name):
    """Preprocessing variants applied before an engine (bench.py --prep):
    clahe  grey + CLAHE + enlarge (ocrMarkings.enhance; raw/frame only)
    up2    Lanczos x2 + unsharp mask (cheap 'super-resolution')
    close  grey, x2, 5x5 morphological closing of the bright marking (merges dot-matrix dots into strokes)
    otsu   grey, x2, Otsu binarisation on CLAHE, white text on black kept as is"""
    if not name:
        return img
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if name == "clahe":
        return enhance(toBgr(g))
    up = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    if name == "up2":
        blur = cv2.GaussianBlur(up, (0, 0), 2)
        return cv2.addWeighted(up, 1.6, blur, -0.6, 0)
    if name == "close":
        return cv2.morphologyEx(up, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    if name == "otsu":
        c = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(up)
        return cv2.threshold(c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    raise ValueError(name)
