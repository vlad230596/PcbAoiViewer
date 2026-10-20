"""CLI: local OCR of the part markings as a second, independent reading, and the images the readings come from.

    venv/Scripts/python.exe ocrMarkings.py out/captured_51.5 [--refs R1,U3]

For every part in markings.json (markingSheets.py -> reading -> markings.json):
  1. marking_img/<ref>.jpg — the image a marking is read from: the part crop (parts/<ref>.jpg) in grey (L of Lab),
     lightly blurred, CLAHE (local contrast: faint laser marking on a dark mould), enlarged so the part is ~420 px,
     turned by the reading's `turn` so the text stands upright. The viewer shows it in the part card.
  2. RapidOCR (PaddleOCR PP-OCRv3 det + rec, ONNX; models ship inside the rapidocr_onnxruntime wheel, no download)
     on that image in all four turns, without looking at the first reading; the turn with the best mean score wins.
     Boxes are grouped into lines by their vertical centre, left to right.
  3. Comparison with the first reading, after folding look-alike characters (O/0/D, I/L/1, S/5, B/8, Z/2, G/6):
     same | close (one character off) | partial (every OCR line is one of the first reading's lines) | differs |
     ocr-only (OCR read text, the first reading has none) | none. --compare-only redoes this step alone.
Written into markings.json parts[ref]: img, ocr {lines, score, turn, agree}. Prints the disagreements.
"""
import argparse
import json
import os
import re
import time

import cv2
import numpy as np

ROT = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
FOLD = str.maketrans({"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "|": "1", "S": "5", "B": "8", "Z": "2",
                      "G": "6"})


def turn(img, deg):
    code = ROT[deg % 360]
    return img if code is None else cv2.rotate(img, code)   # cv2.ROTATE_90_CLOCKWISE == 0: test None, not falsiness


def enhance(crop, target=420):
    """Grey + CLAHE + enlarge: the reading image of a part crop."""
    L = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[:, :, 0]
    L = cv2.GaussianBlur(L, (0, 0), 0.7)
    L = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(3, 3)).apply(L)
    s = min(3.0, max(1.0, target / max(L.shape)))
    return cv2.resize(L, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if s != 1.0 else L


def ocrLines(engine, grey):
    """RapidOCR on a grey image -> ([line text], mean score)."""
    res, _ = engine(cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR), use_cls=False)
    if not res:
        return [], 0.0
    boxes = []
    for box, text, score in res:
        pts = np.array(box, float)
        boxes.append((pts[:, 1].mean(), pts[:, 1].max() - pts[:, 1].min(), pts[:, 0].min(), text.strip(), float(score)))
    boxes = [b for b in boxes if b[3]]
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
    text = [" ".join(x[3] for x in sorted(l, key=lambda x: x[2])) for l in lines]
    return text, float(np.mean([b[4] for b in boxes]))


def fold(lines):
    return re.sub(r"[^0-9A-Z]", "", " ".join(lines).upper()).translate(FOLD)


def editDistance(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cbb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cbb))
    return d[-1]


def agree(first, ocr):
    a, b = fold(first), fold(ocr)
    if not a and not b:
        return "none"
    if not a:
        return "ocr-only"
    if not b:
        return "differs"
    if a == b:
        return "same"
    if editDistance(a, b) <= max(1, len(a) // 6):
        return "close"
    # OCR got some of the lines (often the big one): each OCR line matches one of the first reading's lines
    firstLines = [fold([x]) for x in first if fold([x])]
    ocrLines_ = [fold([x]) for x in ocr if len(fold([x])) >= 3]
    if ocrLines_ and all(any(o == f or (len(f) >= 4 and editDistance(o, f) <= 1) for f in firstLines) for o in ocrLines_):
        return "partial"
    return "differs"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("panorama", help="folder with board.json, parts/ and markings.json")
    ap.add_argument("--refs", help="comma-separated designators (default: all in markings.json)")
    ap.add_argument("--min-score", type=float, default=0.5, help="OCR results below this mean score are dropped")
    ap.add_argument("--compare-only", action="store_true",
                    help="no OCR: compare the stored OCR lines with the first reading again (after editing markings.json)")
    a = ap.parse_args()

    if a.compare_only:
        path = os.path.join(a.panorama, "markings.json")
        mk = json.load(open(path, encoding="utf-8"))
        counts = {}
        for ref, m in mk["parts"].items():
            if "ocr" in m:
                m["ocr"]["agree"] = agree(m["lines"], m["ocr"]["lines"])
                counts[m["ocr"]["agree"]] = counts.get(m["ocr"]["agree"], 0) + 1
        mk.setdefault("ocr", {})["summary"] = counts
        json.dump(mk, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
        return

    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR()
    b = json.load(open(os.path.join(a.panorama, "board.json"), encoding="utf-8"))
    path = os.path.join(a.panorama, "markings.json")
    mk = json.load(open(path, encoding="utf-8"))
    folder = os.path.join(a.panorama, "marking_img")
    os.makedirs(folder, exist_ok=True)
    refs = a.refs.split(",") if a.refs else list(mk["parts"])
    t0 = time.time()
    counts = {}
    for ref in refs:
        m = mk["parts"][ref]
        crop = cv2.imread(os.path.join(a.panorama, b["align"]["crops"][ref]["file"]))
        g = enhance(crop)
        name = re.sub(r"[^\w.+-]", "_", ref) + ".jpg"
        cv2.imwrite(os.path.join(folder, name), turn(g, m.get("turn", 0)), [cv2.IMWRITE_JPEG_QUALITY, 90])
        m["img"] = f"marking_img/{name}"
        best, tries = ([], 0.0, 0), []
        for deg in (0, 90, 180, 270):   # independent of the first reading's turn
            lines, score = ocrLines(engine, turn(g, deg))
            if lines and score >= a.min_score:
                tries.append((deg, lines, score))
            if lines and score > best[1]:
                best = (lines, score, deg)
        lines, score, deg = best
        if score < a.min_score:
            lines, deg = [], 0
        m["ocr"] = {"lines": lines, "score": round(score, 3), "turn": deg, "agree": agree(m["lines"], lines)}
        # the best-scoring turn is not always the upright one (X1, U29): does any other turn agree?
        rank = {"same": 0, "close": 1}
        alt = min(((rank[agree(m["lines"], l)], d, l) for d, l, s in tries if agree(m["lines"], l) in rank), default=None)
        if alt and m["ocr"]["agree"] not in ("same",) and (m["ocr"]["agree"] != "close" or alt[0] == 0):
            m["ocr"]["otherTurn"] = {"turn": alt[1], "lines": alt[2], "agree": "same" if alt[0] == 0 else "close"}
        counts[m["ocr"]["agree"]] = counts.get(m["ocr"]["agree"], 0) + 1
    mk["ocr"] = {"engine": "RapidOCR (rapidocr_onnxruntime, PP-OCRv3 det/rec ONNX)", "minScore": a.min_score,
                 "summary": counts, "sec": round(time.time() - t0, 1)}
    json.dump(mk, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(refs)} parts in {time.time() - t0:.0f} s: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    for ref in refs:
        m = mk["parts"][ref]
        if m["ocr"]["agree"] in ("differs", "close", "ocr-only"):
            print(f"  {m['ocr']['agree']:8} {ref:12} read {' / '.join(m['lines']) or '—':28} "
                  f"OCR {' / '.join(m['ocr']['lines']) or '—'} ({m['ocr']['score']:.2f}, turn {m['ocr']['turn']})")


if __name__ == "__main__":
    main()
