"""CLI: contact sheets of the part crops (alignBoard.py -> out/<name>/parts/) for reading the markings by eye or by a
vision model, and the template of markings.json that the reading fills in.

    venv/Scripts/python.exe markingSheets.py out/captured_51.5 [--groups Resistors,ICs] [--no-flip]

Each tile: the crop as cut (the part's own frame) and, for small parts, the same crop turned 180 deg next to it, since
a code can be printed either way round. Tiles are labelled with the designator only: no BOM value or expected code,
so a reading is not steered. Writes out/<name>/marking_sheets/<group>_<n>.jpg and sheets.json (tile -> ref).

markings.json (filled by the reader), next to board.json:
    {"version": 1, "method": "...", "parts": {ref: {
        "present": true | false | null,     part on its place / place empty / cannot tell
        "lines": ["01C"],                   marking text top to bottom as read, [] = no marking on the part
        "turn": 0 | 90 | 180 | 270,         how the text is turned in the crop (0 = reads as cut)
        "conf": "high" | "medium" | "low",
        "note": ""}}}
"""
import argparse
import json
import math
import os
import re

import cv2
import numpy as np

SKIP_PACKAGES = re.compile(r"^(PAD|M3_Plated_Hole|Solder Jumper|TP.*|FID.*)$", re.I)
# tile height (px) per group: small chips are enlarged so a 0.5 mm code is ~40 px tall
TILE_H = {"Resistors": 170, "Capacitors": 150, "Inductors": 220, "Diodes & protection": 200, "Transistors": 200,
          "LEDs & optocouplers": 150, "ICs": 420, "Crystals & oscillators": 320, "Connectors": 300,
          "Buttons & switches": 300, "Other": 200}
SHEET_W = 1800


def label(img, text):
    pad = np.full((26, img.shape[1], 3), 32, np.uint8)
    cv2.putText(pad, text, (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([pad, img])


def tile(crop, h, flip):
    s = h / crop.shape[0]
    w = max(1, int(round(crop.shape[1] * s)))
    if w > SHEET_W // (2 if flip else 1) - 20:   # very long parts: fit the width instead
        w = SHEET_W // (2 if flip else 1) - 20
        s = w / crop.shape[1]
        h = max(1, int(round(crop.shape[0] * s)))
    im = cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    if flip:
        im = np.hstack([im, np.full((h, 6, 3), 32, np.uint8), cv2.rotate(im, cv2.ROTATE_180)])
    return im


def sheets(tiles, gap=8):
    """Row-pack the tiles into sheets of SHEET_W width and at most ~1800 px height."""
    out, rows, row, rowW = [], [], [], 0
    for t in tiles:
        if row and rowW + t.shape[1] + gap > SHEET_W:
            rows.append(row); row, rowW = [], 0
        row.append(t); rowW += t.shape[1] + gap
    if row:
        rows.append(row)
    page, pageH, pages = [], 0, []
    for r in rows:
        h = max(t.shape[0] for t in r)
        if page and pageH + h + gap > 1800:
            pages.append(page); page, pageH = [], 0
        page.append(r); pageH += h + gap
    if page:
        pages.append(page)
    for pg in pages:
        lines = []
        for r in pg:
            h = max(t.shape[0] for t in r)
            parts = []
            for t in r:
                parts += [np.vstack([t, np.full((h - t.shape[0], t.shape[1], 3), 18, np.uint8)]),
                          np.full((h, gap, 3), 18, np.uint8)]
            ln = np.hstack(parts)
            lines += [np.hstack([ln, np.full((h, SHEET_W - ln.shape[1], 3), 18, np.uint8)]) if ln.shape[1] < SHEET_W
                      else ln[:, :SHEET_W], np.full((gap, SHEET_W, 3), 18, np.uint8)]
        out.append(np.vstack(lines))
    return out, [[len(r) for r in pg] for pg in pages]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("panorama", help="alignBoard output folder (board.json with align.crops, parts/)")
    ap.add_argument("--groups", help="comma-separated BOM groups (default: all)")
    ap.add_argument("--no-flip", action="store_true", help="no 180 deg copy next to each crop")
    a = ap.parse_args()

    b = json.load(open(os.path.join(a.panorama, "board.json"), encoding="utf-8"))
    crops = b["align"]["crops"]
    want = set(a.groups.split(",")) if a.groups else None
    byGroup, skipped = {}, []
    for ref, c in b["components"].items():
        if ref not in crops:
            continue
        line = b["lines"][c["line"]] if c["line"] is not None else None
        if line is None and SKIP_PACKAGES.match(c.get("package", "")):
            skipped.append(ref)
            continue
        g = line["cat"]["group"] if line else "Other"
        if want and g not in want:
            continue
        byGroup.setdefault(g, []).append(ref)
    folder = os.path.join(a.panorama, "marking_sheets")
    os.makedirs(folder, exist_ok=True)
    for f in os.listdir(folder):
        os.remove(os.path.join(folder, f))
    index = {"skipped": skipped, "sheets": {}}
    key = lambda r: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", r)]
    for g, refs in byGroup.items():
        refs.sort(key=key)
        h = TILE_H.get(g, 200)
        flip = not a.no_flip and g not in ("ICs", "Connectors", "Crystals & oscillators", "Buttons & switches")
        tiles = [label(tile(cv2.imread(os.path.join(a.panorama, crops[r]["file"])), h, flip), r) for r in refs]
        pages, layout = sheets(tiles)
        n = 0
        for i, (img, lay) in enumerate(zip(pages, layout)):
            name = f"{re.sub(r'[^A-Za-z]+', '_', g).strip('_')}_{i + 1}.jpg"
            cv2.imwrite(os.path.join(folder, name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            cnt = sum(lay)
            index["sheets"][name] = {"group": g, "refs": refs[n:n + cnt], "rows": lay, "flip": flip}
            n += cnt
            print(f"{name}: {cnt} parts, {img.shape[1]} x {img.shape[0]}")
    with open(os.path.join(folder, "sheets.json"), "w", encoding="utf-8") as fp:
        json.dump(index, fp, ensure_ascii=False, indent=1)
    print(f"skipped (pads, holes, jumpers): {len(skipped)}")


if __name__ == "__main__":
    main()
