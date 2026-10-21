"""CLI: every source view (frame x sharpest height) of the marked parts, for reading their marking.

    venv/Scripts/python.exe cutMarkingViews.py out/captured_51.5 [--stack out/stack] [--refs U17,Q2] [--sheets]

Needs an alignBoard.py panorama folder (board.json) and a stackFocus.py folder of the same reference stitch, built at
--scale 1 (report.json, index.html, focus.jpg). Writes into the panorama folder:
    marking_views/<ref>/<frame>.jpg   body crop from that frame, upright like parts/<ref>.jpg, registered onto
                                      focus.jpg; black where the frame does not reach
    marking_views/<ref>/pano.jpg      the same crop of panorama.jpg, for comparison
    marking_views.json                {ref: {w, h, pxPerMm, pano, views: [{frame, z, focus, cover, file}]}}
    marking_views/sheets/<ref>.jpg    with --sheets: panorama + every view, colour over CLAHE, for reading by eye
Default parts: fitted, own side, groups ICs / Transistors / Diodes / Crystals, tantalum capacitors, power chokes.
See MarkingViews.py for the geometry.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from MarkingViews import MarkingViews, safeName


def clahe(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[..., 0]
    return cv2.cvtColor(cv2.createCLAHE(3.0, (4, 4)).apply(g), cv2.COLOR_GRAY2BGR)


def sheet(t, views, path, side=400):
    tiles = ([("panorama", t.pano)] if t.pano is not None else []) + \
            [(f"{v['frame']} z{v['z']} cover {v['cover']:.2f}", np.where(v["valid"][..., None] > 0, v["img"], 0))
             for v in views]
    s = side / max(t.w, t.h)
    row = []
    for label, img in tiles:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        p = cv2.copyMakeBorder(np.vstack([img, clahe(img)]), 22, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(p, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        row.append(p)
    cv2.imwrite(path, np.hstack(row), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("panorama", help="alignBoard.py output folder (board.json, panorama.jpg)")
    ap.add_argument("--stack", default="out/stack", help="stackFocus.py output folder of the same reference stitch")
    ap.add_argument("--refs", help="comma-separated designators (default: the marked groups, see the header)")
    ap.add_argument("--sheets", action="store_true", help="contact sheets for reading by eye")
    a = ap.parse_args()

    t0 = time.time()
    board = json.load(open(os.path.join(a.panorama, "board.json"), encoding="utf-8"))
    mv = MarkingViews(board, a.stack)
    refs = a.refs.split(",") if a.refs else mv.defaultRefs()
    focus = cv2.imread(os.path.join(a.stack, "focus.jpg"), cv2.IMREAD_COLOR)
    pano = cv2.imread(os.path.join(a.panorama, "panorama.jpg"), cv2.IMREAD_COLOR)
    if focus is None or pano is None or focus.shape != pano.shape:
        raise SystemExit("focus.jpg and panorama.jpg must exist and share the reference canvas")
    targets = mv.collect(refs, focus, pano)
    folder = os.path.join(a.panorama, "marking_views")
    rec = mv.write(targets, folder)
    path = os.path.join(a.panorama, "marking_views.json")
    old = json.load(open(path, encoding="utf-8")) if a.refs and os.path.exists(path) else {}
    old.update(rec)
    json.dump(old, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if a.sheets:
        os.makedirs(os.path.join(folder, "sheets"), exist_ok=True)
        for t in targets:
            sheet(t, mv.ranked(t), os.path.join(folder, "sheets", safeName(t.ref) + ".jpg"))
    none = [t.ref for t in targets if not t.views]
    print(f"{len(targets)} parts -> {folder}, {time.time() - t0:.0f} s" + (f"; no view: {', '.join(none)}" if none else ""))


if __name__ == "__main__":
    main()
