"""Synthetic resistor-code glyphs in the laser dot-matrix / 7-segment style of the real ones (ResistorCodes.py).

The dataset of one board has many 0 / 1 / C, a handful of 4 7 9 R 6 8, and none of E F H S X Y Z (EIA-96 letters).
Each glyph is drawn from segment strokes (GLYPHS: polylines in a unit box, x right, y down) as a row of laser dots,
blurred and normalised like a real character cell (normChar: 24 x 32, 0 background .. 1 glyph). The look is measured
from the real cells (measure): glyph top / bottom and width, stroke width, glyph and background brightness; the dot
pitch, blur and stroke scale are then picked so the synthetic glyphs of the classes that have real samples land
closest to them in feature space (fit). Each synthetic glyph is jittered (stroke, pitch, width, slant, brightness,
blur, noise).

    stats = measure(chars); params = fit(chars, stats)
    img = render("E", params, rng)              # 32 x 24 float, like a real normalised cell
Stored by resistorDataset.py synth into dataset/resistor_codes/synth/ (origin synth); never used as a test sample.
"""
import numpy as np
import cv2

import ResistorCodes as rc

T, M, B, L, R = 0.0, 0.5, 1.0, 0.0, 1.0
A_ = [(L, T), (R, T)]
B_ = [(R, T), (R, M)]
C_ = [(R, M), (R, B)]
D_ = [(L, B), (R, B)]
E_ = [(L, M), (L, B)]
F_ = [(L, T), (L, M)]
G_ = [(L, M), (R, M)]
GLYPHS = {
    "0": [A_, B_, C_, D_, E_, F_],
    "1": [[(0.5, T), (0.5, B)], [(0.5, T), (0.15, 0.18)]],
    "2": [A_, B_, G_, E_, D_],
    "3": [A_, B_, [(0.25, M), (R, M)], C_, D_],
    "4": [F_, G_, [(R, T), (R, B)]],
    "5": [A_, F_, G_, C_, D_],
    "6": [A_, F_, E_, D_, C_, G_],
    "7": [A_, B_, C_],
    "8": [A_, B_, C_, D_, E_, F_, G_],
    "9": [A_, B_, F_, G_, C_, D_],
    "A": [A_, B_, C_, E_, F_, G_],
    "B": [[(L, T), (0.75, T), (R, 0.12), (R, 0.38), (0.8, M), (R, 0.62), (R, 0.88), (0.75, B), (L, B)],
          [(L, T), (L, B)], [(L, M), (0.8, M)]],
    "C": [A_, F_, E_, D_],
    "D": [[(L, T), (0.7, T), (R, 0.2), (R, 0.8), (0.7, B), (L, B)], [(L, T), (L, B)]],
    "E": [A_, F_, E_, D_, [(L, M), (0.8, M)]],
    "F": [A_, F_, E_, [(L, M), (0.8, M)]],
    "H": [F_, E_, B_, C_, G_],
    "R": [A_, F_, E_, B_, G_, [(0.55, M), (R, B)]],
    "S": [A_, F_, G_, C_, D_],
    "X": [[(L, T), (R, B)], [(R, T), (L, B)]],
    "Y": [[(L, T), (0.5, M)], [(R, T), (0.5, M)], [(0.5, M), (0.5, B)]],
    "Z": [A_, [(R, T), (L, B)], D_],
}
NARROW = {"1": 0.45}            # glyph width relative to the others (the real '1' with its flag is ~half as wide)


def measure(chars):
    """Real normalised cells [(img, label, ...)] -> statistics of the look (pixels of the 24 x 32 cell)."""
    tops, bots, widths, strokes, ink, bg = [], [], {}, [], [], []
    for x, c, *_ in chars:
        m = (x > 0.4).astype(np.uint8)
        ys, xs = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
        if len(ys) < 5 or len(xs) < 2:
            continue
        tops.append(ys[0]); bots.append(ys[-1])
        widths.setdefault(c, []).append(xs[-1] - xs[0] + 1)
        d = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        strokes.append(2 * np.percentile(d[m > 0], 90))
        ink.append(x[m > 0].mean()); bg.append(x[m == 0].mean())
    wide = [w for c, ws in widths.items() if c not in NARROW for w in ws]
    return {"top": float(np.median(tops)), "bottom": float(np.median(bots)), "width": float(np.median(wide)),
            "widths": {c: float(np.median(ws)) for c, ws in widths.items()},
            "stroke": float(np.median(strokes)), "ink": float(np.median(ink)), "inkStd": float(np.std(ink)),
            "bg": float(np.median(bg))}


def render(ch, p, rng=None, jitter=True):
    """One glyph -> CHAR_H x CHAR_W float image. p: measure() stats + fit() params (pitch, blur, strokeScale)."""
    rng = rng or np.random.default_rng()
    j = (lambda a, b: rng.uniform(a, b)) if jitter else (lambda a, b: (a + b) / 2.0)
    S = 4                                                      # draw at 4x, then shrink (dots and blur look right)
    H, W = rc.CHAR_H * S, rc.CHAR_W * S
    top, bot = p["top"] + j(-0.6, 0.6), p["bottom"] + j(-0.6, 0.6)
    width = p["widths"].get(ch, p["width"] * NARROW.get(ch, 1.0)) * j(0.92, 1.08)
    stroke = p["stroke"] * p["strokeScale"] * j(0.85, 1.15)
    r = stroke * S / 2.0
    # the strokes' centre lines sit half a stroke inside the glyph box
    x0, x1 = (rc.CHAR_W - width) / 2.0 + stroke / 2, (rc.CHAR_W + width) / 2.0 - stroke / 2
    y0, y1 = top + stroke / 2, bot - stroke / 2
    slant = np.tan(np.radians(j(-2.5, 2.5)))
    img = np.zeros((H, W), np.float32)
    pitch = max(1.0, p["pitch"] * stroke * S * j(0.9, 1.1))
    for poly in GLYPHS[ch]:
        pts = [((x0 + u * (x1 - x0)) * S + slant * (v - 0.5) * (y1 - y0) * S, (y0 + v * (y1 - y0)) * S) for u, v in poly]
        for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
            n = max(1, int(round(np.hypot(bx - ax, by - ay) / pitch)))
            for k in range(n + 1):
                t = k / n
                cv2.circle(img, (int(round(ax + t * (bx - ax))), int(round(ay + t * (by - ay)))), max(1, int(round(r))),
                           float(np.clip(p["ink"] + rng.normal(0, p["inkStd"] + 0.03), 0.3, 1.0)), -1, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), p["blur"] * S * j(0.8, 1.2))
    img = cv2.resize(img, (rc.CHAR_W, rc.CHAR_H), interpolation=cv2.INTER_AREA)
    img += rng.normal(0, 0.03, img.shape).astype(np.float32) + p["bg"] * j(0.5, 1.5)
    hi = np.percentile(img, 99)
    return np.clip(img / max(hi, 0.3), 0, 1).astype(np.float32)


def fit(chars, stats, n=6, seed=0):
    """Pick dot pitch, blur and stroke scale so synthetic glyphs of the classes with real samples sit closest to them
    (mean distance to the nearest real cell of the same class, in recognizer features)."""
    real = {}
    for x, c, *_ in chars:
        real.setdefault(c, []).append(rc.features(x))
    known = [c for c, f in real.items() if len(f) >= 3 and c in GLYPHS]
    real = {c: np.array(real[c]) for c in known}
    best = None
    for pitch in (0.5, 0.7, 1.0):
        for blur in (0.15, 0.3, 0.5):
            for sc in (0.5, 0.6, 0.7, 0.85):
                p = dict(stats, pitch=pitch, blur=blur, strokeScale=sc)
                rng = np.random.default_rng(seed)
                d = [np.linalg.norm(real[c] - rc.features(render(c, p, rng)), axis=1).min()
                     for c in known for _ in range(n)]
                if best is None or np.mean(d) < best[0]:
                    best = (float(np.mean(d)), p)
    best[1]["fitDistance"] = round(best[0], 3)
    return best[1]
