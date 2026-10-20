"""Local recognizer of SMD resistor marking codes (laser dot-matrix, white on a black body): '01C', '103', '22R0', 'R200', '0'.

    rec = Recognizer.fromDataset("dataset/resistor_codes")   # the dataset's characters (+ jittered copies), ~0.6 s
    rec.read(partCrop, turns=(0, 180))                        # BGR parts/<ref>.jpg -> {code, score, margin, turn, ...}

Steps: findBody (dark brownish body vs blue solder mask by R - B) -> textLine (white top-hat + Otsu, the line of
character-sized components) -> segmentN (the font is proportional: characters are column runs of ink; for a hypothesis
of n characters wide runs are cut, extra runs merged or dropped at an end, at a cost) -> normChar / features (24 x 32,
HOG + coarse grid + width) -> per-class nearest neighbours -> log-probabilities, with a reject class of glyphs upside
down. A single character is weak on its own (dot-matrix 0 / D / 8 / B differ by a few dots, some classes have one or two
samples), so the code is chosen as a whole: every valid code of length 1, 3 or 4 (EIA-96 '01C', '103' / '1002',
R-notation '4R7' / 'R200', '0') is scored by the sum of its characters' log-probabilities plus a small prior; the
hypothesis (turn x length) with the best mean log-probability minus the segmentation cost wins.
score = geometric mean of the code's character probabilities, margin = its log-score lead over the runner-up code.
Dataset: resistorDataset.py; reading and evaluation: readResistorCodes.py.
"""
import csv
import os

import cv2
import numpy as np

from Components import E96

REJECT = "~"                    # class of glyphs upside down (never part of a code)
SYMMETRIC = set("0125689HSXZ")  # read as themselves upside down (7-segment-like font: 6 <-> 9; "512" upside down reads
#                                 "215": the 1's flag decides). 3 upside down looks like E, but keeping 3 in the reject
#                                 class wins 2 parts on captured_51.5 (turn of 333 / 223 / 513), E is only ever last.
ROT = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
EIA96_LETTERS = "ZYRXSABHCDEF"
CHAR_W, CHAR_H = 24, 32         # normalised character box (glyph height fills it, width scaled alike, centred)
BODY_MARGIN = 0.06              # fraction of the body cut off each side before looking for glyphs (body edge, pad shine)
STROKE_MAX = 0.16               # half stroke width above this fraction of the character height: a blob, not a glyph
BODY_COVER = 0.5                # body box: rows / columns at least this much covered by the body component
AUGMENT_N = 6                   # jittered copies of every training character
KNN = 2                         # class distance = mean of its KNN nearest samples
TEMP = 0.15                     # softmax temperature, in units of the typical same-class distance
UNSEEN_PENALTY = 8.0            # log-score of a character class the dataset has no sample of
SEG_W = 0.3                     # weight of the segmentation cost against the mean character log-probability
TOPHAT = 0.2                    # top-hat kernel, fraction of the body height (wider than a stroke)
SEG_OK = 0.6                    # a training sample whose cut costs more than this is 'badseg'
DROP_END = 0.3                  # segmentation cost of leaving out a run at an end of the line
LENGTH_PRIOR = {1: -0.3, 3: 0.0, 4: -0.15}   # 3-character codes are the most common on chips
CONTEXT = 0.6                   # context crop: body height fraction added above and below (orientation bar)
BAR_W_RANGE = (0.3, 0.9)        # orientation bar width, in character heights
BAR_GAP = 0.15                  # the bar starts at most this many character heights below the line (above: turned)
BAR_THICK = (0.11, 0.3)         # bar thickness, in character heights (edge glints are thinner)
BAR_PRIOR = 1.0                 # quality bonus of the turn the bar says is upright (penalty for the other)
SYNTH_KEY = "synth:"            # part key of synthetic glyphs (never a test sample)
SYNTH_N = 20                    # synthetic glyphs per class added to training (0: none)
SYNTH_RARE = 10                 # ... for classes with fewer real characters than this
SYNTH_FOR = "rare"              # "rare" or "all" classes
SYNTH_DIST = 1.0                # distance factor of synthetic samples (> 1: real ones win when as close)
SYNTH_AUGMENT = 0               # jittered copies of each synthetic glyph (they are jittered when drawn)
DARK = 95                       # max(B, G, R) below this is body (or board); glyphs and pads are brighter


def turn(img, deg):
    """Turn clockwise by a multiple of 90 deg (cv2.ROTATE_90_CLOCKWISE is 0: test the angle, not the code)."""
    return img if deg % 360 == 0 else cv2.rotate(img, ROT[deg % 360])


# ---------------------------------------------------------------- body and characters

def findBody(crop):
    """Upright BGR part crop -> (grey body crop, (x, y, w, h) in the crop) or (None, None).
    The body is the dark brownish component nearest the centre: board solder mask is blue (B > R), pads and glyphs bright."""
    h, w = crop.shape[:2]
    c = crop.astype(np.int16)
    v = c.max(2)
    rb = c[:, :, 2] - c[:, :, 0]
    # threshold half-way between the board (the crop's border strips) and the dark centre; frames differ in white balance
    bw_ = max(3, min(h, w) // 12)
    border = np.concatenate([rb[:bw_].ravel(), rb[-bw_:].ravel(), rb[:, :bw_].ravel(), rb[:, -bw_:].ravel()])
    cy, cx = h // 2, w // 2
    centre = rb[cy - h // 6:cy + h // 6 + 1, cx - w // 6:cx + w // 6 + 1]
    cv_ = v[cy - h // 6:cy + h // 6 + 1, cx - w // 6:cx + w // 6 + 1]
    centreRb = float(np.median(centre[cv_ < DARK])) if (cv_ < DARK).sum() > 10 else 15.0
    boardRb = float(np.median(border))
    thr = (boardRb + centreRb) / 2 if centreRb - boardRb > 8 else -8
    bodyish = (rb > thr) & (v < DARK)
    s = min(h, w)
    m = bodyish.astype(np.uint8) * 255
    # glyph strokes are bright holes in the body: close them, then open hard to cut the body off dark bits of the pads
    closeK = max(5, int(round(s * 0.07)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closeK, closeK)))
    openK = max(5, int(round(s * 0.14)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (openK, openK)))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
    best, bestScore = None, -1.0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 0.02 * h * w or bw > 0.97 * w or bh > 0.97 * h:
            continue
        fill = area / float(bw * bh)
        d = np.hypot((cent[i][0] - w / 2) / w, (cent[i][1] - h / 2) / h)
        score = area * fill / (1 + 4 * d)
        if score > bestScore:
            best, bestScore = i, score
    if best is None:
        return None, None
    # trim to the rows / columns the component mostly covers (cuts off board and pad bits stuck to the body)
    comp = np.zeros((h, w), np.uint8)
    cs, _ = cv2.findContours((lab == best).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(comp, cs, -1, 1, cv2.FILLED)     # glyphs enclosed by the body are body too
    comp = comp > 0
    x0, y0, bw, bh, _ = stats[best]
    x1, y1 = x0 + bw, y0 + bh
    for _ in range(3):
        rows = np.nonzero(comp[y0:y1, x0:x1].mean(1) >= BODY_COVER)[0]
        if len(rows) == 0:
            return None, None
        y0, y1 = y0 + rows[0], y0 + rows[-1] + 1
        cols = np.nonzero(comp[y0:y1, x0:x1].mean(0) >= BODY_COVER)[0]
        if len(cols) == 0:
            return None, None
        x0, x1 = x0 + cols[0], x0 + cols[-1] + 1
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[:, :, 0]
    return grey[y0:y1, x0:x1].copy(), (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def bodyAndContext(crop):
    """Upright BGR part crop -> (grey context crop, body box (x, y, w, h) inside it) or (None, None). The context is the
    body's columns, CONTEXT body heights more above and below: the orientation bar lies outside the tight body box."""
    body, rect = findBody(crop)
    if body is None:
        return None, None
    x, y, w, h = rect
    e = int(round(h * CONTEXT))
    ya, yb = max(0, y - e), min(crop.shape[0], y + h + e)
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[:, :, 0]
    return grey[ya:yb, x:x + w].copy(), (0, y - ya, w, h)


def bodyOf(ctx, box):
    x, y, w, h = box
    return ctx[y:y + h, x:x + w]


def turnSample(ctx, box, deg):
    """Context crop and body box turned by 0 or 180 deg."""
    if deg % 360 == 0:
        return ctx, box
    H, W = ctx.shape[:2]
    x, y, w, h = box
    return turn(ctx, 180), (W - x - w, H - y - h, w, h)


def glyphMask(body, marginY=BODY_MARGIN):
    """Grey body crop -> binary mask of the bright glyph pixels (dots of the dot matrix joined) inside the body margin."""
    h, w = body.shape
    mx, my = int(w * BODY_MARGIN), int(h * marginY)
    g = cv2.GaussianBlur(body, (0, 0), 0.8)
    # white top-hat: laser strokes are thin, pads and glare inside the box are wide -> only the strokes stay bright
    k = max(5, int(round(min(h, w) * TOPHAT)) | 1)
    g = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    inner = g[my:h - my, mx:w - mx]
    m = np.zeros_like(body)
    if inner.size == 0:
        return m
    t, _ = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(t, float(np.percentile(inner, 30)) + 20)
    m[my:h - my, mx:w - mx] = (inner > t).astype(np.uint8) * 255
    k = max(2, int(round(h * 0.025)))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def _runs(profile, minVal):
    on = profile > minVal
    runs, start = [], None
    for i, o in enumerate(on):
        if o and start is None:
            start = i
        if not o and start is not None:
            runs.append((start, i)); start = None
    if start is not None:
        runs.append((start, len(on)))
    return runs


def textLine(body, mask=None):
    """Grey body crop -> (glyph mask restricted to the text line, (x0, y0, x1, y1) of the line) or (None, None).
    Connected components of the glyph mask; slivers of the pads at the body's ends, specks and underlines (thin bars
    below a character) are dropped; the line is the row band where the character-sized components agree."""
    m = glyphMask(body) if mask is None else mask
    h, w = m.shape
    mx = int(w * BODY_MARGIN)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    thick = np.zeros(n, np.float32)
    np.maximum.at(thick, lab.ravel(), dist.ravel())
    comps, edge = [], []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < max(4, h * h * 0.002) or bh > 0.85 * h:
            continue    # a speck, or as tall as the body (a pad's edge)
        c = (i, x, y, x + bw, y + bh, area)
        (edge if (x <= mx + 1 or x + bw >= w - mx - 1) and bw < 0.3 * bh else comps).append(c)
    if not comps:
        comps, edge = edge, []
    big = [c for c in comps if (c[4] - c[2]) > 0.25 * h]
    if not big:
        return None, None
    hMed = float(np.median([c[4] - c[2] for c in big]))
    # blobs (pad shine, solder) are thicker than a laser stroke
    comps = [c for c in comps if thick[c[0]] <= STROKE_MAX * hMed]
    edge = [c for c in edge if thick[c[0]] <= STROKE_MAX * hMed]
    big = [c for c in comps if (c[4] - c[2]) > 0.25 * h]
    if not big:
        return None, None
    hMed = float(np.median([c[4] - c[2] for c in big]))
    keep = [c for c in big if (c[4] - c[2]) < 1.35 * hMed]
    y0 = int(np.median([c[2] for c in keep]))
    y1 = int(np.median([c[4] for c in keep]))
    th = y1 - y0
    if th < 0.15 * h:
        return None, None
    # a component touching the body's end is a character only if it sits on the line like one (else: pad shine)
    for c in edge:
        ch, fill = c[4] - c[2], c[5] / float((c[3] - c[1]) * (c[4] - c[2]))
        if abs(c[2] - y0) < 0.2 * th and abs(c[4] - y1) < 0.2 * th and fill < 0.8:
            comps.append(c)
    out = np.zeros_like(m)
    for i, x0_, cy0, x1_, cy1, area in comps:
        ch = cy1 - cy0
        inside = min(cy1, y1) - max(cy0, y0)
        if inside < 0.5 * ch or ch > 1.35 * th:
            continue    # outside the line, or taller than a character (pad, body edge)
        if ch < 0.3 * th and (x1_ - x0_) > 2 * ch and cy0 > y0 + 0.6 * th:
            continue    # underline
        out[lab == i] = 255
    out[:max(0, y0 - th // 8)] = 0
    out[y1 + th // 8:] = 0
    xs = np.nonzero(out.sum(0))[0]
    if len(xs) == 0:
        return None, None
    return out, (int(xs[0]), y0, int(xs[-1]) + 1, y1)


def barSide(ctx, box, line):
    """Orientation bar: some 3-digit series carry a short thin bar under the middle digit (220, 272, 333, 621).
    ctx / box: context crop and the body box in it (bodyAndContext), line: textLine's line in body coordinates.
    -> +1 bar below the line (text upright), -1 above (turned 180 deg), 0 none or both.
    Rows just below (above) the line are scanned on the unclosed top-hat (the bar often touches the digit): a bar row is
    bright under the middle third of the line and dark under the outer thirds; the bright run must be about a glyph wide,
    a few rows thick, with dark rows past it. The bar never reaches the character cells (textLine drops it)."""
    if line is None:
        return 0
    bx, by = box[0], box[1]
    x0, y0, x1, y1 = line[0] + bx, line[1] + by, line[2] + bx, line[3] + by
    th = float(y1 - y0)
    W = x1 - x0
    if th < 6 or W < th:
        return 0
    g = cv2.GaussianBlur(ctx, (0, 0), 0.8)
    k = max(5, int(round(box[3] * TOPHAT)) | 1)
    g = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    inner = g[by:by + box[3], bx + int(box[2] * BODY_MARGIN):bx + box[2] - int(box[2] * BODY_MARGIN)]
    t, _ = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = g > max(t, float(np.percentile(inner, 30)) + 20)
    # the middle character's columns (cut as 3 characters; '1' is narrow, so not simply the middle third)
    a3, b3 = int(x0 + W / 3), int(x1 - W / 3)
    mask, _ = textLine(bodyOf(ctx, box))
    if mask is not None:
        cells, _ = segmentN(mask, line, 3)
        if cells is not None:
            a3, b3 = cells[1][0] + bx, cells[1][2] + bx
            if b3 - a3 < 0.3 * th:      # a narrow '1' in the middle: widen to a glyph's width
                c = (a3 + b3) / 2.0
                a3, b3 = int(c - 0.3 * th), int(c + 0.3 * th)

    def side(rows):
        """rows (top to bottom, moving away from the line) -> True when a bar is found among them."""
        run = []
        for r in rows:
            if not 0 <= r < bright.shape[0]:
                break
            mid = bright[r, a3:b3].mean()
            outer = max(bright[r, x0:a3].mean() if a3 > x0 else 0.0, bright[r, b3:x1].mean() if x1 > b3 else 0.0)
            if not run and mid >= 0.35 and outer <= 0.15:
                run.append(r)           # bright in the middle only: a bar starts
            elif run and mid >= 0.25 and outer <= 0.3:
                run.append(r)
            elif run:
                break                   # the run ended: check what follows it
        if not run or not BAR_THICK[0] * th <= len(run) <= BAR_THICK[1] * th or                 abs(run[0] - rows[0]) > (BAR_GAP + 0.05) * th:
            return False    # none, too thin / thick, or too far from the line (glints on the body edge)
        nxt = rows[rows.index(run[-1]) + 1:rows.index(run[-1]) + 1 + max(2, int(0.1 * th))]
        if any(0 <= r < bright.shape[0] and bright[r, a3:b3].mean() > 0.2 for r in nxt):
            return False    # not isolated (body edge glare continues)
        cols = bright[min(run):max(run) + 1, x0:x1].any(0)
        xs = np.nonzero(cols)[0]
        width = xs[-1] - xs[0] + 1 if len(xs) else 0
        centre = x0 + (xs[0] + xs[-1]) / 2.0 if len(xs) else -1
        return BAR_W_RANGE[0] * th <= width <= BAR_W_RANGE[1] * th and abs(centre - (a3 + b3) / 2.0) < 0.25 * th

    reach = int(2 * BAR_GAP * th)
    below = side(list(range(int(y1 - 0.05 * th), int(y1 + reach))))
    above = side(list(range(int(y0 + 0.05 * th), int(y0 - reach), -1)))
    return 1 if below and not above else (-1 if above and not below else 0)


def normChar(grey, box, bandH, ink=None):
    """Character cell -> CHAR_H x CHAR_W float image, 0 background .. 1 glyph: the text band height fills the box
    height, the width is scaled alike and centred. ink: the cell's glyph mask (same size as grey) - everything else in
    the cell (neighbours' fragments, body texture) is dimmed away."""
    x0, y0, x1, y1 = box
    pad = max(1, bandH // 12)
    H, W = grey.shape
    ya, yb, xa, xb = max(0, y0 - pad), min(H, y1 + pad), max(0, x0 - 1), min(W, x1 + 1)
    c = grey[ya:yb, xa:xb].astype(np.float32)
    if c.size == 0:
        return np.zeros((CHAR_H, CHAR_W), np.float32)
    lo, hi = np.percentile(c, 5), np.percentile(c, 99)
    c = np.clip((c - lo) / max(hi - lo, 1.0), 0, 1)
    if ink is not None:
        k = max(3, (bandH // 10) | 1)
        soft = cv2.GaussianBlur(cv2.dilate(ink[ya:yb, xa:xb], np.ones((k, k), np.uint8)).astype(np.float32) / 255,
                                (0, 0), k / 3.0)
        c = c * np.clip(soft * 1.5, 0, 1)
    s = CHAR_H / float(c.shape[0])
    nw = max(1, min(CHAR_W, int(round(c.shape[1] * s))))
    r = cv2.resize(c, (nw, CHAR_H), interpolation=cv2.INTER_AREA)
    out = np.zeros((CHAR_H, CHAR_W), np.float32)
    x = (CHAR_W - nw) // 2
    out[:, x:x + nw] = r
    return out


def segmentN(mask, line, n):
    """Cut the text line into n characters. The font is proportional ('1' is narrow and sits close to its neighbours),
    so characters are the column runs of glyph ink; a run too wide for one character is cut at its weakest column, and
    when there are more runs than n the closest narrow pair is merged (a character broken in two).
    -> (n boxes (x0, y0, x1, y1), cost) or (None, inf); cost grows with ink cut, merges and odd widths, per height."""
    x0, y0, x1, y1 = line
    th = float(y1 - y0)
    prof = (mask[y0:y1] > 0).sum(0).astype(np.float32) / th
    runs = [[a, b] for a, b in _runs(prof, 0)]
    def tall(a, b):
        ys = np.nonzero(mask[y0:y1, a:b].any(1))[0]
        return len(ys) and ys[-1] - ys[0] + 1 >= 0.45 * th
    runs = [r for r in runs if tall(*r)]     # specks and dots are not characters
    if not runs:
        return None, np.inf
    cost = 0.0
    while len(runs) > n:
        # merge the adjacent pair giving the narrowest result, or drop an end run (pad glare at the body's end)
        widths = [runs[i + 1][1] - runs[i][0] for i in range(len(runs) - 1)]
        i = int(np.argmin(widths))
        gap = runs[i + 1][0] - runs[i][1]
        merge = 0.4 + gap / th + max(0.0, widths[i] / th - 0.75) * 2
        # a run touching the glyph search margin is most likely glare on the pad, cheap to leave out
        lim0, lim1 = int(mask.shape[1] * BODY_MARGIN) + 2, mask.shape[1] - int(mask.shape[1] * BODY_MARGIN) - 2
        drops = [((DROP_END * (0.2 if (a <= lim0 or b >= lim1) else 1.0)) + float(prof[a:b].sum()) * 0.5, j)
                 for j, (a, b) in ((0, runs[0]), (-1, runs[-1]))]
        drop, j = min(drops)
        if drop < merge:
            cost += drop
            runs.pop(j)
        else:
            cost += merge
            runs[i:i + 2] = [[runs[i][0], runs[i + 1][1]]]
    while len(runs) < n:
        # split the widest run at its weakest column near the middle
        i = int(np.argmax([b - a for a, b in runs]))
        a, b = runs[i]
        if b - a < 4:
            return None, np.inf
        lo, hi = a + (b - a) // 4, b - (b - a) // 4
        c = lo + int(np.argmin(np.convolve(prof[lo:hi], np.ones(3) / 3, "same"))) if hi > lo + 1 else (a + b) // 2
        cost += 0.15 + float(prof[c])
        runs[i:i + 1] = [[a, c], [c, b]]
    for a, b in runs:
        cost += max(0.0, (b - a) / th - 0.85) * 2     # wider than a character
    return [(int(a), y0, int(b), y1) for a, b in runs], cost


def cellInk(mask, line, boxes):
    """Character boxes -> (boxes tightened to their ink, per-character glyph masks clipped to the box)."""
    out, inks = [], []
    for a, y0, b, y1 in boxes:
        ink = np.zeros_like(mask)
        ink[:, a:b] = mask[:, a:b]
        xs = np.nonzero(ink.any(0))[0]
        out.append((int(xs[0]), y0, int(xs[-1]) + 1, y1) if len(xs) else (a, y0, b, y1))
        inks.append(ink)
    return out, inks


_HOG = cv2.HOGDescriptor((CHAR_W, CHAR_H), (8, 8), (4, 4), (4, 4), 9)


def features(norm):
    """Normalised character -> HOG + a coarse 5 x 4 intensity grid + width (narrow '1' vs the rest)."""
    u8 = (norm * 255).astype(np.uint8)
    hog = _HOG.compute(u8).ravel()
    grid = cv2.resize(norm, (4, 5), interpolation=cv2.INTER_AREA).ravel()
    cols = np.nonzero(norm.max(0) > 0.3)[0]
    width = (cols[-1] - cols[0] + 1) / CHAR_W if len(cols) else 0.0
    return np.concatenate([hog, grid * 0.6, [width * 1.5]]).astype(np.float32)


# ---------------------------------------------------------------- augmentation

def augment(norm, rng, n):
    """n jittered copies of a normalised character: shift, scale, rotation +-3 deg, blur, contrast, gamma."""
    out = []
    for _ in range(n):
        a = rng.uniform(-3, 3)
        s = rng.uniform(0.92, 1.08)
        tx, ty = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
        M = cv2.getRotationMatrix2D((CHAR_W / 2, CHAR_H / 2), a, s)
        M[:, 2] += (tx, ty)
        x = cv2.warpAffine(norm, M, (CHAR_W, CHAR_H), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if rng.random() < 0.5:
            x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.4, 1.0))
        x = np.clip(x * rng.uniform(0.75, 1.15) + rng.uniform(-0.08, 0.08), 0, 1) ** rng.uniform(0.8, 1.25)
        out.append(x.astype(np.float32))
    return out


# ---------------------------------------------------------------- grammar

def validCodes():
    """All codes a chip resistor may carry, by length: {n: [code]}."""
    codes = {"0"}
    for i in range(1, 97):
        for L in EIA96_LETTERS:
            codes.add(f"{i:02d}{L}")
    for n in (3, 4):
        for v in range(10 ** n):
            codes.add(f"{v:0{n}d}")
        # R-notation: one R, n-1 digits, R anywhere ('R10', '4R7', '47R', 'R200', '22R0', '100R'
        for pos in range(n):
            for v in range(10 ** (n - 1)):
                d = f"{v:0{n - 1}d}"
                codes.add(d[:pos] + "R" + d[pos:])
    out = {}
    for c in codes:
        out.setdefault(len(c), []).append(c)
    return {k: sorted(v) for k, v in out.items()}


def codePrior(code):
    """Small log-prior: odd codes ('000', '0R00', leading-zero 3/4-digit codes other than EIA-96) are unlikely."""
    if code == "0":
        return 0.0
    if code.isdigit():
        if code[0] == "0" and len(code) >= 3:
            return -3.0 if set(code) != {"0"} else -1.0
        return 0.0
    if "R" in code:
        if code.startswith("0") or code.endswith("R") and len(code) == 4:
            return -2.0
        return -0.5
    return 0.0


# ---------------------------------------------------------------- recognizer

def charCells(body, n):
    """Grey body crop -> (n tight character boxes, their glyph masks, segmentation cost, line) or (None, None, inf, None)."""
    mask, line = textLine(body)
    if line is None:
        return None, None, np.inf, None
    cells, cost = segmentN(mask, line, n)
    if cells is None:
        return None, None, np.inf, line
    boxes, inks = cellInk(mask, line, cells)
    return boxes, inks, cost, line


def charImages(body, n):
    """Grey body crop -> ([n normalised characters], segmentation cost, boxes) or (None, inf, None)."""
    boxes, inks, cost, line = charCells(body, n)
    if boxes is None:
        return None, np.inf, None
    return [normChar(body, b, line[3] - line[1], k) for b, k in zip(boxes, inks)], cost, boxes


class Recognizer:
    """Per-class nearest neighbours on character features -> log-probabilities per character; the code is the best
    valid code over the length hypotheses (1, 3, 4 cells) and the candidate turns."""

    def __init__(self, feats, labels, keys):
        self.feats = np.asarray(feats, np.float32)
        self.labels = np.asarray(labels)
        self.keys = np.asarray(keys)
        self.classes = sorted(set(self.labels.tolist()) - {REJECT}) + ([REJECT] if REJECT in self.labels else [])
        self.onehot = np.stack([self.labels == c for c in self.classes])
        self.synth = np.char.startswith(self.keys.astype(str), SYNTH_KEY)
        self.tau = self._tau()
        self.grammar = {}
        for n, codes in validCodes().items():
            self.grammar[n] = (codes, np.array([[self._ci(ch) for ch in c] for c in codes], np.int32),
                               np.array([codePrior(c) for c in codes], np.float32))

    def _ci(self, ch):
        return self.classes.index(ch) if ch in self.classes and ch != REJECT else len(self.classes)   # unseen -> extra

    def _tau(self):
        """Distance scale: median distance from a sample to its nearest same-class sample of another part."""
        d = []
        rng = np.random.default_rng(1)
        for i in rng.choice(len(self.feats), min(300, len(self.feats)), replace=False):
            same = (self.labels == self.labels[i]) & (self.keys != self.keys[i])
            if same.any():
                d.append(np.linalg.norm(self.feats[same] - self.feats[i], axis=1).min())
        return float(np.median(d)) if d else 1.0

    @classmethod
    def fromChars(cls, chars, augmentN=AUGMENT_N, seed=0):
        """chars: [(normalised char, label, part key)] -> recognizer (each character plus augmentN jittered copies)."""
        rng = np.random.default_rng(seed)
        feats, labels, keys = [], [], []
        for norm, lab, key in chars:
            synth = key.startswith(SYNTH_KEY)
            for x in [norm] + augment(norm, rng, SYNTH_AUGMENT if synth else augmentN):
                feats.append(features(x)); labels.append(lab); keys.append(key)
            # the same glyph upside down is the reject class: a code read in the wrong turn then scores low
            if lab not in SYMMETRIC:
                for x in [norm[::-1, ::-1].copy()] + augment(norm[::-1, ::-1].copy(), rng, augmentN // 2):
                    feats.append(features(x)); labels.append(REJECT); keys.append(key)
        return cls(feats, labels, keys)

    @classmethod
    def fromDataset(cls, folder, keep=None, augmentN=AUGMENT_N, synthN=None):
        """keep(row) -> bool selects the code samples to train on (default: all usable). Synthetic glyphs
        (synth/, SynthGlyphs.py) are added for the classes with fewer than SYNTH_RARE real characters (SYNTH_FOR =
        "all": every class), synthN (default SYNTH_N) per class; 0 = none."""
        real = [(x, ch, partKey(row)) for x, ch, row in loadChars(folder) if keep is None or keep(row)]
        return cls.fromChars(real + synthChars(folder, real, SYNTH_N if synthN is None else synthN), augmentN)

    def charLogProb(self, norm, exclude=None):
        """Normalised character -> log-probability per class, plus an 'unseen class' column at a fixed low value.
        exclude: a part key whose samples are ignored (leave-one-part-out)."""
        d = np.linalg.norm(self.feats - features(norm), axis=1)
        if SYNTH_DIST != 1.0:
            d = np.where(self.synth, d * SYNTH_DIST, d)     # synthetic glyphs count as a little farther
        if exclude is not None:
            d = np.where(self.keys == exclude, np.inf, d)
        per = np.empty(len(self.classes))
        for k in range(len(self.classes)):
            dk = np.sort(d[self.onehot[k]])[:KNN]
            per[k] = dk.mean() if len(dk) and np.isfinite(dk).all() else np.inf
        z = np.where(np.isfinite(per), -per / (self.tau * TEMP), -1e9)
        z = np.append(z, z.max() - UNSEEN_PENALTY)
        z -= z.max()
        return z - np.log(np.exp(z).sum())

    def decode(self, logp, top=5):
        """logp: (n, classes+1) -> [(code, summed log-score incl. prior)] best first."""
        n = len(logp)
        if n not in self.grammar:
            return []
        codes, ci, prior = self.grammar[n]
        s = logp[np.arange(n)[None, :], ci].sum(1) + prior
        best = np.argsort(-s)[:top]
        return [(codes[i], float(s[i])) for i in best]

    def readBody(self, body, exclude=None, lengths=(1, 3, 4), bar=0):
        """Grey upright body crop -> best hypothesis {code, score, margin, quality, n, chars, cells, bar} or None.
        bar: barSide of this turn (+1 upright, -1 turned), a prior on the turn."""
        hyps = []
        for n in lengths:
            norms, cost, boxes = charImages(body, n)
            if norms is None:
                continue
            logp = np.array([self.charLogProb(x, exclude) for x in norms])
            top = self.decode(logp)
            if not top:
                continue
            code, s = top[0]
            mean = s / n
            hyps.append({"code": code, "score": float(np.exp(mean)),
                         "quality": float(mean - SEG_W * cost + LENGTH_PRIOR.get(n, 0.0) + BAR_PRIOR * bar),
                         "runnerUp": top[1][0] if len(top) > 1 else "",
                         "margin": float(s - top[1][1]) if len(top) > 1 else 99.0,
                         "n": n, "segCost": float(cost), "cells": boxes,
                         "chars": "".join(self.classes[i] if i < len(self.classes) else "?" for i in logp.argmax(1))})
        if not hyps:
            return None
        hyps.sort(key=lambda h: -h["quality"])
        best = hyps[0]
        best["alt"] = [(h["code"], round(h["quality"], 3)) for h in hyps[1:]]
        best["bar"] = bar
        return best

    def readSample(self, ctx, box, exclude=None, useBar=True):
        """Context crop + body box (upright or not) -> readBody with the orientation bar's prior."""
        body = bodyOf(ctx, box)
        bar = barSide(ctx, box, textLine(body)[1]) if useBar else 0
        return self.readBody(body, exclude, bar=bar)

    def read(self, crop, turns=(0, 180), exclude=None):
        """BGR part crop (its own frame) -> best reading over the candidate clockwise turns, or None.
        turnMargin = quality lead over the best reading in another turn."""
        reads = []
        for t in turns:
            ctx, box = bodyAndContext(turn(crop, t))
            if ctx is None:
                continue
            r = self.readSample(ctx, box, exclude)
            if r is not None:
                r["turn"] = t
                reads.append(r)
        if not reads:
            return None
        reads.sort(key=lambda r: -r["quality"])
        best = reads[0]
        best["turnMargin"] = float(best["quality"] - reads[1]["quality"]) if len(reads) > 1 else 99.0
        return best

    def labelQuality(self, body, label, exclude=None):
        """How well a known label fits a body crop (mean log-probability of its characters minus the segmentation cost)."""
        norms, cost, _ = charImages(body, len(label))
        if norms is None:
            return -np.inf
        logp = np.array([self.charLogProb(x, exclude) for x in norms])
        return float(np.mean([logp[i, self._ci(ch)] for i, ch in enumerate(label)]) - SEG_W * cost)


# ---------------------------------------------------------------- dataset access

def partKey(row):
    return f"{row['source']}/{row['ref']}"


def loadManifest(folder):
    path = os.path.join(folder, "manifest.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def loadSample(folder, row):
    """Manifest row -> (context crop, body box) or (None, None)."""
    ctx = cv2.imread(os.path.join(folder, row["image"]), cv2.IMREAD_GRAYSCALE)
    if ctx is None:
        return None, None
    box = tuple(int(v) for v in row["bodyBox"].split()) if row.get("bodyBox") else (0, 0, ctx.shape[1], ctx.shape[0])
    return ctx, box


def synthChars(folder, real, n):
    """Synthetic glyphs from <folder>/synth/manifest.csv -> [(img, label, 'synth:<label>')], n per eligible class."""
    path = os.path.join(folder, "synth", "manifest.csv")
    if n <= 0 or not os.path.exists(path):
        return []
    count = {}
    for _, ch, _ in real:
        count[ch] = count.get(ch, 0) + 1
    out, taken = [], {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ch = row["label"]
            if (SYNTH_FOR != "all" and count.get(ch, 0) >= SYNTH_RARE) or taken.get(ch, 0) >= n:
                continue
            img = cv2.imread(os.path.join(folder, "synth", row["image"]), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                out.append((img.astype(np.float32) / 255, ch, SYNTH_KEY + ch))
                taken[ch] = taken.get(ch, 0) + 1
    return out


def loadChars(folder):
    """-> [(normalised char, label char, manifest row)] for every usable code sample (status ok)."""
    out = []
    for row in loadManifest(folder):
        if row["status"] != "ok":
            continue
        ctx, box = loadSample(folder, row)
        if ctx is None:
            continue
        norms, _, _ = charImages(bodyOf(ctx, box), len(row["label"]))
        if norms is None:
            continue
        out.extend((x, ch, row) for x, ch in zip(norms, row["label"]))
    return out
