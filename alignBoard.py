"""CLI: align a board (importBoard.py -> boards/<name>/board.json) to a stitched panorama (stitchGrid.py -> out/<name>/).

    venv/Scripts/python.exe alignBoard.py out/captured_51.5 boards/Antilatency.SocketTester.Main-r79 [--side top]

Model: one affine M (2x3) from board mm (IPC-2581 Datum, Y up) to panorama px. The camera only translates, so a
similarity would do; the affine also absorbs the small px/mm anisotropy of the calibration (77.66 / 77.35).

Both sides are compared as edge maps (gradient magnitude of a blurred image), which needs no assumption about
colours: the board is rendered from board.json (pads of that side, holes, outline, component bodies), the panorama
is turned to grey. Steps:
  1. coarse: rotation 0..359 deg x mirror, render at ~4 px/mm, matchTemplate over the downscaled panorama;
  2. fine: every drilled hole (plated, non-plated, vias) looked for at full resolution near its predicted place
     (its rim matched as an edge ring), RANSAC affine; search 2 mm -> 0.3 -> 0.15 mm. Holes are at board level,
     so tall bodies (parallax) do not pull the fit; pads are not used, solder and leads hide their edges.
Writes <out>/board.json (the board data + "align": {M, residuals}) and <out>/datasheets/ for the viewer (index.html
shows a "Board (CAD)" layer when board.json is there), align_check.jpg and align_crops/ for a look by eye.
"""
import argparse
import json
import math
import os
import re
import shutil
import time

import cv2
import numpy as np

import BoardData


# ---------------------------------------------------------------- SVG path (M L A Z, absolute) -> polylines

def flattenPath(d, step=0.25):
    """SVG path in mm -> [np.array Nx2] closed rings. Arcs (rx = ry, no x-rotation) flattened every ~step mm."""
    rings, cur, pos = [], [], None
    for cmd, args in re.findall(r"([MLAZ])([^MLAZ]*)", d):
        v = [float(x) for x in args.replace(",", " ").split()]
        if cmd == "M":
            if cur:
                rings.append(np.array(cur))
            pos = (v[0], v[1]); cur = [pos]
        elif cmd == "L":
            pos = (v[0], v[1]); cur.append(pos)
        elif cmd == "A":
            r, large, sweep, x, y = v[0], int(v[3]), int(v[4]), v[5], v[6]
            cur += _arc(pos, (x, y), r, large, sweep, step)[1:]
            pos = (x, y)
        elif cmd == "Z":
            if cur:
                rings.append(np.array(cur))
            cur = []
    if cur:
        rings.append(np.array(cur))
    return rings


def _arc(p0, p1, r, large, sweep, step):
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = (x1 - x0) / 2, (y1 - y0) / 2
    d2 = dx * dx + dy * dy
    if d2 < 1e-12 or r < 1e-9:
        return [p0, p1]
    r = max(r, math.sqrt(d2))
    h = math.sqrt(max(0.0, r * r - d2))
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    nx, ny = -dy / math.sqrt(d2), dx / math.sqrt(d2)
    sign = 1 if large != sweep else -1   # sweep 1 = CCW in math space (Y up)
    cx, cy = mx + sign * h * nx, my + sign * h * ny
    a0, a1 = math.atan2(y0 - cy, x0 - cx), math.atan2(y1 - cy, x1 - cx)
    span = (a1 - a0) % (2 * math.pi) if sweep else -((a0 - a1) % (2 * math.pi))
    n = max(2, int(abs(span) * r / step) + 1)
    return [(cx + r * math.cos(a0 + span * i / n), cy + r * math.sin(a0 + span * i / n)) for i in range(n + 1)]


# ---------------------------------------------------------------- board features in mm

class BoardShapes:
    def __init__(self, board, side):
        self.side = side
        self.outline = flattenPath(board["board"]["outline"]) if board["board"]["outline"] else []
        self.holes = [(h[0], h[1], h[2]) for h in board["board"]["holes"] if h[3] != "v" or h[2] >= 0.5]
        self.pads, self.bodies = [], []
        for ref, c in board["components"].items():
            for p in c["pins"]:
                if p["side"] == side or p["side"] == "both":
                    x0, y0, x1, y1 = p["bb"]
                    self.pads.append(np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
            if c["side"] == side and c["fit"] in ("fitted", "optional") and c.get("outline"):
                self.bodies.extend(flattenPath(c["outline"]))

    def points(self):
        """Every vertex, for extents."""
        pts = [r for r in self.outline] + self.pads + self.bodies
        pts.append(np.array([(x, y) for x, y, _ in self.holes]) if self.holes else np.zeros((0, 2)))
        return np.vstack([p for p in pts if len(p)])


def applyM(M, pts):
    pts = np.asarray(pts, float)
    return pts @ M[:, :2].T + M[:, 2]


def render(shapes, M, size, bodies=True, outline=True):
    """Filled mask (float32 0..1) of the board features under M (mm -> px) on a canvas of size (w, h)."""
    w, h = size
    img = np.zeros((h, w), np.uint8)
    sh = 4  # sub-pixel precision for cv2 drawing
    k = 1 << sh
    scale = math.sqrt(abs(np.linalg.det(M[:, :2])))
    if outline:
        for r in shapes.outline:
            cv2.polylines(img, [np.round(applyM(M, r) * k).astype(np.int32)], True, 255, max(1, int(scale * 0.15)),
                          cv2.LINE_AA, sh)
    if bodies:
        for r in shapes.bodies:
            cv2.polylines(img, [np.round(applyM(M, r) * k).astype(np.int32)], True, 160, max(1, int(scale * 0.1)),
                          cv2.LINE_AA, sh)
    for p in shapes.pads:
        cv2.fillPoly(img, [np.round(applyM(M, p) * k).astype(np.int32)], 255, cv2.LINE_AA, sh)
    for x, y, d in shapes.holes:
        c = applyM(M, [(x, y)])[0]
        cv2.circle(img, (int(round(c[0] * k)), int(round(c[1] * k))), max(1, int(round(d / 2 * scale * k))), 90, -1,
                   cv2.LINE_AA, sh)
    return img.astype(np.float32) / 255.0


def edges(img, sigma):
    """Gradient magnitude of the blurred image, normalised by its local level so bright and dark areas weigh alike."""
    g = cv2.GaussianBlur(img, (0, 0), sigma)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = cv2.magnitude(gx, gy)
    local = cv2.GaussianBlur(m, (0, 0), sigma * 8) + 1e-3
    return m / local


def panoGrey(pano):
    lab = cv2.cvtColor(pano, cv2.COLOR_BGR2LAB)
    return lab[:, :, 0].astype(np.float32) / 255.0


def similarity(scale, angleDeg, mirror):
    """mm (Y up) -> px (Y down) linear part: scale * R(angle) * flipY * [mirror X]."""
    a = math.radians(angleDeg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    F = np.diag([-1.0 if mirror else 1.0, -1.0])
    return scale * R @ F


# ---------------------------------------------------------------- 1. coarse search

def coarse(shapes, grey, pxPerMm, workPxPerMm=4.0, log=print):
    f = workPxPerMm / pxPerMm
    small = cv2.resize(grey, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    E = edges(small, 1.2)
    # the staircase border of the covered area is a strong edge that is not on the board
    E[cv2.dilate((small < 0.02).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0] = 0
    pts = shapes.points()
    best = None
    t0 = time.time()
    for mirror in (False, True):
        for ang in range(0, 360, 2):
            L = similarity(workPxPerMm, ang, mirror)
            q = pts @ L.T
            mn, mx = q.min(0) - 4, q.max(0) + 4
            w, h = int(mx[0] - mn[0]) + 1, int(mx[1] - mn[1]) + 1
            if w >= E.shape[1] or h >= E.shape[0]:
                continue
            M = np.hstack([L, -mn[:, None]])
            T = edges(render(shapes, M, (w, h)), 1.2)
            r = cv2.matchTemplate(E, T, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(r)
            if best is None or score > best[0]:
                best = (score, ang, mirror, M, loc)
    score, ang, mirror, M, loc = best
    # refine the angle around the best in 0.25 deg steps
    for a in np.arange(ang - 2, ang + 2.01, 0.25):
        L = similarity(workPxPerMm, a, mirror)
        q = pts @ L.T
        mn, mx = q.min(0) - 4, q.max(0) + 4
        w, h = int(mx[0] - mn[0]) + 1, int(mx[1] - mn[1]) + 1
        Mi = np.hstack([L, -mn[:, None]])
        T = edges(render(shapes, Mi, (w, h)), 1.2)
        r = cv2.matchTemplate(E, T, cv2.TM_CCOEFF_NORMED)
        _, s, _, lc = cv2.minMaxLoc(r)
        if s > score:
            score, ang, M, loc = s, a, Mi, lc
    Mw = M.copy()
    Mw[:, 2] += loc
    Mfull = Mw / f
    log(f"coarse: angle {ang:.2f} deg, mirror {mirror}, score {score:.3f}, {time.time() - t0:.1f} s")
    return Mfull, {"angle": float(ang), "mirror": bool(mirror), "score": float(score)}


# ---------------------------------------------------------------- 2. fine: drilled holes

def _peak(r, loc):
    """Sub-pixel peak of a matchTemplate result by a parabola in x and y."""
    lx, ly = loc
    d = [0.0, 0.0]
    if 0 < lx < r.shape[1] - 1:
        lo, mid, hi = r[ly, lx - 1], r[ly, lx], r[ly, lx + 1]
        den = lo - 2 * mid + hi
        d[0] = 0.5 * (lo - hi) / den if den else 0.0
    if 0 < ly < r.shape[0] - 1:
        lo, mid, hi = r[ly - 1, lx], r[ly, lx], r[ly + 1, lx]
        den = lo - 2 * mid + hi
        d[1] = 0.5 * (lo - hi) / den if den else 0.0
    return lx + d[0], ly + d[1]


def findHole(grey, u, v, rPx, searchPx):
    """Centre of a drilled hole of radius rPx near (u, v): its rim as an edge ring. -> (x, y, score) or None."""
    H, W = grey.shape
    tr = int(rPx * 1.6) + 2
    half = tr + searchPx
    x0, y0 = int(round(u)) - half, int(round(v)) - half
    if x0 < 0 or y0 < 0 or x0 + 2 * half >= W or y0 + 2 * half >= H:
        return None
    T = np.zeros((2 * tr + 1, 2 * tr + 1), np.float32)
    cv2.circle(T, (tr * 16, tr * 16), int(round(rPx * 16)), 1.0, -1, cv2.LINE_AA, 4)
    sig = max(1.0, rPx * 0.08)
    r = cv2.matchTemplate(edges(grey[y0:y0 + 2 * half + 1, x0:x0 + 2 * half + 1], sig), edges(T, sig),
                          cv2.TM_CCOEFF_NORMED)
    _, s, _, loc = cv2.minMaxLoc(r)
    px, py = _peak(r, loc)
    return x0 + px + tr, y0 + py + tr, float(s)


FINE_STEPS = (  # (search mm, hole kinds, min drill mm): plated/non-plated first, vias once close
    (2.0, ("p", "n"), 0.5),
    (0.3, ("p", "n", "v"), 0.25),
    (0.15, ("p", "n", "v"), 0.25),
)


def fine(holes, grey, M, pxPerMm, steps=FINE_STEPS, minScore=0.3, log=print):
    """Holes are at board level (no parallax) and crisp: each is looked for near where M puts it, then a RANSAC
    affine over all of them; the search narrows pass by pass."""
    info = {}
    for it, (searchMm, kinds, minD) in enumerate(steps):
        src, dst = [], []
        for x, y, d, k, *_ in holes:
            if k not in kinds or d < minD:
                continue
            u, v = applyM(M, [(x, y)])[0]
            hit = findHole(grey, u, v, d / 2 * pxPerMm, int(searchMm * pxPerMm))
            if hit and hit[2] >= minScore:
                src.append((x, y)); dst.append(hit[:2])
        if len(src) < 6:
            raise RuntimeError(f"fine pass {it + 1}: only {len(src)} holes found; is the coarse result right?")
        src, dst = np.array(src, np.float32), np.array(dst, np.float32)
        A, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=max(2.0, 0.04 * pxPerMm),
                                      maxIters=5000, confidence=0.999, refineIters=20)
        inl = inl.ravel().astype(bool)
        res = np.linalg.norm(applyM(A, src[inl]) - dst[inl], axis=1)
        log(f"fine pass {it + 1} (search {searchMm} mm): {len(src)} holes found, {inl.sum()} inliers, "
            f"residual median {np.median(res):.2f} px, p90 {np.percentile(res, 90):.2f}, max {res.max():.2f} px")
        M = A
        info = {"holes": int(len(src)), "inliers": int(inl.sum()), "medianPx": float(np.median(res)),
                "p90Px": float(np.percentile(res, 90)), "maxPx": float(res.max()),
                "points": [[round(float(a), 4), round(float(b), 4), round(float(c), 2), round(float(d), 2), bool(k)]
                           for (a, b), (c, d), k in zip(src, dst, inl)]}
    return M, info


# ---------------------------------------------------------------- checks

def drawCheck(pano, shapes, M, path, maxSide=5000):
    f = min(1.0, maxSide / max(pano.shape[:2]))
    img = cv2.resize(pano, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    Mf = M * f
    k = 16
    for r in shapes.outline:
        cv2.polylines(img, [np.round(applyM(Mf, r) * k).astype(np.int32)], True, (255, 255, 0), 2, cv2.LINE_AA, 4)
    for p in shapes.pads:
        cv2.polylines(img, [np.round(applyM(Mf, p) * k).astype(np.int32)], True, (0, 255, 255), 1, cv2.LINE_AA, 4)
    for r in shapes.bodies:
        cv2.polylines(img, [np.round(applyM(Mf, r) * k).astype(np.int32)], True, (255, 0, 255), 1, cv2.LINE_AA, 4)
    for x, y, d in shapes.holes:
        c = applyM(Mf, [(x, y)])[0]
        cv2.circle(img, (int(c[0] * k), int(c[1] * k)), int(d / 2 * math.sqrt(abs(np.linalg.det(Mf[:, :2]))) * k),
                   (0, 128, 255), 1, cv2.LINE_AA, 4)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def drawCrops(pano, board, shapes, M, folder, refs, sizeMm=6.0):
    os.makedirs(folder, exist_ok=True)
    s = math.sqrt(abs(np.linalg.det(M[:, :2])))
    half = int(sizeMm / 2 * s)
    for ref in refs:
        c = board["components"].get(ref)
        if not c:
            continue
        cx, cy = (c["bb"][0] + c["bb"][2]) / 2, (c["bb"][1] + c["bb"][3]) / 2
        u, v = applyM(M, [(cx, cy)])[0]
        x0, y0 = int(u) - half, int(v) - half
        crop = pano[max(0, y0):y0 + 2 * half, max(0, x0):x0 + 2 * half].copy()
        Mc = M.copy()
        Mc[:, 2] -= (max(0, x0), max(0, y0))
        for p in shapes.pads:
            cv2.polylines(crop, [np.round(applyM(Mc, p) * 16).astype(np.int32)], True, (0, 255, 255), 1, cv2.LINE_AA, 4)
        if c.get("outline"):
            for r in flattenPath(c["outline"]):
                cv2.polylines(crop, [np.round(applyM(Mc, r) * 16).astype(np.int32)], True, (255, 0, 255), 2,
                              cv2.LINE_AA, 4)
        cv2.imwrite(os.path.join(folder, f"{ref}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])


def cropParts(pano, board, M, side, folder, marginMm=0.5, maxSide=1400):
    """Every component of the side as an upright image: the part's own frame (its rotation undone, top view),
    outline + pads + margin, at the panorama's px/mm (big connectors scaled down to maxSide).
    -> {ref: {file, w, h, pxPerMm}} with file relative to the panorama folder."""
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    L, t = M[:, :2], M[:, 2]
    native = math.sqrt(abs(np.linalg.det(L)))
    crops = {}
    for ref, c in board["components"].items():
        if c["side"] != side:
            continue
        pts = [r for r in flattenPath(c["outline"])] if c.get("outline") else []
        for p in c["pins"]:
            x0, y0, x1, y1 = p["bb"]
            pts.append(np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
        if not pts:
            x0, y0, x1, y1 = c["bb"]
            pts.append(np.array([(x0, y0), (x1, y1)]))
        C = np.array([c["x"], c["y"]])
        a = math.radians(c["rot"])
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        local = (np.vstack(pts) - C) @ R            # board -> part frame (rotation undone)
        (u0, v0), (u1, v1) = local.min(0) - marginMm, local.max(0) + marginMm
        k = min(native, maxSide / max(u1 - u0, v1 - v0))
        w, h = int(math.ceil((u1 - u0) * k)), int(math.ceil((v1 - v0) * k))
        # output px (i, j) -> part frame (u0 + i/k, v1 - j/k) -> board C + R (u, v) -> panorama M
        W = np.hstack([L @ R @ np.diag([1 / k, -1 / k]), (L @ (C + R @ np.array([u0, v1])) + t)[:, None]])
        img = cv2.warpAffine(pano, W, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT)
        name = re.sub(r"[^\w.+-]", "_", ref) + ".jpg"
        cv2.imwrite(os.path.join(folder, name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        crops[ref] = {"file": f"{os.path.basename(folder)}/{name}", "w": w, "h": h, "pxPerMm": round(k, 3)}
    return crops


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("panorama", help="stitchGrid output folder (panorama.jpg, report.json)")
    ap.add_argument("board", help="boards/<name> folder or its board.json")
    ap.add_argument("--side", choices=["top", "bottom"], default="top", help="board side seen in the panorama")
    ap.add_argument("--image", default="panorama.jpg", help="image in the panorama folder (focus.jpg for a stack)")
    ap.add_argument("--px-per-mm", type=float, help="override the calibration from report.json")
    ap.add_argument("--crops", default="U1,U28,R103,C73,X1,U29,J28", help="components to crop for a visual check")
    a = ap.parse_args()

    t0 = time.time()
    board = BoardData.loadBoard(a.board)
    pano = cv2.imread(os.path.join(a.panorama, a.image), cv2.IMREAD_COLOR)
    if pano is None:
        raise SystemExit(f"cannot read {os.path.join(a.panorama, a.image)}")
    pxPerMm = a.px_per_mm
    if pxPerMm is None:
        rep = json.load(open(os.path.join(a.panorama, "report.json"), encoding="utf-8"))
        cal = rep["calibration"]   # full-resolution px/mm; the image may be a scaled panorama
        pxPerMm = (cal["pxPerMmX"] + cal["pxPerMmY"]) / 2 * pano.shape[1] / rep["panoramaSizeFullResPx"][0]
    print(f"panorama {pano.shape[1]} x {pano.shape[0]} px, {pxPerMm:.2f} px/mm")
    shapes = BoardShapes(board, a.side)
    grey = panoGrey(pano)
    M, co = coarse(shapes, grey, pxPerMm)
    M, fi = fine(board["board"]["holes"], grey, M, pxPerMm)
    L = M[:, :2]
    sx, sy = np.linalg.norm(L[:, 0]), np.linalg.norm(L[:, 1])
    angle = math.degrees(math.atan2(L[1, 0], L[0, 0]))
    skew = math.degrees(math.acos(abs(float(L[:, 0] @ L[:, 1]) / (sx * sy)))) if sx and sy else 0
    print(f"M: {sx:.3f} / {sy:.3f} px/mm, angle {angle:.2f} deg, axes {skew:.2f} deg apart, "
          f"{'MIRRORED' if np.linalg.det(L) > 0 else 'not mirrored'}")   # mm Y up -> px Y down: det < 0 is a plain view
    um = 1000 / ((sx + sy) / 2)
    print(f"residual median {fi['medianPx'] * um:.0f} um, p90 {fi['p90Px'] * um:.0f} um")

    out = dict(board)
    out["align"] = {"side": a.side, "image": a.image, "M": M.tolist(), "pxPerMm": [sx, sy], "angleDeg": angle,
                    "axesDeg": skew, "coarse": co, "fine": {k: v for k, v in fi.items() if k != "points"},
                    "points": fi["points"], "source": os.path.abspath(a.board)}
    t1 = time.time()
    out["align"]["crops"] = cropParts(pano, board, M, a.side, os.path.join(a.panorama, "parts"))
    print(f"{len(out['align']['crops'])} part crops in parts/, {time.time() - t1:.1f} s")
    with open(os.path.join(a.panorama, "board.json"), "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, separators=(",", ":"))
    # the viewer links the datasheet copies by their board-relative paths: put them next to it
    boardDir = a.board if os.path.isdir(a.board) else os.path.dirname(a.board)
    if os.path.isdir(os.path.join(boardDir, "datasheets")):
        shutil.copytree(os.path.join(boardDir, "datasheets"), os.path.join(a.panorama, "datasheets"), dirs_exist_ok=True)
    drawCheck(pano, shapes, M, os.path.join(a.panorama, "align_check.jpg"))
    drawCrops(pano, board, shapes, M, os.path.join(a.panorama, "align_crops"), [r for r in a.crops.split(",") if r])
    print(f"wrote {os.path.join(a.panorama, 'board.json')}, align_check.jpg, align_crops/ in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
