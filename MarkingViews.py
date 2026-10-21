"""Every source view of a part top, for reading its marking.

Laser marking on IC bodies shows only at some angles to the light: one frame shows "K177" clearly where its
neighbour (the one the panorama took) shows a blank body. A part is seen by 2-4 neighbouring frames (grid step 10 mm,
frame ~25 x 14 mm) at every height of the stack, so instead of the panorama, the body is cut out of each source frame:

    part frame (CAD outline, rotation undone, ~77.5 px/mm) -> panorama px (board.json align M)
    -> reference frame px (frame position, ortho magnification D/(D-h) about the frame centre at the CAD height h)
    -> layer frame px (one similarity per height, stack report: scale + centre shift)

then registered onto the all-in-focus panorama (ECC, affine, over the frame's coverage) so all views are pixel-aligned.
Per frame the sharpest height is kept (Laplacian energy over the body); which frame shows the marking is left to
the reader (see ranked).

Geometry inputs come from a stackFocus.py output folder: report.json (layers, camera distance), index.html (the
reference frame positions the viewer embeds), focus.jpg (registration template). The board comes from an
alignBoard.py panorama folder of the same reference stitch (board.json with align.M).
"""
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from alignBoard import flattenPath

GROUPS = ("ICs", "Transistors", "Diodes & protection", "Crystals & oscillators")
SUBGROUPS = ("polarized", "chokes")          # tantalum capacitors, power inductors: marked too


def safeName(ref):
    return re.sub(r"[^\w.+-]", "_", ref)


def viewerData(stackDir):
    """The data block BuildViewer embeds in index.html (frame names, paths, canvas positions)."""
    html = open(os.path.join(stackDir, "index.html"), encoding="utf-8").read()
    i = html.index('"frames": [{')
    start = html.rfind('{"width"', 0, i)
    data, _ = json.JSONDecoder().raw_decode(html[start:])
    return data


class Target:
    def __init__(self, ref, comp, W, w, h, inner):
        self.ref = ref
        self.comp = comp
        self.W = W                # 2x3: part px -> panorama px
        self.w, self.h = w, h
        self.inner = inner        # bool mask: body interior (coverage is measured over it)
        self.corners = (W[:, :2] @ np.array([[0, 0], [w, 0], [0, h], [w, h]]).T + W[:, 2:]).T   # panorama px
        self.template = None      # all-in-focus crop (registration target)
        self.pano = None          # reference panorama crop (what the panorama shows)
        self.views = {}           # frame -> view at the sharpest height (dict)


class MarkingViews:
    def __init__(self, board, stackDir, marginMm=0.3, minCover=0.5, threads=8, log=print):
        self.board = board
        self.stackDir = stackDir
        self.marginMm = marginMm
        self.minCover = minCover
        self.threads = threads
        self.log = log
        rep = json.load(open(os.path.join(stackDir, "report.json"), encoding="utf-8"))["stack"]
        self.refZ = rep["refZ"]
        self.distance = rep["cameraDistanceMm"]
        self.layers = [(l["z"], l["folder"], l["scale"], np.array(l["centreShiftPx"], np.float64))
                       for l in rep["layers"] if l["ok"]]
        data = viewerData(stackDir)
        if data.get("scale", 1.0) != 1.0:
            raise ValueError("the stack must be built at --scale 1 (frame positions are in full-resolution px)")
        self.frameSize = (data["frameW"], data["frameH"])
        self.frames = {f["name"]: np.array([f["px"], f["py"]], np.float64) for f in data["frames"]}
        self.M = np.array(board["align"]["M"], np.float64)
        self.k = math.sqrt(abs(np.linalg.det(self.M[:, :2])))
        self.timings = {}

    # ------------------------------------------------------------------ targets
    def defaultRefs(self):
        out = []
        for ref, c in self.board["components"].items():
            line = self.board["lines"][c["line"]] if c.get("line") is not None else None
            if c["side"] != self.board["align"]["side"] or c.get("fit") != "fitted" or not line:
                continue
            if line["cat"]["group"] in GROUPS or line["cat"].get("sub") in SUBGROUPS:
                out.append(ref)
        return out

    def target(self, ref):
        """Upright crop of the body (CAD outline + margin; pads when there is no outline) and its interior."""
        c = self.board["components"][ref]
        pts = [r for r in flattenPath(c["outline"])] if c.get("outline") else []
        if not pts:
            for p in c["pins"]:
                x0, y0, x1, y1 = p["bb"]
                pts.append(np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
        C = np.array([c["x"], c["y"]])
        a = math.radians(c["rot"])
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        local = (np.vstack(pts) - C) @ R
        (u0, v0), (u1, v1) = local.min(0) - self.marginMm, local.max(0) + self.marginMm
        k = self.k
        w, h = int(math.ceil((u1 - u0) * k)), int(math.ceil((v1 - v0) * k))
        L, t = self.M[:, :2], self.M[:, 2]
        # output px (i, j) -> part frame (u0 + i/k, v1 - j/k) -> board C + R (u, v) -> panorama M (as alignBoard)
        W = np.hstack([L @ R @ np.diag([1 / k, -1 / k]), (L @ (C + R @ np.array([u0, v1])) + t)[:, None]])
        inner = np.zeros((h, w), bool)
        m = int(round((self.marginMm + 0.15) * k))
        inner[m:h - m, m:w - m] = True
        return Target(ref, c, W, w, h, inner)

    # ------------------------------------------------------------------ geometry
    def toLayerFrame(self, t, pos, z, scale, shift):
        """2x3: part px -> px of the frame at canvas position pos, height z (ortho at the part's CAD height)."""
        w, h = self.frameSize
        c = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
        Dz = self.distance + (z - self.refZ)
        hp = t.comp.get("height") or 1.0
        m = Dz / (Dz - min(hp, 0.5 * Dz))
        A = t.W[:, :2] * m / scale
        b = c + ((t.W[:, 2] - pos - c) * m - shift) / scale
        return np.hstack([A, b[:, None]])

    def covers(self, t, pos):
        d = t.corners - pos
        w, h = self.frameSize
        return d[:, 0].max() > 0 and d[:, 1].max() > 0 and d[:, 0].min() < w and d[:, 1].min() < h

    # ------------------------------------------------------------------ measures
    @staticmethod
    def grey(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def focusScore(self, img, mask):
        g = cv2.GaussianBlur(self.grey(img), (0, 0), 1.0)
        v = np.abs(cv2.Laplacian(g, cv2.CV_32F)[mask])
        return float(v.mean()) if v.size else 0.0

    def register(self, img, valid, template):
        """ECC (affine) of a view onto the all-in-focus crop over the view's coverage. -> (img, valid) or None."""
        g0 = cv2.GaussianBlur(self.grey(template), (0, 0), 1.5)
        g1 = cv2.GaussianBlur(self.grey(img), (0, 0), 1.5)
        mask = cv2.erode(valid, np.ones((9, 9), np.uint8))
        Mw = np.eye(2, 3, dtype=np.float32)
        try:
            _, Mw = cv2.findTransformECC(g0, g1, Mw, cv2.MOTION_AFFINE,
                                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), mask, 5)
        except cv2.error:
            return None
        if np.hypot(Mw[0, 2], Mw[1, 2]) > 25 or np.abs(Mw[:, :2] - np.eye(2)).max() > 0.05:
            return None
        h, w = template.shape[:2]
        f = cv2.WARP_INVERSE_MAP
        return (cv2.warpAffine(img, Mw, (w, h), flags=cv2.INTER_LINEAR | f, borderMode=cv2.BORDER_REPLICATE),
                cv2.warpAffine(mask, Mw, (w, h), flags=cv2.INTER_NEAREST | f))

    # ------------------------------------------------------------------ run
    def collect(self, refs, focusImg, panoImg=None):
        t0 = time.time()
        targets = [self.target(r) for r in refs]
        for t in targets:
            f = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
            t.template = cv2.warpAffine(focusImg, t.W, (t.w, t.h), flags=f)
            if panoImg is not None:
                t.pano = cv2.warpAffine(panoImg, t.W, (t.w, t.h), flags=f)
        fw, fh = self.frameSize
        ones = np.ones((fh, fw), np.uint8)
        for z, folder, scale, shift in self.layers:
            def oneFrame(name):
                pos = self.frames[name]
                need = [t for t in targets if self.covers(t, pos)]
                if not need:
                    return []
                img = cv2.imread(os.path.join(folder, name + ".png"), cv2.IMREAD_COLOR)
                if img is None:
                    return []
                out = []
                for t in need:
                    A = self.toLayerFrame(t, pos, z, scale, shift)
                    f = cv2.WARP_INVERSE_MAP
                    crop = cv2.warpAffine(img, A, (t.w, t.h), flags=cv2.INTER_CUBIC | f)
                    valid = cv2.warpAffine(ones, A, (t.w, t.h), flags=cv2.INTER_NEAREST | f, borderValue=0)
                    if (valid[t.inner] > 0).mean() < self.minCover:
                        continue
                    reg = self.register(crop, valid, t.template)
                    if reg is None:
                        continue
                    crop, valid = reg
                    held = t.inner & (valid > 0)
                    if held.sum() < self.minCover * t.inner.sum():
                        continue
                    out.append((t, {"frame": name, "z": z, "focus": self.focusScore(crop, held),
                                    "cover": float(held.sum() / t.inner.sum()), "img": crop, "valid": valid}))
                return out

            with ThreadPoolExecutor(self.threads) as pool:
                for res in pool.map(oneFrame, list(self.frames)):
                    for t, v in res:
                        old = t.views.get(v["frame"])
                        if old is None or v["focus"] > old["focus"]:
                            t.views[v["frame"]] = v
        self.timings["collect"] = time.time() - t0
        n = sum(len(t.views) for t in targets)
        self.log(f"{len(targets)} parts, {n} views ({n / max(1, len(targets)):.1f} per part), {time.time() - t0:.1f} s")
        return targets

    @staticmethod
    def ranked(t):
        """Most of the body first. No ranking by image measures: scratches and glare outweigh the faint marking
        (band-pass, top-hat and DoG measures all put the readable view last on U17, Q2, U15), so every view
        goes to the reader and the readings vote."""
        return sorted(t.views.values(), key=lambda v: (-round(v["cover"], 1), -v["focus"]))

    def write(self, targets, folder):
        """marking_views/<ref>/<frame>.jpg (registered view, sharpest height) + pano.jpg; -> {ref: record}."""
        os.makedirs(folder, exist_ok=True)
        out = {}
        for t in targets:
            d = os.path.join(folder, safeName(t.ref))
            os.makedirs(d, exist_ok=True)
            for f in os.listdir(d):
                os.remove(os.path.join(d, f))
            rec = {"w": t.w, "h": t.h, "pxPerMm": round(self.k, 3), "views": []}
            if t.pano is not None:
                cv2.imwrite(os.path.join(d, "pano.jpg"), t.pano, [cv2.IMWRITE_JPEG_QUALITY, 92])
                rec["pano"] = f"{os.path.basename(folder)}/{safeName(t.ref)}/pano.jpg"
            for v in self.ranked(t):
                img = np.where(v["valid"][..., None] > 0, v["img"], 0)
                name = f"{v['frame']}.jpg"
                cv2.imwrite(os.path.join(d, name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                rec["views"].append({"frame": v["frame"], "z": v["z"], "focus": round(v["focus"], 2),
                                     "cover": round(v["cover"], 2),
                                     "file": f"{os.path.basename(folder)}/{safeName(t.ref)}/{name}"})
            out[t.ref] = rec
        return out
