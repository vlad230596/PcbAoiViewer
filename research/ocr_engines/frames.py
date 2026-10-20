"""Multi-frame material for fusion: every part cut out of every source frame that sees its body whole.

    venvs/rapid3/Scripts/python.exe frames.py            -> fusion/offsets.json, fusion/<ref>/<frame>.jpg, fusion/<ref>/median.jpg

The stitcher does not store frame placements, so each source frame (sources/<x>_<y>.jpg, full resolution, same
scale and orientation as the panorama) is located in panorama.jpg by template matching: coarse at 1/4 scale, then
refined at full resolution on the central region (pure translation; its NCC is printed as a check).
A part crop maps output px -> panorama px by the same affine as alignBoard.cropParts; minus the frame offset it
maps into the frame. Crops are upright by the reference turn (the benchmark's "given" turn). median.jpg is the
per-pixel median over the frames (suppresses glare that moves between frames).
"""
import json
import math
import os

import cv2
import numpy as np

import common as C

OUT = os.path.join(C.HERE, "fusion")


def locate(pano, panoSmall, frame, k=4):
    fs = cv2.resize(frame, None, fx=1 / k, fy=1 / k, interpolation=cv2.INTER_AREA)
    r = cv2.matchTemplate(panoSmall, fs, cv2.TM_CCOEFF_NORMED)
    _, _, _, (x, y) = cv2.minMaxLoc(r)
    h, w = frame.shape[:2]
    cx0, cy0 = w // 4, h // 4
    tpl = frame[cy0:cy0 + h // 2, cx0:cx0 + w // 2]
    m = 2 * k + 4
    X, Y = x * k + cx0 - m, y * k + cy0 - m
    X0, Y0 = max(0, X), max(0, Y)
    win = pano[Y0:Y + tpl.shape[0] + 2 * m, X0:X + tpl.shape[1] + 2 * m]
    r = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
    _, ncc, _, (dx, dy) = cv2.minMaxLoc(r)
    return X0 + dx - cx0, Y0 + dy - cy0, float(ncc)


def cropTransforms(board, refs):
    """ref -> (W 2x3 output px -> panorama px, w, h): alignBoard.cropParts geometry."""
    M = np.array(board["align"]["M"], float)
    L, t = M[:, :2], M[:, 2]
    out = {}
    for ref in refs:
        c = board["components"][ref]
        pts = [C.flattenOutline(c["outline"])] if c.get("outline") else []
        for p in c["pins"]:
            x0, y0, x1, y1 = p["bb"]
            pts.append(np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
        Cc = np.array([c["x"], c["y"]])
        a = math.radians(c["rot"])
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        local = (np.vstack(pts) - Cc) @ R
        (u0, v0), (u1, v1) = local.min(0) - 0.5, local.max(0) + 0.5
        k = board["align"]["crops"][ref]["pxPerMm"]
        w, h = int(math.ceil((u1 - u0) * k)), int(math.ceil((v1 - v0) * k))
        W = np.hstack([L @ R @ np.diag([1 / k, -1 / k]), (L @ (Cc + R @ np.array([u0, v1])) + t)[:, None]])
        out[ref] = (W, w, h)
    return out


def main():
    mk, board = C.load()
    os.makedirs(OUT, exist_ok=True)
    pano = cv2.imread(os.path.join(C.PANO, "panorama.jpg"))
    panoSmall = cv2.resize(pano, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    offPath = os.path.join(OUT, "offsets.json")
    offsets = json.load(open(offPath)) if os.path.exists(offPath) else {}
    srcDir = os.path.join(C.PANO, "sources")
    frames = {}
    for f in sorted(os.listdir(srcDir)):
        name = f[:-4]
        img = cv2.imread(os.path.join(srcDir, f))
        frames[name] = img
        if name not in offsets:
            x, y, ncc = locate(pano, panoSmall, img)
            offsets[name] = [int(x), int(y), round(ncc, 3)]
            print(f"{name}: offset {x},{y} ncc {ncc:.3f}", flush=True)
    json.dump(offsets, open(offPath, "w"), indent=0)
    refs = C.refsWithLines(mk)
    T = cropTransforms(board, refs)
    counts = []
    for ref in refs:
        W, w, h = T[ref]
        x0, y0, x1, y1 = C.bodyBox(ref, board)               # the body must be inside the frame; pads may be cut
        corners = W @ np.array([[x0, x1, x0, x1], [y0, y0, y1, y1], [1, 1, 1, 1]], float)
        d = os.path.join(OUT, ref)
        os.makedirs(d, exist_ok=True)
        stack = []
        for name, (ox, oy, ncc) in offsets.items():
            fh, fw = frames[name].shape[:2]
            fx, fy = corners[0] - ox, corners[1] - oy
            if ncc < 0.8 or fx.min() < 2 or fy.min() < 2 or fx.max() > fw - 3 or fy.max() > fh - 3:
                continue
            Wf = W.copy(); Wf[:, 2] -= (ox, oy)
            crop = cv2.warpAffine(frames[name], Wf, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_REPLICATE)
            crop = C.turn(crop, mk[ref].get("turn", 0))
            cv2.imwrite(os.path.join(d, name + ".jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            stack.append(crop)
        if stack:
            cv2.imwrite(os.path.join(d, "median.jpg"), np.median(np.stack(stack), 0).astype(np.uint8),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        counts.append(len(stack))
    print(f"{len(refs)} parts, frames per part: min {min(counts)}, median {int(np.median(counts))}, max {max(counts)}")


if __name__ == "__main__":
    main()
