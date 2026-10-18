"""Focus stacking over grids captured at several camera heights Z.

The camera moves up by dz -> the plane in focus moves up by dz and the image shrinks about (nearly) the frame
centre. Between heights the machine XY is repeatable to <1 px, so ONE similarity per height (scale + shift,
estimated from AKAZE matches of the same-named frames) maps every frame of that height onto the reference height.
All heights are then composited with the reference geometry (positions, seams, parallax regions), so the layer
panoramas are pixel-aligned on the board plane and can be stacked:

    reference GridStitcher (full pipeline) -> per height: load, flat field, warp to the reference frame geometry,
    composite -> focus measure per layer (Laplacian energy, cells of cellPx) -> depth = Z of best focus
    (sub-layer parabola) -> board plane fit -> height above the board, holes filled
    -> all-in-focus panorama (per pixel: layers near the local depth).

Layers whose frames do not match the reference (another XY offset, e.g. captured_45..49.5 of SocketTester2) are
dropped by the inlier test.
"""
import copy
import glob
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from GridStitcher import GridStitcher

LAYER_RE = re.compile(r'^(.*?)(-?\d+(?:\.\d+)?)$')


def findLayers(pattern):
    """Folders matching a glob like 'SocketTester2/captured_*', sorted by the trailing number (Z, mm)."""
    out = []
    for d in glob.glob(pattern):
        m = LAYER_RE.match(os.path.basename(os.path.normpath(d)))
        if os.path.isdir(d) and m:
            out.append((float(m.group(2)), d))
    return sorted(out)


class Layer:
    def __init__(self, z, folder):
        self.z = z
        self.folder = folder
        self.M = None             # 2x3 work px: layer frame -> reference frame
        self.scale = None         # image scale relative to the reference (>1 above it)
        self.shift = None         # full-res px, shift of the frame centre
        self.inliers = 0
        self.spread = None        # full-res px, disagreement of the per-frame estimates
        self.pano = None          # uint8 BGR, reference canvas
        self.focus = None         # float32 log focus measure per cell
        self.ok = False
        self.reason = ''


class FocusStack:
    def __init__(self, ref, refZ, cellPx=8.0, marginPx=40.0, focusSigmaPx=12.0, minInliers=300,
                 confMin=0.6, depthSigmaMm=0.3, focusFloor=4.0, log=print):
        self.ref = ref                    # GridStitcher, optimized + layoutCanvas (+ detectParallax) done
        self.refZ = refZ
        s = ref.scale
        self.factor = max(1, int(round(cellPx * s)))     # work px per depth cell
        self.margin = marginPx * s       # frames scaled down at lower Z get empty borders
        self.focusSigma = focusSigmaPx * s / self.factor  # in cells
        self.minInliers = minInliers
        self.confMin = confMin            # min (log) ratio of the focus peak over the median layer
        self.focusFloor = focusFloor      # min ratio of the focus peak over the textureless floor of the panorama
        self.spikeMinMm = 0.6             # raised blobs above this height ...
        self.spikeMinAreaMm2 = 0.5        # ... and smaller than this are measurement spikes (0603 is 1.3 mm2)
        self.depthSigma = depthSigmaMm
        self.log = log
        self.layers = []
        self.timings = {}
        self.names = [f.name for f in ref.frames]
        self.measureFactor = min(1.0, 0.5 / s)
        self.akaze = cv2.AKAZE_create(threshold=0.0005)
        self.refFeatures = {}

    # ---------------------------------------------------------- registration
    def features(self, gray):
        g = gray if self.measureFactor == 1.0 else cv2.resize(gray, None, fx=self.measureFactor,
                                                              fy=self.measureFactor, interpolation=cv2.INTER_AREA)
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
        kp, desc = self.akaze.detectAndCompute(g, None)
        return np.array([k.pt for k in kp], dtype=np.float32).reshape(-1, 2), desc

    def sampleFrames(self, count=12):
        """Frames in the middle of the grid (board, not the mat), evenly spread."""
        P = np.array([f.position for f in self.ref.frames])
        c = np.median(P, axis=0)
        d = np.linalg.norm(P - c, axis=1)
        inner = np.argsort(d)[:max(count, len(P) // 2)]
        return sorted(inner[np.linspace(0, len(inner) - 1, count).astype(int)])

    def register(self, layer, st):
        """One similarity for the whole layer: median of per-frame estimates (AKAZE + RANSAC)."""
        sample = self.sampleFrames()
        for k in sample:
            if k not in self.refFeatures:
                self.refFeatures[k] = self.features(self.ref.frames[k].gray)
        byName = {f.name: f for f in st.frames}

        def one(k):
            fr = byName.get(self.names[k])
            if fr is None:
                return None
            p, d = self.features(fr.gray)
            pr, dr = self.refFeatures[k]
            if d is None or dr is None or len(d) < 20 or len(dr) < 20:
                return None
            knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d, dr, k=2)
            good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
            if len(good) < 20:
                return None
            M, inl = cv2.estimateAffinePartial2D(p[[g.queryIdx for g in good]], pr[[g.trainIdx for g in good]],
                                                 ransacReprojThreshold=2.0)
            if M is None:
                return None
            return M, int(inl.sum())

        with ThreadPoolExecutor(self.ref.threads) as pool:
            res = [r for r in pool.map(one, sample) if r is not None and r[1] >= self.minInliers]
        if len(res) < max(3, len(sample) // 3):
            layer.reason = f'only {len(res)}/{len(sample)} frames match the reference'
            return False
        P = np.array([[M[0, 0], M[1, 0], M[0, 2], M[1, 2]] for M, _ in res])   # a, b, tx, ty (measure px)
        a, b, tx, ty = np.median(P, axis=0)
        f = self.measureFactor
        w, h = self.ref.frameSize
        c = np.array([w, h]) * f / 2.0
        L = np.array([[a, -b], [b, a]])
        # per-frame disagreement: where each estimate puts the frame corners
        corners = np.array([[0, 0], [w * f, 0], [0, h * f], [w * f, h * f]])
        spread = max(np.abs(corners @ np.array([[p[0], p[1]], [-p[1], p[0]]]) + p[2:] -
                            (corners @ L.T + [tx, ty])).max() for p in P)
        layer.M = np.hstack([L, np.array([[tx], [ty]]) / f])     # work px
        layer.scale = float(math.hypot(a, b))
        layer.shift = ((L @ c + [tx, ty] - c) / f / self.ref.scale).tolist()
        layer.inliers = int(np.median([n for _, n in res]))
        layer.spread = float(spread / f / self.ref.scale)
        return True

    # ---------------------------------------------------------- layers
    def layerStitcher(self, folder):
        st = GridStitcher(scale=self.ref.scale, flatField=self.ref.flatField, threads=self.ref.threads,
                          log=lambda *a: None)
        st.loadFolder(folder)
        st.loadImages()
        st.computeFlatField()
        return st

    def layerFrames(self, layer, gray=False):
        """Frames of one height warped into the reference frame geometry (same order as the reference),
        or None if the height does not register."""
        ref = self.ref
        if layer.z == self.refZ:
            layer.M = np.float64([[1, 0, 0], [0, 1, 0]])
            layer.scale, layer.shift, layer.spread, layer.inliers = 1.0, [0.0, 0.0], 0.0, 0
            return [copy.copy(f) for f in ref.frames]
        st = self.layerStitcher(layer.folder)
        if layer.M is None and not self.register(layer, st):
            return None
        byName = {f.name: f for f in st.frames}
        missing = [n for n in self.names if n not in byName]
        if missing:
            layer.reason = f'{len(missing)} frames missing'
            return None
        w, h = ref.frameSize
        gains = ref.gains.astype(np.float32) if ref.gains is not None else None

        def warp(k):
            fr = byName[self.names[k]]
            img = cv2.warpAffine(fr.image, layer.M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            if gains is not None:
                img = np.clip(img.astype(np.float32) * gains[k] + 0.5, 0, 255).astype(np.uint8)
            fr.image = img
            fr.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if gray else None
            fr.position = ref.frames[k].position.copy()
            return fr

        with ThreadPoolExecutor(ref.threads) as pool:
            return list(pool.map(warp, range(len(self.names))))

    def detectParallax(self, layers):
        """Parallax regions of the reference, measured on the reference AND on the given heights, where the tops
        of tall components are sharp (on the board-focused layer they are blurred and give no shift)."""
        t0 = time.time()
        sets, zs = [], []
        for z, folder in layers:
            if z == self.refZ:
                continue
            frames = self.layerFrames(Layer(z, folder), gray=True)
            if frames is not None:
                sets.append([f.gray for f in frames])
                zs.append(z)
        self.ref.detectParallax(graySets=sets)
        self.parallaxZ = [self.refZ] + zs
        self.timings['parallax'] = time.time() - t0

    def buildLayer(self, layer, heights=None):
        """Composite one height onto the reference canvas. Frames are warped into the reference frame geometry,
        so positions, seams and parallax regions of the reference apply unchanged."""
        t0 = time.time()
        frames = self.layerFrames(layer)
        if frames is None:
            return False
        st = copy.copy(self.ref)
        st.frames = frames
        st.timings = {}
        if heights is not None:
            # ortho correction; the layer frames were scaled to the reference, so a point at height h is
            # magnified by (D + dz) / (D + dz - h) relative to the board
            st.override = None
            st.heightMap, st.heightFactor = heights, self.factor
            st.orthoDistance = self.distance + (layer.z - self.refZ)
        layer.pano = None
        layer.pano = st.composite(blend='multiband', band=12.0, margin=self.margin)
        layer.focus = self.focusMap(layer.pano)
        layer.ok = True
        self.timings[f'layer_{layer.z}'] = self.timings.get(f'layer_{layer.z}', 0) + time.time() - t0
        self.log(f'Z {layer.z}: scale {layer.scale:.4f}, centre shift {np.round(layer.shift, 1).tolist()} px, '
                 f'spread {layer.spread:.1f} px, {time.time() - t0:.1f} s')
        return True

    def focusMap(self, pano, bandCells=64):
        """log of the Laplacian energy, averaged over cells of factor x factor work px and blurred."""
        f = self.factor
        H, W = pano.shape[:2]
        hs, ws = int(math.ceil(H / f)), int(math.ceil(W / f))
        out = np.zeros((hs, ws), np.float32)
        pad = 8
        sigma = max(0.6, 1.0 * self.ref.scale)

        def run(r0):
            r1 = min(hs, r0 + bandCells)
            y0, y1 = r0 * f, min(H, r1 * f)
            ya, yb = max(0, y0 - pad), min(H, y1 + pad)
            g = cv2.cvtColor(pano[ya:yb], cv2.COLOR_BGR2GRAY).astype(np.float32)
            g = cv2.GaussianBlur(g, (0, 0), sigma)
            e = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
            e *= e
            e = e[y0 - ya:y1 - ya]
            full = np.zeros(((r1 - r0) * f, ws * f), np.float32)
            full[:e.shape[0], :W] = e
            out[r0:r1] = full.reshape(r1 - r0, f, ws, f).mean(axis=(1, 3))

        with ThreadPoolExecutor(self.ref.threads) as pool:
            list(pool.map(run, range(0, hs, bandCells)))
        out = cv2.GaussianBlur(out, (0, 0), self.focusSigma)
        return np.log(out + 1e-3)

    def run(self, layers, orthoPasses=1):
        """layers: list of (z, folder). Pass 1 composites every layer that registers onto the reference with the
        plain geometry and measures the depth. Every further pass re-composites ortho-corrected with the heights
        of the previous pass (tall tops then agree between frames and sit at their true XY) and re-measures."""
        self.layers = [Layer(z, folder) for z, folder in layers]
        for layer in self.layers:
            if not self.buildLayer(layer):
                self.log(f'Z {layer.z}: skipped, {layer.reason}')
        self.used = [l for l in self.layers if l.ok]
        self.zs = np.array([l.z for l in self.used])
        self.estimateDistance()
        self.log(f'{len(self.used)} layers stacked: Z {self.zs.min()}..{self.zs.max()}, '
                 f'camera {self.distance:.1f} mm above the board')
        self.depth()
        for p in range(orthoPasses):
            heights = self.orthoHeights()
            self.log(f'ortho pass {p + 1}')
            for layer in self.used:
                self.buildLayer(layer, heights)
            self.orthoDone = True
            self.depth()
        self.orthoPasses = orthoPasses

    def estimateDistance(self):
        """Projection centre to board distance D (mm): moving the camera up by dz scales the image by D / (D + dz),
        i.e. the registration scale is s = 1 + dz / D."""
        dz = self.zs - self.refZ
        ds = np.array([l.scale for l in self.used]) - 1.0
        self.distance = float((dz * dz).sum() / (dz * ds).sum())
        self.distanceFitResidual = float(np.abs(ds - dz / self.distance).max())

    # ---------------------------------------------------------- depth
    def peakFit(self, L, half=2):
        """Z of the focus peak per cell: least-squares parabola on log focus over the argmax layer and up to
        `half` layers on each side (the focus curve is ~2 mm wide, three points alone are noisy)."""
        K = len(self.used)
        zs = self.zs
        k = np.argmax(L, axis=0)
        lo = np.clip(k - half, 0, max(0, K - 1 - 2 * half))
        zc = zs[k]
        Sx = [np.zeros(k.shape) for _ in range(5)]
        Sy = [np.zeros(k.shape) for _ in range(3)]
        for o in range(min(K, 2 * half + 1)):
            idx = np.clip(lo + o, 0, K - 1)
            x = zs[idx] - zc
            y = np.take_along_axis(L, idx[None], 0)[0].astype(np.float64)
            for p in range(5):
                Sx[p] += x ** p
            for p in range(3):
                Sy[p] += y * x ** p
        # normal equations of y = a x^2 + b x + c
        M = np.stack([np.stack([Sx[4], Sx[3], Sx[2]], -1), np.stack([Sx[3], Sx[2], Sx[1]], -1),
                      np.stack([Sx[2], Sx[1], Sx[0]], -1)], -2)
        M[..., 0, 0] += 1e-9
        sol = np.linalg.solve(M, np.stack([Sy[2], Sy[1], Sy[0]], -1)[..., None])[..., 0]
        a, b = sol[..., 0], sol[..., 1]
        with np.errstate(divide='ignore', invalid='ignore'):
            zv = zc - b / (2 * a)
        step = float(np.median(np.diff(zs))) if K > 1 else 1.0
        ok = (a < 0) & np.isfinite(zv)
        zv = np.where(ok, np.clip(zv, zc - step, zc + step), zc)
        return k, np.clip(zv, zs[0], zs[-1]).astype(np.float32)

    def cellLabels(self):
        """Frame whose pixels the panorama uses at every depth cell (seams of the last composite)."""
        ref = self.ref
        saved = ref.override
        if getattr(self, 'orthoDone', False):
            ref.override = None
        try:
            return ref.labelMap(self.factor, margin=self.margin)
        finally:
            ref.override = saved

    def boardSurface(self, zmap, sel, lab, plane, sigmaCells=20.0, samples=200000):
        """Board Z per cell = B(x, y) + F(u, v) (+ smoothed local residual): B is a cubic board warp over the
        panorama, F the tilted focus field of the camera in frame coordinates, shared by all frames."""
        hs, ws = zmap.shape
        f = self.factor
        w, h = self.ref.frameSize
        P = np.array([fr.position for fr in self.ref.frames], np.float32)
        L = np.where(lab >= 0, lab, 0)
        yy, xx = np.mgrid[0:hs, 0:ws].astype(np.float32)
        u = ((xx + 0.5) * f - P[L, 0] - w / 2.0) / w
        v = ((yy + 0.5) * f - P[L, 1] - h / 2.0) / w
        X, Y = (xx - ws / 2.0) / ws, (yy - hs / 2.0) / ws

        def design(x, y, uu, vv):
            return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y, x ** 3, x * x * y, x * y * y, y ** 3,
                             uu, vv, uu * uu, uu * vv, vv * vv], -1)

        cand = sel & (lab >= 0) & (zmap > plane - 0.7) & (zmap < plane + 0.8)
        idx = np.flatnonzero(cand)
        if len(idx) < 1000:
            self.boardFrames = None
            return plane.copy()
        idx = np.random.default_rng(0).choice(idx, min(samples, len(idx)), replace=False)
        A = design(X.flat[idx], Y.flat[idx], u.flat[idx], v.flat[idx])
        y = zmap.flat[idx]
        keep = np.ones(len(y), bool)
        for tol in (0.5, 0.3, 0.25, 0.2):
            c = np.linalg.lstsq(A[keep], y[keep], rcond=None)[0]
            keep = np.abs(y - A @ c) < tol
        model = (design(X, Y, u, v) @ c).astype(np.float32)
        model = np.where(lab >= 0, model, plane)
        # local residual (warp finer than the cubic), smooth in canvas coordinates
        board = (cand & (np.abs(zmap - model) < 0.25)).astype(np.float32)
        num = cv2.GaussianBlur(np.where(board > 0, zmap - model, 0).astype(np.float32), (0, 0), sigmaCells)
        den = cv2.GaussianBlur(board, (0, 0), sigmaCells)
        base = model + np.where(den > 0.05, num / np.maximum(den, 1e-6), 0.0)
        fu = design(np.zeros(3), np.zeros(3), np.array([-0.5, 0.5, 0.0]), np.zeros(3)) @ c
        fv = design(np.zeros(3), np.zeros(3), np.zeros(3), np.array([-0.5, 0.5, 0.0])) @ c
        self.boardFrames = {
            'fieldCoef_u_v_uu_uv_vv': np.round(c[10:], 3).tolist(),
            'fieldSpanMm_u_v': [round(float(fu[1] - fu[0]), 3), round(float(fv[1] - fv[0]), 3)],
            'fitKeptFraction': round(float(keep.mean()), 3),
            'fitResidualStdMm': round(float((y - A @ c)[keep].std()), 3),
        }
        return base.astype(np.float32)

    def depth(self):
        """Z of best focus per cell, confidence, board surface and height above it. Heights in mm."""
        t0 = time.time()
        L = np.stack([l.focus for l in self.used])          # K x h x w
        K = len(self.used)
        zs = self.zs
        k, zmap = self.peakFit(L)
        lmax = np.take_along_axis(L, k[None], 0)[0]
        conf = lmax - np.median(L, axis=0)
        covered = np.isfinite(lmax) & (lmax > math.log(1e-2))
        # absolute texture floor: on smooth spots (bare mask, glare) every layer is at the noise level and the
        # relative peak is noise; the peak must be focusFloor times the textureless floor of the panorama
        floor = float(np.percentile(lmax[covered], 2)) + math.log(self.focusFloor)
        textured = lmax > floor
        top = (k == K - 1) & covered & textured   # still sharpening at the highest layer: taller than the range
        bottom = (k == 0) & covered               # below the range (mat beyond the board edge)
        valid = covered & textured & (conf > self.confMin) & ~bottom

        # global board plane: robust fit on the most common depth (summary, fallback for the local board)
        hs, ws = zmap.shape
        yy, xx = np.mgrid[0:hs, 0:ws].astype(np.float32)
        sel = valid & ~top
        hist, edges = np.histogram(zmap[sel], bins=np.arange(zs.min(), zs.max() + 0.1, 0.1))
        mode = edges[np.argmax(hist)] + 0.05
        keep = sel & (np.abs(zmap - mode) < 0.5)
        coef = np.array([0.0, 0.0, mode])
        for tol in (0.4, 0.25, 0.2):
            X = np.stack([xx[keep], yy[keep], np.ones(keep.sum(), np.float32)], 1)
            coef = np.linalg.lstsq(X, zmap[keep], rcond=None)[0]
            plane = coef[0] * xx + coef[1] * yy + coef[2]
            keep = sel & (np.abs(zmap - plane) < tol)
        plane = (coef[0] * xx + coef[1] * yy + coef[2]).astype(np.float32)
        base = self.boardSurface(zmap, sel, self.cellLabels(), plane)
        boardCells = sel & (np.abs(zmap - base) < 0.25)

        # holes: normalized convolution, smallest scale with enough support wins, else board.
        # tops above the range get one height per connected region (median of the lower bound)
        clipped = np.zeros(zmap.shape, np.float32)
        joined = cv2.morphologyEx(top.astype(np.uint8), cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        n, comp = cv2.connectedComponents(joined, connectivity=8)
        if n > 1:
            bound = (zs[-1] - base).ravel()
            lab = comp.ravel()
            order = np.argsort(lab, kind='stable')
            starts = np.searchsorted(lab[order], np.arange(n))
            med = np.array([np.median(bound[order[a:b]]) if b > a else 0.0
                            for a, b in zip(starts, list(starts[1:]) + [len(order)])], np.float32)
            clipped = np.where(top, med[comp], 0.0).astype(np.float32)

        def fill(valid):
            weight = valid.astype(np.float32) * np.clip(conf / (2 * self.confMin), 0, 1)
            rel = np.where(valid, np.where(top, clipped, zmap - base), 0.0).astype(np.float32)
            height = rel.copy()
            todo = ~valid
            for sigma, minSupport in ((1.5, 0.08), (4.0, 0.12), (10.0, 0.3)):
                den = cv2.GaussianBlur(weight, (0, 0), sigma)
                num = cv2.GaussianBlur(rel * weight, (0, 0), sigma)
                ok = todo & (den > minSupport)
                height[ok] = num[ok] / den[ok]
                todo &= ~ok
            height[todo] = 0.0
            return cv2.medianBlur(height, 5)

        height = fill(valid)
        # spikes: small raised blobs (glints) are dropped and filled from around
        mmPerCell = self.ref.calibrationSummary()['umPerPx'] / 1000.0 / self.ref.scale * self.factor
        minCells = self.spikeMinAreaMm2 / mmPerCell ** 2
        n, comp, stats, _ = cv2.connectedComponentsWithStats((height > self.spikeMinMm).astype(np.uint8), connectivity=8)
        small = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] < minCells)
        small = small[small > 0]
        spikes = np.isin(comp, small) & valid
        if spikes.any():
            valid = valid & ~cv2.dilate(spikes.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            height = fill(valid)
        self.spikes = int(len(small))
        self.zmap, self.conf, self.valid, self.top, self.bottom = zmap, conf, valid, top, bottom
        self.covered = covered
        self.plane, self.planeCoef, self.board = plane, coef, base
        self.height = height                         # mm above the local board, holes filled
        self.focusZ = base + height                  # Z of best focus used for fusion
        px = self.ref.frameSize[0] / self.ref.scale
        self.depthStats = {
            'cells': [int(ws), int(hs)], 'cellFullResPx': self.factor / self.ref.scale,
            'validFraction': round(float(valid.mean()), 3),
            'texturelessFraction': round(float((covered & ~textured).mean()), 3),
            'aboveRangeFraction': round(float(top.mean()), 3),
            'belowRangeFraction': round(float(bottom.mean()), 3),
            'boardZ_mm': round(float(np.median(plane)), 3),
            'boardTilt_mmPerFrameWidth': [round(float(c * px / (self.factor / self.ref.scale)), 3) for c in coef[:2]],
            'globalPlaneResidualMm': round(float(np.std((zmap - plane)[keep])), 3),
            'localBoardResidualMm': round(float(np.std((zmap - base)[boardCells])), 3),
            'boardPerFrame': self.boardFrames,
            'heightP99_mm': round(float(np.percentile(height, 99)), 2),
            'spikesRemoved': self.spikes,
        }
        self.timings['depth'] = self.timings.get('depth', 0) + time.time() - t0
        self.log(f'depth: {valid.mean() * 100:.0f}% cells measured, {(covered & ~textured).mean() * 100:.0f}% '
                 f'textureless, board Z {np.median(plane):.2f} mm, residual global plane '
                 f'{self.depthStats["globalPlaneResidualMm"]} -> local board {self.depthStats["localBoardResidualMm"]} '
                 f'mm, focus field span u/v {self.boardFrames and self.boardFrames["fieldSpanMm_u_v"]} mm, '
                 f'{top.mean() * 100:.1f}% above range')

    def orthoHeights(self, matMm=-0.5, closePx=48.0):
        """Heights for the ortho composite: 0 on the mat, clipped to >= 0 elsewhere."""
        mat = (self.bottom | (self.height < matMm)).astype(np.uint8)
        r = max(1, int(round(closePx * self.ref.scale / self.factor)))
        mat = cv2.morphologyEx(mat, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        return np.where(mat > 0, 0.0, np.maximum(self.height, 0.0)).astype(np.float32)

    # ---------------------------------------------------------- fusion
    def fuse(self, band=512):
        """All-in-focus panorama: every pixel mixes the layers whose Z is near the local depth
        (Gaussian in Z, depthSigmaMm), weights upsampled smoothly from the depth cells."""
        t0 = time.time()
        H, W = self.used[0].pano.shape[:2]
        f = self.factor
        target = cv2.GaussianBlur(self.focusZ, (0, 0), 1.0)
        wk = np.stack([np.exp(-0.5 * ((target - z) / self.depthSigma) ** 2) for z in self.zs]).astype(np.float32)
        wk /= np.maximum(wk.sum(axis=0), 1e-6)
        out = np.zeros((H, W, 3), np.uint8)

        def run(y0):
            bh = min(band, H - y0)
            Ml = np.float32([[1.0 / f, 0, 0.5 / f - 0.5], [0, 1.0 / f, (y0 + 0.5) / f - 0.5]])
            acc = np.zeros((bh, W, 3), np.float32)
            tot = np.zeros((bh, W), np.float32)
            for a, layer in enumerate(self.used):
                w = cv2.warpAffine(wk[a], Ml, (W, bh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_REPLICATE)
                if w.max() < 1e-3:
                    continue
                acc += layer.pano[y0:y0 + bh].astype(np.float32) * w[..., None]
                tot += w
            acc /= np.maximum(tot, 1e-6)[..., None]
            out[y0:y0 + bh] = np.clip(acc + 0.5, 0, 255).astype(np.uint8)

        with ThreadPoolExecutor(self.ref.threads) as pool:
            list(pool.map(run, range(0, H, band)))
        self.timings['fuse'] = time.time() - t0
        return out

    # ---------------------------------------------------------- marking
    def markingRegions(self, minMm=0.45, maxMm=4.0, minSideMm=1.5, maxSideMm=15.0, minFill=0.6, padPx=24.0,
                       openPx=56.0):
        """Raised parts with a top worth reading (ICs, tantalums, ...): connected cells minMm..maxMm above the board
        (the mat excluded), the minimal rotated rectangle between minSideMm and maxSideMm and filled to minFill.
        Returns dicts with the block (x0, y0, bw, bh, work px, padded by padPx) and the cell mask over it."""
        h = self.orthoHeights()
        f = self.factor
        s = self.ref.scale
        mmPerCell = self.ref.calibrationSummary()['umPerPx'] / 1000.0 / s * f
        raised = ((h > minMm) & (h < maxMm)).astype(np.uint8)
        # opening wider than the gap between neighbouring parts: two ICs joined by a low strip are two regions
        r = max(1, int(round(openPx * s / f / 2)))
        raised = cv2.morphologyEx(raised, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(raised, connectivity=8)
        W, H = self.ref.canvasSize
        pad = int(round(padPx * s))
        out = []
        for i in range(1, n):
            x, y, w, hh, area = stats[i]
            pts = np.column_stack(np.nonzero(lab[y:y + hh, x:x + w] == i)[::-1]).astype(np.float32)
            (_, _), (rw, rh), angle = cv2.minAreaRect(pts)
            rw, rh = rw + 1, rh + 1
            if not (minSideMm <= min(rw, rh) * mmPerCell and max(rw, rh) * mmPerCell <= maxSideMm
                    and area >= minFill * rw * rh):
                continue
            x0, y0 = max(0, x * f - pad), max(0, y * f - pad)
            x1, y1 = min(W, (x + w) * f + pad), min(H, (y + hh) * f + pad)
            out.append({'id': len(out), 'block': (x0, y0, x1 - x0, y1 - y0), 'cells': lab == i,
                        'sizeMm': [round(float(max(rw, rh) * mmPerCell), 1), round(float(min(rw, rh) * mmPerCell), 1)],
                        'heightMm': round(float(np.median(h[lab == i])), 2)})
        return out

    def regionMask(self, r, growPx):
        """Cells of region r upsampled to its block (float 0/1), grown by growPx full-res px (negative shrinks)."""
        x0, y0, bw, bh = r['block']
        f = self.factor
        M = np.float32([[1.0 / f, 0, (x0 + 0.5) / f - 0.5], [0, 1.0 / f, (y0 + 0.5) / f - 0.5]])
        m = cv2.warpAffine(r['cells'].astype(np.float32), M, (bw, bh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        m = (m > 0.5).astype(np.uint8)
        g = int(round(abs(growPx) * self.ref.scale))
        if g:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))
            m = cv2.dilate(m, kernel) if growPx > 0 else cv2.erode(m, kernel)
        return m.astype(np.float32)

    def markingScore(self, img, inner):
        """Marking contrast: std of a 4-25 px band-pass relative to local brightness over the inner top."""
        s = self.ref.scale
        g = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        lo = cv2.GaussianBlur(g, (0, 0), 25 * s, borderType=cv2.BORDER_REFLECT)
        band = cv2.GaussianBlur(g, (0, 0), max(1.0, 4.0 * s), borderType=cv2.BORDER_REFLECT) - lo
        v = (band / np.maximum(lo, 8.0))[inner > 0]
        if v.size < 50:
            return 0.0
        a, b = np.percentile(v, [1, 99])
        return float(np.clip(v, a, b).std() * 1000)

    def alignView(self, img, cover, ref, refCover, inner, maxShiftPx=6.0):
        """Shift of a marking view onto the best one (ECC, translation, over the top both hold): the ortho
        correction leaves the tops of different frames 1-3 px apart, enough to double 8 px strokes."""
        s = self.ref.scale
        mask = ((inner > 0) & (cover >= 1) & (refCover >= 1)).astype(np.uint8)
        if mask.sum() < 400:
            return img, cover
        sig = max(0.7, 1.5 * s)
        g = [cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), sig) for x in (ref, img)]
        M = np.eye(2, 3, dtype=np.float32)
        try:
            _, M = cv2.findTransformECC(g[0], g[1], M, cv2.MOTION_TRANSLATION,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4), mask, 5)
        except cv2.error:
            return img, cover
        if np.hypot(M[0, 2], M[1, 2]) > maxShiftPx * s:
            return img, cover
        h, w = img.shape[:2]
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        return (cv2.warpAffine(img, M, (w, h), flags=flags, borderMode=cv2.BORDER_REPLICATE),
                cv2.warpAffine(cover, M, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT))

    def markingLayer(self, fused, edgePx=10.0, featherPx=4.0, innerPx=-16.0, minCover=0.5, borderPx=24.0, power=8.0):
        """All-in-focus panorama with every raised part top replaced by a mix of its best-scored frame views
        (best height per frame, weight (score / best)^power). Returns (marking BGR, region records)."""
        t0 = time.time()
        s = self.ref.scale
        regions = self.markingRegions()
        heights = self.orthoHeights()
        geo = copy.copy(self.ref)
        geo.override, geo.heightMap, geo.heightFactor = None, heights, self.factor
        for r in regions:
            x0, y0, bw, bh = r['block']
            r['h'] = geo.blockHeights(x0, y0, bw, bh)
            r['paste'] = self.regionMask(r, edgePx)
            r['inner'] = self.regionMask(r, innerPx)
            r['cands'] = self.ref.framesInRect(x0, y0, x0 + bw, y0 + bh)
            r['fusedScore'] = self.markingScore(fused[y0:y0 + bh, x0:x0 + bw].astype(np.float32), r['inner'])
            r['views'] = {}               # frame -> (score, z, uint8 BGR, float32 coverage)
        fade = max(1, int(round(borderPx * s)))
        for layer in self.used:
            frames = self.layerFrames(layer)
            if frames is None:
                continue
            st = copy.copy(geo)
            st.frames = frames
            st.orthoDistance = self.distance + (layer.z - self.refZ)

            def one(r):
                x0, y0, bw, bh = r['block']
                inner = r['inner'] > 0
                for k in r['cands']:
                    img, _, ok = st.orthoWarp(k, x0, y0, bw, bh, r['h'], withLow=False)
                    # the frame border is a straight line on the canvas: distance to it fades the view out
                    cover = np.clip(cv2.distanceTransform((ok > 0).astype(np.uint8), cv2.DIST_L2, 3) / fade, 0, 1)
                    held = inner & (cover >= 1)
                    if held.sum() < minCover * inner.sum():
                        continue
                    sc = self.markingScore(img, held)
                    if k not in r['views'] or sc > r['views'][k][0]:
                        r['views'][k] = (sc, layer.z, np.clip(img + 0.5, 0, 255).astype(np.uint8), cover)

            with ThreadPoolExecutor(self.ref.threads) as pool:
                list(pool.map(one, regions))
            del frames
        out = fused.copy()
        sig = max(0.5, featherPx * s)
        records = []
        for r in regions:
            x0, y0, bw, bh = r['block']
            base = fused[y0:y0 + bh, x0:x0 + bw].astype(np.float32)
            rec = {'id': r['id'], 'block': [int(v) for v in r['block']], 'sizeMm': r['sizeMm'],
                   'heightMm': r['heightMm'], 'fusedScore': round(r['fusedScore'], 1)}
            views = sorted(r['views'].items(), key=lambda kv: -kv[1][0])
            if views and views[0][1][0] > r['fusedScore']:
                top = views[0][1][0]
                acc = base * (r['fusedScore'] / top) ** power
                tot = np.full((bh, bw), (r['fusedScore'] / top) ** power, np.float32)
                shares = []
                bestImg, bestCover = views[0][1][2], views[0][1][3]
                for k, (sc, z, img, cover) in views:
                    w = (sc / top) ** power
                    if w < 1e-3:
                        continue
                    if k != views[0][0]:
                        img, cover = self.alignView(img, cover, bestImg, bestCover, r['inner'])
                    w = cover * w
                    sel = (r['paste'] > 0) & (cover > 0)
                    img = img.astype(np.float32)
                    gain = base[sel].mean(axis=0) / np.maximum(img[sel].mean(axis=0), 1.0)
                    acc += img * gain * w[..., None]
                    tot += w
                    shares.append((k, sc, z, float((w * r['inner']).sum())))
                acc /= tot[..., None]
                m = cv2.GaussianBlur(r['paste'], (0, 0), sig)[..., None]
                out[y0:y0 + bh, x0:x0 + bw] = np.clip(acc * m + base * (1 - m) + 0.5, 0, 255).astype(np.uint8)
                allw = float((tot * r['inner']).sum())
                rec['views'] = [{'frame': self.names[k], 'z': z, 'score': round(sc, 1), 'share': round(sh / allw, 2)}
                                for k, sc, z, sh in shares if sh / allw >= 0.01]
                main = max(rec['views'], key=lambda v: v['share'])
                rec.update({'frame': main['frame'], 'z': main['z'], 'score': main['score']})
            records.append(rec)
        self.markingRecords = records
        self.timings['marking'] = time.time() - t0
        better = [r for r in records if 'score' in r]
        self.log(f'marking: {len(records)} parts, {len(better)} replaced, median contrast '
                 f'{np.median([r["fusedScore"] for r in better]) if better else 0:.0f} -> '
                 f'{np.median([r["score"] for r in better]) if better else 0:.0f}, {time.time() - t0:.1f} s')
        return out, records

    # ---------------------------------------------------------- outputs
    def heightColor(self, vmax=None):
        """BGRA colour map of the height (turbo), transparent where nothing was measured."""
        vmax = vmax or max(0.5, float(np.ceil(self.zs.max() - np.median(self.plane))))
        v = np.clip(self.height / vmax, 0, 1)
        rgb = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        a = np.where(self.valid | self.top, 255, 70).astype(np.uint8)
        return np.dstack([rgb, a]), vmax

    def meshHeights(self, kernel=9, step=0.02, shapeKernel=11, tallMm=2.0):
        """Height for the 3D relief: median over kernel x kernel cells (removes single-cell spikes from specular
        highlights), done on heights quantized to step mm; NaN where no frame covers the canvas."""
        lo = float(self.height.min())
        q = np.clip(np.round((self.height - lo) / step), 0, 255).astype(np.uint8)
        top = lo + 255 * step
        med = cv2.medianBlur(q, kernel).astype(np.float32) * step + lo
        med = np.where(self.height > top, self.height, med)
        # tall parts only: opening cuts side spikes, closing fills notches
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shapeKernel, shapeKernel))
        opened = cv2.dilate(cv2.erode(med, k), k)
        med = np.where((med > tallMm) & (opened < med - 0.3), opened, med)
        closed = cv2.erode(cv2.dilate(med, k), k)
        med = np.where((closed > tallMm) & (closed > med + 0.3), closed, med)
        med = cv2.GaussianBlur(med, (0, 0), 0.7)
        covered = cv2.erode(self.covered.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        return np.where(covered, med, np.nan).astype(np.float32)

    def focusCurve(self, x, y):
        """Log focus of every layer at canvas work px (x, y) - for diagnostics."""
        f = self.factor
        return [float(l.focus[int(y // f), int(x // f)]) for l in self.used]

    def report(self):
        return {
            'refZ': self.refZ,
            'cameraDistanceMm': round(self.distance, 2),
            'scaleFitMaxResidual': round(self.distanceFitResidual, 5),
            'orthoPasses': self.orthoPasses,
            'cellWorkPx': self.factor,
            'layers': [{'z': l.z, 'folder': l.folder, 'ok': l.ok, 'reason': l.reason,
                        'scale': None if l.scale is None else round(l.scale, 5),
                        'centreShiftPx': None if l.shift is None else [round(v, 1) for v in l.shift],
                        'spreadPx': None if l.spread is None else round(l.spread, 2),
                        'inliers': l.inliers} for l in self.layers],
            'depth': getattr(self, 'depthStats', None),
            'marking': getattr(self, 'markingRecords', None),
            'timingsSec': {k: round(v, 2) for k, v in self.timings.items()},
        }
