"""Grid stitcher for frames captured on a translating gantry.

Every frame is named "{x}_{y}.png" where x, y are machine coordinates in mm.
The camera only translates, so the position of a frame on the panorama is
    p = A * (x_mm, y_mm) + t
with one 2x2 matrix A shared by all frames (px/mm, camera rotation, axis signs).

Pipeline:
    loadFolder -> loadImages -> computeFlatField -> calibrate (A from AKAZE matches)
    -> measurePairs (phase correlation in the predicted overlaps)
    -> optimize (weighted least squares + machine-coordinate prior, Huber IRLS)
    -> compensateGains (per-frame, per-channel exposure / white balance)
    -> [detectParallax: tall components taken whole from one frame, optional]
    -> composite (multiband / voronoi / feather / hard), done block by block to keep memory low.
"""
import glob
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

FRAME_NAME_RE = re.compile(r'^(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)\.(png|jpg|jpeg|bmp|tif|tiff)$', re.IGNORECASE)


class Frame:
    def __init__(self, path, x, y):
        self.path = path
        self.x = x
        self.y = y
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.image = None       # BGR uint8, work scale, flat-field (and exposure gain) corrected
        self.low = None         # float32 BGR, blurred and downscaled by lowFactor (multiband blending)
        self.gray = None        # uint8, work scale
        self.position = None    # top-left corner on the canvas, work px (float)
        self.predicted = None   # A * (x, y), work px

    def machine(self):
        return np.array([self.x, self.y], dtype=np.float64)


class Pair:
    def __init__(self, i, j, kind):
        self.i = i
        self.j = j
        self.kind = kind          # 'x', 'y' or 'diag'
        self.predicted = None     # p_j - p_i predicted by A, work px
        self.measured = None      # p_j - p_i measured, work px
        self.response = 0.0
        self.ncc = 0.0
        self.overlap = (0, 0)
        self.valid = False
        self.rejected = False
        self.rejectReason = ''
        self.weight = 0.0
        self.residual = None      # (p_j - p_i) - measured after optimization


class GridStitcher:
    def __init__(self, scale=1.0, flatField=True, refine=True, priorSigmaPx=25.0, huberDeltaPx=1.5,
                 rejectPx=6.0, nccMin=0.3, nccWeak=0.12, maxDeviationPx=30.0,
                 nearDeviationPx=15.0, minOverlapPx=96, threads=None, log=print):
        self.scale = scale
        self.flatField = flatField
        self.refine = refine
        # all thresholds below are given in full-resolution px and converted to work px
        self.priorSigma = priorSigmaPx * scale
        self.huberDelta = huberDeltaPx * scale
        self.rejectDistance = rejectPx * scale
        self.nccMin = nccMin
        self.nccWeak = nccWeak
        self.nearDeviation = nearDeviationPx * scale
        self.maxDeviation = maxDeviationPx * scale
        self.minOverlap = max(24, int(minOverlapPx * scale))
        self.threads = threads or max(1, (os.cpu_count() or 4) - 1)
        self.lowBand = 150.0      # full-res px, cross-fade half-width of the low band in 'multiband' blending
        self.lowFactor = 8
        self.gains = None
        self.gainStats = None
        self.override = None      # parallax regions: label map (frame index or -1) at canvas / overrideFactor
        self.overrideFactor = 4
        self.heightMap = None     # ortho correction (multiband only): mm above the board, cells of heightFactor work px
        self.heightFactor = 8
        self.orthoDistance = None  # mm from the projection centre to the board plane
        self.regions = []
        self.parallaxStats = None
        self.log = log

        self.frames = []
        self.pairs = []
        self.stepX = None
        self.stepY = None
        self.frameSize = None     # (w, h) work px
        self.flat = None          # float32 HxWx3 gain map (work scale)
        self.A = None             # 2x2, work px / mm
        self.calibrationPairs = []
        self.canvasSize = None    # (w, h)
        self.fitA = None
        self.fitT = None
        self.timings = {}

    # ------------------------------------------------------------------ loading
    def loadFolder(self, folder):
        paths = []
        for path in glob.glob(os.path.join(folder, '*')):
            m = FRAME_NAME_RE.match(os.path.basename(path))
            if m:
                paths.append((float(m.group(1)), float(m.group(2)), path))
        if not paths:
            raise RuntimeError(f'No "x_y.png" frames found in {folder}')
        paths.sort(key=lambda p: (p[1], p[0]))
        self.frames = [Frame(p, x, y) for x, y, p in paths]
        xs = np.unique([f.x for f in self.frames])
        ys = np.unique([f.y for f in self.frames])
        self.stepX = float(np.min(np.diff(xs))) if len(xs) > 1 else 1.0
        self.stepY = float(np.min(np.diff(ys))) if len(ys) > 1 else 1.0
        self.log(f'{len(self.frames)} frames, X {xs.min()}..{xs.max()} step {self.stepX}, '
                 f'Y {ys.min()}..{ys.max()} step {self.stepY}')

    def loadImages(self):
        t0 = time.time()

        def load(frame):
            image = cv2.imread(frame.path, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f'Cannot read {frame.path}')
            if self.scale != 1.0:
                image = cv2.resize(image, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            frame.image = image

        with ThreadPoolExecutor(self.threads) as pool:
            list(pool.map(load, self.frames))
        h, w = self.frames[0].image.shape[:2]
        self.frameSize = (w, h)
        self.timings['load'] = time.time() - t0

    def computeFlatField(self):
        """Vignetting estimate: per-pixel median of all frames (content averages out), heavily blurred."""
        t0 = time.time()
        w, h = self.frameSize
        if self.flatField:
            f = 8
            small = np.stack([cv2.resize(fr.image, (w // f, h // f), interpolation=cv2.INTER_AREA)
                              for fr in self.frames]).astype(np.float32)
            med = np.median(small, axis=0)
            med = cv2.GaussianBlur(med, (0, 0), sigmaX=w / f / 12)
            med = np.maximum(med, 1.0)
            gain = med.reshape(-1, 3).mean(axis=0) / med   # brightness gain per pixel and channel
            gain = np.clip(gain, 0.5, 3.0)
            self.flat = cv2.resize(gain, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            self.flatStats = {'gainMin': float(gain.min()), 'gainMax': float(gain.max())}

            def apply(frame):
                frame.image = np.clip(frame.image.astype(np.float32) * self.flat + 0.5, 0, 255).astype(np.uint8)

            with ThreadPoolExecutor(self.threads) as pool:
                list(pool.map(apply, self.frames))
        for fr in self.frames:
            fr.gray = cv2.cvtColor(fr.image, cv2.COLOR_BGR2GRAY)
        self.timings['flatField'] = time.time() - t0

    # ------------------------------------------------------------- calibration
    def neighbourIndex(self):
        index = {}
        for k, f in enumerate(self.frames):
            index[(round(f.x / self.stepX), round(f.y / self.stepY))] = k
        return index

    @staticmethod
    def translationRansac(vectors, tolerance):
        if len(vectors) == 0:
            return None, 0
        d = np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=2)
        counts = (d < tolerance).sum(axis=1)
        best = int(np.argmax(counts))
        inliers = vectors[d[best] < tolerance]
        return np.median(inliers, axis=0), int(counts[best])

    def calibrate(self):
        """Estimate A from AKAZE feature matches between grid neighbours (translation-only RANSAC)."""
        t0 = time.time()
        measureFactor = min(1.0, 0.5 / self.scale)
        akaze = cv2.AKAZE_create(threshold=0.0005)

        def features(frame):
            g = frame.gray if measureFactor == 1.0 else cv2.resize(frame.gray, None, fx=measureFactor,
                                                                    fy=measureFactor, interpolation=cv2.INTER_AREA)
            g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
            kp, desc = akaze.detectAndCompute(g, None)
            return np.array([k.pt for k in kp], dtype=np.float32).reshape(-1, 2), desc

        with ThreadPoolExecutor(self.threads) as pool:
            feats = list(pool.map(features, self.frames))

        index = self.neighbourIndex()
        candidates = []
        for (gx, gy), i in index.items():
            for (dx, dy), kind in (((1, 0), 'x'), ((0, 1), 'y')):
                j = index.get((gx + dx, gy + dy))
                if j is not None:
                    candidates.append((i, j, kind))

        def match(c):
            i, j, kind = c
            (p1, d1), (p2, d2) = feats[i], feats[j]
            if d1 is None or d2 is None or len(d1) < 10 or len(d2) < 10:
                return None
            knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
            good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
            if len(good) < 8:
                return None
            v = np.array([p2[g.trainIdx] - p1[g.queryIdx] for g in good], dtype=np.float64)
            disp, count = self.translationRansac(v, 3.0)
            if count < 12 or count < 0.15 * len(good):
                return None
            # p_j - p_i = -displacement of the content
            return i, j, kind, -disp / measureFactor, count

        with ThreadPoolExecutor(self.threads) as pool:
            results = [r for r in pool.map(match, candidates) if r is not None]
        if len(results) < 3:
            raise RuntimeError('Calibration failed: too few matched neighbour pairs')

        dc = np.array([self.frames[j].machine() - self.frames[i].machine() for i, j, _, _, _ in results])
        m = np.array([r[3] for r in results])
        # initial A from medians of pure X and pure Y moves
        A = np.zeros((2, 2))
        for col, kind, step in ((0, 'x', self.stepX), (1, 'y', self.stepY)):
            sel = [k for k, r in enumerate(results) if r[2] == kind]
            if sel:
                A[:, col] = np.median(m[sel] / dc[sel, col:col + 1], axis=0)
        # robust LS refit
        for _ in range(3):
            res = np.linalg.norm(m - dc @ A.T, axis=1)
            keep = res < max(20.0 * self.scale, 3.0 * np.median(res))
            A = np.linalg.lstsq(dc[keep], m[keep], rcond=None)[0].T
        res = np.linalg.norm(m - dc @ A.T, axis=1)
        self.A = A
        self.calibrationPairs = [{'a': self.frames[r[0]].name, 'b': self.frames[r[1]].name, 'kind': r[2],
                                  'inliers': r[4], 'deviationPx': float(res[k] / self.scale)}
                                 for k, r in enumerate(results)]
        self.calibrationKept = int(keep.sum())
        self.calibrationDeviation = float(np.median(res[keep]) / self.scale)
        for f in self.frames:
            f.predicted = A @ f.machine()
            f.position = f.predicted.copy()
        self.timings['calibrate'] = time.time() - t0
        self.log(f'calibration: {len(results)}/{len(candidates)} neighbour pairs matched, {keep.sum()} kept, '
                 f'median deviation {self.calibrationDeviation:.1f} px')
        self.log('A (full-res px/mm) =\n' + str(A / self.scale))

    def calibrationSummary(self):
        A = self.A / self.scale
        sx = np.linalg.norm(A[:, 0])
        sy = np.linalg.norm(A[:, 1])
        angX = math.degrees(math.atan2(A[1, 0], A[0, 0]))
        angY = math.degrees(math.atan2(A[1, 1], A[0, 1]))
        return {
            'A_fullResPxPerMm': A.tolist(),
            'pxPerMmX': sx, 'pxPerMmY': sy,
            'machineXAxisAngleDeg': angX,      # direction of machine +X on the image (0 = image right, 90 = image down)
            'machineYAxisAngleDeg': angY,
            'axesAngleDeg': (angY - angX + 180) % 360 - 180,  # +-90 if axes orthogonal; sign = handedness
            'umPerPx': 1000.0 / ((sx + sy) / 2),
        }

    # ---------------------------------------------------------- measurements
    def buildPairs(self, minOverlapFraction=0.12):
        w, h = self.frameSize
        self.pairs = []
        n = len(self.frames)
        for i in range(n):
            for j in range(i + 1, n):
                o = self.frames[j].predicted - self.frames[i].predicted
                ow, oh = w - abs(o[0]), h - abs(o[1])
                if ow < self.minOverlap or oh < self.minOverlap:
                    continue
                if ow * oh < minOverlapFraction * w * h:
                    continue
                dx = round((self.frames[j].x - self.frames[i].x) / self.stepX)
                dy = round((self.frames[j].y - self.frames[i].y) / self.stepY)
                kind = 'x' if dy == 0 else ('y' if dx == 0 else 'diag')
                p = Pair(i, j, kind)
                p.predicted = o
                self.pairs.append(p)

    @staticmethod
    def highPass(img, sigma):
        f = img.astype(np.float32)
        return f - cv2.GaussianBlur(f, (0, 0), sigma)

    def overlapCrops(self, gi, gj, o, factor=1.0):
        """Crops of the overlap of two frames given offset o = p_j - p_i (in the crops' px)."""
        h, w = gi.shape[:2]
        ox, oy = int(round(o[0])), int(round(o[1]))
        x0, x1 = max(0, ox), min(w, w + ox)
        y0, y1 = max(0, oy), min(h, h + oy)
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None, None, (ox, oy)
        ci = gi[y0:y1, x0:x1]
        cj = gj[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        return ci, cj, (ox, oy)

    def phaseStep(self, gi, gj, o, sigma):
        ci, cj, (ox, oy) = self.overlapCrops(gi, gj, o)
        if ci is None:
            return None
        hi = self.highPass(ci, sigma)
        hj = self.highPass(cj, sigma)
        win = cv2.createHanningWindow((hi.shape[1], hi.shape[0]), cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(hi, hj, win)
        return np.array([ox - dx, oy - dy]), response

    def nccAt(self, gi, gj, o, sigma):
        ci, cj, _ = self.overlapCrops(gi, gj, o)
        if ci is None:
            return 0.0, (0, 0)
        hi = self.highPass(ci, sigma)
        hj = self.highPass(cj, sigma)
        hi -= hi.mean()
        hj -= hj.mean()
        den = math.sqrt(float((hi * hi).sum()) * float((hj * hj).sum())) + 1e-9
        return float((hi * hj).sum()) / den, (ci.shape[1], ci.shape[0])

    def measurePairs(self):
        t0 = time.time()
        self.buildPairs()
        coarse = min(1.0, 0.5 / self.scale)
        coarseGray = [cv2.resize(f.gray, None, fx=coarse, fy=coarse, interpolation=cv2.INTER_AREA)
                      if coarse != 1.0 else f.gray for f in self.frames]

        def measure(p):
            gi, gj = self.frames[p.i].gray, self.frames[p.j].gray
            # coarse: two iterations on the downscaled images
            o = p.predicted * coarse
            for _ in range(2):
                r = self.phaseStep(coarseGray[p.i], coarseGray[p.j], o, 3.0)
                if r is None:
                    return
                o = r[0]
            o = o / coarse
            # fine: work resolution
            r = self.phaseStep(gi, gj, o, 3.0 / coarse)
            if r is None:
                return
            o, p.response = r
            # confidence on the coarse images: full-res NCC is dominated by sensor noise and defocus
            p.ncc, _ = self.nccAt(coarseGray[p.i], coarseGray[p.j], o * coarse, 3.0)
            p.overlap = (gi.shape[1] - abs(o[0]), gi.shape[0] - abs(o[1]))
            p.measured = o
            deviation = np.linalg.norm(o - p.predicted)
            # weak matches (plain solder mask, defocused or parallax-heavy overlaps) are kept only when
            # they agree with the machine prediction; strong ones may correct larger machine errors
            if p.ncc < self.nccWeak or (p.ncc < self.nccMin and deviation > self.nearDeviation):
                p.rejected, p.rejectReason = True, f'low ncc {p.ncc:.2f} (deviation {deviation / self.scale:.0f}px)'
            elif deviation > self.maxDeviation:
                p.rejected, p.rejectReason = True, f'deviation from machine prediction {deviation / self.scale:.0f}px'
            else:
                p.valid = True
                p.weight = p.ncc ** 2

        with ThreadPoolExecutor(self.threads) as pool:
            list(pool.map(measure, self.pairs))
        self.timings['measurePairs'] = time.time() - t0
        nValid = sum(p.valid for p in self.pairs)
        self.log(f'pairs: {len(self.pairs)} candidates, {nValid} measured ok')

    # ------------------------------------------------------------- optimization
    def solve(self, weights):
        n = len(self.frames)
        rows, rhs, ws = [], [], []
        active = [p for p in self.pairs if p.valid]
        for p, w in zip(active, weights):
            if w <= 0:
                continue
            r = np.zeros(n)
            r[p.j], r[p.i] = 1.0, -1.0
            rows.append(r)
            rhs.append(p.measured)
            ws.append(w)
        wPrior = 1.0 / (self.priorSigma / max(self.huberDelta, 1e-6)) ** 2
        for k, f in enumerate(self.frames):
            r = np.zeros(n)
            r[k] = 1.0
            rows.append(r)
            rhs.append(f.predicted)
            ws.append(wPrior)
        M = np.array(rows)
        b = np.array(rhs)
        sw = np.sqrt(np.array(ws))[:, None]
        return np.linalg.lstsq(M * sw, b * sw, rcond=None)[0]

    def optimize(self):
        t0 = time.time()
        active = [p for p in self.pairs if p.valid]
        if not self.refine or not active:
            for f in self.frames:
                f.position = f.predicted.copy()
            self.computeResiduals()
            self.timings['optimize'] = time.time() - t0
            return
        base = np.array([p.weight for p in active])
        weights = base.copy()
        P = None
        for it in range(100):
            P = self.solve(weights)
            res = np.array([np.linalg.norm(P[p.j] - P[p.i] - p.measured) for p in active])
            huber = np.where(res <= self.huberDelta, 1.0, self.huberDelta / np.maximum(res, 1e-9))
            newWeights = base * huber
            if it >= 10:
                # hard rejection of gross outliers, one worst at a time
                bad = np.where((res > self.rejectDistance) & (newWeights > 0))[0]
                if len(bad):
                    # drop the worst ones (a bad pair also distorts its neighbours' residuals)
                    worst = bad[res[bad] >= max(self.rejectDistance, 0.7 * res[bad].max())]
                    for wIdx in worst:
                        newWeights[wIdx] = 0.0
                        base[wIdx] = 0.0
                        active[wIdx].rejected = True
                        active[wIdx].rejectReason = f'residual {res[wIdx] / self.scale:.1f}px after optimization'
                elif np.allclose(newWeights, weights, atol=1e-4):
                    break
            weights = newWeights
        for k, f in enumerate(self.frames):
            f.position = P[k]
        for p, w in zip(active, base):
            if w == 0:
                p.valid = False
        self.computeResiduals()
        self.timings['optimize'] = time.time() - t0

    def computeResiduals(self):
        for p in self.pairs:
            if p.measured is not None:
                p.residual = self.frames[p.j].position - self.frames[p.i].position - p.measured
        # global affine fit of the final positions (diagnostic)
        C = np.array([np.append(f.machine(), 1.0) for f in self.frames])
        P = np.array([f.position for f in self.frames])
        X = np.linalg.lstsq(C, P, rcond=None)[0]
        self.fitA = X[:2].T
        self.fitT = X[2]
        self.machineError = P - C @ X   # deviation of every frame from a perfect affine grid, work px

    # ------------------------------------------------------------ exposure
    def overlapMeans(self, small, f):
        """Per used pair: (i, j, pixel count, mean BGR of i, mean BGR of j) over unsaturated overlap pixels."""
        out = []
        for p in self.pairs:
            if not p.valid:
                continue
            o = (self.frames[p.j].position - self.frames[p.i].position) * f
            ci, cj, _ = self.overlapCrops(small[p.i], small[p.j], o)
            if ci is None:
                continue
            ok = np.all((ci > 8) & (ci < 247) & (cj > 8) & (cj < 247), axis=2)
            n = int(ok.sum())
            if n < 100:
                continue
            out.append((p.i, p.j, n, ci[ok].mean(axis=0), cj[ok].mean(axis=0)))
        return out

    def compensateGains(self, sigmaN=10.0, sigmaG=0.1):
        """Per-frame, per-channel gain (camera auto exposure / white balance drift), Brown & Lowe:
        minimize sum N_ij * ((g_i I_ij - g_j I_ji)^2 / sigmaN^2 + (1 - g_i)^2 / sigmaG^2) over used pairs."""
        t0 = time.time()
        f = min(1.0, 0.25 / self.scale)
        small = [cv2.resize(fr.image, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) for fr in self.frames]
        stats = self.overlapMeans(small, f)
        n = len(self.frames)
        gains = np.ones((n, 3))
        before = self.seamMismatch(stats)
        for c in range(3):
            M = np.zeros((n, n))
            b = np.zeros(n)
            for i, j, N, mi, mj in stats:
                a, d = mi[c], mj[c]
                M[i, i] += N * a * a / sigmaN ** 2
                M[j, j] += N * d * d / sigmaN ** 2
                M[i, j] -= N * a * d / sigmaN ** 2
                M[j, i] -= N * a * d / sigmaN ** 2
                for k in (i, j):
                    M[k, k] += N / sigmaG ** 2
                    b[k] += N / sigmaG ** 2
            for k in range(n):
                if M[k, k] == 0:
                    M[k, k], b[k] = 1.0, 1.0
            gains[:, c] = np.linalg.solve(M, b)
        self.gains = gains
        g32 = gains.astype(np.float32)

        def apply(k):
            fr = self.frames[k]
            fr.image = np.clip(fr.image.astype(np.float32) * g32[k] + 0.5, 0, 255).astype(np.uint8)

        with ThreadPoolExecutor(self.threads) as pool:
            list(pool.map(apply, range(n)))
        small = [cv2.resize(fr.image, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) for fr in self.frames]
        after = self.seamMismatch(self.overlapMeans(small, f))
        self.gainStats = {'gainMin': gains.min(axis=0).round(3).tolist(), 'gainMax': gains.max(axis=0).round(3).tolist(),
                          'overlapMeanDiffBefore': before, 'overlapMeanDiffAfter': after}
        self.timings['gains'] = time.time() - t0
        self.log(f'gains BGR min {gains.min(axis=0).round(3)}, max {gains.max(axis=0).round(3)}; '
                 f'overlap mean |diff| {before} -> {after}')

    @staticmethod
    def seamMismatch(stats):
        """Median and max over pairs of the mean BGR difference in the overlap (grey levels)."""
        if not stats:
            return None
        d = np.array([np.abs(mi - mj).max() for _, _, _, mi, mj in stats])
        return {'median': round(float(np.median(d)), 2), 'max': round(float(d.max()), 2)}

    # ---------------------------------------------------------------- canvas
    def layoutCanvas(self):
        w, h = self.frameSize
        P = np.array([f.position for f in self.frames])
        mn = np.floor(P.min(axis=0))
        for f in self.frames:
            f.position = f.position - mn
        mx = np.array([f.position for f in self.frames]).max(axis=0)
        self.canvasSize = (int(math.ceil(mx[0])) + w, int(math.ceil(mx[1])) + h)
        self.origin = mn

    def centres(self):
        w, h = self.frameSize
        return np.array([f.position + np.array([(w - 1) / 2.0, (h - 1) / 2.0]) for f in self.frames])

    def framesInRect(self, x0, y0, x1, y1):
        w, h = self.frameSize
        out = []
        for k, f in enumerate(self.frames):
            px, py = f.position
            if px < x1 and px + w > x0 and py < y1 and py + h > y0:
                out.append(k)
        return out

    def blendBlock(self, x0, y0, bw, bh, blend, band, margin):
        """Returns uint8 BGR block of the panorama [y0:y0+bh, x0:x0+bw]."""
        w, h = self.frameSize
        idx = self.framesInRect(x0, y0, x0 + bw, y0 + bh)
        out = np.zeros((bh, bw, 3), np.uint8)
        if not idx:
            return out
        centres = self.centres()
        edge, d2 = self.blockGeometry(idx, x0, y0, bw, bh, margin)
        inside = edge > 0
        if self.override is not None and blend in ('multiband', 'voronoi', 'hard'):
            # parallax regions: the seam goes along an explicit label map instead of the plain voronoi rule
            sigmas = [0.0 if blend == 'hard' else band / 2.0]
            if blend == 'multiband':
                sigmas.append(self.lowBand * self.scale / 2.0)
            ws = self.labelWeights(idx, x0, y0, bw, bh, margin, sigmas)
            if blend == 'multiband':
                wHigh, wLow = ws
            else:
                weights = ws[0]
        if blend == 'multiband':
            # two bands: details switch at a narrow voronoi seam (keeps tall parts sharp),
            # low frequencies (brightness, colour cast) cross-fade over a wide band
            if self.override is None:
                wHigh = self.blendWeights('voronoi', band, idx, inside, edge, d2, centres)
                wLow = self.blendWeights('voronoi', self.lowBand * self.scale, idx, inside, edge, d2, centres)
            k8 = self.lowFactor
            heights = self.blockHeights(x0, y0, bw, bh)
            samples = [None] * len(idx)
            if heights is not None:
                # ortho samples outside the frame are junk: give those pixels to the neighbours
                hi0, lo0 = wHigh.copy(), wLow.copy()
                for a, k in enumerate(idx):
                    if np.any(wHigh[a]) or np.any(wLow[a]):
                        full, low, ok = self.orthoWarp(k, x0, y0, bw, bh, heights)
                        samples[a] = (full, low)
                        wHigh[a] *= ok
                        wLow[a] *= ok
                lost = (wHigh.sum(axis=0) == 0) | (wLow.sum(axis=0) == 0)
                wHigh[:, lost] = hi0[:, lost]
                wLow[:, lost] = lo0[:, lost]
            tHigh, tLow = wHigh.sum(axis=0), wLow.sum(axis=0)
            valid = (tHigh > 0) & (tLow > 0)
            wHigh /= np.where(valid, tHigh, 1.0)
            wLow /= np.where(valid, tLow, 1.0)
            acc = np.zeros((bh, bw, 3), np.float32)
            for a, k in enumerate(idx):
                if not (np.any(wHigh[a]) or np.any(wLow[a])):
                    continue
                px, py = self.frames[k].position
                if samples[a] is not None:
                    full, low = samples[a]
                else:
                    M = np.float32([[1, 0, px - x0], [0, 1, py - y0]])
                    full = cv2.warpAffine(self.frames[k].image, M, (bw, bh), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
                    # low band is stored downscaled: block px (u, v) -> low px ((u + x0 - px + 0.5) / k8 - 0.5, ...)
                    Ml = np.float32([[1.0 / k8, 0, (x0 - px + 0.5) / k8 - 0.5],
                                     [0, 1.0 / k8, (y0 - py + 0.5) / k8 - 0.5]])
                    low = cv2.warpAffine(self.frames[k].low, Ml, (bw, bh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                         borderMode=cv2.BORDER_REPLICATE)
                acc += (full - low) * wHigh[a][..., None] + low * wLow[a][..., None]
            np.clip(acc, 0, 255, out=acc)
            out[valid] = (acc[valid] + 0.5).astype(np.uint8)
            return out
        if self.override is None or blend == 'feather':
            weights = self.blendWeights(blend, band, idx, inside, edge, d2, centres)
        total = weights.sum(axis=0)
        valid = total > 0
        acc = np.zeros((bh, bw, 3), np.float32)
        for a, k in enumerate(idx):
            wa = weights[a]
            if not np.any(wa):
                continue
            px, py = self.frames[k].position
            M = np.float32([[1, 0, px - x0], [0, 1, py - y0]])
            warped = cv2.warpAffine(self.frames[k].image, M, (bw, bh), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
            acc += warped.astype(np.float32) * wa[..., None]
        acc[valid] /= total[valid][:, None]
        np.clip(acc, 0, 255, out=acc)
        out[valid] = (acc[valid] + 0.5).astype(np.uint8)
        return out

    def blockHeights(self, x0, y0, bw, bh):
        """Height above the board (mm) over canvas px [y0:y0+bh, x0:x0+bw], upsampled from self.heightMap
        (cells of heightFactor work px); None when ortho correction is off."""
        if self.heightMap is None:
            return None
        f = self.heightFactor
        M = np.float32([[1.0 / f, 0, (x0 + 0.5) / f - 0.5], [0, 1.0 / f, (y0 + 0.5) / f - 0.5]])
        return cv2.warpAffine(self.heightMap, M, (bw, bh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)

    def orthoWarp(self, k, x0, y0, bw, bh, heights, withLow=True):
        """Frame k sampled for canvas px of the block as seen from straight above (orthographic): a point at
        height h is imaged magnified by D / (D - h) about the frame centre (central projection from distance
        orthoDistance mm), so the frame px is c + (board px - c) * D / (D - h). Returns (full, low) float32 BGR
        (low is None without withLow) and a float32 mask, 1 where the sample lies inside the frame (minus a 1 px rim)."""
        w, h = self.frameSize
        px, py = self.frames[k].position
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        D = self.orthoDistance
        m = D / (D - np.clip(heights, -0.5 * D, 0.5 * D))
        u = np.arange(x0, x0 + bw, dtype=np.float32)[None, :] - px - cx
        v = np.arange(y0, y0 + bh, dtype=np.float32)[:, None] - py - cy
        mapX = (cx + u * m).astype(np.float32)
        mapY = (cy + v * m).astype(np.float32)
        full = cv2.remap(self.frames[k].image, mapX, mapY, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        low = None
        if withLow:
            k8 = self.lowFactor
            low = cv2.remap(self.frames[k].low, (mapX + 0.5) / k8 - 0.5, (mapY + 0.5) / k8 - 0.5, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
        ok = ((mapX >= 1) & (mapX <= w - 2) & (mapY >= 1) & (mapY <= h - 2)).astype(np.float32)
        return full, low, ok

    def blockGeometry(self, idx, x0, y0, bw, bh, margin):
        """Per frame in idx: distance to the frame border minus margin (>0 inside) and squared distance
        to the frame centre, over canvas pixels [y0:y0+bh, x0:x0+bw]."""
        w, h = self.frameSize
        u = np.arange(x0, x0 + bw, dtype=np.float32)[None, :]
        v = np.arange(y0, y0 + bh, dtype=np.float32)[:, None]
        centres = self.centres()
        edge = np.empty((len(idx), bh, bw), np.float32)
        d2 = np.empty((len(idx), bh, bw), np.float32)
        for a, k in enumerate(idx):
            px, py = self.frames[k].position
            ex = np.minimum(u - px, px + w - 1 - u)
            ey = np.minimum(v - py, py + h - 1 - v)
            edge[a] = np.minimum(ex, ey) - margin
            cx, cy = centres[k]
            d2[a] = (u - cx) ** 2 + (v - cy) ** 2
        return edge, d2

    def labelWeights(self, idx, x0, y0, bw, bh, margin, sigmas):
        """Blend weights from the label map (nearest centre, replaced by self.override in parallax regions):
        one-hot masks blurred by every sigma (0 = hard), computed on a padded block so blocks join seamlessly."""
        pad = int(math.ceil(3 * max(sigmas))) + 1
        X0, Y0, BW, BH = x0 - pad, y0 - pad, bw + 2 * pad, bh + 2 * pad
        edge, d2 = self.blockGeometry(idx, X0, Y0, BW, BH, margin)
        inside = edge > 0
        d2m = np.where(inside, d2, np.inf)
        winner = np.argmin(d2m, axis=0)
        covered = np.isfinite(np.min(d2m, axis=0))
        f = self.overrideFactor
        oy = np.clip((np.arange(Y0, Y0 + BH) // f).astype(int), 0, self.override.shape[0] - 1)
        ox = np.clip((np.arange(X0, X0 + BW) // f).astype(int), 0, self.override.shape[1] - 1)
        ov = self.override[oy[:, None], ox[None, :]]
        for a, k in enumerate(idx):
            sel = (ov == k) & inside[a]
            winner[sel] = a
        out = []
        for sigma in sigmas:
            ws = np.zeros((len(idx), bh, bw), np.float32)
            for a in range(len(idx)):
                m = ((winner == a) & covered).astype(np.float32)
                if not m.any():
                    continue
                if sigma > 0:
                    if sigma > 16:
                        # wide blur on a downscaled mask
                        d = 8
                        small = cv2.resize(m, (max(1, BW // d), max(1, BH // d)), interpolation=cv2.INTER_AREA)
                        small = cv2.GaussianBlur(small, (0, 0), sigma / d, borderType=cv2.BORDER_REPLICATE)
                        m = cv2.resize(small, (BW, BH), interpolation=cv2.INTER_LINEAR)
                    else:
                        m = cv2.GaussianBlur(m, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
                ws[a] = m[pad:pad + bh, pad:pad + bw] * inside[a, pad:pad + bh, pad:pad + bw]
            out.append(ws)
        return out

    # ------------------------------------------------------- parallax regions
    def detectParallax(self, stridePx=32, windowPx=96, shiftPx=10.0, minResponse=0.08, minAreaPx=6000, growPx=48,
                       closePx=96, coverFraction=0.97, graySets=None):
        """Find tall components: places where the two nearest frames, already aligned on the board plane, still
        show a local shift (parallax; ~25 px for a 3 mm part at 10 mm frame step). Every such region is then
        taken whole from ONE frame - the one whose centre is nearest and which contains the whole region.
        graySets: extra lists of frame grays in this geometry (other focus heights, where tall tops are sharp);
        the largest shift over all sets counts.
        Result: self.override (label map at canvas / overrideFactor, -1 = plain nearest-centre rule), self.regions."""
        t0 = time.time()
        if self.canvasSize is None:
            self.layoutCanvas()
        W, H = self.canvasSize
        w, h = self.frameSize
        f = max(2, int(round(stridePx * self.scale)))
        win = max(16, int(round(windowPx * self.scale)))
        lw, lh = int(math.ceil(W / f)), int(math.ceil(H / f))
        centres = self.centres()
        margin = 2.0
        # nearest and second nearest frame (whose window fits) for every grid cell centre
        u = (np.arange(lw, dtype=np.float32) * f + f / 2.0)[None, :]
        v = (np.arange(lh, dtype=np.float32) * f + f / 2.0)[:, None]
        best = np.full((2, lh, lw), np.inf, np.float32)
        lab = np.full((2, lh, lw), -1, np.int32)
        m = win / 2.0 + 1
        for k, c in enumerate(centres):
            px, py = self.frames[k].position
            inside = (u >= px + m) & (u <= px + w - 1 - m) & (v >= py + m) & (v <= py + h - 1 - m)
            d = np.where(inside, (u - c[0]) ** 2 + (v - c[1]) ** 2, np.inf)
            first = d < best[0]
            second = ~first & (d < best[1])
            best[1] = np.where(first, best[0], np.where(second, d, best[1]))
            lab[1] = np.where(first, lab[0], np.where(second, k, lab[1]))
            best[0] = np.where(first, d, best[0])
            lab[0] = np.where(first, k, lab[0])
        hann = cv2.createHanningWindow((win, win), cv2.CV_32F)
        shift = np.zeros((lh, lw), np.float32)
        sets = [[fr.gray for fr in self.frames]] + list(graySets or [])

        def row(r):
            for c in range(lw):
                cell(r, c)

        def cell(r, c):
            for grays in sets:
                i, j = lab[0, r, c], lab[1, r, c]
                if i < 0 or j < 0:
                    continue
                cx, cy = float(u[0, c]), float(v[r, 0])
                pi, pj = self.frames[i].position, self.frames[j].position
                a = cv2.getRectSubPix(grays[i], (win, win), (cx - pi[0], cy - pi[1]), patchType=cv2.CV_32F)
                b = cv2.getRectSubPix(grays[j], (win, win), (cx - pj[0], cy - pj[1]), patchType=cv2.CV_32F)
                if a.std() < 4 or b.std() < 4:
                    continue          # no texture, nothing to measure
                (dx, dy), resp = cv2.phaseCorrelate(a - a.mean(), b - b.mean(), hann)
                if resp >= minResponse:
                    shift[r, c] = max(shift[r, c], math.hypot(dx, dy) / self.scale)

        with ThreadPoolExecutor(self.threads) as pool:
            list(pool.map(row, range(lh)))
        diff = shift
        # flat board gives 0-6 px of noise; tall parts give 15-60 px but only where their top has texture,
        # so keep cells with at least one more strong neighbour and fill the holes between them
        strong = (shift > shiftPx).astype(np.float32)
        dense = cv2.boxFilter(strong, -1, (3, 3), normalize=False) >= 2
        mask = (dense & (strong > 0)).astype(np.uint8)
        closeR = max(1, int(round(closePx * self.scale / f)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * closeR + 1, 2 * closeR + 1)))
        grow = max(1, int(round(growPx * self.scale / f)))
        growKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        count, comp, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        override = np.full((lh, lw), -1, np.int32)
        regions = []
        minArea = minAreaPx * (self.scale / f) ** 2
        skipped = 0
        # big regions first, so small neighbours drawn later keep their own frame where grown margins overlap
        order = sorted(range(1, count), key=lambda c: -stats[c, cv2.CC_STAT_AREA])
        for c in order:
            if stats[c, cv2.CC_STAT_AREA] < minArea:
                continue
            region = cv2.dilate((comp == c).astype(np.uint8), growKernel)    # each region grown on its own
            ys, xs = np.nonzero(region)
            cx, cy = (xs.mean() + 0.5) * f, (ys.mean() + 0.5) * f
            best_k, best_d = -1, np.inf
            for k, fr in enumerate(self.frames):
                px, py = fr.position
                ins = (((xs + 0.5) * f >= px + margin) & ((xs + 0.5) * f <= px + w - 1 - margin) &
                       ((ys + 0.5) * f >= py + margin) & ((ys + 0.5) * f <= py + h - 1 - margin))
                if ins.mean() < coverFraction:
                    continue
                d = (centres[k][0] - cx) ** 2 + (centres[k][1] - cy) ** 2
                if d < best_d:
                    best_k, best_d = k, d
            if best_k < 0:
                skipped += 1      # no single frame holds it (e.g. off-board background): keep plain voronoi
                continue
            override[ys, xs] = best_k
            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            poly = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 1.0, True).reshape(-1, 2) * f
            regions.append({'frame': int(best_k), 'polygon': poly.astype(float).tolist(),
                            'areaPx': int(len(xs) * f * f / self.scale ** 2)})
        # pixels outside the chosen frame stay on the plain rule
        self.override = override
        self.overrideFactor = f
        self.regions = regions
        self.parallaxDiff = diff
        self.parallaxStats = {'regions': len(regions), 'skippedNoSingleFrame': skipped,
                              'shiftPx': shiftPx, 'growPx': growPx, 'stridePx': stridePx}
        self.timings['parallax'] = time.time() - t0
        self.log(f'parallax regions: {len(regions)} assigned to one frame, {skipped} skipped (no frame holds them)')

    def drawParallaxOverlay(self, pano, maxSide=4000):
        H, W = pano.shape[:2]
        f = min(1.0, maxSide / max(W, H))
        img = cv2.resize(pano, (int(W * f), int(H * f)), interpolation=cv2.INTER_AREA)
        img = (img * 0.7).astype(np.uint8)
        for r in self.regions:
            poly = (np.array(r['polygon']) * f).astype(np.int32)
            cv2.polylines(img, [poly], True, (255, 0, 255), 2, cv2.LINE_AA)
            c = poly.mean(axis=0).astype(int)
            cv2.putText(img, self.frames[r['frame']].name, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                        cv2.LINE_AA)
        return img

    @staticmethod
    def blendWeights(blend, band, idx, inside, edge, d2, centres):
        n, bh, bw = edge.shape
        if blend == 'feather':
            return np.where(inside, edge, 0.0).astype(np.float32)
        d2m = np.where(inside, d2, np.inf)
        winner = np.argmin(d2m, axis=0)
        covered = np.isfinite(np.min(d2m, axis=0))
        if blend == 'hard':
            return (np.arange(n)[:, None, None] == winner[None]).astype(np.float32) * covered
        cIdx = centres[idx]
        ckx = cIdx[winner, 0].astype(np.float32)
        cky = cIdx[winner, 1].astype(np.float32)
        dkk = np.take_along_axis(d2m, winner[None], axis=0)[0]
        weights = np.zeros((n, bh, bw), np.float32)
        tau = max(band / 3.0, 0.5)
        ramp = max(band, 1.0)
        np.seterr(invalid='ignore')
        for a in range(n):
            dist = np.sqrt((cIdx[a, 0] - ckx) ** 2 + (cIdx[a, 1] - cky) ** 2)
            delta = np.where(dist > 0, (d2m[a] - dkk) / (2.0 * np.maximum(dist, 1e-6)), 0.0)
            wa = np.exp(-np.minimum(delta, 50 * tau) / tau)
            wa *= np.clip(edge[a] / ramp, 0.0, 1.0)     # fade out near the frame border
            wa[winner == a] = np.maximum(wa[winner == a], 1e-3)
            weights[a] = np.where(inside[a], wa, 0.0)
            weights[a][delta > 50 * tau] = 0.0
        return weights

    def prepareLowBand(self, sigmaPx=24.0, factor=8):
        """Blurred, downscaled copy of every frame for the low band of 'multiband' blending."""
        self.lowFactor = factor
        sigma = max(1.0, sigmaPx * self.scale / factor)

        def make(fr):
            small = cv2.resize(fr.image, None, fx=1.0 / factor, fy=1.0 / factor, interpolation=cv2.INTER_AREA)
            fr.low = cv2.GaussianBlur(small.astype(np.float32), (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)

        with ThreadPoolExecutor(self.threads) as pool:
            list(pool.map(make, self.frames))

    def composite(self, blend='multiband', band=12.0, margin=2.0, blockSize=512):
        """band: half-width (full-res px) of the seam transition for voronoi blending."""
        t0 = time.time()
        if self.canvasSize is None:
            self.layoutCanvas()
        if blend == 'multiband':
            self.prepareLowBand()
        W, H = self.canvasSize
        pano = np.zeros((H, W, 3), np.uint8)
        blocks = [(x, y) for y in range(0, H, blockSize) for x in range(0, W, blockSize)]

        def run(b):
            x, y = b
            bw, bh = min(blockSize, W - x), min(blockSize, H - y)
            pano[y:y + bh, x:x + bw] = self.blendBlock(x, y, bw, bh, blend, band * self.scale, margin)

        with ThreadPoolExecutor(self.threads) as pool:
            list(pool.map(run, blocks))
        self.timings['composite'] = time.time() - t0
        return pano

    def labelMap(self, factor, margin=2.0):
        """Voronoi label (index of the frame whose centre is nearest) on a grid downscaled by factor; -1 = empty."""
        W, H = self.canvasSize
        w, h = self.frameSize
        lw, lh = int(math.ceil(W / factor)), int(math.ceil(H / factor))
        u = (np.arange(lw, dtype=np.float32) * factor)[None, :]
        v = (np.arange(lh, dtype=np.float32) * factor)[:, None]
        best = np.full((lh, lw), np.inf, np.float32)
        label = np.full((lh, lw), -1, np.int32)
        for k, c in enumerate(self.centres()):
            px, py = self.frames[k].position
            inside = (u >= px + margin) & (u <= px + w - 1 - margin) & (v >= py + margin) & (v <= py + h - 1 - margin)
            d = (u - c[0]) ** 2 + (v - c[1]) ** 2
            better = inside & (d < best)
            best[better] = d[better]
            label[better] = k
        if self.override is not None:
            of = self.overrideFactor
            oy = np.clip((v[:, 0] // of).astype(int), 0, self.override.shape[0] - 1)
            ox = np.clip((u[0] // of).astype(int), 0, self.override.shape[1] - 1)
            ov = self.override[oy[:, None], ox[None, :]]
            for k in np.unique(ov[ov >= 0]):
                px, py = self.frames[k].position
                inside = (u >= px + margin) & (u <= px + w - 1 - margin) & (v >= py + margin) & (v <= py + h - 1 - margin)
                label[(ov == k) & inside] = k
        return label

    # ---------------------------------------------------------------- report
    def pairStats(self):
        res = np.array([np.linalg.norm(p.residual) / self.scale for p in self.pairs if p.valid and p.residual is not None])
        if len(res) == 0:
            return {}
        return {
            'count': int(len(res)),
            'mean': float(res.mean()), 'median': float(np.median(res)),
            'p90': float(np.percentile(res, 90)), 'max': float(res.max()),
            'rms': float(np.sqrt((res ** 2).mean())),
        }

    def report(self):
        s = self.scale
        return {
            'frames': len(self.frames),
            'scale': s,
            'frameSizeWork': list(self.frameSize),
            'grid': {'stepX': self.stepX, 'stepY': self.stepY},
            'calibration': self.calibrationSummary(),
            'calibrationKeptPairs': self.calibrationKept,
            'calibrationMedianDeviationPx': self.calibrationDeviation,
            'finalAffineFit_fullResPxPerMm': (self.fitA / s).tolist(),
            'frameDeviationFromAffineGridPx': {
                'median': float(np.median(np.linalg.norm(self.machineError, axis=1)) / s),
                'max': float(np.max(np.linalg.norm(self.machineError, axis=1)) / s),
            },
            'refine': self.refine,
            'flatField': self.flatField,
            'flatFieldGain': getattr(self, 'flatStats', None),
            'exposureGains': self.gainStats,
            'parallaxRegions': self.parallaxStats,
            'canvasSizeWork': list(self.canvasSize) if self.canvasSize else None,
            'pairResidualsPx_validPairs': self.pairStats(),
            'pairs': {
                'candidates': len(self.pairs),
                'used': sum(p.valid for p in self.pairs),
                'rejected': [{'a': self.frames[p.i].name, 'b': self.frames[p.j].name, 'kind': p.kind,
                              'reason': p.rejectReason, 'ncc': round(p.ncc, 3)}
                             for p in self.pairs if p.rejected],
                'unmeasured': [{'a': self.frames[p.i].name, 'b': self.frames[p.j].name}
                               for p in self.pairs if p.measured is None],
            },
            'timingsSec': {k: round(v, 2) for k, v in self.timings.items()},
        }

    def pairList(self):
        out = []
        for p in self.pairs:
            out.append({
                'a': p.i, 'b': p.j, 'kind': p.kind,
                'valid': bool(p.valid), 'rejected': bool(p.rejected), 'reason': p.rejectReason,
                'ncc': round(float(p.ncc), 3), 'response': round(float(p.response), 3),
                'residualPx': None if p.residual is None else round(float(np.linalg.norm(p.residual)) / self.scale, 2),
                'deviationFromMachinePx': None if p.measured is None else
                round(float(np.linalg.norm(p.measured - p.predicted)) / self.scale, 1),
            })
        return out

    def drawDebugOverlay(self, pano, maxSide=4000):
        H, W = pano.shape[:2]
        f = min(1.0, maxSide / max(W, H))
        img = cv2.resize(pano, (int(W * f), int(H * f)), interpolation=cv2.INTER_AREA)
        img = (img * 0.6).astype(np.uint8)
        w, h = self.frameSize
        for fr in self.frames:
            px, py = fr.position * f
            cv2.rectangle(img, (int(px), int(py)), (int(px + w * f), int(py + h * f)), (200, 200, 200), 1)
        centres = self.centres() * f
        for p in self.pairs:
            a, b = centres[p.i], centres[p.j]
            if p.valid:
                r = np.linalg.norm(p.residual) / self.scale
                color = (0, 220, 0) if r < 1.0 else ((0, 220, 220) if r < 2.5 else (0, 0, 255))
                thick = 2
            elif p.rejected:
                color, thick = (255, 0, 255), 1
            else:
                color, thick = (128, 128, 128), 1
            cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), color, thick, cv2.LINE_AA)
            if p.valid:
                mid = ((a + b) / 2).astype(int)
                cv2.putText(img, f'{np.linalg.norm(p.residual) / self.scale:.1f}', tuple(mid),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        for k, fr in enumerate(self.frames):
            c = centres[k].astype(int)
            cv2.circle(img, tuple(c), 3, (255, 255, 255), -1)
            cv2.putText(img, fr.name, (c[0] + 4, c[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
        return img
