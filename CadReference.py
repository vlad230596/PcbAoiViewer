"""CAD reference for the focus-stack height map: the board's STEP model (stepToMesh.py -> model/mesh.npz +
parts.json) rasterized as a top view (max z per depth cell) onto the stack's height grid and compared with the
measured height.

Frames: STEP XY == IPC-2581 Datum mm (checked by CadModel.frameCheck), z = 0 on the board top. Board mm -> panorama
px is the alignBoard.py affine (holes, board level); panorama px -> depth cell is / cellFullResPx. The ortho pass of
the stack puts every top at its vertical projection, so the CAD top view needs no parallax model.

Measured height = Z of best focus - local board surface (FocusStack.depth), valid only on textured cells; cells whose
focus still rises at the top layer are 'above range' (height clipped to Z max - board).
"""
import json
import math
import os
import re

import cv2
import numpy as np


# ---------------------------------------------------------------- model
class CadModel:
    """Triangles of the STEP model with the designator owning each triangle (index into self.names, 0 = board)."""

    def __init__(self, boardDir):
        mdir = os.path.join(boardDir, 'model')
        m = np.load(os.path.join(mdir, 'mesh.npz'))
        self.vertices = m['vertices'].astype(np.float64)
        self.triangles = m['triangles'].astype(np.int64)
        with open(os.path.join(mdir, 'parts.json'), encoding='utf-8') as fp:
            self.leaves = json.load(fp)
        self.names = ['Board']
        index = {'Board': 0}
        self.triPart = np.zeros(len(self.triangles), np.int32)
        for leaf in self.leaves:
            ref = self.designator(leaf)
            if ref not in index:
                index[ref] = len(self.names)
                self.names.append(ref)
            a = leaf['triangleStart']
            self.triPart[a:a + leaf['triangleCount']] = index[ref]
        self.index = index

    @staticmethod
    def designator(leaf):
        p = leaf['path']
        return p[2] if len(p) > 2 else leaf['name']

    def bboxes(self):
        """designator -> (min xyz, max xyz) over its leaves."""
        out = {}
        for leaf in self.leaves:
            ref = self.designator(leaf)
            lo, hi = np.array(leaf['bboxMin']), np.array(leaf['bboxMax'])
            if ref in out:
                out[ref] = (np.minimum(out[ref][0], lo), np.maximum(out[ref][1], hi))
            else:
                out[ref] = (lo, hi)
        return out

    def frameCheck(self, board):
        """STEP frame vs IPC-2581 Datum: bbox centre of every part vs its pick-and-place XY (bodies of asymmetric
        parts are off-centre, so the median and the share within 0.05 mm decide), also against mirrored X."""
        comps = board['components']
        bb = self.bboxes()
        d, dm = [], []
        for ref, (lo, hi) in bb.items():
            c = comps.get(ref)
            if c is None:
                continue
            cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
            d.append((cx - c['x'], cy - c['y']))
            dm.append((-cx - c['x'], cy - c['y']))
        d, dm = np.array(d), np.array(dm)
        r = np.hypot(d[:, 0], d[:, 1])
        mirrorShift = np.median(dm, 0)
        rm = np.hypot(*(dm - mirrorShift).T)
        return {'parts': int(len(d)), 'medianOffsetMm': np.round(np.median(d, 0), 4).tolist(),
                'within0.05mm': round(float((r < 0.05).mean()), 3),
                'mirroredXWithin0.05mm': round(float((rm < 0.05).mean()), 3),
                'missingInModel': sorted(set(comps) - set(bb)), 'notOnBoard': sorted(set(bb) - set(comps) - {'Board'})}


# ---------------------------------------------------------------- raster
class TopRaster:
    """Max z (top view) of triangles sampled at integer points of a W x H grid, with the owner of the top."""

    def __init__(self, width, height, chunk=4_000_000):
        self.w, self.h = width, height
        self.chunk = chunk
        self.z = np.full(height * width, -np.inf, np.float32)
        self.owner = np.full(height * width, -1, np.int32)

    def update(self, idx, z, own):
        if len(idx) == 0:
            return
        order = np.lexsort((z, idx))
        idx, z, own = idx[order], z[order], own[order]
        last = np.r_[idx[1:] != idx[:-1], True]
        idx, z, own = idx[last], z[last], own[last]
        better = z > self.z[idx]
        self.z[idx[better]] = z[better]
        self.owner[idx[better]] = own[better]

    def samples(self, P, Z, own, x0, y0, s):
        """Grid points of an s x s window from (x0, y0) inside each triangle, z interpolated."""
        oy, ox = np.mgrid[0:s, 0:s]
        X = x0[:, None] + ox.ravel()[None]
        Y = y0[:, None] + oy.ravel()[None]
        ax, ay, bx, by, cx, cy = (P[:, i, j][:, None] for i in range(3) for j in range(2))
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        den = np.where(np.abs(den) > 1e-12, den, np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            l0 = ((by - cy) * (X - cx) + (cx - bx) * (Y - cy)) / den
            l1 = ((cy - ay) * (X - cx) + (ax - cx) * (Y - cy)) / den
            l2 = 1 - l0 - l1
            eps = -1e-6
            ok = (l0 >= eps) & (l1 >= eps) & (l2 >= eps) & (X >= 0) & (Y >= 0) & (X < self.w) & (Y < self.h)
            z = l0 * Z[:, 0:1] + l1 * Z[:, 1:2] + l2 * Z[:, 2:3]
        o = np.broadcast_to(own[:, None], X.shape)
        return (Y[ok] * self.w + X[ok]).astype(np.int64), z[ok].astype(np.float32), o[ok]

    def render(self, P, Z, own):
        """P: (T, 3, 2) grid coordinates, Z: (T, 3), own: (T,)."""
        # vertices first: thin features (pins, walls) that cover no grid point still leave their tops
        vi = np.round(P.reshape(-1, 2)).astype(np.int64)
        ok = (vi[:, 0] >= 0) & (vi[:, 1] >= 0) & (vi[:, 0] < self.w) & (vi[:, 1] < self.h)
        self.update((vi[ok, 1] * self.w + vi[ok, 0]), Z.reshape(-1)[ok].astype(np.float32), np.repeat(own, 3)[ok])
        lo, hi = P.min(1), P.max(1)
        x0, y0 = np.ceil(lo[:, 0]).astype(np.int64), np.ceil(lo[:, 1]).astype(np.int64)
        nx = np.floor(hi[:, 0]).astype(np.int64) - x0 + 1
        ny = np.floor(hi[:, 1]).astype(np.int64) - y0 + 1
        n = np.maximum(nx, ny)
        keep = (nx > 0) & (ny > 0)
        s = 1
        while True:
            sel = keep & (n <= s) & (n > s // 2) if s > 1 else keep & (n == 1)
            idx = np.flatnonzero(sel)
            if s > 64:
                idx = np.flatnonzero(keep & (n > 64))
                for t in idx:      # large faces (board top, big lids) one by one
                    gx, gy = max(0, x0[t]), max(0, y0[t])
                    sx, sy = min(self.w, x0[t] + nx[t]) - gx, min(self.h, y0[t] + ny[t]) - gy
                    if sx <= 0 or sy <= 0:
                        continue
                    oy, ox = np.mgrid[0:sy, 0:sx]
                    self.oneTriangle(P[t], Z[t], own[t], gx + ox.ravel(), gy + oy.ravel())
                break
            step = max(1, self.chunk // (s * s))
            for a in range(0, len(idx), step):
                t = idx[a:a + step]
                self.update(*self.samples(P[t], Z[t], own[t], x0[t], y0[t], s))
            s *= 2

    def oneTriangle(self, p, z, own, X, Y):
        (ax, ay), (bx, by), (cx, cy) = p
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-12:
            return
        l0 = ((by - cy) * (X - cx) + (cx - bx) * (Y - cy)) / den
        l1 = ((cy - ay) * (X - cx) + (ax - cx) * (Y - cy)) / den
        l2 = 1 - l0 - l1
        ok = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
        zz = (l0 * z[0] + l1 * z[1] + l2 * z[2])[ok].astype(np.float32)
        self.update((Y[ok] * self.w + X[ok]).astype(np.int64), zz, np.full(len(zz), own, np.int32))

    def result(self):
        z = self.z.reshape(self.h, self.w).copy()
        z[~np.isfinite(z)] = np.nan
        return z, self.owner.reshape(self.h, self.w).copy()


# ---------------------------------------------------------------- classes
def partClass(ref, comp, cadTop):
    pkg = (comp.get('package') or '') + ' ' + (comp.get('footprint') or '')
    if cadTop < 0.05:
        return 'pad'
    if re.match(r'B\d', ref):
        return 'hole'
    if re.search(r'0402|YC124', pkg):
        return 'passive0402'
    if re.search(r'0603', pkg):
        return 'passive0603'
    if cadTop >= 1.8:
        return 'tall'
    if re.search(r'0805|1206|1210', pkg):
        return 'passiveLarge'
    if re.match(r'(U|Q|X|D|RGB)', ref):
        return 'ic'
    return 'other'


def robustLine(x, y, iters=6, k=3.0):
    """y = a x + b, least squares re-fit on |residual| < k * 1.4826 MAD."""
    keep = np.ones(len(x), bool)
    a, b = 1.0, 0.0
    for _ in range(iters):
        if keep.sum() < 10:
            break
        A = np.stack([x[keep], np.ones(keep.sum())], 1)
        a, b = np.linalg.lstsq(A, y[keep], rcond=None)[0]
        r = y - (a * x + b)
        mad = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        keep = np.abs(r) < max(k * mad, 0.02)
    return float(a), float(b), keep


def errStats(e):
    e = e[np.isfinite(e)]
    if len(e) == 0:
        return {'cells': 0}
    a = np.abs(e)
    return {'cells': int(len(e)), 'bias': round(float(np.median(e)), 3), 'mae': round(float(a.mean()), 3),
            'medianAbs': round(float(np.median(a)), 3), 'p90Abs': round(float(np.percentile(a, 90)), 3),
            'rms': round(float(np.sqrt((e * e).mean())), 3)}


# ---------------------------------------------------------------- comparison
class CadComparison:
    def __init__(self, stackDir, boardDir, alignDir, markings=None, log=print):
        self.log = log
        self.stackDir = stackDir
        self.height = np.load(os.path.join(stackDir, 'height.npy')).astype(np.float32)
        self.valid = np.load(os.path.join(stackDir, 'valid.npy'))
        self.boardZ = np.load(os.path.join(stackDir, 'board_z.npy')).astype(np.float32)
        H, W = self.height.shape
        mp = os.path.join(stackDir, 'measured.bin')
        measured = np.fromfile(mp, np.uint8).reshape(H, W).astype(bool) if os.path.exists(mp) else self.valid
        self.top = measured & ~self.valid           # focus still rising at the highest layer
        cp = os.path.join(stackDir, 'confidence.png')    # focus peak over the median layer, x60 (stackFocus.py)
        conf = cv2.imread(cp, cv2.IMREAD_GRAYSCALE) if os.path.exists(cp) else None
        self.conf = conf.astype(np.float32) / 60.0 if conf is not None and conf.shape == (H, W) else None
        with open(os.path.join(stackDir, 'report.json'), encoding='utf-8') as fp:
            self.stackReport = json.load(fp)
        st = self.stackReport['stack']
        self.zMax = max(l['z'] for l in st['layers'] if l['ok'])
        self.zMin = min(l['z'] for l in st['layers'] if l['ok'])
        self.cellPx = st['depth']['cellFullResPx']
        self.rangeTop = (self.zMax - self.boardZ).astype(np.float32)   # highest measurable height per cell
        self.mmPerCell = self.stackReport['reference']['calibration']['umPerPx'] / 1000.0 * self.cellPx
        with open(os.path.join(alignDir, 'board.json'), encoding='utf-8') as fp:
            ab = json.load(fp)
        self.board = ab
        self.M = np.array(ab['align']['M'], np.float64)
        with open(os.path.join(alignDir, 'report.json'), encoding='utf-8') as fp:
            ar = json.load(fp)
        self.panoToFull = 1.0 / ar.get('scale', 1.0)
        self.checks = {
            'stackCanvasWork': self.stackReport['reference']['canvasSizeWork'],
            'stackScale': self.stackReport['scale'],
            'alignCanvasWork': ar['canvasSizeWork'], 'alignScale': ar.get('scale', 1.0),
            'stackPairResidualMedianPx': self.stackReport['reference']['pairResidualsPx_validPairs']['median'],
            'alignPairResidualMedianPx': ar['pairResidualsPx_validPairs']['median'],
        }
        sw = np.array(self.checks['stackCanvasWork']) / self.checks['stackScale']
        aw = np.array(self.checks['alignCanvasWork']) / self.checks['alignScale']
        self.checks['sameCanvas'] = bool(np.allclose(sw, aw, atol=2))
        if not self.checks['sameCanvas']:
            log(f'WARNING: stack canvas {sw} != aligned panorama canvas {aw} (full-res px)')
        self.model = CadModel(boardDir)
        self.frame = self.model.frameCheck(ab)
        self.excluded = self.exclusions(markings)
        self.shift = np.zeros(2)

    def exclusions(self, markings):
        """Parts of the model not on the photographed board: bottom side, not fitted (nc / none) unless the
        marking reading saw them, or seen absent."""
        present = {}
        if markings and os.path.exists(markings):
            with open(markings, encoding='utf-8') as fp:
                present = {k: v.get('present') for k, v in json.load(fp).get('parts', {}).items()}
        out = {}
        for ref in self.model.names[1:]:
            c = self.board['components'].get(ref)
            if c is None:
                out[ref] = 'not in board.json'
            elif c['side'] != 'top':
                out[ref] = 'bottom side'
            elif present.get(ref) is False:
                out[ref] = 'absent (markings.json)'
            elif c.get('fit') in ('nc', 'none') and present.get(ref) is not True and c.get('height', 0) > 0.05:
                out[ref] = f'fit {c.get("fit")}'
        return out

    # ------------------------------------------------------------ raster
    def cellAffine(self, shift=(0.0, 0.0)):
        """mm -> grid coordinates where integer (i, j) is the centre of cell (i, j)."""
        A = self.M * (self.panoToFull / self.cellPx)
        A[:, 2] -= 0.5
        A[:, 2] += shift
        return A

    def rasterize(self, shift=(0.0, 0.0)):
        m = self.model
        drop = np.array([n in self.excluded for n in m.names])
        keepTri = ~drop[m.triPart]
        V = m.vertices
        zt = V[m.triangles, 2]
        keepTri &= zt.max(1) >= -1e-4            # above the board top surface plane
        T = m.triangles[keepTri]
        A = self.cellAffine(shift)
        xy = V[:, :2] @ A[:, :2].T + A[:, 2]
        P = xy[T]
        Z = V[T, 2]
        H, W = self.height.shape
        r = TopRaster(W, H)
        r.render(P, Z, m.triPart[keepTri])
        cad, owner = r.result()
        # board model is the outline only (no holes): everything inside it is at least the board top
        return cad, owner

    # ------------------------------------------------------------ fits
    def comparable(self, cad):
        """Cells where measured and CAD can be compared: measured (not filled), within the stack range."""
        return self.valid & ~self.top & np.isfinite(cad) & (cad < self.rangeTop - 0.25)

    def fitShift(self, cad, owner, radius=6):
        """Integer then sub-cell XY shift of the CAD raster that best matches the measured height near parts."""
        parts = cv2.dilate((owner > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        sel = self.comparable(cad) & parts
        sel[:radius] = sel[-radius:] = False
        sel[:, :radius] = sel[:, -radius:] = False
        ys, xs = np.nonzero(sel)
        c = cad[ys, xs]
        cost = np.zeros((2 * radius + 1, 2 * radius + 1))
        hmeas = self.height
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                # CAD shifted by (dx, dy): CAD at x - d compared with measured at x
                e = np.abs(hmeas[ys, xs] - cad[ys - dy, xs - dx])
                cost[dy + radius, dx + radius] = np.mean(np.minimum(np.nan_to_num(e, nan=1.0), 1.0))
        j, i = np.unravel_index(np.argmin(cost), cost.shape)

        def sub(a, b, c):
            d = a - 2 * b + c
            return 0.0 if d <= 0 else 0.5 * (a - c) / d
        sx = sub(*cost[j, i - 1:i + 2]) if 0 < i < 2 * radius else 0.0
        sy = sub(*cost[j - 1:j + 2, i]) if 0 < j < 2 * radius else 0.0
        self.shiftCost = cost
        return np.array([i - radius + sx, j - radius + sy]), float(cost[radius, radius]), float(cost.min())

    def partShift(self, cad, sl, mask, radius=4):
        """Local shift (cells) of one part: CAD part window moved over the measured one."""
        y0, y1, x0, x1 = sl
        H, W = cad.shape
        if y0 - radius < 0 or x0 - radius < 0 or y1 + radius > H or x1 + radius > W:
            return None
        m = self.comparable(cad)[y0:y1, x0:x1]
        if m.sum() < 30:
            return None
        best = None
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                c = cad[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
                e = np.abs(self.height[y0:y1, x0:x1] - c)[m & np.isfinite(c)]
                if len(e) < 30:
                    continue
                v = float(np.mean(np.minimum(e, 1.0)))
                if best is None or v < best[0]:
                    best = (v, dx, dy)
        return best

    # ------------------------------------------------------------ run
    def run(self, fitShift=True):
        cad, owner = self.rasterize()
        self.log(f'CAD raster: {np.isfinite(cad).mean() * 100:.0f}% of cells on the board, '
                 f'{len(self.excluded)} parts excluded')
        self.shift = np.zeros(2)
        if fitShift:
            self.shift, c0, c1 = self.fitShift(cad, owner)
            self.log(f'XY shift {np.round(self.shift, 2).tolist()} cells '
                     f'({np.round(self.shift * self.mmPerCell * 1000).tolist()} um), cost {c0:.3f} -> {c1:.3f} mm')
            cad, owner = self.rasterize(self.shift)
        self.cad, self.owner = cad, owner
        self.err = np.where(np.isfinite(cad), self.height - cad, np.nan).astype(np.float32)
        self.partTable()
        self.fitScale()
        self.summary = self.summarize()
        return self.summary

    def interiors(self):
        """Per part: (slice, own cells, interior cells eroded 1..2 cells by size)."""
        n = len(self.model.names)
        H, W = self.owner.shape
        ys, xs = np.nonzero(self.owner > 0)
        o = self.owner[ys, xs]
        order = np.argsort(o, kind='stable')
        starts = np.searchsorted(o[order], np.arange(n + 1))
        for k in range(1, n):
            a, b = starts[k], starts[k + 1]
            if b <= a:
                continue
            py, px = ys[order[a:b]], xs[order[a:b]]
            pad = 6
            y0, y1 = max(0, py.min() - pad), min(H, py.max() + pad + 1)
            x0, x1 = max(0, px.min() - pad), min(W, px.max() + pad + 1)
            own = self.owner[y0:y1, x0:x1] == k
            side = min(py.max() - py.min(), px.max() - px.min()) + 1
            r = 1 if side < 12 else 2
            inner = cv2.erode(own.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
            if inner.sum() < 4:
                inner = own
            yield k, (y0, y1, x0, x1), own, inner

    def partTable(self):
        comps = self.board['components']
        rows = []
        for k, sl, own, inner in self.interiors():
            ref = self.model.names[k]
            y0, y1, x0, x1 = sl
            c = self.cad[y0:y1, x0:x1]
            h = self.height[y0:y1, x0:x1]
            v = self.valid[y0:y1, x0:x1] & inner
            t = self.top[y0:y1, x0:x1] & inner
            rt = self.rangeTop[y0:y1, x0:x1]
            cadTop = float(np.percentile(c[own], 99))
            beyond = inner & (c >= rt - 0.25)            # CAD above the stack range: cannot be compared
            cmp = v & ~beyond & ~t
            cadMed = float(np.median(c[cmp] if cmp.sum() >= 5 else c[inner]))
            comp = comps.get(ref, {})
            row = {'ref': ref, 'class': partClass(ref, comp, cadTop), 'package': comp.get('package', ''),
                   'bomHeight': comp.get('height'), 'cadTop': round(cadTop, 3), 'cadMedian': round(cadMed, 3),
                   'cells': int(inner.sum()), 'validFraction': round(float(v.sum() / max(1, inner.sum())), 3),
                   'aboveRangeFraction': round(float(t.sum() / max(1, inner.sum())), 3),
                   'rangeTop': round(float(np.median(rt[inner])), 2),
                   'saturated': bool(beyond.sum() > 0.5 * inner.sum()),
                   'filledMedian': round(float(np.median(h[inner])), 3)}
            if self.conf is not None:
                row['confMedian'] = round(float(np.median(self.conf[y0:y1, x0:x1][inner])), 2)
            if cmp.sum() >= 5:
                hv = h[cmp]
                row.update({'measMedian': round(float(np.median(hv)), 3),
                            'measP90': round(float(np.percentile(hv, 90)), 3),
                            'error': round(float(np.median(hv - c[cmp])), 3)})
            else:
                row.update({'measMedian': None, 'measP90': None, 'error': None})
            best = self.partShift(self.cad, sl, inner)
            if best is not None:
                row['localShift'] = [best[1], best[2]]
                row['localShiftCost'] = round(best[0], 3)
            cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
            row['cell'] = [int(cx), int(cy)]
            rows.append(row)
        self.rows = rows

    def fitScale(self):
        """measured = a * CAD + b: on interior cells (all parts in range + board) and on per-part medians."""
        sel = self.comparable(self.cad) & self.innerMask() & np.isfinite(self.err)
        x, y = self.cad[sel].astype(np.float64), self.height[sel].astype(np.float64)
        rng = np.random.default_rng(0)
        if len(x) > 400000:
            i = rng.choice(len(x), 400000, replace=False)
            x, y = x[i], y[i]
        # equal weight to raised cells and board cells, else the board dominates the slope
        hi = x > 0.2
        w = np.where(hi, 1.0, hi.mean() / max(1e-6, (~hi).mean()))
        pick = rng.random(len(x)) < w
        a, b, keep = robustLine(x[pick], y[pick])
        self.cellFit = {'scale': round(a, 4), 'bias': round(b, 4), 'cells': int(pick.sum()),
                        'inlierFraction': round(float(keep.mean()), 3)}
        P = [r for r in self.rows if r['measMedian'] is not None and not r['saturated'] and r['validFraction'] >= 0.2
             and r['class'] not in ('pad', 'hole')]
        if len(P) >= 5:
            px = np.array([r['cadMedian'] for r in P])
            py = np.array([r['measMedian'] for r in P])
            a2, b2, k2 = robustLine(px, py)
            self.partFit = {'scale': round(a2, 4), 'bias': round(b2, 4), 'parts': len(P),
                            'inlierFraction': round(float(k2.mean()), 3)}
        else:
            self.partFit = None
        # large parts only (>= 2 mm2 interior): small ones are pulled toward the board by the focus window
        big = [r for r in P if r['cells'] * self.mmPerCell ** 2 >= 2.0]
        if len(big) >= 5:
            a3, b3, k3 = robustLine(np.array([r['cadMedian'] for r in big]), np.array([r['measMedian'] for r in big]))
            self.largeFit = {'scale': round(a3, 4), 'bias': round(b3, 4), 'parts': len(big),
                             'inlierFraction': round(float(k3.mean()), 3),
                             'outliers': [r['ref'] for r, k in zip(big, k3) if not k]}
        else:
            self.largeFit = None

    def groupStats(self, key):
        """Per-part errors (measured median - CAD median over valid interior cells) grouped by a row field."""
        groups = {}
        for r in self.rows:
            groups.setdefault(r[key], []).append(r)
        out = {}
        for c, R in sorted(groups.items()):
            E = [r for r in R if r['error'] is not None and not r['saturated'] and r['validFraction'] >= 0.1]
            e = np.array([r['error'] for r in E])
            g = {'parts': len(R), 'measured': len(E), 'saturated': sum(r['saturated'] for r in R),
                 'cadMedian': round(float(np.median([r['cadMedian'] for r in R])), 3),
                 'medianValidFraction': round(float(np.median([r['validFraction'] for r in R])), 3)}
            if len(e):
                g.update({'measMedian': round(float(np.median([r['measMedian'] for r in E])), 3),
                          'medianError': round(float(np.median(e)), 3), 'mae': round(float(np.abs(e).mean()), 3),
                          'medianAbsError': round(float(np.median(np.abs(e))), 3),
                          'maxAbsError': round(float(np.abs(e).max()), 3)})
            out[c] = g
        return out

    def innerMask(self):
        m = np.zeros(self.cad.shape, bool)
        for k, sl, own, inner in self.interiors():
            y0, y1, x0, x1 = sl
            m[y0:y1, x0:x1] |= inner
        return m | self.boardMask()

    def boardMask(self, partGap=3, edge=6):
        """Bare-board cells: board owner, >= partGap cells from any part, >= edge cells inside the outline."""
        onBoard = np.isfinite(self.cad).astype(np.uint8)
        inside = cv2.erode(onBoard, np.ones((2 * edge + 1, 2 * edge + 1), np.uint8)) > 0
        parts = cv2.dilate((self.owner > 0).astype(np.uint8), np.ones((2 * partGap + 1, 2 * partGap + 1), np.uint8)) > 0
        return inside & ~parts & (self.owner == 0)

    def summarize(self):
        comp = self.comparable(self.cad)
        board = self.boardMask() & comp
        inner = self.innerMask() & ~self.boardMask() & comp
        a, b = self.cellFit['scale'], self.cellFit['bias']
        out = {
            'checks': self.checks, 'stepFrame': self.frame,
            'excluded': self.excluded,
            'shiftCells': np.round(self.shift, 2).tolist(),
            'shiftUm': np.round(self.shift * self.mmPerCell * 1000).tolist(),
            'mmPerCell': round(self.mmPerCell, 4),
            'zRange': [self.zMin, self.zMax],
            'rangeTopMm': {'median': round(float(np.median(self.rangeTop)), 2),
                           'min': round(float(self.rangeTop.min()), 2), 'max': round(float(self.rangeTop.max()), 2)},
            'fitCells': self.cellFit, 'fitParts': self.partFit, 'fitLargeParts': self.largeFit,
            'cellsBoard': errStats(self.err[board]),
            'cellsParts': errStats(self.err[inner]),
            'cellsPartsAfterFit': errStats((self.height - (a * self.cad + b))[inner]),
            'validFraction': {'board': round(float((self.valid & self.boardMask()).sum() / max(1, self.boardMask().sum())), 3),
                              'parts': round(float((self.valid & self.innerMask() & ~self.boardMask()).sum() /
                                                   max(1, (self.innerMask() & ~self.boardMask()).sum())), 3)},
        }
        out['byClass'] = self.groupStats('class')
        out['byPackage'] = self.groupStats('package')
        good = [r for r in self.rows if r['error'] is not None and not r['saturated'] and r['validFraction'] >= 0.1
                and r['class'] not in ('pad',)]
        out['worstParts'] = [{k: r.get(k) for k in ('ref', 'class', 'cadMedian', 'measMedian', 'error', 'validFraction',
                                                    'confMedian')}
                             for r in sorted(good, key=lambda r: -abs(r['error']))[:15]]
        out['saturatedParts'] = [{k: r[k] for k in ('ref', 'class', 'cadTop', 'rangeTop', 'filledMedian',
                                                    'aboveRangeFraction')} for r in self.rows if r['saturated']]
        ls = np.array([r['localShift'] for r in self.rows if 'localShift' in r], float)
        if len(ls):
            out['localShiftCells'] = {'parts': len(ls), 'median': np.median(ls, 0).tolist(),
                                      'p90Abs': np.percentile(np.abs(ls), 90, 0).tolist()}
        return out

    # ------------------------------------------------------------ output
    def save(self, outDir, focusPath=None, crops=()):
        os.makedirs(outDir, exist_ok=True)
        np.save(os.path.join(outDir, 'cad_height.npy'), self.cad)
        np.save(os.path.join(outDir, 'cad_owner.npy'), self.owner.astype(np.int16))
        np.save(os.path.join(outDir, 'error.npy'), self.err)
        # viewer: int16 um, -32768 = off board / not measured
        cadUm = np.where(np.isfinite(self.cad), np.clip(np.round(np.nan_to_num(self.cad) * 1000), -32000, 32000), -32768)
        cadUm.astype('<i2').tofile(os.path.join(outDir, 'cad_height.bin'))
        errMask = np.isfinite(self.err)
        e = np.where(errMask, np.clip(np.round(np.nan_to_num(self.err) * 1000), -32000, 32000), -32768)
        e.astype('<i2').tofile(os.path.join(outDir, 'error.bin'))
        # per cell: 2 = compared (measured, in range), 1 = filled or above range, 0 = off board
        state = np.where(self.comparable(self.cad), 2, np.where(np.isfinite(self.cad), 1, 0)).astype(np.uint8)
        state.tofile(os.path.join(outDir, 'compared.bin'))
        H, W = self.cad.shape
        viewer = {'cells': [W, H], 'shiftCells': self.summary['shiftCells'], 'fit': self.cellFit,
                  'cadMax': round(float(np.nanmax(self.cad)), 2), 'rangeTop': self.summary['rangeTopMm']['median'],
                  'files': {'cad': 'cad/cad_height.bin', 'error': 'cad/error.bin', 'state': 'cad/compared.bin'}}
        with open(os.path.join(outDir, 'viewer.json'), 'w', encoding='utf-8') as fp:
            json.dump(viewer, fp)
        with open(os.path.join(outDir, 'summary.json'), 'w', encoding='utf-8') as fp:
            json.dump(self.summary, fp, indent=1, ensure_ascii=False)
        with open(os.path.join(outDir, 'parts.json'), 'w', encoding='utf-8') as fp:
            json.dump(self.rows, fp, indent=1, ensure_ascii=False)
        cols = ['ref', 'class', 'package', 'bomHeight', 'cadTop', 'cadMedian', 'measMedian', 'measP90', 'error',
                'filledMedian', 'validFraction', 'confMedian', 'aboveRangeFraction', 'rangeTop', 'saturated', 'cells', 'localShift']
        with open(os.path.join(outDir, 'parts.csv'), 'w', encoding='utf-8') as fp:
            fp.write(','.join(cols) + '\n')
            for r in self.rows:
                vals = [r.get(c) for c in cols]
                fp.write(','.join('' if v is None else (' '.join(map(str, v)) if isinstance(v, list) else
                                                        f'"{v}"' if isinstance(v, str) and ',' in v else str(v))
                                  for v in vals) + '\n')
        Images(self).write(outDir, focusPath, crops)


# ---------------------------------------------------------------- images
def turboLut():
    return cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(1, -1), cv2.COLORMAP_TURBO)[0]


def divergingLut():
    """BGR, 0 = blue (measured below CAD), 128 = white, 255 = red (measured above)."""
    pts = np.array([0, 64, 128, 192, 255])
    rgb = np.array([[5, 48, 97], [67, 147, 195], [247, 247, 247], [214, 96, 77], [103, 0, 31]], float)
    lut = np.stack([np.interp(np.arange(256), pts, rgb[:, c]) for c in range(3)], 1)
    return lut[:, ::-1].astype(np.uint8)


class Images:
    def __init__(self, cmp):
        self.c = cmp

    def heightImage(self, h, vmax, mask=None):
        lut = turboLut()
        v = np.clip(np.nan_to_num(h) / vmax * 255, 0, 255).astype(np.uint8)
        img = lut[v]
        img[~np.isfinite(h)] = (40, 40, 40)
        if mask is not None:
            img[~mask] = (img[~mask] * 0.35 + 30).astype(np.uint8)
        return img

    def errorImage(self, err, compared, span=1.0):
        lut = divergingLut()
        v = np.clip((np.nan_to_num(err) / span + 1) * 127.5, 0, 255).astype(np.uint8)
        img = lut[v].copy()
        img[~compared] = (img[~compared] * 0.25 + 60).astype(np.uint8)   # filled / above range: dimmed
        img[~np.isfinite(err)] = (25, 25, 25)
        return img

    @staticmethod
    def label(img, text, y=22, scale=0.6):
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def colorbar(self, lut, lo, hi, unit, w=360, h=46):
        bar = np.zeros((h, w, 3), np.uint8) + 255
        idx = np.linspace(0, 255, w - 20).astype(int)
        bar[6:24, 10:w - 10] = lut[idx][None]
        for t in np.linspace(lo, hi, 5):
            x = int(10 + (t - lo) / (hi - lo) * (w - 21))
            cv2.putText(bar, f'{t:g}', (max(0, x - 10), 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(bar, unit, (w - 40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        return bar

    def write(self, outDir, focusPath, crops):
        c = self.c
        comp = c.comparable(c.cad)
        vmax = 4.0
        cadImg = self.heightImage(c.cad, vmax)
        measImg = self.heightImage(np.where(np.isfinite(c.cad), c.height, np.nan), vmax, c.valid | c.top)
        errImg = self.errorImage(c.err, comp)
        cv2.imwrite(os.path.join(outDir, 'cad_height.png'), cadImg)
        cv2.imwrite(os.path.join(outDir, 'error.png'), errImg)
        sbs = np.concatenate([self.label(measImg.copy(), 'measured (dim = filled)', 30, 1.0),
                              self.label(cadImg.copy(), 'CAD top view', 30, 1.0)], 1)
        cv2.imwrite(os.path.join(outDir, 'heights_side_by_side.jpg'), sbs, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(os.path.join(outDir, 'colorbar_height.png'), self.colorbar(turboLut(), 0, vmax, 'mm'))
        cv2.imwrite(os.path.join(outDir, 'colorbar_error.png'), self.colorbar(divergingLut(), -1, 1, 'mm'))
        cv2.imwrite(os.path.join(outDir, 'scatter.png'), self.scatter())
        cv2.imwrite(os.path.join(outDir, 'board_error_hist.png'), self.histogram())
        if focusPath and os.path.exists(focusPath) and crops:
            self.crops(outDir, focusPath, crops, vmax)

    def scatter(self, size=640):
        """Per part: CAD median height (x) vs measured median (y), coloured by class; dashed: stack range top."""
        c = self.c
        hmax = float(max(5, math.ceil(max(r['cadMedian'] for r in c.rows) + 0.5)))
        img = np.full((size, size, 3), 255, np.uint8)
        m0 = 50
        sc = (size - m0 - 20) / hmax

        def P(x, y):
            return int(m0 + x * sc), int(size - m0 - y * sc)
        for t in range(0, int(hmax) + 1):
            cv2.line(img, P(t, 0), P(t, hmax), (230, 230, 230), 1)
            cv2.line(img, P(0, t), P(hmax, t), (230, 230, 230), 1)
            cv2.putText(img, str(t), (P(t, 0)[0] - 4, size - m0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.putText(img, str(t), (m0 - 18, P(0, t)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.line(img, P(0, 0), P(hmax, hmax), (150, 150, 150), 1)
        rt = c.summary['rangeTopMm']['median']
        for x in range(0, int(hmax * sc), 8):
            cv2.line(img, (m0 + x, P(0, rt)[1]), (m0 + x + 4, P(0, rt)[1]), (0, 0, 200), 1)
        colors = {'passive0402': (200, 120, 0), 'passive0603': (0, 160, 0), 'passiveLarge': (0, 200, 200),
                  'ic': (200, 0, 200), 'tall': (0, 0, 230), 'other': (120, 120, 120), 'hole': (0, 0, 0),
                  'pad': (180, 180, 180)}
        for r in c.rows:
            y = r['measMedian'] if r['measMedian'] is not None else r['filledMedian']
            col = colors.get(r['class'], (0, 0, 0))
            p = P(min(r['cadMedian'], hmax), min(max(y, 0), hmax))
            if r['measMedian'] is None:
                cv2.circle(img, p, 3, col, 1, cv2.LINE_AA)
            else:
                cv2.circle(img, p, 3, col, -1, cv2.LINE_AA)
            if r['class'] in ('tall', 'ic') or (r['error'] is not None and abs(r['error']) > 0.5):
                cv2.putText(img, r['ref'], (p[0] + 4, p[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1, cv2.LINE_AA)
        y = 20
        for k, col in colors.items():
            cv2.circle(img, (size - 130, y - 4), 4, col, -1)
            cv2.putText(img, k, (size - 120, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            y += 16
        cv2.putText(img, 'x: CAD median mm, y: measured median mm (open = no valid cells)', (m0, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(img, 'red dashes: stack range top', (m0, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
        return img

    def histogram(self, w=640, h=300, span=0.6, bins=120):
        c = self.c
        comp = c.comparable(c.cad)
        groups = [('board', c.boardMask() & comp, (80, 80, 80)),
                  ('parts', c.innerMask() & ~c.boardMask() & comp, (0, 0, 220))]
        img = np.full((h, w, 3), 255, np.uint8)
        edges = np.linspace(-span, span, bins + 1)
        for name, m, col in groups:
            hist = np.histogram(np.clip(c.err[m], -span, span), edges)[0].astype(float)
            hist /= max(1, hist.max())
            pts = [(int(20 + i * (w - 40) / bins), int(h - 30 - v * (h - 60))) for i, v in enumerate(hist)]
            cv2.polylines(img, [np.array(pts, np.int32)], False, col, 2, cv2.LINE_AA)
        for t in np.linspace(-span, span, 7):
            x = int(20 + (t + span) / (2 * span) * (w - 40))
            cv2.line(img, (x, h - 30), (x, h - 25), (0, 0, 0), 1)
            cv2.putText(img, f'{t:.1f}', (x - 12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(img, 'measured - CAD, mm (compared cells, normalised): grey board, red part interiors',
                    (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        return img

    def crops(self, outDir, focusPath, refs, vmax, sizeMm=None):
        c = self.c
        focus = cv2.imread(focusPath, cv2.IMREAD_COLOR)
        cell = c.cellPx * c.checks['stackScale']     # focus.jpg px per depth cell
        cdir = os.path.join(outDir, 'crops')
        os.makedirs(cdir, exist_ok=True)
        rows = {r['ref']: r for r in c.rows}
        cadImg = self.heightImage(c.cad, vmax)
        measImg = self.heightImage(np.where(np.isfinite(c.cad), c.height, np.nan), vmax, c.valid | c.top)
        errImg = self.errorImage(c.err, c.comparable(c.cad))
        for ref in refs:
            if ref not in c.model.index or ref not in rows:
                continue
            k = c.model.index[ref]
            ys, xs = np.nonzero(c.owner == k)
            pad = max(12, int(0.4 * max(ys.ptp(), xs.ptp())))
            H, W = c.cad.shape
            y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
            x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
            target = 320
            f = target / max(y1 - y0, x1 - x0)
            size = (int((x1 - x0) * f), int((y1 - y0) * f))
            photo = focus[int(y0 * cell):int(y1 * cell), int(x0 * cell):int(x1 * cell)]
            photo = cv2.resize(photo, size, interpolation=cv2.INTER_AREA)
            own = cv2.resize((c.owner[y0:y1, x0:x1] == k).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
            cnts, _ = cv2.findContours(own, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            panels = []
            for img, name in ((photo, 'photo + CAD outline'), (measImg[y0:y1, x0:x1], 'measured'),
                              (cadImg[y0:y1, x0:x1], 'CAD'), (errImg[y0:y1, x0:x1], 'meas - CAD')):
                p = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST if img is not photo else cv2.INTER_AREA)
                p = np.ascontiguousarray(p)
                cv2.drawContours(p, cnts, -1, (255, 255, 255) if img is not photo else (0, 255, 255), 1, cv2.LINE_AA)
                panels.append(self.label(p, name, 16, 0.45))
            strip = np.concatenate(panels, 1)
            r = rows[ref]
            info = (f"{ref} {r['class']}  CAD {r['cadMedian']:.2f} (top {r['cadTop']:.2f})  meas "
                    f"{r['measMedian'] if r['measMedian'] is not None else '-'} (filled {r['filledMedian']:.2f})  "
                    f"valid {r['validFraction']:.2f}  range {r['rangeTop']:.2f}")
            bar = np.full((24, strip.shape[1], 3), 255, np.uint8)
            cv2.putText(bar, info, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            name = re.sub(r'[^\w.+-]', '_', ref)
            cv2.imwrite(os.path.join(cdir, f'{name}.jpg'), np.concatenate([bar, strip], 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
