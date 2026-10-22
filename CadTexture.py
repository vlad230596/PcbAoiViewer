"""Photo texture of the board on its STEP model (stepToMesh.py -> boards/<board>/model/mesh.npz + parts.json).

Frames:
  * STEP mm = IPC-2581 Datum mm (checked on SocketTester r79: part bbox centres vs board.json bb, median offset
    < 0.0001 mm, no mirror), z up, board top at z = 0, so board.json "align".M maps STEP x, y to panorama px.
  * focus.jpg of the stack is ortho-corrected with the measured height map: the top of a part of height h appears
    at its true board XY. Where the true height H is above the measured one hm (tall parts above the depth range,
    walls blurred by the focus measure), the pixel q of a model point p (= M x, y) solves
        q = c + (p - c) (D - hm(q)) / (D - H)
    with c the centre of the frame that supplied q (nearest centre, the Voronoi rule of the stitcher) and D the
    camera distance. PanoramaProjector iterates it; with hm = H it is q = p.

Outputs (TexturedModelWriter): one board-aligned texture (mm grid, so the uv of a point at board level is linear in
x, y; texture size up to 8k), a GLB with the textured mesh (TEXCOORD_0 photo uv; custom vertex attributes
_STEPCOLOR (STEP colour), _BODYCOLOR (median photo colour of the part top, for its walls), _PHOTO (1 where the photo may be used: top side, seen from above), _PART (part index)),
model_parts.json, the measured relief on a mm grid (model_relief.bin) and model.html.
"""
import json
import math
import os
import struct
from collections import OrderedDict

import cv2
import numpy as np


def readEmbeddedData(htmlPath):
    """The `const DATA = {...}` object BuildViewer.py embeds in index.html (frame positions, sizes)."""
    text = open(htmlPath, encoding='utf-8').read()
    i = text.index('const DATA = ') + len('const DATA = ')
    obj, _ = json.JSONDecoder().raw_decode(text[i:])
    return obj


# ---------------------------------------------------------------- model

class CadModel:
    """mesh.npz + parts.json, grouped by designator (path[2]); board.json component data attached."""

    def __init__(self, modelDir, components):
        m = np.load(os.path.join(modelDir, 'mesh.npz'))
        self.vertices = m['vertices'].astype(np.float32)
        self.normals = m['normals'].astype(np.float32)
        self.triangles = m['triangles'].astype(np.uint32)
        self.leaves = json.load(open(os.path.join(modelDir, 'parts.json'), encoding='utf-8'))
        self.components = components
        self.parts = OrderedDict()           # designator -> {leaves, side, fit, ...}
        for leaf in self.leaves:
            ref = leaf['path'][2] if len(leaf['path']) > 2 else leaf['name']
            self.parts.setdefault(ref, []).append(leaf)
        self.refs = list(self.parts)
        n = len(self.vertices)
        self.partOfVertex = np.zeros(n, np.uint16)
        self.colours = np.zeros((n, 3), np.float32)
        for i, ref in enumerate(self.refs):
            for leaf in self.parts[ref]:
                s, c = leaf['vertexStart'], leaf['vertexCount']
                self.partOfVertex[s:s + c] = i
                self.colours[s:s + c] = leaf['color'] or (0.6, 0.6, 0.6)

    def side(self, ref):
        if ref == 'Board':
            return 'board'
        c = self.components.get(ref)
        return c['side'] if c else 'top'

    def fit(self, ref):
        c = self.components.get(ref)
        return c['fit'] if c else ('board' if ref == 'Board' else 'unknown')

    def bbox(self, ref):
        lo = np.min([l['bboxMin'] for l in self.parts[ref]], 0)
        hi = np.max([l['bboxMax'] for l in self.parts[ref]], 0)
        return lo, hi

    def topSideVertices(self):
        """True for vertices that can be seen by a camera above the board: not of a bottom-side part, above the
        board's underside."""
        bottom = np.array([self.side(r) == 'bottom' for r in self.refs])
        return ~bottom[self.partOfVertex] & (self.vertices[:, 2] > -0.05)

    def partTable(self):
        out = []
        for i, ref in enumerate(self.refs):
            c = self.components.get(ref, {})
            lo, hi = self.bbox(ref)
            out.append({'ref': ref, 'side': self.side(ref), 'fit': self.fit(ref), 'comment': c.get('comment', ''),
                        'package': c.get('package', ''), 'description': c.get('description', ''),
                        'step': self.parts[ref][0]['name'], 'leaves': len(self.parts[ref]),
                        'bboxMin': [round(float(v), 3) for v in lo], 'bboxMax': [round(float(v), 3) for v in hi]})
        return out


class TopVisibility:
    """Z-buffer of the model seen from straight above (max surface z per cell), rasterised from non-vertical
    top-side triangles; each triangle is filled with its lowest z (conservative), painter's order."""

    def __init__(self, model, rect, pxPerMm=20.0, steepNz=0.3):
        self.rect, self.s = rect, pxPerMm
        x0, y0, x1, y1 = rect
        self.W, self.H = int(math.ceil((x1 - x0) * pxPerMm)), int(math.ceil((y1 - y0) * pxPerMm))
        V, T = model.vertices, model.triangles
        topSide = model.topSideVertices()
        fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
        nz = fn[:, 2] / (np.linalg.norm(fn, axis=1) + 1e-12)
        sel = (nz > steepNz) & topSide[T].all(1)
        tri = T[sel]
        zmin = V[tri, 2].min(1)
        order = np.argsort(zmin)
        pts = np.round(self.toGrid(V[:, :2]) * 16).astype(np.int32)   # 4 fractional bits
        buf = np.full((self.H, self.W), -10.0, np.float32)
        for t, z in zip(tri[order], zmin[order]):
            cv2.fillConvexPoly(buf, pts[t], float(z), cv2.LINE_8, 4)
        self.zbuf = cv2.erode(buf, np.ones((3, 3), np.uint8))   # min over 0.15 mm: no aliasing at face edges

    def toGrid(self, xy):
        x0, y0, x1, y1 = self.rect
        return np.stack([(xy[:, 0] - x0) * self.s, (y1 - xy[:, 1]) * self.s], 1)

    def visible(self, xyz, tol=0.1):
        g = np.clip(np.round(self.toGrid(xyz[:, :2])).astype(int), 0, [self.W - 1, self.H - 1])
        return self.zbuf[g[:, 1], g[:, 0]] <= xyz[:, 2] + tol


# ---------------------------------------------------------------- photo

class PanoramaProjector:
    """Model mm (x, y, z above the board top) -> panorama px of the ortho-corrected stack image."""

    def __init__(self, M, centres, distance, heights, cellPx, matMm=-0.5, closePx=48.0):
        self.M = np.asarray(M, np.float64)
        self.centres = np.asarray(centres, np.float64)
        self.D = float(distance)
        self.cell = cellPx
        self.raw = np.nan_to_num(heights.astype(np.float32), nan=-10.0)
        self.h = self.orthoHeights(self.raw, matMm, closePx / cellPx)

    @staticmethod
    def orthoHeights(h, matMm, closeCells):
        """The height map the ortho composite used (FocusStack.orthoHeights, approximately: from the final height
        map, the below-range mask is not saved): 0 on the mat, clipped to >= 0 elsewhere."""
        r = max(1, int(round(closeCells)))
        mat = cv2.morphologyEx((h < matMm).astype(np.uint8), cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        return np.where(mat > 0, 0.0, np.maximum(h, 0.0)).astype(np.float32)

    def boardPx(self, xy):
        return xy @ self.M[:, :2].T + self.M[:, 2]

    def heightAt(self, q, raw=False):
        """Bilinear height at panorama px: the ortho-composite map, or the measured one (raw)."""
        f = self.cell
        h = self.raw if raw else self.h
        H, W = h.shape
        mx = np.clip((q[:, 0] + 0.5) / f - 0.5, 0, W - 1.001)
        my = np.clip((q[:, 1] + 0.5) / f - 0.5, 0, H - 1.001)
        ix, iy = mx.astype(int), my.astype(int)
        fx, fy = mx - ix, my - iy
        return ((h[iy, ix] * (1 - fx) + h[iy, ix + 1] * fx) * (1 - fy)
                + (h[iy + 1, ix] * (1 - fx) + h[iy + 1, ix + 1] * fx) * fy)

    def nearestCentre(self, p):
        out = np.empty_like(p)
        for i in range(0, len(p), 200000):
            b = p[i:i + 200000]
            d2 = ((b[:, None, :] - self.centres[None, :, :]) ** 2).sum(2)
            out[i:i + 200000] = self.centres[np.argmin(d2, 1)]
        return out

    def project(self, xyz, parallax=True, iterations=4):
        p = self.boardPx(xyz[:, :2].astype(np.float64))
        if not parallax:
            return p
        H = np.clip(xyz[:, 2].astype(np.float64), 0, 0.5 * self.D)
        c = self.nearestCentre(p)
        q = p.copy()
        for _ in range(iterations):
            hm = self.heightAt(q)
            q = c + (p - c) * ((self.D - hm) / (self.D - H))[:, None]
        return q


class BoardTexture:
    """The panorama resampled onto a board-aligned mm grid: texture px (i, j) <-> mm x = x0 + (i + .5) / s,
    y = y1 - (j + .5) / s. uv = ((x - x0) / w, (y1 - y) / h), glTF convention (v down, flipY = false)."""

    def __init__(self, rect, longSide):
        self.rect = rect
        x0, y0, x1, y1 = rect
        self.s = longSide / max(x1 - x0, y1 - y0)
        self.W = int(round((x1 - x0) * self.s))
        self.H = int(round((y1 - y0) * self.s))

    def texToPano(self, M):
        """2x3 affine texture px -> panorama px."""
        x0, y0, x1, y1 = self.rect
        s = self.s
        T = np.array([[1 / s, 0, x0 + 0.5 / s], [0, -1 / s, y1 - 0.5 / s], [0, 0, 1]])   # tex px -> mm
        M3 = np.vstack([np.asarray(M, np.float64), [0, 0, 1]])
        return (M3 @ T)[:2]

    def render(self, pano, M):
        A = self.texToPano(M)
        return cv2.warpAffine(pano, A, (self.W, self.H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    def uvOfPano(self, q, M):
        """panorama px -> uv (through the inverse of M)."""
        M3 = np.vstack([np.asarray(M, np.float64), [0, 0, 1]])
        mm = (np.linalg.inv(M3)[:2] @ np.vstack([q.T, np.ones(len(q))])).T
        x0, y0, x1, y1 = self.rect
        return np.stack([(mm[:, 0] - x0) / (x1 - x0), (y1 - mm[:, 1]) / (y1 - y0)], 1)


def bodyColours(model, uv, texture, topTriangles, erodePx=6):
    """Per part: median photo colour (RGB 0..1) over its top faces (topTriangles(ref) -> triangle ids), rasterised
    in texture space through the uv and eroded, so the walls can take the colour of the body they belong to.
    Bottom-side parts and parts without a visible top keep their STEP colour."""
    H, W = texture.shape[:2]
    out = []
    for ref in model.refs:
        leaf = model.parts[ref][0]
        colour = np.array(leaf['color'] or (0.6, 0.6, 0.6), np.float64)
        if model.side(ref) != 'bottom':
            tri = topTriangles(ref)[0]
            if len(tri):
                T = model.triangles[tri]
                px = uv[T] * [W, H]
                lo = np.maximum(np.floor(px.reshape(-1, 2).min(0)).astype(int) - 2, 0)
                hi = np.minimum(np.ceil(px.reshape(-1, 2).max(0)).astype(int) + 2, [W, H])
                if (hi - lo).min() > 2 * erodePx + 2:
                    mask = np.zeros((hi[1] - lo[1], hi[0] - lo[0]), np.uint8)
                    for t in np.round((px - lo) * 16).astype(np.int32):
                        cv2.fillConvexPoly(mask, t, 255, cv2.LINE_8, 4)
                    mask = cv2.erode(mask, np.ones((2 * erodePx + 1,) * 2, np.uint8))
                    sel = texture[lo[1]:hi[1], lo[0]:hi[0]][mask > 0]
                    if len(sel) > 20:
                        colour = np.median(sel, 0)[::-1] / 255.0     # BGR -> RGB
        out.append(colour)
    return np.array(out, np.float32)


# ---------------------------------------------------------------- registration check

class RegistrationCheck:
    """Per part: rasterise the top faces of the STEP body (upward, within 0.25 mm of its highest point) into the
    panorama through the projector and walk its outline. At every outline point the strongest photo edge along the
    outline normal (+-searchPx) gives an offset t; a robust least-squares fit of t = n . (dx, dy) + grow separates
    a shift of the photo against the CAD (dx, dy) from a uniform size difference of the body (grow > 0: the photo
    body is larger than the CAD top face)."""

    def __init__(self, model, projector, grey, searchPx=25):
        self.model, self.proj, self.searchPx = model, projector, searchPx
        self.grey = cv2.GaussianBlur(grey, (0, 0), 1.5).astype(np.float32)
        V, T = model.vertices, model.triangles
        fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
        self.nz = fn[:, 2] / (np.linalg.norm(fn, axis=1) + 1e-12)

    def topTriangles(self, ref):
        V, T = self.model.vertices, self.model.triangles
        idx = np.concatenate([np.arange(l['triangleStart'], l['triangleStart'] + l['triangleCount'])
                              for l in self.model.parts[ref]])
        zmax = V[T[idx], 2].max()
        keep = idx[(self.nz[idx] > 0.9) & (V[T[idx], 2].min(1) >= zmax - 0.25)]
        return keep, zmax

    def outline(self, ref, parallax):
        """Outline points (panorama px) and outward normals of the projected top faces."""
        tri, zmax = self.topTriangles(ref)
        if len(tri) == 0:
            return None, None, zmax
        T = self.model.triangles[tri]
        vid = np.unique(T)
        q = self.proj.project(self.model.vertices[vid], parallax)
        lut = np.full(len(self.model.vertices), -1, np.int64)
        lut[vid] = np.arange(len(vid))
        lo = np.floor(q.min(0)).astype(int) - 4
        hi = np.ceil(q.max(0)).astype(int) + 4
        mask = np.zeros((hi[1] - lo[1], hi[0] - lo[0]), np.uint8)
        pts = np.round((q - lo) * 16).astype(np.int32)
        for t in T:
            cv2.fillConvexPoly(mask, pts[lut[t]], 255, cv2.LINE_8, 4)
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        P, N = [], []
        for c in cs:
            c = c[:, 0, :].astype(np.float64)
            if len(c) < 40:
                continue
            k = 5
            tang = np.roll(c, -k, 0) - np.roll(c, k, 0)
            n = np.stack([tang[:, 1], -tang[:, 0]], 1)
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
            # outward: a step along n leaves the mask
            probe = np.round(c + 3 * n).astype(int)
            inside = mask[np.clip(probe[:, 1], 0, mask.shape[0] - 1), np.clip(probe[:, 0], 0, mask.shape[1] - 1)] > 0
            n[inside] *= -1
            P.append(c[::3] + lo)
            N.append(n[::3])
        if not P:
            return None, None, zmax
        return np.vstack(P), np.vstack(N), zmax

    def measure(self, ref, parallax):
        P, N, zmax = self.outline(ref, parallax)
        if P is None or len(P) < 30:
            return None
        S = self.searchPx
        t = np.arange(-S, S + 1, dtype=np.float64)
        X = (P[:, None, 0] + N[:, None, 0] * t[None, :]).astype(np.float32)
        Y = (P[:, None, 1] + N[:, None, 1] * t[None, :]).astype(np.float32)
        prof = cv2.remap(self.grey, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        d = np.abs(np.gradient(prof, axis=1))
        best = np.argmax(d, 1)
        strength = d[np.arange(len(d)), best] / (np.median(d, 1) + 1.0)
        off = t[best]
        ok = strength > 3
        if ok.sum() < 20:
            return None
        A = np.hstack([N, np.ones((len(N), 1))])[ok]
        y = off[ok]
        w = np.ones(len(y))
        for _ in range(6):     # IRLS, Huber-like
            sol = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
            r = y - A @ sol
            w = 1.0 / np.maximum(1.0, np.abs(r) / 3.0)
        inl = np.abs(y - A @ sol) < 4
        return {'ref': ref, 'heightMm': round(float(zmax), 2), 'dxPx': round(float(sol[0]), 1),
                'dyPx': round(float(sol[1]), 1), 'growPx': round(float(sol[2]), 1),
                'shiftPx': round(math.hypot(sol[0], sol[1]), 1), 'score': round(float(inl.mean()), 3),
                'points': int(ok.sum())}

    def run(self, refs):
        out = []
        for ref in refs:
            a = self.measure(ref, False)
            b = self.measure(ref, True)
            if a and b:
                out.append({'ref': ref, 'heightMm': a['heightMm'], 'plain': a, 'parallax': b})
        return out


# ---------------------------------------------------------------- export

class GlbWriter:
    """Minimal glTF 2.0 binary writer: one mesh, one primitive, external JPEG texture."""

    def __init__(self):
        self.bin = bytearray()
        self.views, self.accessors = [], []

    def add(self, arr, componentType, typ, normalized=False, target=None, minmax=False):
        arr = np.ascontiguousarray(arr)
        while len(self.bin) % 4:
            self.bin.append(0)
        off = len(self.bin)
        self.bin += arr.tobytes()
        view = {'buffer': 0, 'byteOffset': off, 'byteLength': arr.nbytes}
        if target:
            view['target'] = target
        self.views.append(view)
        acc = {'bufferView': len(self.views) - 1, 'componentType': componentType, 'count': int(arr.shape[0]),
               'type': typ}
        if normalized:
            acc['normalized'] = True
        if minmax:
            acc['min'] = [float(v) for v in arr.min(0)]
            acc['max'] = [float(v) for v in arr.max(0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def write(self, path, attributes, indices, textureUri, extras):
        F, U8, U16, U32 = 5126, 5121, 5123, 5125
        attr = {}
        for name, (arr, ctype, typ, norm) in attributes.items():
            attr[name] = self.add(arr, ctype, typ, norm, 34962, minmax=(name == 'POSITION'))
        ind = self.add(indices.reshape(-1), U32, 'SCALAR', target=34963)
        gltf = {
            'asset': {'version': '2.0', 'generator': 'PcbAoiViewer CadTexture.py'},
            'scene': 0, 'scenes': [{'nodes': [0]}],
            # model is z up in mm; glTF is y up in m: rotate -90 deg about x, scale 1/1000
            'nodes': [{'mesh': 0, 'rotation': [-math.sqrt(0.5), 0, 0, math.sqrt(0.5)], 'scale': [0.001] * 3}],
            'meshes': [{'primitives': [{'attributes': attr, 'indices': ind, 'material': 0}]}],
            'materials': [{'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}, 'metallicFactor': 0,
                                                    'roughnessFactor': 0.8}, 'doubleSided': False}],
            'textures': [{'source': 0, 'sampler': 0}],
            'samplers': [{'magFilter': 9729, 'minFilter': 9987, 'wrapS': 33071, 'wrapT': 33071}],
            'images': [{'uri': textureUri}],
            'buffers': [{'byteLength': len(self.bin)}],
            'bufferViews': self.views, 'accessors': self.accessors, 'extras': extras,
        }
        js = json.dumps(gltf, separators=(',', ':')).encode()
        js += b' ' * (-len(js) % 4)
        while len(self.bin) % 4:
            self.bin.append(0)
        with open(path, 'wb') as fp:
            fp.write(struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(self.bin)))
            fp.write(struct.pack('<II', len(js), 0x4E4F534A) + js)
            fp.write(struct.pack('<II', len(self.bin), 0x004E4942) + bytes(self.bin))


def reliefGrid(projector, rect, stepMm, valid=None):
    """Measured height (mm above the board) sampled on a board mm grid; NaN where the panorama has no frames."""
    x0, y0, x1, y1 = rect
    xs = np.arange(x0, x1 + 1e-9, stepMm)
    ys = np.arange(y1, y0 - 1e-9, -stepMm)
    X, Y = np.meshgrid(xs, ys)
    p = projector.boardPx(np.stack([X.ravel(), Y.ravel()], 1))
    h = projector.heightAt(p, raw=True).reshape(X.shape)
    if valid is not None:
        f = projector.cell
        ix = np.clip((p[:, 0] / f).astype(int), 0, valid.shape[1] - 1)
        iy = np.clip((p[:, 1] / f).astype(int), 0, valid.shape[0] - 1)
        h[~valid[iy, ix].reshape(X.shape)] = np.nan
    return h, (float(xs[0]), float(ys[0]), float(stepMm))
