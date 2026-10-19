"""Board geometry from the IPC-2581 output of an Altium build (`IPC2581/*.cvg`, Altium, rev B).

Ported from Antilatency.Electronics.Registry.Server `tools/ipc2581.py` (INDEX_VERSION 3): `parse` and `resolve` are
the server's code unchanged, so fixes can be copied both ways; the server's cache files (pcb/index.json) are dropped.

Stdlib only. Streams the XML (iterparse), so 80+ MB files fit in memory. parse(path) -> (index, geometry):

    index      board bbox, layers, components (refdes, side, x, y, rot, package, part, height, description,
               outline, bb, pins), pins (net, name, x, y, bb, side), nets, holes summary, warnings
    geometry   per layer per net: tracks, pours, pad instances; pad shapes; holes; slots; board outline (SVG path)

Coordinates: mm from Step/Datum, Y up (math), angles CCW, view from the top. Bottom side is NOT mirrored
(the file is "top view through the board"); a component's `side` says which face it sits on.

Pitfalls handled (see the ipc-2581 skill): Datum, units (MILLIMETER/INCH/MICRON scaled to mm), UserSpecial at any
depth, clockwise="false" = CCW, duplicate holes (PadStack/LayerHole vs LayerFeature/Set/Hole), Oval/RectRound/
RectCham pad shapes, Features with a bare primitive + Location/Xform, SlotCavity slots. Polygon rings are
re-oriented (outer CCW, cutouts CW) so a whole layer fills with the nonzero rule. Negative layers are only
flagged (`polarity`), not inverted.
"""
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

INDEX_VERSION = 3
NS = "{http://webstds.ipc.org/2581}"
UNITS = {"MILLIMETER": 1.0, "MICRON": 0.001, "INCH": 25.4}
COPPER = {"SIGNAL", "PLANE", "MIXED", "CONDUCTOR", "CONDFILM", "CONDFOIL", "POWER_GROUND"}
KINDS = {"LEGEND": "silk", "SOLDERMASK": "mask", "PASTEMASK": "paste", "DOCUMENT": "doc"}
SHAPES = ("Circle", "RectCenter", "RectRound", "RectCham", "Oval", "Contour", "Ellipse")


def _t(e):
    return e.tag[len(NS):] if e.tag.startswith(NS) else e.tag


def fmt(v):
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


# ---------------------------------------------------------------- geometry helpers

def _arc_cmd(x1, y1, x2, y2, cx, cy, ccw):
    """SVG arc from (x1,y1) to (x2,y2) around (cx,cy); full circle when start == end. Math space (Y up)."""
    r = math.hypot(x1 - cx, y1 - cy)
    if r < 1e-9:
        return f"L{fmt(x2)} {fmt(y2)}"
    sweep = 1 if ccw else 0
    if abs(x1 - x2) < 1e-6 and abs(y1 - y2) < 1e-6:  # full circle: two halves
        mx, my = 2 * cx - x1, 2 * cy - y1
        return (f"A{fmt(r)} {fmt(r)} 0 0 {sweep} {fmt(mx)} {fmt(my)}"
                f"A{fmt(r)} {fmt(r)} 0 0 {sweep} {fmt(x2)} {fmt(y2)}")
    a1, a2 = math.atan2(y1 - cy, x1 - cx), math.atan2(y2 - cy, x2 - cx)
    span = (a2 - a1) % (2 * math.pi) if ccw else (a1 - a2) % (2 * math.pi)
    large = 1 if span > math.pi else 0
    return f"A{fmt(r)} {fmt(r)} 0 {large} {sweep} {fmt(x2)} {fmt(y2)}"


class Ring:
    """Closed ring: start point + segments ('L', x, y) / ('A', x, y, cx, cy, ccw)."""
    __slots__ = ("x0", "y0", "segs")

    def __init__(self, x0, y0, segs):
        self.x0, self.y0, self.segs = x0, y0, segs

    def area(self):
        pts = [(self.x0, self.y0)] + [(s[1], s[2]) for s in self.segs]
        a = 0.0
        for (xa, ya), (xb, yb) in zip(pts, pts[1:] + pts[:1]):
            a += xa * yb - xb * ya
        # arcs: add the circular segment area (signed) so rounded shapes orient right
        px, py = self.x0, self.y0
        for s in self.segs:
            if s[0] == "A":
                x, y, cx, cy, ccw = s[1:]
                r = math.hypot(px - cx, py - cy)
                a1, a2 = math.atan2(py - cy, px - cx), math.atan2(y - cy, x - cx)
                th = (a2 - a1) % (2 * math.pi) if ccw else -((a1 - a2) % (2 * math.pi))
                if abs(px - x) < 1e-6 and abs(py - y) < 1e-6:
                    th = 2 * math.pi if ccw else -2 * math.pi
                a += r * r * (th - math.sin(th))
            px, py = s[1], s[2]
        return a / 2

    def reversed(self):
        pts = [(self.x0, self.y0)] + [(s[1], s[2]) for s in self.segs]
        segs = []
        for i in range(len(self.segs) - 1, -1, -1):
            s, (x, y) = self.segs[i], pts[i]
            segs.append(("L", x, y) if s[0] == "L" else ("A", x, y, s[3], s[4], not s[5]))
        return Ring(pts[-1][0], pts[-1][1], segs)

    def path(self):
        out = [f"M{fmt(self.x0)} {fmt(self.y0)}"]
        px, py = self.x0, self.y0
        for s in self.segs:
            if s[0] == "L":
                out.append(f"L{fmt(s[1])} {fmt(s[2])}")
            else:
                out.append(_arc_cmd(px, py, *s[1:]))
            px, py = s[1], s[2]
        out.append("Z")
        return "".join(out)

    def bbox(self):
        xs = [self.x0] + [s[1] for s in self.segs]
        ys = [self.y0] + [s[2] for s in self.segs]
        for s in self.segs:  # arcs: conservative circle box
            if s[0] == "A":
                r = math.hypot(s[1] - s[3], s[2] - s[4])
                xs += [s[3] - r, s[3] + r]; ys += [s[4] - r, s[4] + r]
        return [min(xs), min(ys), max(xs), max(ys)]


def _ring(poly, tr):
    """Polygon/Cutout element -> Ring with coordinates mapped by tr(x, y)."""
    x0 = y0 = None
    segs = []
    for ch in poly:
        k = _t(ch)
        if k == "PolyBegin":
            x0, y0 = tr(float(ch.get("x")), float(ch.get("y")))
        elif k == "PolyStepSegment":
            segs.append(("L",) + tr(float(ch.get("x")), float(ch.get("y"))))
        elif k == "PolyStepCurve":
            x, y = tr(float(ch.get("x")), float(ch.get("y")))
            cx, cy = tr(float(ch.get("centerX")), float(ch.get("centerY")))
            segs.append(("A", x, y, cx, cy, ch.get("clockwise") == "false"))
    if x0 is None:
        return None
    if segs and abs(segs[-1][1] - x0) < 1e-9 and abs(segs[-1][2] - y0) < 1e-9 and segs[-1][0] == "L":
        segs.pop()  # closing segment is implied by Z
    return Ring(x0, y0, segs)


def _contour_path(c, tr):
    """Contour (Polygon + Cutout*) -> (path with outer CCW / cutouts CW, bbox of the outer ring)."""
    parts, bb = [], None
    for ch in c:
        k = _t(ch)
        if k not in ("Polygon", "Cutout"):
            continue
        r = _ring(ch, tr)
        if r is None:
            continue
        a = r.area()
        if (k == "Polygon") != (a > 0):
            r = r.reversed()
        if k == "Polygon" and bb is None:
            bb = r.bbox()
        parts.append(r.path())
    return "".join(parts), bb


def _rect_path(w, h, rx=0.0, cham=0.0):
    x, y = w / 2, h / 2
    if rx > 1e-9:
        rx = min(rx, x, y)
        return (f"M{fmt(x - rx)} {fmt(-y)}A{fmt(rx)} {fmt(rx)} 0 0 1 {fmt(x)} {fmt(-y + rx)}L{fmt(x)} {fmt(y - rx)}"
                f"A{fmt(rx)} {fmt(rx)} 0 0 1 {fmt(x - rx)} {fmt(y)}L{fmt(-x + rx)} {fmt(y)}"
                f"A{fmt(rx)} {fmt(rx)} 0 0 1 {fmt(-x)} {fmt(y - rx)}L{fmt(-x)} {fmt(-y + rx)}"
                f"A{fmt(rx)} {fmt(rx)} 0 0 1 {fmt(-x + rx)} {fmt(-y)}Z")
    if cham > 1e-9:
        c = min(cham, x, y)
        return (f"M{fmt(-x + c)} {fmt(-y)}L{fmt(x - c)} {fmt(-y)}L{fmt(x)} {fmt(-y + c)}L{fmt(x)} {fmt(y - c)}"
                f"L{fmt(x - c)} {fmt(y)}L{fmt(-x + c)} {fmt(y)}L{fmt(-x)} {fmt(y - c)}L{fmt(-x)} {fmt(-y + c)}Z")
    return f"M{fmt(-x)} {fmt(-y)}L{fmt(x)} {fmt(-y)}L{fmt(x)} {fmt(y)}L{fmt(-x)} {fmt(y)}Z"


def _shape(e, k):
    """Standard primitive (local coordinates, mm) -> (SVG path, half-extents box [x0,y0,x1,y1])."""
    f = lambda a, d=0.0: float(e.get(a, d)) * _S
    if k == "Circle":
        r = f("diameter") / 2
        return (f"M{fmt(r)} 0A{fmt(r)} {fmt(r)} 0 0 1 {fmt(-r)} 0A{fmt(r)} {fmt(r)} 0 0 1 {fmt(r)} 0Z",
                [-r, -r, r, r])
    if k in ("RectCenter", "RectRound", "RectCham", "Oval", "Ellipse"):
        w, h = f("width"), f("height")
        if k == "Oval" or k == "Ellipse":
            p = _rect_path(w, h, rx=min(w, h) / 2)
        elif k == "RectRound":
            p = _rect_path(w, h, rx=f("radius"))
        elif k == "RectCham":
            p = _rect_path(w, h, cham=f("chamfer"))
        else:
            p = _rect_path(w, h)
        return p, [-w / 2, -h / 2, w / 2, h / 2]
    if k == "Contour":
        p, bb = _contour_path(e, lambda x, y: (x * _S, y * _S))
        return p, bb or [0, 0, 0, 0]
    return None, None


_S = 1.0  # unit scale of the file being parsed (module-level for the helpers; parse is not reentrant)


def _rot_box(bb, x, y, rot):
    """Box of a local box rotated by rot degrees and moved to (x, y)."""
    a = math.radians(rot)
    c, s = math.cos(a), math.sin(a)
    pts = [(px * c - py * s, px * s + py * c) for px in (bb[0], bb[2]) for py in (bb[1], bb[3])]
    return [x + min(p[0] for p in pts), y + min(p[1] for p in pts), x + max(p[0] for p in pts), y + max(p[1] for p in pts)]


def _grow(a, b):
    if a is None:
        return list(b)
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def _r3(b):
    return [round(v, 3) for v in b]


# ---------------------------------------------------------------- parse

def find_input(build):
    build = Path(build)
    for pat in ("*.cvg", "*.xml", "*.CVG", "*.XML"):
        f = next(iter(sorted((build / "IPC2581").glob(pat))), None)
        if f:
            return f
    return None


def parse(path):
    """IPC-2581 file -> (index dict, geometry dict). Streams the file."""
    global _S
    _S = 1.0
    datum = [0.0, 0.0]
    T = lambda x, y: ((x * _S) - datum[0], (y * _S) - datum[1])  # file coords -> mm from datum
    shapes, shape_ix, shape_box = [], {}, []  # dictionary id -> index in shapes
    colors, layers, layer_order = {}, {}, {}
    geo = {}  # layer -> net -> {"l": {w: [path parts]}, "p": [paths], "d": [[shape, x, y, rot, pin]]}
    pads_bb = {}  # "REF.PIN" -> box (all layers)
    pins, comps, packages, bom = {}, {}, {}, {}
    holes, hole_seen, slots, slot_seen = [], set(), [], set()
    outline, warnings, cur_layer, steps = "", [], None, 0
    counts = {"lines": 0, "arcs": 0, "contours": 0, "pads": 0}
    skipped = {}

    def G(layer, net):
        return geo.setdefault(layer, {}).setdefault(net, {"l": {}, "p": [], "d": [], "b": None, "a": 0.0})

    def add_hole(x, y, d, plating, name, net=""):
        key = (round(x, 4), round(y, 4), round(d, 4))
        if key in hole_seen:
            return
        hole_seen.add(key)
        holes.append([round(x, 4), round(y, 4), round(d, 4), {"VIA": "v", "PLATED": "p", "NONPLATED": "n"}.get(plating, "p"), name, net])

    def features(fe, layer, net):
        g = G(layer, net)
        loc = fe.find(NS + "Location")
        for e in fe.iter():
            k = _t(e)
            if k in ("Line", "Arc"):
                d = e.find(NS + "LineDesc")
                w = fmt(float(d.get("lineWidth", 0)) * _S) if d is not None else "0"
                x1, y1 = T(float(e.get("startX")), float(e.get("startY")))
                x2, y2 = T(float(e.get("endX")), float(e.get("endY")))
                parts = g["l"].setdefault(w, [])
                h = float(w) / 2
                g["b"] = _grow(g["b"], [min(x1, x2) - h, min(y1, y2) - h, max(x1, x2) + h, max(y1, y2) + h])
                if k == "Line":
                    parts.append(f"M{fmt(x1)} {fmt(y1)}L{fmt(x2)} {fmt(y2)}")
                    counts["lines"] += 1
                else:
                    cx, cy = T(float(e.get("centerX")), float(e.get("centerY")))
                    parts.append(f"M{fmt(x1)} {fmt(y1)}" + _arc_cmd(x1, y1, x2, y2, cx, cy, e.get("clockwise") == "false"))
                    r = math.hypot(x1 - cx, y1 - cy) + h
                    g["b"] = _grow(g["b"], [cx - r, cy - r, cx + r, cy + r])
                    counts["arcs"] += 1
            elif k == "Contour":
                p, bb = _contour_path(e, T)
                if p:
                    g["p"].append(p)
                    g["b"] = _grow(g["b"], bb or [0, 0, 0, 0])
                    if bb:
                        g["a"] += (bb[2] - bb[0]) * (bb[3] - bb[1])
                    counts["contours"] += 1
            elif k in SHAPES and k != "Contour" and loc is not None and e in list(fe):
                # bare primitive placed by the Features' own Location / Xform
                p, bb = _shape(e, k)
                if p:
                    sid = len(shapes); shapes.append(p); shape_box.append(bb)
                    xf = fe.find(NS + "Xform")
                    x, y = T(float(loc.get("x")), float(loc.get("y")))
                    rot = float(xf.get("rotation", 0)) if xf is not None else 0.0
                    g["d"].append([sid, round(x, 4), round(y, 4), rot, ""])
                    g["b"] = _grow(g["b"], _rot_box(bb, x, y, rot))
                    counts["pads"] += 1
            elif k in ("Polyline", "Text", "UserPrimitiveRef", "LineDescRef"):
                skipped[k] = skipped.get(k, 0) + 1

    ctx = ET.iterparse(str(path), events=("start", "end"))
    stack = []
    for ev, e in ctx:
        k = _t(e)
        if ev == "start":
            stack.append(k)
            if k == "LayerFeature":
                cur_layer = e.get("layerRef")
            elif k == "Step":
                steps += 1
            continue
        stack.pop()
        parent = stack[-1] if stack else ""
        if steps > 1 and "Step" in stack:  # panel: only the first Step
            if k == "Step":
                warnings.append("several Steps — using the first")
            e.clear()
            continue
        if k == "CadHeader":
            u = e.get("units", "MILLIMETER")
            if u not in UNITS:
                raise ValueError(f"unsupported units: {u}")
            _S = UNITS[u]
        elif k == "EntryStandard":
            ch = next(iter(e), None)
            if ch is not None:
                p, bb = _shape(ch, _t(ch))
                if p:
                    shape_ix[e.get("id")] = len(shapes); shapes.append(p); shape_box.append(bb)
                else:
                    skipped["shape:" + _t(ch)] = skipped.get("shape:" + _t(ch), 0) + 1
            e.clear()
        elif k == "EntryColor":
            c = e.find(NS + "Color")
            if c is not None and e.get("id", "").startswith("LAYER_COLOR_"):
                colors[e.get("id")[12:]] = "#%02x%02x%02x" % (int(c.get("r")), int(c.get("g")), int(c.get("b")))
        elif k == "Layer" and parent == "CadData":
            layers[e.get("name")] = {"function": e.get("layerFunction", ""), "side": e.get("side", ""),
                                     "polarity": e.get("polarity", "POSITIVE")}
        elif k == "StackupLayer":
            layer_order[e.get("layerOrGroupRef")] = int(e.get("sequence", 0) or 0)
        elif k == "BomItem":
            for rd in e.findall(NS + "RefDes"):
                bom[rd.get("name")] = {"description": e.get("description", ""), "partNumber": e.get("OEMDesignNumberRef", "")}
            e.clear()
        elif k == "Datum" and parent == "Step":
            datum[:] = [float(e.get("x")) * _S, float(e.get("y")) * _S]
        elif k == "Profile" and parent == "Step":
            outline = _contour_path(e, T)[0]  # Profile holds Polygon + Cutout* directly
            e.clear()
        elif k == "PadStack":
            net = e.get("net") or ""
            net = "" if net == "No Net" else net
            for lh in e.findall(NS + "LayerHole"):
                x, y = T(float(lh.get("x")), float(lh.get("y")))
                add_hole(x, y, float(lh.get("diameter")) * _S, lh.get("platingStatus", ""), lh.get("name", ""), net)
            for lp in e.findall(NS + "LayerPad"):
                loc, ref = lp.find(NS + "Location"), lp.find(NS + "StandardPrimitiveRef")
                if loc is None or ref is None:
                    continue
                sid = shape_ix.get(ref.get("id"))
                if sid is None:
                    skipped["pad-ref"] = skipped.get("pad-ref", 0) + 1
                    continue
                xf, pr = lp.find(NS + "Xform"), lp.find(NS + "PinRef")
                rot = float(xf.get("rotation", 0)) if xf is not None else 0.0
                if xf is not None and xf.get("mirror") == "true":
                    rot = -rot  # TODO: mirrored pad shapes are not flipped (not seen in Altium exports)
                x, y = T(float(loc.get("x")), float(loc.get("y")))
                pk = ""
                if pr is not None:
                    pk = f"{pr.get('componentRef')}.{pr.get('pin')}"
                    bb = _rot_box(shape_box[sid], x, y, rot)
                    p = pins.setdefault(pk, {"net": net, "layers": [], "bb": None})
                    p["bb"] = _grow(p["bb"], bb)
                    p["layers"].append(lp.get("layerRef"))
                    if net and not p["net"]:
                        p["net"] = net
                g = G(lp.get("layerRef"), net)
                g["d"].append([sid, round(x, 4), round(y, 4), rot, pk])
                g["b"] = _grow(g["b"], _rot_box(shape_box[sid], x, y, rot))
                counts["pads"] += 1
            e.clear()
        elif k == "Package":
            pk = {"outline": None, "pins": {}, "height": float(e.get("height", 0) or 0) * _S}
            ol = e.find(NS + "Outline")
            if ol is not None:
                poly = ol.find(NS + "Polygon")
                if poly is not None:
                    pk["outline"] = _ring(poly, lambda x, y: (x * _S, y * _S))
            for pin in e.findall(NS + "Pin"):
                loc = pin.find(NS + "Location")
                pk["pins"][pin.get("number")] = {"name": pin.get("name", ""),
                                                 "x": float(loc.get("x")) * _S if loc is not None else 0.0,
                                                 "y": float(loc.get("y")) * _S if loc is not None else 0.0}
            packages[e.get("name")] = pk
            e.clear()
        elif k == "Component":
            xf, loc = e.find(NS + "Xform"), e.find(NS + "Location")
            x, y = T(float(loc.get("x")), float(loc.get("y"))) if loc is not None else (0.0, 0.0)
            comps[e.get("refDes")] = {
                "package": e.get("packageRef", ""), "part": e.get("part", ""), "layer": e.get("layerRef", ""),
                "x": x, "y": y, "rot": float(xf.get("rotation", 0)) if xf is not None else 0.0,
                "mirror": xf is not None and xf.get("mirror") == "true",
                "height": float(e.get("height", 0) or 0) * _S, "mount": e.get("mountType", "")}
            e.clear()
        elif k == "Features" and parent == "Set" and cur_layer:
            pass  # handled with its Set (needs the net)
        elif k == "Set" and parent == "LayerFeature":
            net = e.get("net") or ""
            net = "" if net == "No Net" else net
            for fe in e.findall(NS + "Features"):
                features(fe, cur_layer, net)
            for h in e.findall(NS + "Hole"):
                x, y = T(float(h.get("x")), float(h.get("y")))
                add_hole(x, y, float(h.get("diameter")) * _S, h.get("platingStatus", ""), h.get("name", ""), net)
            for sc in e.findall(NS + "SlotCavity"):
                poly = sc.find(f"{NS}Outline/{NS}Polygon")
                r = _ring(poly, T) if poly is not None else None
                if r is not None:
                    key = (sc.get("name"), round(r.x0, 3), round(r.y0, 3))
                    if key not in slot_seen:
                        slot_seen.add(key)
                        slots.append(r.path())
            e.clear()
        elif k == "LayerFeature":
            cur_layer = None
            e.clear()

    # ------------------------------------------------ components: outline, pins (names from the package)
    def _pin_side(pad_layers, comp_side):
        """'both' for a pin with copper on both outer sides (through-hole), else the component's side."""
        sides = {layers.get(L, {}).get("side") for L in pad_layers if layers.get(L, {}).get("function") in COPPER}
        return "both" if {"TOP", "BOTTOM"} <= sides else comp_side

    out_comps, out_pins = {}, {}
    pins_of = {}
    for pk in pins:
        ref, _, num = pk.rpartition(".")
        pins_of.setdefault(ref, []).append(num)
    for ref, c in comps.items():
        pkg = packages.get(c["package"], {})
        a = math.radians(c["rot"])
        ca, sa = math.cos(a), math.sin(a)
        mx = -1.0 if c["mirror"] else 1.0

        def place(px, py, c=c, ca=ca, sa=sa, mx=mx):
            # rotate CCW, then mirror X (bottom), then move: checked against the LayerPad positions of 1182 pins
            # (MainModuleM r59) — "mirror, then rotate" puts 140 bottom pins wrong
            return (c["x"] + mx * (px * ca - py * sa), c["y"] + px * sa + py * ca)

        side = "bottom" if c["mirror"] or layers.get(c["layer"], {}).get("side") == "BOTTOM" else "top"
        ol, bb = None, None
        r = pkg.get("outline")
        if r is not None:
            pts = [place(r.x0, r.y0)] + [place(s[1], s[2]) for s in r.segs]
            ol = "M" + "L".join(f"{fmt(px)} {fmt(py)}" for px, py in pts) + "Z"
            bb = [min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)]
        nums = sorted(pins_of.get(ref, []), key=lambda s: (len(s), s))
        for num in nums:
            p = pins[f"{ref}.{num}"]
            bb = _grow(bb, p["bb"])
        if bb is None:
            bb = [c["x"] - 0.25, c["y"] - 0.25, c["x"] + 0.25, c["y"] + 0.25]
        out_comps[ref] = {
            "side": side, "x": round(c["x"], 4), "y": round(c["y"], 4), "rot": c["rot"], "layer": c["layer"],
            "package": c["package"], "part": c["part"], "height": round(c["height"], 3),
            "description": bom.get(ref, {}).get("description", ""), "outline": ol, "bb": _r3(bb),
            "pins": [f"{ref}.{n}" for n in nums]}
        for num in nums:
            p = pins[f"{ref}.{num}"]
            pp = pkg.get("pins", {}).get(num, {})
            cx, cy = (p["bb"][0] + p["bb"][2]) / 2, (p["bb"][1] + p["bb"][3]) / 2
            out_pins[f"{ref}.{num}"] = {"net": p["net"], "name": pp.get("name", ""), "x": round(cx, 4), "y": round(cy, 4),
                                        "bb": _r3(p["bb"]), "side": _pin_side(p["layers"], side)}
    for pk, p in pins.items():  # pads whose component is missing from the Component list
        if pk not in out_pins:
            cx, cy = (p["bb"][0] + p["bb"][2]) / 2, (p["bb"][1] + p["bb"][3]) / 2
            out_pins[pk] = {"net": p["net"], "name": "", "x": round(cx, 4), "y": round(cy, 4), "bb": _r3(p["bb"]), "side": "both"}

    # ------------------------------------------------ layers
    def kind(name, info):
        fn = info["function"]
        if fn in COPPER:
            return "copper"
        if fn in ("DRILL", "DIELCORE", "DIELPREG", "DIELBASE", "DIELADHV") or name.lower().startswith("drill drawing"):
            return None
        return KINDS.get(fn, "doc")

    def side_of(name, info):
        s = info["side"]
        if s in ("TOP", "BOTTOM"):
            return s.lower()
        n = name.lower()
        return "top" if n.startswith("top") else "bottom" if n.startswith("bottom") or " bottom " in f" {n} " else "inner"

    out_layers, out_geo = [], {}
    names = sorted(layers, key=lambda n: (layer_order.get(n, 10_000), n))
    for n in names:
        info = layers[n]
        kd = kind(n, info)
        g = geo.get(n, {})
        if kd is None or not any(v["l"] or v["p"] or v["d"] for v in g.values()):
            continue
        lnets = {}
        cnt = 0
        for net, v in g.items():
            if not (v["l"] or v["p"] or v["d"]):
                continue
            item = {}
            if v["l"]:
                item["l"] = {w: "".join(parts) for w, parts in v["l"].items()}
            if v["p"]:
                item["p"] = v["p"]
            if v["d"]:
                item["d"] = v["d"]
            item["b"] = _r3(v["b"] or [0, 0, 0, 0])
            if v["a"] > 1:
                item["a"] = round(v["a"])  # mm², sum of pour boxes: tells planes / pours from signal nets
            cnt += sum(len(parts) for parts in v["l"].values()) + len(v["p"]) + len(v["d"])
            lnets[net] = item
        out_geo[n] = lnets
        out_layers.append({"name": n, "kind": kd, "side": side_of(n, info), "function": info["function"],
                           "polarity": info["polarity"], "color": colors.get(n, ""), "items": cnt,
                           "nets": sorted(k for k in lnets if k)})

    nets = {}
    for pk, p in out_pins.items():
        if p["net"]:
            nets.setdefault(p["net"], {"pins": [], "layers": []})["pins"].append(pk)
    for L in out_layers:
        if L["kind"] == "copper":
            for net in L["nets"]:
                nets.setdefault(net, {"pins": [], "layers": []})["layers"].append(L["name"])
    for v in nets.values():
        v["pins"].sort(key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)])

    bb = None
    for s in re.findall(r"[MLA]([^MLAZ]+)", outline):  # every command ends with its end point
        v = s.split()
        x, y = float(v[-2]), float(v[-1])
        bb = _grow(bb, [x, y, x, y])
    for c in out_comps.values():
        bb = _grow(bb, c["bb"])
    if bb is None:
        bb = [0, 0, 100, 100]
    if skipped:
        warnings.append("skipped: " + ", ".join(f"{k} ×{v}" for k, v in sorted(skipped.items())))
    neg = [L["name"] for L in out_layers if L["polarity"] == "NEGATIVE"]
    if neg:
        warnings.append("negative layers shown as positive: " + ", ".join(neg))

    index = {
        "version": INDEX_VERSION, "units": "mm", "datum": [round(datum[0], 6), round(datum[1], 6)],
        "bbox": _r3(bb), "layers": out_layers, "components": out_comps, "pins": out_pins, "nets": nets,
        "holes": {"total": len(holes), "via": sum(1 for h in holes if h[3] == "v"),
                  "plated": sum(1 for h in holes if h[3] == "p"), "nonplated": sum(1 for h in holes if h[3] == "n"),
                  "slots": len(slots)},
        "counts": counts, "warnings": warnings,
    }
    geometry = {"version": INDEX_VERSION, "outline": outline, "shapes": shapes, "layers": out_geo,
                "holes": holes, "slots": slots}
    return index, geometry


def resolve(idx, tokens):
    """Selection tokens (REF, REF.PIN, REF.PINNAME, net) -> {components, pins, nets, unknown}. Mirrors pcb.js."""
    out = {"components": [], "pins": [], "nets": [], "unknown": []}
    add = lambda k, v: out[k].append(v) if v not in out[k] else None
    for t in tokens:
        t = t.strip().rstrip(",;")
        if not t:
            continue
        if t in idx["components"]:
            add("components", t); continue
        if t in idx["nets"]:
            add("nets", t); continue
        ref, dot, pin = t.rpartition(".")
        if dot and ref:
            keys = [t] if t in idx["pins"] else [k for k, v in idx["pins"].items()
                                                if k.startswith(ref + ".") and v["name"].upper() == pin.upper()]
            if keys:
                for k in keys:
                    add("pins", k)
                    if idx["pins"][k]["net"]:
                        add("nets", idx["pins"][k]["net"])
                continue
        out["unknown"].append(t)
    return out
