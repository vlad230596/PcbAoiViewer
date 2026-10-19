"""Files of an Altium build exported by Antilatency.Electronics.Registry.Server.

The server's "artifacts.zip" (GET /api/builds/<key>/artifacts.zip, also the "Download all" button) keeps the CI
folder layout; the zip may also be one folder deeper. Only what the AOI needs is taken out of it:

    BOM/*.csv                     lines: Designator, Name, Description, Footprint, LibRef, Layer, Quantity (ANSI)
    Pick Place/*.txt              per component: layer, X, Y, rotation (fixed width; the old .csv form is read too)
    IPC2581/*.cvg                 board outline, holes, components with outlines, pads, heights (Ipc2581.py)
    WireListNetlist/*.NET         symbol pin names (VDD, EN) for the pad numbers of IPC-2581
    ExportSTEP/*.step             3D model of the assembled board (kept as a file, not parsed)
    PCB Print/Assembly*.pdf, PCB 3D Print/*.pdf, Schematic Print/*.pdf   for a human
    build.json                    server's build record: board, path, ref, rev, author

Stdlib only. Readers are the server's (server.py read_bom/decode_ansi, tools/schpdf.py load_wirelist) with the
server plumbing taken out.
"""
import csv
import io
import json
import re
import zipfile
from pathlib import Path

# kind -> glob inside the build folder; the first match wins
FILES = {
    "bom": "BOM/*.csv",
    "pickPlace": "Pick Place/*.txt",
    "ipc2581": "IPC2581/*.cvg",
    "wirelist": "WireListNetlist/*.NET",
    "step": "ExportSTEP/*.step",
    "assembly": "PCB Print/Assembly*.PDF",
    "front3d": "PCB 3D Print/*Front*.PDF",
    "back3d": "PCB 3D Print/*Back*.PDF",
    "schematic": "Schematic Print/*.PDF",
    "build": "build.json",
}


def _match(rel, pattern):
    """Case-insensitive glob of a build-relative posix path."""
    return Path(rel.lower()).match(pattern.lower()) and len(Path(rel).parts) == len(Path(pattern).parts)


def extractArchive(zipPath, outDir):
    """Unpacks only the FILES of an artifacts.zip into outDir (layout kept). -> {kind: Path}."""
    outDir = Path(outDir)
    found = {}
    with zipfile.ZipFile(zipPath) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        # the build root is where BOM/ or IPC2581/ sits; strip whatever is above it
        roots = set()
        for n in names:
            parts = n.split("/")
            i = next((i for i, p in enumerate(parts[:-1]) if p in ("BOM", "IPC2581", "Pick Place", "ExportSTEP")), None)
            if i is not None:
                roots.add("/".join(parts[:i]))
        root = min(roots, key=len) if roots else ""
        prefix = root + "/" if root else ""
        for kind, pat in FILES.items():
            for n in sorted(names):
                if not n.startswith(prefix):
                    continue
                rel = n[len(prefix):]
                if _match(rel, pat):
                    dst = outDir / rel
                    if ".." in Path(rel).parts:
                        raise ValueError(f"unsafe path in archive: {n}")
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(n) as src, open(dst, "wb") as f:
                        while chunk := src.read(1 << 20):
                            f.write(chunk)
                    found[kind] = dst
                    break
    return found


def findFiles(buildDir):
    """{kind: Path} of FILES present in an unpacked build folder."""
    buildDir = Path(buildDir)
    rels = sorted(f.relative_to(buildDir).as_posix() for f in buildDir.rglob("*") if f.is_file())   # one walk
    found = {}
    for kind, pat in FILES.items():
        hit = next((r for r in rels if _match(r, pat)), None)
        if hit:
            found[kind] = buildDir / hit
    return found


def readBuildRecord(path):
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: rec.get(k) for k in ("key", "board", "boardId", "repoName", "path", "ref", "rev", "author", "label",
                                    "finishedAt", "svnUrl")}


# ---------------------------------------------------------------- BOM
# Comment column is headed "Name"; one row per Comment+Footprint+LibRef group; a blank line after the header;
# quoted fields may contain newlines. Altium writes the ANSI code page (cp1252: ± ° — ™; cp1251 for Cyrillic).

BOM_FIELDS = {"designator": "designators", "name": "name", "comment": "name", "description": "description",
              "footprint": "footprint", "libref": "libRef", "layer": "layer", "quantity": "qty"}


def decodeAnsi(b):
    if b.startswith(b"\xef\xbb\xbf"):
        return b[3:].decode("utf-8"), "utf-8"
    try:
        return b.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    # Cyrillic text in cp1251 is mostly bytes 0xC0-0xFF in runs; cp1252 text has them only as rare accents
    runs = len(re.findall(rb"[\xc0-\xff]{3,}", b))
    enc = "cp1251" if runs >= 3 else "cp1252"
    return b.decode(enc, errors="replace"), enc


def designatorKey(d):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", d)]


def readBom(path):
    """-> {encoding, header, items: [{designators[], name, description, footprint, libRef, layer, qty, ...}]}."""
    text, enc = decodeAnsi(Path(path).read_bytes())
    rows = [r for r in csv.reader(io.StringIO(text, newline="")) if any(c.strip() for c in r)]
    if not rows:
        raise ValueError(f"BOM is empty: {path}")
    header = [h.strip() for h in rows[0]]
    cols = [BOM_FIELDS.get(h.lower().replace(" ", ""), h) for h in header]
    items = []
    for r in rows[1:]:
        it = {c: " ".join(v.split()) for c, v in zip(cols, r)}  # fold multi-line cells
        des = sorted((x.strip() for x in it.get("designators", "").split(",") if x.strip()), key=designatorKey)
        it["designators"] = des
        try:
            it["qty"] = int(it.get("qty") or len(des))
        except ValueError:
            it["qty"] = len(des)
        items.append(it)
    items.sort(key=lambda it: designatorKey(it["designators"][0]) if it["designators"] else [])
    return {"encoding": enc, "header": header, "items": items}


# ---------------------------------------------------------------- Pick and Place
# Text form (CI OutJob): a preamble ("Date:", "Revision:", "Variant:", "Units used: mm"), then a header line and
# fixed-width columns: Designator Comment Layer Footprint Center-X(mm) Center-Y(mm) Rotation Description.
# Comment and Footprint may contain spaces ("MICRO-MATCH FOB.12P"), so columns are cut at the header's positions.
# CSV form (older exports, PickPlace/test.csv): the same preamble, then a quoted CSV with the same columns.

PNP_FIELDS = {"designator": "designator", "comment": "comment", "layer": "layer", "footprint": "footprint",
              "center-x(mm)": "x", "center-y(mm)": "y", "rotation": "rot", "description": "description",
              "center-x(mil)": "xMil", "center-y(mil)": "yMil"}


def _openQuote(s):
    s = s.strip()
    return s.startswith('"') and (len(s) < 2 or not s.endswith('"'))


def readPickPlace(path):
    """-> {meta: {Date, Revision, Variant, Units used, ...}, units, components: {ref: {comment, layer, side,
    footprint, x, y, rot, description}}}. x/y in mm (mil files are converted)."""
    text, _ = decodeAnsi(Path(path).read_bytes())
    lines = text.splitlines()
    meta, start = {}, None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if re.match(r'^"?Designator"?[\s,]', s):
            start = i
            break
        m = re.match(r"^([A-Za-z][A-Za-z ]*?):\s*(.*)$", s)
        if m and not re.match(r"^[A-Z]:\\", s):
            meta[m.group(1)] = m.group(2).strip()
    if start is None:
        raise ValueError(f"no header line in {path}")
    head = lines[start]
    rows = []
    if head.lstrip().startswith('"') or "," in head:
        rdr = csv.reader(io.StringIO("\n".join(lines[start:]), newline=""))
        header = next(rdr)
        rows = [dict(zip(header, r)) for r in rdr if any(c.strip() for c in r)]
    else:
        cols = [(m.start(), m.group(0)) for m in re.finditer(r"\S+", head)]
        desc = next((pos for pos, name in cols if name.lower() == "description"), None)
        body, i = lines[start + 1:], 0
        while i < len(body):
            ln = body[i]
            i += 1
            if not ln.strip():
                continue
            # a quoted Description may go on over several lines; inner quotes are not escaped (0.100" (2.54mm)),
            # so it is open until a line ends with the quote
            while desc is not None and _openQuote(ln[desc:]) and i < len(body):
                ln += " " + body[i].strip()
                i += 1
            row = {}
            for j, (pos, name) in enumerate(cols):
                end = cols[j + 1][0] if j + 1 < len(cols) else None
                row[name] = ln[pos:end].strip()
            rows.append(row)
    comps = {}
    mil = False
    for r in rows:
        it = {PNP_FIELDS.get(k.strip().lower(), k.strip()): (v or "").strip() for k, v in r.items() if k}
        if "xMil" in it:
            mil = True
            it["x"], it["y"] = it.pop("xMil"), it.pop("yMil")
        ref = it.pop("designator", "")
        if not ref:
            continue
        k = 0.0254 if mil else 1.0
        it["x"], it["y"] = round(float(it["x"]) * k, 4), round(float(it["y"]) * k, 4)
        it["rot"] = float(it.get("rot") or 0)
        it["description"] = it.get("description", "").strip('"')
        it["side"] = "bottom" if it.get("layer", "").lower().startswith("bottom") else "top"
        comps[ref] = it
    return {"meta": meta, "units": "mm", "components": comps}


# ---------------------------------------------------------------- WireList netlist (pin names)

_NODE = re.compile(r"^\[(\d+)\]\s+(.*\S)\s*$")


def readWirelist(path):
    """Altium Wirelist -> (components {ref: value}, pins {"REF.PIN": {"net", "name"}}).
    The Wire List truncates designators to 11 columns (C100_DC_5V -> C100_DC_5); the Component List keeps
    them whole, so truncated refs are restored by unique prefix."""
    comps, raw = {}, []
    sec = net = None
    text, _ = decodeAnsi(Path(path).read_bytes())   # Altium writes the ANSI code page
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("<<< Component List"):
            sec = "c"; continue
        if s.startswith("<<< Wire List"):
            sec = "w"; continue
        if not s:
            continue
        if sec == "c":
            value, ref = ln[:32].strip(), ln[32:47].strip()
            if ref:
                comps[ref] = value
        elif sec == "w":
            m = _NODE.match(s)
            if m:
                net = m.group(2); continue
            if s.startswith("NODE"):
                continue
            ref, pin, name = ln[8:19].strip(), ln[19:27].strip(), ln[27:42].strip()
            if ref:
                raw.append((ref, pin, name, net))
    pins = {}
    for ref, pin, name, net in raw:
        if ref not in comps:
            cand = [c for c in comps if c.startswith(ref)]
            ref = cand[0] if len(cand) == 1 else ref
        pins[f"{ref}.{pin}"] = {"net": net, "name": name}
    return comps, pins
