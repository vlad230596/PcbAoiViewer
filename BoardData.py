"""Board description for inspection, built from an Altium build export: boards/<name>/board.json.

    importBoard.py <artifacts.zip | build folder>   ->   boards/<project>-r<rev>/
        board.json          everything below, one file
        export/             the export files it was built from (BOM, Pick Place, IPC-2581, STEP, PDFs)
        datasheets/         copies of the matched datasheet .md files (+ their images), paths as in the library

board.json:
    version, source {board, project, ref, rev, files{kind: rel path}}, units "mm",
    frame       "IPC-2581 Datum, Y up, viewed from the top; bottom components are NOT mirrored"
    board       {bbox, outline (SVG path, mm), holes [[x, y, d, kind v/p/n, name, net]], slots [SVG path]}
    variant     Pick Place "Variant:" line ("No variations" when the project has none)
    lines       BOM lines [{designators, name, fit, fittedName, description, footprint, libRef, qty,
                            cat {group, sub, source, hint}, value {kind, value, unit, text, marking|voltage...},
                            ds {doc, method, mpn, status, vendor, title, file, excursus, sections{marking, ordering,
                                package: [{title, page, line, text}]}}}]
    components  {ref: {line (index into lines or null), side, x, y, rot, layer, package, part, footprint, comment,
                       height, outline (SVG path), bb [x0 y0 x1 y1], fit, fiducial,
                       pnp {x, y, rot} (Pick Place, for the cross-check),
                       pins [{pin, name, net, x, y, bb, side}]}}
    checks      {pnpDelta {max, median, worst[]}, notInBom[], notOnBoard[], warnings[]}

fit: 'fitted' | 'nc' (must stay empty) | 'optional' (Name "X/NC": X on some builds, empty on others) | 'none'
(component with no BOM line: pads, fiducials, mounting holes, test points).
"""
import json
import re
import statistics
from pathlib import Path

import AltiumExport
import Components
import Ipc2581

VERSION = 1
FIDUCIAL = re.compile(r"fiduc|^fid|\bfid\b|mark_?point", re.I)


def _sections(lib, docId):
    secs = lib.sections(docId)
    return {k: v for k, v in secs.items() if v}


def buildBoard(files, outDir, library=None, log=print):
    """files {kind: Path} (AltiumExport.FILES kinds, inside outDir/export) -> board dict, written to board.json."""
    outDir = Path(outDir)
    if "ipc2581" not in files and "pickPlace" not in files:
        raise FileNotFoundError("the export has neither IPC-2581 nor Pick Place: nothing to place components by")
    rec = AltiumExport.readBuildRecord(files["build"]) if "build" in files else {}
    warnings = []

    # ---- BOM lines: fit, type, value, datasheet
    bom = AltiumExport.readBom(files["bom"]) if "bom" in files else {"items": []}
    if "bom" not in files:
        warnings.append("no BOM in the export: components have no values")
    lines, lineOf = [], {}
    dsDir = outDir / "datasheets"
    for it in bom["items"]:
        fit, fittedName = Components.parseFit(it.get("name", ""))
        ds, docMeta = None, None
        if library:
            m = library.match(it)
            if m and not m.get("missing"):
                docMeta = library.byId.get(m["doc"])
                copy = library.bundle(m["doc"], dsDir)
                ds = {"doc": m["doc"], "method": m["method"], "mpn": m.get("mpn"), "token": m.get("token"),
                      "others": m.get("others", []), "status": docMeta["status"], "vendor": docMeta["vendor"],
                      "family": docMeta["family"], "title": docMeta["doc"], "revision": docMeta["revision"],
                      "file": copy.relative_to(outDir).as_posix(), "excursus": library.excursus(m["doc"]),
                      "sections": _sections(library, m["doc"])}
        cat = Components.classifyLine(it, docMeta)
        value = Components.parseValue(fittedName, cat["group"], cat["sub"]) if fittedName else None
        line = {"designators": it["designators"], "name": it.get("name", ""), "fit": fit, "fittedName": fittedName,
                "description": it.get("description", ""), "footprint": it.get("footprint", ""),
                "libRef": it.get("libRef", ""), "qty": it["qty"], "cat": cat, "value": value, "ds": ds}
        for ref in it["designators"]:
            lineOf[ref] = len(lines)
        lines.append(line)

    # ---- geometry: IPC-2581 first, Pick Place for the cross-check (or alone when there is no IPC-2581)
    pnp = AltiumExport.readPickPlace(files["pickPlace"]) if "pickPlace" in files else {"meta": {}, "components": {}}
    board = {"bbox": None, "outline": "", "holes": [], "slots": []}
    comps = {}
    if "ipc2581" in files:
        log(f"parsing {files['ipc2581'].name} ({files['ipc2581'].stat().st_size / 1e6:.1f} MB)...")
        idx, geo = Ipc2581.parse(files["ipc2581"])
        warnings += idx["warnings"]
        board = {"bbox": idx["bbox"], "outline": geo["outline"], "holes": geo["holes"], "slots": geo["slots"]}
        wpins = {}
        if "wirelist" in files:
            try:
                _, wpins = AltiumExport.readWirelist(files["wirelist"])
            except Exception as e:  # names are a nicety; never fail the board for them
                warnings.append(f"pin names from Wirelist not read: {e}")
        for ref, c in idx["components"].items():
            pins = []
            for pk in c["pins"]:
                p = idx["pins"][pk]
                num = pk.rpartition(".")[2]
                name = p["name"]
                w = wpins.get(pk)
                if w and w.get("name") and (not name or name == num):
                    name = w["name"]
                pins.append({"pin": num, "name": name, "net": p["net"], "x": p["x"], "y": p["y"], "bb": p["bb"],
                             "side": p["side"]})
            comps[ref] = {"side": c["side"], "x": c["x"], "y": c["y"], "rot": c["rot"], "layer": c["layer"],
                          "package": c["package"], "part": c["part"], "height": c["height"],
                          "description": c["description"], "outline": c["outline"], "bb": c["bb"], "pins": pins}
    for ref, p in pnp["components"].items():
        c = comps.get(ref)
        if c is None:
            if "ipc2581" in files:
                warnings.append(f"{ref}: in Pick Place, not in IPC-2581")
            c = comps[ref] = {"side": p["side"], "x": p["x"], "y": p["y"], "rot": p["rot"], "layer": p["layer"],
                              "package": p.get("footprint", ""), "part": "", "height": 0.0,
                              "description": p.get("description", ""), "outline": None,
                              "bb": [p["x"] - 0.25, p["y"] - 0.25, p["x"] + 0.25, p["y"] + 0.25], "pins": []}
        c["footprint"], c["comment"] = p.get("footprint", ""), p.get("comment", "")
        c["pnp"] = {"x": p["x"], "y": p["y"], "rot": p["rot"]}

    # ---- link components to BOM lines
    for ref, c in comps.items():
        li = lineOf.get(ref)
        c["line"] = li
        c["fit"] = lines[li]["fit"] if li is not None else "none"
        c["fiducial"] = bool(FIDUCIAL.search(" ".join((ref, c.get("package", ""), c.get("footprint", ""),
                                                        c.get("comment", "")))))
    deltas = []
    for ref, c in comps.items():
        if "pnp" in c and c.get("outline") is not None:
            d = ((c["x"] - c["pnp"]["x"]) ** 2 + (c["y"] - c["pnp"]["y"]) ** 2) ** 0.5
            deltas.append((round(d, 4), ref))
    deltas.sort(reverse=True)
    checks = {
        "pnpDelta": {"max": deltas[0][0] if deltas else None,
                     "median": round(statistics.median(d for d, _ in deltas), 4) if deltas else None,
                     "worst": [{"ref": r, "mm": d} for d, r in deltas[:10]]},
        "notInBom": sorted((r for r, c in comps.items() if c["line"] is None), key=AltiumExport.designatorKey),
        "notOnBoard": sorted((r for r in lineOf if r not in comps), key=AltiumExport.designatorKey),
        "warnings": warnings,
    }

    out = {
        "version": VERSION,
        "source": {"board": rec.get("board"), "project": Path(files.get("ipc2581") or files["pickPlace"]).stem,
                   "path": rec.get("path"), "ref": rec.get("ref"), "rev": rec.get("rev"), "author": rec.get("author"),
                   "finishedAt": rec.get("finishedAt"), "bomEncoding": bom.get("encoding"),
                   "datasheets": str(library.root) if library else None,
                   "files": {k: Path(v).relative_to(outDir).as_posix() for k, v in files.items()}},
        "units": "mm",
        "frame": "IPC-2581 Datum, Y up, viewed from the top; bottom components are NOT mirrored",
        "board": board,
        "variant": pnp["meta"].get("Variant"),
        "pickPlace": pnp["meta"],
        "lines": lines,
        "components": dict(sorted(comps.items(), key=lambda kv: AltiumExport.designatorKey(kv[0]))),
        "checks": checks,
    }
    (outDir / "board.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def loadBoard(path):
    """board.json (or its folder) -> dict."""
    p = Path(path)
    return json.loads((p / "board.json" if p.is_dir() else p).read_text(encoding="utf-8"))


def componentsOf(board, group=None, side=None, fit=None):
    """[(ref, component, line)] filtered by line group ("Resistors"…), side ("top"/"bottom"), fit."""
    out = []
    for ref, c in board["components"].items():
        line = board["lines"][c["line"]] if c["line"] is not None else None
        if group and (not line or line["cat"]["group"] != group):
            continue
        if side and c["side"] != side:
            continue
        if fit and c["fit"] != fit:
            continue
        out.append((ref, c, line))
    return out
