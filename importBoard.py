"""Import an Altium build exported by the registry server into boards/<project>-r<rev>/board.json.

    python importBoard.py <artifacts.zip | build folder> [--out boards/<name>] [--datasheets <library>]
                          [--links <datasheet-links.json>] [--no-datasheets]

<artifacts.zip>   "Download all" of a build on the server (GET /api/builds/<key>/artifacts.zip)
<build folder>    an unpacked one, or the server's own data/builds/<repo>/.../r<rev>
See BoardData.py for what board.json holds.
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import AltiumExport
import BoardData
import Datasheets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="artifacts.zip or a build folder")
    ap.add_argument("--out", help="output folder (default boards/<project>-r<rev>)")
    ap.add_argument("--datasheets", default=str(Datasheets.DEFAULT_ROOT), help="datasheet library working copy")
    ap.add_argument("--links", default=str(Datasheets.DEFAULT_LINKS),
                    help="manual BOM line -> document links of the server (data/datasheet-links.json)")
    ap.add_argument("--no-datasheets", action="store_true", help="skip datasheet matching")
    a = ap.parse_args()

    src = Path(a.input)
    tmp = Path(tempfile.mkdtemp(prefix="importBoard-"))
    try:
        files = AltiumExport.extractArchive(src, tmp) if src.is_file() else AltiumExport.findFiles(src)
        if not files:
            sys.exit(f"no export files found in {src}")
        primary = files.get("ipc2581") or files.get("bom") or files.get("pickPlace")
        if primary is None:
            sys.exit(f"{src}: no IPC-2581, BOM or Pick Place in the export (found: {', '.join(files)})")
        rec = AltiumExport.readBuildRecord(files["build"]) if "build" in files else {}
        project = Path(primary).stem
        project = project.replace("Bill of Materials Print-", "").replace("Pick Place for ", "")
        name = f"{project}-r{rec['rev']}" if rec.get("rev") else project
        out = Path(a.out) if a.out else Path(__file__).resolve().parent / "boards" / name
        export = out / "export"
        if out.exists() and any(out.iterdir()) and not (out / "board.json").exists():
            sys.exit(f"{out} exists and is not a board folder (no board.json): refusing to overwrite it")
        if export.exists():
            shutil.rmtree(export)
        if (out / "datasheets").exists():
            shutil.rmtree(out / "datasheets")
        root = tmp if src.is_file() else src
        kept = {}
        for kind, f in files.items():
            dst = export / Path(f).relative_to(root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            kept[kind] = dst
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    lib = None
    if not a.no_datasheets:
        try:
            lib = Datasheets.DatasheetLibrary(a.datasheets, a.links)
        except FileNotFoundError as e:
            print(f"! {e}; continuing without datasheets")
    board = BoardData.buildBoard(kept, out, lib)

    comps, lines = board["components"], board["lines"]
    fit = {}
    for c in comps.values():
        fit[c["fit"]] = fit.get(c["fit"], 0) + 1
    groups = {}
    for ln in lines:
        g = ln["cat"]["group"]
        groups[g] = groups.get(g, 0) + len(ln["designators"])
    withDs = [ln for ln in lines if ln["ds"]]
    withMark = [ln for ln in withDs if ln["ds"]["sections"].get("marking")]
    ck = board["checks"]
    print(f"{out}")
    print(f"  board {board['source']['board']} {board['source']['ref']} r{board['source']['rev']}, "
          f"variant: {board['variant']}")
    print(f"  {len(comps)} components ({', '.join(f'{k} {v}' for k, v in sorted(fit.items()))}), "
          f"{len(lines)} BOM lines, {len(board['board']['holes'])} holes")
    print("  " + ", ".join(f"{g} {n}" for g, n in groups.items()))
    print(f"  datasheets: {len(withDs)} of {len(lines)} lines matched, {len(withMark)} with a marking section")
    print(f"  values parsed: {sum(1 for ln in lines if ln['value'])} lines")
    print(f"  Pick Place vs IPC-2581: max {ck['pnpDelta']['max']} mm, median {ck['pnpDelta']['median']} mm")
    if ck["notInBom"]:
        print(f"  not in BOM: {', '.join(ck['notInBom'])}")
    if ck["notOnBoard"]:
        print(f"  in BOM, not on the board: {', '.join(ck['notOnBoard'])}")
    for w in ck["warnings"]:
        print(f"  ! {w}")


if __name__ == "__main__":
    main()
