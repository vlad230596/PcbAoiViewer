"""CLI: the dataset of SMD resistor marking codes (ResistorCodes.py), grown from every panorama whose markings were read.

    venv/Scripts/python.exe resistorDataset.py add out/captured_51.5 --auto       # labels where two readings agree
    venv/Scripts/python.exe resistorDataset.py add out/captured_51.5              # labels = the reference reading
    venv/Scripts/python.exe resistorDataset.py correct out/captured_51.5 R91 512  # one part, the right label
    venv/Scripts/python.exe resistorDataset.py correct out/captured_51.5 R44 -    # not a usable sample
    venv/Scripts/python.exe resistorDataset.py synth --sheet out/synth_sheet.png  # synthetic glyphs (SynthGlyphs.py)
    venv/Scripts/python.exe resistorDataset.py stats

dataset/resistor_codes/ (tracked in git, small):
    manifest.csv            one row per part: image, bodyBox, label, source (panorama folder), board, ref, footprint,
                            turn (clockwise degrees from parts/<ref>.jpg to upright text), origin (auto | reference |
                            manual), split (train | val, by part: crc32(source/ref) % 5 == 0 -> val), status (ok |
                            badseg = the cut for the label needed costly splits / merges, evaluation only | nobody |
                            noline | dropped), segCost, note, added
    codes/<source>/<ref>.png        upright grey crop of the part body with 0.6 body heights more above and below
                                    (the orientation bar); bodyBox "x y w h" is the body inside it (what is read)
    chars/<source>/<ref>_<i>.png    normalised character cells (24 x 32) as cut for the label, for looking at
    review.csv              parts `add --auto` left out because the readings disagree (reference vs Qwen)
    synth/                  synthetic glyphs (manifest.csv: image, label, origin synth; params.json: the fitted look);
                            training only, never a test sample

add: resistors of board.json that markings.json reads as present.
     --auto (origin auto): only where the reference reading (lines) and Qwen (vlm.lines) agree after folding
     look-alikes (ocrMarkings.agree == 'same'), in the same turn, on a grammatical code; the rest goes to review.csv.
     The BOM never takes part in a label. Without --auto (origin reference): the reference reading, conf high / medium.
     Arrays (footprint *ARRAY*: text across the part) get the turn of the four in which the label fits best, judged by
     a recognizer trained on the other parts. Rows from `correct` (origin manual) are kept as they are.
correct: when a reading was wrong (readResistorCodes.py disagrees and the eye says the part reads X), store X for that
     part; '-' marks it unusable (dirty, out of focus). The turn is the reading's (markings.json), for arrays the
     best fit; --turn overrides it.
synth: measure the look of the real cells, fit the dot pitch / blur / stroke of SynthGlyphs to them, draw --per-class
     glyphs of every class (also E F H S X Y Z, which no board has shown yet).
"""
import argparse
import csv
import json
import os
import time
import zlib

import cv2
import numpy as np

import ResistorCodes as rc
from ocrMarkings import agree

FIELDS = ["image", "bodyBox", "label", "source", "board", "ref", "footprint", "turn", "origin", "split", "status",
          "segCost", "note", "added"]
DEFAULT_DIR = os.path.join("dataset", "resistor_codes")
VALID = {c for codes in rc.validCodes().values() for c in codes}


def isArray(footprint):
    return "ARRAY" in (footprint or "").upper()


def splitOf(source, ref):
    return "val" if zlib.crc32(f"{source}/{ref}".encode()) % 5 == 0 else "train"


def saveManifest(folder, rows):
    rows = sorted(rows, key=lambda r: (r["source"], r["ref"]))
    with open(os.path.join(folder, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def loadPanorama(pano):
    b = json.load(open(os.path.join(pano, "board.json"), encoding="utf-8"))
    mk = json.load(open(os.path.join(pano, "markings.json"), encoding="utf-8"))
    src = b.get("source", {})
    board = f"{src.get('board', '')}@r{src.get('rev', '')}" if src else ""
    return b, mk, board


def resistorParts(b, mk):
    """-> [(ref, marking entry, component)] for the resistors of the board that markings.json covers."""
    out = []
    for ref, m in mk["parts"].items():
        c = b["components"].get(ref)
        if not c or c["line"] is None:
            continue
        if b["lines"][c["line"]]["cat"]["group"] == "Resistors":
            out.append((ref, m, c))
    return out


def referenceLabel(m):
    """The reference reading alone (markings.json lines, conf high / medium) -> (label, None) or (None, why)."""
    label = (m.get("lines") or [""])[0].strip().upper()
    if m.get("conf") not in ("high", "medium"):
        return None, "low confidence"
    if label not in VALID:
        return None, "no valid code read"
    return label, None


def autoLabel(m):
    """Label without a human: the reference reading and Qwen (markings.json vlm) must agree exactly after folding
    look-alikes (ocrMarkings.agree == 'same'), in the same turn, on a grammatical code. -> (label, None) or (None, why).
    The BOM never takes part."""
    ref = [x.strip().upper() for x in (m.get("lines") or []) if x.strip()]
    vlm = m.get("vlm") or {}
    q = [x.strip().upper() for x in (vlm.get("lines") or []) if x.strip()]
    if not ref:
        return None, "no reference reading"
    if not q:
        return None, "no Qwen reading"
    if agree(ref, q) != "same":
        return None, "reference and Qwen differ"
    if vlm.get("turn") is not None and int(vlm["turn"]) % 360 != int(m.get("turn", 0)) % 360:
        return None, "reference and Qwen turns differ"
    for cand in (ref[0], q[0]):
        if cand in VALID:
            return cand, None
    return None, "not a resistor code"


def saveReview(folder, source, review):
    """review.csv: the parts the two readings do not agree on, per source (replaced on every add --auto)."""
    path = os.path.join(folder, "review.csv")
    fields = ["source", "ref", "reference", "vlm", "referenceTurn", "vlmTurn", "reason"]
    old = []
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            old = [r for r in csv.DictReader(f) if r["source"] != source]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for r in old + review:
            w.writerow(r)


def cmdSynth(a):
    """Draw synthetic glyphs (SynthGlyphs.py) with the look measured from the real cells into synth/."""
    import SynthGlyphs as sg
    folder = a.dataset
    real = rc.loadChars(folder)
    stats = sg.measure(real)
    p = sg.fit(real, stats)
    out = os.path.join(folder, "synth")
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        if f.endswith(".png"):
            os.remove(os.path.join(out, f))
    rng = np.random.default_rng(a.seed)
    classes = a.classes or "".join(sorted(sg.GLYPHS))
    with open(os.path.join(out, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["image", "label", "origin"])
        w.writeheader()
        for ch in classes:
            for i in range(a.per_class):
                name = f"{ch}_{i:03d}.png"
                cv2.imwrite(os.path.join(out, name), (sg.render(ch, p, rng) * 255).astype(np.uint8))
                w.writerow({"image": name, "label": ch, "origin": "synth"})
    json.dump(p, open(os.path.join(out, "params.json"), "w"), indent=1)
    print(f"synth: {a.per_class} x {len(classes)} glyphs ({classes}), fit distance {p['fitDistance']} "
          f"(pitch {p['pitch']}, blur {p['blur']}, stroke x{p['strokeScale']} of {p['stroke']:.1f} px)")
    if a.sheet:
        by = {}
        for x, c, _ in real:
            by.setdefault(c, []).append(x)
        rows = []
        for ch in classes:
            row = np.zeros((rc.CHAR_H, rc.CHAR_W * 17), np.float32)
            for i, x in enumerate(by.get(ch, [])[:8]):
                row[:, i * rc.CHAR_W:(i + 1) * rc.CHAR_W] = x
            for i in range(8):
                img = cv2.imread(os.path.join(out, f"{ch}_{i:03d}.png"), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    row[:, (9 + i) * rc.CHAR_W:(10 + i) * rc.CHAR_W] = img / 255.0
            row = cv2.cvtColor(cv2.resize((row * 255).astype(np.uint8), None, fx=2, fy=2,
                                          interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            cv2.putText(row, ch, (8 * 2 * rc.CHAR_W + 14, 40), 0, 0.9, (0, 220, 255), 2)
            rows.append(row)
        cv2.imwrite(a.sheet, np.vstack(rows))
        print(f"-> {a.sheet} (real left, synthetic right)")


def writeSample(folder, row, crop, rec=None, turns=None):
    """Cut the body of the part crop in the row's turn (or the best of `turns` for the label), save it and its
    character cells; fills image / turn / status / note."""
    key = f"{row['source']}/{row['ref']}"
    best = None
    for t in turns or [int(row["turn"])]:
        ctx, box = rc.bodyAndContext(rc.turn(crop, t))
        if ctx is None:
            continue
        body = rc.bodyOf(ctx, box)
        q = rec.labelQuality(body, row["label"], exclude=key) if rec is not None and len(turns or []) > 1 else 0.0
        if best is None or q > best[0]:
            best = (q, t, ctx, box)
    codeDir = os.path.join(folder, "codes", row["source"])
    charDir = os.path.join(folder, "chars", row["source"])
    os.makedirs(codeDir, exist_ok=True)
    os.makedirs(charDir, exist_ok=True)
    for f in os.listdir(charDir):
        if f.startswith(row["ref"] + "_"):
            os.remove(os.path.join(charDir, f))
    row["image"] = f"codes/{row['source']}/{row['ref']}.png"
    if best is None:
        row["status"] = "nobody"
        return row
    q, t, ctx, box = best
    body = rc.bodyOf(ctx, box)
    row["turn"] = str(t)
    row["bodyBox"] = " ".join(str(v) for v in box)
    if turns and len(turns) > 1 and rec is not None:
        row["note"] = (row.get("note") or "").split(" [turn")[0] + f" [turn picked by fit {q:.2f}]"
    cv2.imwrite(os.path.join(folder, row["image"]), ctx)
    norms, cost, _ = rc.charImages(body, len(row["label"]))
    if norms is None:
        row["status"] = "noline"
        return row
    row["segCost"] = f"{cost:.2f}"
    if row.get("status") != "dropped":
        # a cut that needed costly splits / merges is likely wrong: keep it for evaluation, not for training
        row["status"] = "ok" if cost <= rc.SEG_OK else "badseg"
    for i, x in enumerate(norms):
        cv2.imwrite(os.path.join(charDir, f"{row['ref']}_{i}.png"), (x * 255).astype(np.uint8))
    return row


def cmdAdd(a):
    folder = a.dataset
    os.makedirs(folder, exist_ok=True)
    b, mk, board = loadPanorama(a.panorama)
    source = a.source or os.path.basename(os.path.normpath(a.panorama))
    rows = {(r["source"], r["ref"]): r for r in rc.loadManifest(folder)}
    want = set(a.refs.split(",")) if a.refs else None
    todo, skipped, review = [], {}, []
    for ref, m, c in resistorParts(b, mk):
        if want and ref not in want:
            continue
        old = rows.get((source, ref))
        if old and old["origin"] == "manual":
            skipped["manual row kept"] = skipped.get("manual row kept", 0) + 1
            continue
        if not m.get("present"):
            skipped["not present"] = skipped.get("not present", 0) + 1
            continue
        label, why = autoLabel(m) if a.auto else referenceLabel(m)
        if why:
            skipped[why] = skipped.get(why, 0) + 1
            if a.auto:
                vlm = m.get("vlm") or {}
                review.append({"source": source, "ref": ref, "reference": " / ".join(m.get("lines") or []),
                               "vlm": " / ".join(vlm.get("lines") or []), "referenceTurn": m.get("turn", ""),
                               "vlmTurn": vlm.get("turn", ""), "reason": why})
            continue
        todo.append({"label": label, "source": source, "board": board, "ref": ref, "footprint": c.get("footprint", ""),
                     "turn": str(int(m.get("turn", 0)) % 360), "origin": "auto" if a.auto else "reference",
                     "split": splitOf(source, ref), "note": f"conf {m.get('conf', '-')}; " + (m.get("note") or "")[:70],
                     "added": time.strftime("%Y-%m-%d")})
    # parts with a trusted turn first; then the arrays, their turn picked by a recognizer trained on everything else
    sure = [r for r in todo if not isArray(r["footprint"])]
    unsure = [r for r in todo if isArray(r["footprint"])]
    for r in sure:
        crop = cv2.imread(os.path.join(a.panorama, "parts", r["ref"] + ".jpg"))
        if crop is None:
            skipped["no part crop"] = skipped.get("no part crop", 0) + 1
            continue
        rows[(source, r["ref"])] = writeSample(folder, r, crop)
    saveManifest(folder, rows.values())
    if unsure:
        rec = rc.Recognizer.fromDataset(folder)
        for r in unsure:
            crop = cv2.imread(os.path.join(a.panorama, "parts", r["ref"] + ".jpg"))
            if crop is None:
                continue
            t0 = int(r["turn"])
            rows[(source, r["ref"])] = writeSample(folder, r, crop, rec, [(t0 + k) % 360 for k in (0, 90, 180, 270)])
        saveManifest(folder, rows.values())
    if a.auto:
        saveReview(folder, source, review)
    st = {}
    for r in todo:
        st[r.get("status", "?")] = st.get(r.get("status", "?"), 0) + 1
    print(f"{source}: {len(todo)} samples ({', '.join(f'{k} {v}' for k, v in sorted(st.items()))}); skipped: "
          + ", ".join(f"{k} {v}" for k, v in sorted(skipped.items())))
    for r in todo:
        if r.get("status") != "ok":
            print(f"  {r['ref']:12} {r['label']:5} {r.get('status')}")
    for r in review:
        print(f"  review {r['ref']:12} reference {r['reference'] or '-':8} Qwen {r['vlm'] or '-':8} {r['reason']}")


def cmdCorrect(a):
    folder = a.dataset
    os.makedirs(folder, exist_ok=True)
    b, mk, board = loadPanorama(a.panorama)
    source = a.source or os.path.basename(os.path.normpath(a.panorama))
    rows = {(r["source"], r["ref"]): r for r in rc.loadManifest(folder)}
    c = b["components"].get(a.ref)
    m = mk["parts"].get(a.ref, {})
    if c is None:
        raise SystemExit(f"{a.ref}: not in board.json")
    label = a.label.strip().upper()
    old = rows.get((source, a.ref), {})
    if label == "-":
        if not old:
            raise SystemExit(f"{a.ref}: not in the dataset")
        old.update(status="dropped", origin="manual", note=(a.note or "dropped by hand"))
        saveManifest(folder, rows.values())
        print(f"{source}/{a.ref}: dropped")
        return
    if label not in VALID:
        raise SystemExit(f"'{label}' is not a resistor code (EIA-96 '01C', '103', '1002', 'R100', '4R7', '0')")
    crop = cv2.imread(os.path.join(a.panorama, "parts", a.ref + ".jpg"))
    if crop is None:
        raise SystemExit(f"{a.ref}: no parts/{a.ref}.jpg")
    row = {"label": label, "source": source, "board": board, "ref": a.ref, "footprint": c.get("footprint", ""),
           "turn": str(a.turn if a.turn is not None else int(m.get("turn", old.get("turn", 0) or 0)) % 360),
           "origin": "manual", "split": splitOf(source, a.ref), "note": a.note or f"corrected (was {old.get('label') or (m.get('lines') or ['-'])[0]})",
           "added": time.strftime("%Y-%m-%d")}
    if a.turn is not None or not isArray(row["footprint"]):
        row = writeSample(folder, row, crop)      # the reading's turn (codes like '512' / '215' fit both ways)
    else:
        rec = rc.Recognizer.fromDataset(folder)
        t0 = int(row["turn"])
        row = writeSample(folder, row, crop, rec, [(t0 + k) % 360 for k in (0, 90, 180, 270)])
    rows[(source, a.ref)] = row
    saveManifest(folder, rows.values())
    print(f"{source}/{a.ref}: {label}, turn {row['turn']}, {row['status']}")


def cmdStats(a):
    rows = rc.loadManifest(a.dataset)
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"{len(rows)} parts, {len(ok)} usable ({sum(r['split'] == 'val' for r in ok)} val), "
          f"sources: {', '.join(sorted({r['source'] for r in rows}))}")
    codes, chars = {}, {}
    for r in ok:
        codes[r["label"]] = codes.get(r["label"], 0) + 1
        for ch in r["label"]:
            chars[ch] = chars.get(ch, 0) + 1
    print("codes: " + ", ".join(f"{k} {v}" for k, v in sorted(codes.items(), key=lambda x: -x[1])))
    print("chars: " + ", ".join(f"{k} {v}" for k, v in sorted(chars.items())))
    missing = sorted(set("0123456789" + rc.EIA96_LETTERS) - set(chars))
    print("no sample yet: " + " ".join(missing))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DEFAULT_DIR)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("add", help="add / refresh the samples of a panorama from its markings.json")
    p.add_argument("panorama")
    p.add_argument("--source", help="source name (default: the panorama folder name)")
    p.add_argument("--refs", help="comma-separated designators")
    p.add_argument("--auto", action="store_true",
                   help="label only where the reference reading and Qwen (vlm) agree; the rest goes to review.csv")
    p = sub.add_parser("correct", help="store the right label of one part ('-' = not usable)")
    p.add_argument("panorama")
    p.add_argument("ref")
    p.add_argument("label")
    p.add_argument("--turn", type=int, help="clockwise turn of parts/<ref>.jpg to upright text (default: the "
                                            "reading's turn; arrays: the best fit of the four)")
    p.add_argument("--source")
    p.add_argument("--note")
    sub.add_parser("stats", help="counts per code and per character")
    p = sub.add_parser("synth", help="draw synthetic glyphs into synth/ (look measured from the real cells)")
    p.add_argument("--per-class", type=int, default=40)
    p.add_argument("--classes", help="characters to draw (default: all of SynthGlyphs.GLYPHS)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sheet", help="also write a contact sheet real vs synthetic per class to this file")
    a = ap.parse_args()
    {"add": cmdAdd, "correct": cmdCorrect, "stats": cmdStats, "synth": cmdSynth}[a.cmd](a)


if __name__ == "__main__":
    main()
