"""CLI: read the SMD resistor marking codes of a panorama with the local recognizer (ResistorCodes.py), or evaluate it.

    venv/Scripts/python.exe readResistorCodes.py out/captured_51.5            # -> markings.json parts[ref].resistorCode
    venv/Scripts/python.exe readResistorCodes.py --eval                       # accuracy on dataset/resistor_codes

Reading: every resistor of board.json that markings.json does not mark absent; parts/<ref>.jpg is tried in turns 0 and
180 (arrays, footprint *ARRAY*: all four), without looking at the first reading. The recognizer is trained on
dataset/resistor_codes (resistorDataset.py) leaving the part itself out (--with-self keeps it), so a part that is in the
dataset is still read by a model that never saw it.
Written into markings.json parts[ref] (nothing else is touched):
    resistorCode {code, score (0..1, geometric mean of the character probabilities), margin (log-score lead over the
                  runner-up code), runnerUp, turn (clockwise, from parts/<ref>.jpg), chars (per-cell best class, no
                  grammar), bar (orientation bar in that turn: 1 under the line, -1 above, 0 none), agree (with lines[0]: same | differs | none = no first reading | unread = nothing found),
                  bom (ok | mismatch | - : the code decoded against the BOM value, as markingCheck.py does), value}
Prints the disagreements with the first reading and with the BOM.

--eval: leave-one-part-out over every labelled real part of the dataset (stored upright crops, both turns tried, so the
turn is judged too; parts whose cut failed count as errors; synthetic glyphs train only), step by step (before / + the
orientation bar / + synthetic glyphs / variants), the bar's hits, the synthetic glyphs read by a real-only model,
train -> val by the manifest split and the per-class confusion; failures go to out/resistor_codes_eval/ (failures.jpg,
eval.json).
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

import ResistorCodes as rc
from markingCheck import decodeResistor
from resistorDataset import DEFAULT_DIR, isArray, resistorParts
from Components import formatSi


def bomCheck(code, line):
    v = (line or {}).get("value") or {}
    if not code or v.get("value") is None:
        return "-", ""
    got = decodeResistor(code)
    if got is None:
        return "-", ""
    ok = abs(got - v["value"]) <= max(1e-9, v["value"] * 0.005)
    return ("ok" if ok else "mismatch"), formatSi(got, "Ω")


def readPanorama(a):
    t0 = time.time()
    rec = rc.Recognizer.fromDataset(a.dataset)
    tTrain = time.time() - t0
    b = json.load(open(os.path.join(a.panorama, "board.json"), encoding="utf-8"))
    path = os.path.join(a.panorama, "markings.json")
    mk = json.load(open(path, encoding="utf-8"))
    source = a.source or os.path.basename(os.path.normpath(a.panorama))
    want = set(a.refs.split(",")) if a.refs else None
    counts, rows = {}, []
    t1 = time.time()
    nRead = 0
    for ref, m, c in resistorParts(b, mk):
        if (want and ref not in want) or m.get("present") is False:
            continue
        crop = cv2.imread(os.path.join(a.panorama, "parts", ref + ".jpg"))
        if crop is None:
            continue
        turns = (0, 90, 180, 270) if isArray(c.get("footprint")) else (0, 180)
        r = rec.read(crop, turns, exclude=None if a.with_self else f"{source}/{ref}")
        nRead += 1
        first = (m.get("lines") or [""])[0].strip().upper()
        line = b["lines"][c["line"]]
        if r is None:
            out = {"code": "", "score": 0.0, "margin": 0.0, "runnerUp": "", "turn": None, "chars": "", "bar": 0,
                   "agree": "unread", "bom": "-", "value": ""}
        else:
            bom, val = bomCheck(r["code"], line)
            out = {"code": r["code"], "score": round(r["score"], 3), "margin": round(r["margin"], 2),
                   "runnerUp": r["runnerUp"], "turn": r["turn"], "chars": r["chars"], "bar": r.get("bar", 0),
                   "agree": "none" if not first else ("same" if first == r["code"] else "differs"),
                   "bom": bom, "value": val}
        m["resistorCode"] = out
        counts[out["agree"]] = counts.get(out["agree"], 0) + 1
        if out["agree"] in ("differs", "unread") or out["bom"] == "mismatch":
            rows.append((ref, first, out, line["name"]))
    tRead = time.time() - t1
    json.dump(mk, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{nRead} resistors read in {tRead:.1f} s (+ {tTrain:.1f} s training); vs first reading: "
          + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    for ref, first, out, bomName in rows:
        print(f"  {ref:12} first {first or '-':5} local {out['code'] or '-':5} score {out['score']:.2f} "
              f"margin {out['margin']:5.1f} turn {out['turn']}  agree {out['agree']:7} "
              f"BOM {bomName}: {out['bom']}{' (' + out['value'] + ')' if out['bom'] == 'mismatch' else ''}")


STEPS = [
    # (name, settings applied to ResistorCodes for this run)
    ("before (no bar, no synth)", {"BAR_PRIOR": 0.0, "SYNTH_N": 0, "SYMMETRIC": set("0125689")}),
    ("+ orientation bar", {"BAR_PRIOR": None, "SYNTH_N": 0, "SYMMETRIC": set("0125689")}),
    ("+ synthetic glyphs (rare classes)", {"BAR_PRIOR": None, "SYNTH_N": None, "SYMMETRIC": None}),
    ("  variant: synth for every class", {"BAR_PRIOR": None, "SYNTH_N": None, "SYMMETRIC": None, "SYNTH_FOR": "all"}),
    ("  variant: 3 out of the reject class", {"BAR_PRIOR": None, "SYNTH_N": None, "SYMMETRIC": set("0125689HSXZ3E")}),
]


def evaluate(a):
    """Leave-one-part-out over every labelled real part (synthetic glyphs train only), step by step."""
    rows = [r for r in rc.loadManifest(a.dataset) if r["status"] in ("ok", "badseg", "noline", "nobody")]
    samples = {rc.partKey(r): rc.loadSample(a.dataset, r) for r in rows}
    origins = {}
    for r in rows:
        origins[r["origin"]] = origins.get(r["origin"], 0) + 1
    print(f"dataset: {len(rows)} labelled parts ({', '.join(f'{k} {v}' for k, v in sorted(origins.items()))}), "
          f"{sum(r['status'] == 'ok' for r in rows)} used for training")
    defaults = {k: getattr(rc, k) for k in ("BAR_PRIOR", "SYNTH_N", "SYMMETRIC", "SYNTH_FOR")}

    def configure(settings):
        for k, v in defaults.items():
            setattr(rc, k, v)
        for k, v in settings.items():
            setattr(rc, k, defaults[k] if v is None else v)

    def readBoth(rec, key):
        """Both turns of the stored upright sample -> best reading (turn 0 = the reference's turn)."""
        ctx, box = samples[key]
        if ctx is None:
            return None
        best = None
        for t in (0, 180):
            c, b = rc.turnSample(ctx, box, t)
            r = rec.readSample(c, b, key, useBar=rc.BAR_PRIOR > 0)
            if r is not None and (best is None or r["quality"] > best["quality"]):
                r["turn"] = t
                best = r
        return best

    def run(rec, test, loo=True):
        res = []
        for r in test:
            key = rc.partKey(r)
            p = readBoth(rec, key if loo else None) if loo else readBoth(rec, key)
            res.append({"ref": r["ref"], "source": r["source"], "label": r["label"], "code": p["code"] if p else "",
                        "turn": p["turn"] if p else None, "bar": p["bar"] if p else 0,
                        "score": round(p["score"], 3) if p else 0.0, "margin": round(p["margin"], 2) if p else 0.0,
                        "chars": p["chars"] if p else "", "ok": bool(p and p["code"] == r["label"]),
                        "footprint": r["footprint"]})
        return res

    table, results = [], {}
    for name, settings in STEPS:
        configure(settings)
        t0 = time.time()
        rec = rc.Recognizer.fromDataset(a.dataset)
        res = run(rec, rows)
        k = sum(x["ok"] for x in res)
        turnRight = sum(1 for x in res if x["turn"] == 0)
        table.append((name, k, len(res), turnRight, time.time() - t0))
        results[name] = res
        sure = [x for x in res if x["score"] >= 0.5 and x["margin"] >= 1.0]
        print(f"{name:36} {k}/{len(res)} = {100.0 * k / len(res):.0f} %   turn = reference {turnRight}/{len(res)}"
              f"   confident (score >= 0.5, margin >= 1) {len(sure)}, {sum(not x['ok'] for x in sure)} wrong"
              f"   ({time.time() - t0:.0f} s)")
    final = STEPS[2][0]
    configure(STEPS[2][1])
    rec = rc.Recognizer.fromDataset(a.dataset)
    res = results[final]
    prev = {x["ref"]: x for x in results[STEPS[0][0]]}

    # the orientation bar: on how many parts it speaks, and whether it ever contradicts the reference turn
    barUp = barDown = 0
    barRefs = []
    for r in rows:
        ctx, box = samples[rc.partKey(r)]
        if ctx is None:
            continue
        s = rc.barSide(ctx, box, rc.textLine(rc.bodyOf(ctx, box))[1])
        if s:
            barRefs.append(f"{r['ref']} {r['label']}{'' if s > 0 else ' (says turned!)'}")
            barUp += s > 0
            barDown += s < 0
    print(f"orientation bar: found on {barUp + barDown} parts ({', '.join(barRefs)}); "
          f"{barDown} contradict the reference turn")

    # domain gap: a model of real characters only, reading the synthetic glyphs of the classes it knows
    configure({"SYNTH_N": 0})
    realOnly = rc.Recognizer.fromDataset(a.dataset)
    synth = rc.synthChars(a.dataset, [], 10 ** 6)
    known = [(x, ch) for x, ch, _ in synth if ch in realOnly.classes]
    hit = sum(realOnly.classes[int(np.argmax(realOnly.charLogProb(x)[:len(realOnly.classes)]))] == ch for x, ch in known)
    if known:
        print(f"synthetic glyphs read by the real-only model: {hit}/{len(known)} = {100.0 * hit / len(known):.0f} % "
              f"(classes {''.join(sorted({ch for _, ch in known}))})")
    configure(STEPS[2][1])

    # held-out split
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    recTrain = rc.Recognizer.fromDataset(a.dataset, keep=lambda r: r["split"] == "train")
    held = run(recTrain, val, loo=False)
    print(f"train ({len(train)}) -> val ({len(val)}): {sum(x['ok'] for x in held)}/{len(held)}")

    # per-class confusion of single characters (cut for the label's length, leave-one-part-out)
    conf = {}
    for r in rows:
        ctx, box = samples[rc.partKey(r)]
        if ctx is None:
            continue
        norms, _, _ = rc.charImages(rc.bodyOf(ctx, box), len(r["label"]))
        if norms is None:
            continue
        for x, ch in zip(norms, r["label"]):
            lp = rec.charLogProb(x, rc.partKey(r))
            got = rec.classes[int(np.argmax(lp[:len(rec.classes)]))]
            conf.setdefault(ch, {}).setdefault(got, 0)
            conf[ch][got] += 1
    print("characters (cut for the label, leave-one-part-out): true -> read")
    for ch in sorted(conf):
        n = sum(conf[ch].values())
        wrong = ", ".join(f"{g} {v}" for g, v in sorted(conf[ch].items(), key=lambda x: -x[1]) if g != ch)
        print(f"    {ch}: {conf[ch].get(ch, 0)}/{n}" + (f"   ({wrong})" if wrong else ""))

    print("failures (final):")
    for x in res:
        if not x["ok"]:
            was = prev[x["ref"]]["code"] or "-"
            print(f"    {x['ref']:12} {x['label']:5} -> {x['code'] or '-':5} turn {x['turn']} score {x['score']:.2f} "
                  f"margin {x['margin']:.1f} cells {x['chars']:5} (before: {was})")
    fixed = [x["ref"] for x in res if x["ok"] and not prev[x["ref"]]["ok"]]
    broke = [x["ref"] for x in res if not x["ok"] and prev[x["ref"]]["ok"]]
    print(f"vs before: fixed {', '.join(fixed) or '-'}; broke {', '.join(broke) or '-'}")
    for thr in (0.5, 0.6):
        kept = [x for x in res if x["score"] >= thr and x["margin"] >= 1.0]
        print(f"    score >= {thr} and margin >= 1: {len(kept)} parts, {sum(not x['ok'] for x in kept)} wrong")

    os.makedirs(a.eval_out, exist_ok=True)
    json.dump({"steps": [{"name": n, "right": k, "parts": p, "turnRight": t} for n, k, p, t, _ in table],
               "leaveOneOut": res, "heldOut": held, "confusion": conf, "bar": barRefs},
              open(os.path.join(a.eval_out, "eval.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    tiles = []
    for x in res:
        if x["ok"]:
            continue
        ctx, box = samples[f"{x['source']}/{x['ref']}"]
        if ctx is None:
            continue
        s = 90.0 / ctx.shape[0]
        img = cv2.cvtColor(cv2.resize(ctx, None, fx=s, fy=s), cv2.COLOR_GRAY2BGR)
        t = np.zeros((120, 240, 3), np.uint8)
        t[26:26 + img.shape[0], :min(240, img.shape[1])] = img[:, :240]
        cv2.putText(t, f"{x['ref']} {x['label']} -> {x['code'] or '-'} t{x['turn']}", (3, 18), 0, 0.45, (0, 200, 255), 1)
        tiles.append(t)
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(os.path.join(a.eval_out, "failures.jpg"),
                    np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    configure({})
    print(f"-> {a.eval_out}/eval.json, failures.jpg")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("panorama", nargs="?", help="folder with board.json, parts/ and markings.json")
    ap.add_argument("--dataset", default=DEFAULT_DIR)
    ap.add_argument("--refs", help="comma-separated designators")
    ap.add_argument("--source", help="dataset source name of this panorama (default: its folder name)")
    ap.add_argument("--with-self", action="store_true", help="do not leave the part itself out of the training set")
    ap.add_argument("--eval", action="store_true", help="evaluate on the dataset instead of reading a panorama")
    ap.add_argument("--eval-out", default=os.path.join("out", "resistor_codes_eval"))
    a = ap.parse_args()
    if a.eval:
        evaluate(a)
    elif a.panorama:
        readPanorama(a)
    else:
        ap.error("a panorama folder or --eval")


if __name__ == "__main__":
    main()
