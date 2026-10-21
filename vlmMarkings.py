"""CLI: second reading of the part markings by a local vision-language model (Qwen3-VL-2B), run in its own venv.

    venv/Scripts/python.exe vlmMarkings.py out/captured_51.5 [--refs R1,U3] [--python <vlm venv python>] [--device cpu]
    venv/Scripts/python.exe vlmMarkings.py out/captured_51.5 --views [--model Qwen/Qwen3-VL-4B-Instruct]

The project venv stays light: vlmWorker.py runs in a separate venv with torch + transformers, started here as a
subprocess; jobs and results go through two JSON files in the panorama folder (vlm_jobs.json, vlm_results.json).

Which venv and weights (first match wins):
    --python / env AOI_VLM_PYTHON            python.exe of a venv with torch, transformers, opencv, pillow
    research/ocr_engines/venvs/gpu           the OCR survey's venv (torch 2.14 cu130, RTX 5080)
    --hf-home / env HF_HOME                  Hugging Face cache with the model
    research/ocr_engines/cache/hf            the survey's cache (Qwen3-VL-2B, 4 GB); run offline from it
A new venv: py -3.13 -m venv <dir>; pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision;
pip install transformers accelerate opencv-python-headless pillow numpy  (CPU: plain `pip install torch`, ~7 s/part).

Input per part: parts/<ref>.jpg turned by markings.json `turn` (the first reading's turn: the text stands upright).
Result into markings.json parts[ref].vlm = {lines, score, turn, agree, model}, agree as in ocrMarkings.py
(same / close / partial / differs / ocr-only / none, look-alike characters folded). Prints the disagreements.
Survey numbers (research/ocr_engines/REPORT.md): 90/116 exact on these crops, resistors 79/80, 0 false readings on
40 unmarked parts, 0.26 s/part on the GPU; ICs only ~4-6/17 (faint laser marking), so read those as a hint.

--views reads the source views of cutMarkingViews.py instead (marking_views.json): every view in all four turns, no
reference turn needed (an unread part has none). Per view the turn with the best score stands; the views then vote:
readings equal after folding look-alikes form a group, a group weighs the sum of its views' scores, the heaviest wins.
Result into parts[ref].vlmViews = {lines, score, frame, turn, support (views agreeing), views (views read),
readings [{frame, turn, lines, score}], agree (vs the first reading), model}.
"""
import argparse
import json
import os
import subprocess
import sys
import re
import time

from ocrMarkings import agree, editDistance, fold

HERE = os.path.dirname(os.path.abspath(__file__))
SURVEY = os.path.join(HERE, "research", "ocr_engines")
TURNS = (0, 90, 180, 270)


def runWorker(a, pano, jobs):
    jobsFile, resFile = os.path.join(pano, "vlm_jobs.json"), os.path.join(pano, "vlm_results.json")
    json.dump(jobs, open(jobsFile, "w", encoding="utf-8"), indent=1)
    env = dict(os.environ, HF_HOME=a.hf_home, PYTHONIOENCODING="utf-8", TRANSFORMERS_VERBOSITY="error")
    if os.path.isdir(os.path.join(a.hf_home, "hub", "models--" + a.model.replace("/", "--"))):
        env.setdefault("HF_HUB_OFFLINE", "1")   # weights are local: no network
    cmd = [a.python, os.path.join(HERE, "vlmWorker.py"), jobsFile, resFile, "--model", a.model]
    if a.device:
        cmd += ["--device", a.device]
    print(f"{len(jobs)} jobs -> {os.path.relpath(a.python, HERE) if a.python.startswith(HERE) else a.python}", flush=True)
    proc = subprocess.run(cmd, env=env)
    if proc.returncode:
        sys.exit(f"vlmWorker.py failed with code {proc.returncode}")
    return json.load(open(resFile, encoding="utf-8"))


def readParts(a, pano, board, mk, refs):
    crops = board["align"]["crops"]
    jobs = [{"ref": r, "image": os.path.join(pano, crops[r]["file"]), "turn": mk["parts"][r].get("turn", 0)}
            for r in refs if r in crops]
    t0 = time.time()
    res = runWorker(a, pano, jobs)
    counts, rows = {}, []
    for j in jobs:
        r, m = j["ref"], mk["parts"][j["ref"]]
        v = res["parts"].get(r, {"lines": [], "score": 0.0})
        lines = v["lines"] if v["score"] >= a.min_score else []
        m["vlm"] = {"lines": lines, "score": v["score"], "turn": j["turn"], "agree": agree(m["lines"], lines),
                    "model": res["model"]}
        counts[m["vlm"]["agree"]] = counts.get(m["vlm"]["agree"], 0) + 1
        if m["vlm"]["agree"] in ("differs", "close", "ocr-only"):
            rows.append(f"  {m['vlm']['agree']:8} {r:12} read {' / '.join(m['lines']) or '—':28} "
                        f"VLM {' / '.join(lines) or '—'} ({v['score']:.2f})")
    mk["vlm"] = {"model": res["model"], "device": res["device"], "minScore": a.min_score, "summary": counts,
                 "sec": res["sec"]}
    print(f"{len(jobs)} parts, {res['device']}, {time.time() - t0:.0f} s total: "
          + ", ".join(f"{k} {n}" for k, n in sorted(counts.items())))
    print("\n".join(rows))


def garbage(lines):
    """Degenerate generations: a character repeated 6+ times in a row, or an overlong line."""
    return any(len(l) > 24 or re.search(r"(.){5,}", l) for l in lines)


def vote(readings):
    """readings: the best turn of every view (the model reads sideways text too, so the turn is no key). Readings
    vote by text: greedy groups, best score first; a reading joins a group whose leader folds (look-alikes merged) to
    the same text, or, from 4 characters on, within an edit distance of len / 5 (short noise like J6 / J8 / JE must
    not pile up). A group weighs the sum of its views' scores, the heaviest wins.
    -> (winner = the group leader, support = views in the group) or (None, 0)."""
    groups = []
    for x in sorted(readings, key=lambda x: -x["score"]):
        key = fold(x["lines"])
        if not key:
            continue
        for g in groups:
            if key == g["key"] or (len(g["key"]) >= 4 and editDistance(key, g["key"]) <= max(1, len(g["key"]) // 5)):
                g["views"][x["frame"]] = max(g["views"].get(x["frame"], 0.0), x["score"])
                break
        else:
            groups.append({"key": key, "lead": x, "views": {x["frame"]: x["score"]}})
    if not groups:
        return None, 0
    best = max(groups, key=lambda g: (sum(g["views"].values()), g["lead"]["score"]))
    return best["lead"], len(best["views"])


def readViews(a, pano, mk, refs):
    views = json.load(open(os.path.join(pano, "marking_views.json"), encoding="utf-8"))
    jobs = []
    for r in refs:
        for v in views.get(r, {}).get("views", []):
            for t in TURNS:
                jobs.append({"ref": f"{r}|{v['frame']}|{t}", "image": os.path.join(pano, v["file"]), "turn": t})
    t0 = time.time()
    res = json.load(open(os.path.join(pano, "vlm_results.json"), encoding="utf-8")) if a.reuse else runWorker(a, pano, jobs)
    counts, rows = {}, []
    for r in refs:
        if r not in views or r not in mk["parts"]:
            continue
        m = mk["parts"][r]
        perView = []
        for v in views[r]["views"]:
            own = []
            for t in TURNS:
                x = res["parts"].get(f"{r}|{v['frame']}|{t}", {"lines": [], "score": 0.0})
                if x["lines"] and x["score"] >= a.min_score and not garbage(x["lines"]):
                    own.append({"frame": v["frame"], "turn": t, "lines": x["lines"], "score": x["score"]})
            if own:
                perView.append(max(own, key=lambda x: x["score"]))
        win, support = vote(perView)
        lines = win["lines"] if win else []
        m["vlmViews"] = {"lines": lines, "score": win["score"] if win else 0.0, "frame": win and win["frame"],
                         "turn": win and win["turn"], "support": support, "views": len(views[r]["views"]),
                         "readings": perView, "agree": agree(m.get("lines", []), lines), "model": res["model"]}
        k = m["vlmViews"]["agree"]
        counts[k] = counts.get(k, 0) + 1
        rows.append(f"  {k:8} {r:12} read {' / '.join(m.get('lines', [])) or '—':28} views {' / '.join(lines) or '—'}"
                    + (f" ({win['score']:.2f}, {win['frame']} turn {win['turn']}, {support}/{len(views[r]['views'])})"
                       if win else ""))
    mk["vlmViews"] = {"model": res["model"], "device": res["device"], "minScore": a.min_score, "summary": counts,
                      "sec": res["sec"]}
    print(f"{len(refs)} parts, {len(jobs)} jobs, {res['device']}, {time.time() - t0:.0f} s total: "
          + ", ".join(f"{k} {n}" for k, n in sorted(counts.items())))
    print("\n".join(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("panorama", help="folder with board.json, parts/ and markings.json")
    ap.add_argument("--refs", help="comma-separated designators (default: all in markings.json / marking_views.json)")
    ap.add_argument("--views", action="store_true", help="read the source views of cutMarkingViews.py, all turns")
    ap.add_argument("--reuse", action="store_true", help="with --views: vote again over the last vlm_results.json")
    ap.add_argument("--python", default=os.environ.get("AOI_VLM_PYTHON") or os.path.join(SURVEY, "venvs", "gpu", "Scripts",
                                                                                           "python.exe"))
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME") or os.path.join(SURVEY, "cache", "hf"))
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--device", help="cuda | cpu (default: cuda when available)")
    ap.add_argument("--min-score", type=float, default=0.5, help="readings below this mean token probability are dropped")
    a = ap.parse_args()

    if not os.path.isfile(a.python):
        sys.exit(f"no VLM venv python at {a.python}; pass --python or set AOI_VLM_PYTHON (see the header for setup)")
    pano = os.path.abspath(a.panorama)
    board = json.load(open(os.path.join(pano, "board.json"), encoding="utf-8"))
    path = os.path.join(pano, "markings.json")
    mk = json.load(open(path, encoding="utf-8"))
    if a.views:
        allViews = json.load(open(os.path.join(pano, "marking_views.json"), encoding="utf-8"))
        readViews(a, pano, mk, a.refs.split(",") if a.refs else list(allViews))
    else:
        readParts(a, pano, board, mk, a.refs.split(",") if a.refs else list(mk["parts"]))
    json.dump(mk, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
