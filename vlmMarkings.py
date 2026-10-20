"""CLI: second reading of the part markings by a local vision-language model (Qwen3-VL-2B), run in its own venv.

    venv/Scripts/python.exe vlmMarkings.py out/captured_51.5 [--refs R1,U3] [--python <vlm venv python>] [--device cpu]

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
"""
import argparse
import json
import os
import subprocess
import sys
import time

from ocrMarkings import agree

HERE = os.path.dirname(os.path.abspath(__file__))
SURVEY = os.path.join(HERE, "research", "ocr_engines")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("panorama", help="folder with board.json, parts/ and markings.json")
    ap.add_argument("--refs", help="comma-separated designators (default: all in markings.json)")
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
    refs = a.refs.split(",") if a.refs else list(mk["parts"])
    crops = board["align"]["crops"]
    jobs = [{"ref": r, "image": os.path.join(pano, crops[r]["file"]), "turn": mk["parts"][r].get("turn", 0)}
            for r in refs if r in crops]
    jobsFile, resFile = os.path.join(pano, "vlm_jobs.json"), os.path.join(pano, "vlm_results.json")
    json.dump(jobs, open(jobsFile, "w", encoding="utf-8"), indent=1)

    env = dict(os.environ, HF_HOME=a.hf_home, PYTHONIOENCODING="utf-8", TRANSFORMERS_VERBOSITY="error")
    if os.path.isdir(os.path.join(a.hf_home, "hub")):
        env.setdefault("HF_HUB_OFFLINE", "1")   # weights are local: no network
    cmd = [a.python, os.path.join(HERE, "vlmWorker.py"), jobsFile, resFile, "--model", a.model]
    if a.device:
        cmd += ["--device", a.device]
    t0 = time.time()
    print(f"{len(jobs)} parts -> {os.path.relpath(a.python, HERE) if a.python.startswith(HERE) else a.python}", flush=True)
    proc = subprocess.run(cmd, env=env)
    if proc.returncode:
        sys.exit(f"vlmWorker.py failed with code {proc.returncode}")
    res = json.load(open(resFile, encoding="utf-8"))

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
    json.dump(mk, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(jobs)} parts, {res['device']}, {time.time() - t0:.0f} s total: "
          + ", ".join(f"{k} {n}" for k, n in sorted(counts.items())))
    print("\n".join(rows))


if __name__ == "__main__":
    main()
