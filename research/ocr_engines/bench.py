"""Runs one OCR engine over the benchmark set and writes results/<tag>.jsonl (one row per part x source x rotation).

    venvs/<venv>/Scripts/python.exe bench.py <engine> [--variant v] [--sources raw,mimg,body] [--rots 0,90,180,270]
                                              [--refs R2,U3] [--blank] [--tag name] [--threads 4] [--prep up2]

<engine> is a module in engines/ with make(variant) -> object with .read(img) -> (lines, score); img is upright
BGR or grey. Rows: {ref, group, src, rot, lines, score, sec}. --blank adds 40 unmarked parts (mimg, rot 0 only) to count
false readings. Already present rows are skipped, so an interrupted run resumes.
"""
import argparse
import importlib
import json
import os
import sys
import time


def threads():
    """--threads N before any numeric library loads (OpenMP / MKL / onnxruntime read these at import)."""
    n = sys.argv[sys.argv.index("--threads") + 1] if "--threads" in sys.argv else "4"
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BENCH_THREADS"):
        os.environ[k] = n
    return int(n)


threads()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("--variant", default="")
    ap.add_argument("--sources", default="raw,mimg,body")
    ap.add_argument("--rots", default="0,90,180,270")
    ap.add_argument("--refs")
    ap.add_argument("--blank", action="store_true")
    ap.add_argument("--tag")
    ap.add_argument("--prep", default="", help="preprocessing variant (common.prep): clahe | up2 | close | otsu")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads per engine (default 4; timings use this)")
    a = ap.parse_args()
    tag = a.tag or (a.engine + (f"-{a.variant}" if a.variant else "") + (f"+{a.prep}" if a.prep else ""))
    os.makedirs(C.RESULTS, exist_ok=True)
    path = os.path.join(C.RESULTS, tag + ".jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            done.add((r["ref"], r["src"], r["rot"]))

    mk, board = C.load()
    refs = a.refs.split(",") if a.refs else C.refsWithLines(mk)
    jobs = [(r, s, int(d)) for r in refs for s in a.sources.split(",") for d in a.rots.split(",")]
    if a.blank:
        jobs += [(r, "mimg", 0) for r in C.refsBlank(mk)]
    jobs = [j for j in jobs if j not in done]
    t0 = time.time()
    try:
        import torch
        torch.set_num_threads(a.threads)
    except ImportError:
        pass
    eng = importlib.import_module("engines." + a.engine).make(a.variant)
    print(f"{tag}: engine ready in {time.time() - t0:.1f} s, {len(jobs)} jobs", flush=True)
    warm = C.image(refs[0], "mimg", mk, board)
    eng.read(warm)                                              # first call pays lazy init; keep it out of timings
    out = open(path, "a", encoding="utf-8")
    t1 = time.time()
    for i, (ref, src, rot) in enumerate(jobs):
        img = C.turn(C.prep(C.image(ref, src, mk, board), a.prep), rot)
        ts = time.perf_counter()
        lines, score = eng.read(img)
        sec = time.perf_counter() - ts
        row = {"ref": ref, "group": C.group(ref), "src": src, "rot": rot, "lines": lines, "score": round(float(score), 4),
               "sec": round(sec, 3), "blank": not mk[ref].get("lines")}
        out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
        if i % 50 == 0:
            print(f"  {i}/{len(jobs)} {ref} {src} {rot}: {lines} ({score:.2f}, {sec:.2f} s)", flush=True)
    print(f"{tag}: {len(jobs)} reads in {time.time() - t1:.0f} s", flush=True)


if __name__ == "__main__":
    main()
