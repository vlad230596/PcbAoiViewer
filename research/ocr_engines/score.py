"""Scores every results/*.jsonl against the reference reading (markings.json) -> results/summary.csv, summary.json.

    python score.py [--min-score 0.5] [--tags a,b] [--print]

Per tag x source x mode (given = rot 0, unknown = best mean score of the 4 rotations, as ocrMarkings.py picks it)
x group: n, same (exact after folding look-alikes), same+close, partial, differs, empty, CER (edit distance of the
folded strings / reference length; an empty reading costs 1.0), mean s per read; false readings on blank parts.
Also: same on the high-confidence references only, and "oracle" (any rotation same — what a better turn picker could get).
"""
import argparse
import collections
import csv
import glob
import json
import os

import common as C


def pick(rows, minScore):
    """Best-scoring rotation; below min score counts as no reading."""
    best = max(rows, key=lambda r: (r["score"] if r["lines"] else -1))
    if not best["lines"] or best["score"] < minScore:
        return {**best, "lines": []}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--tags")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()
    mk, _ = C.load()
    out, detail = [], {}
    for path in sorted(glob.glob(os.path.join(C.RESULTS, "*.jsonl"))):
        tag = os.path.basename(path)[:-6]
        if a.tags and tag not in a.tags.split(","):
            continue
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        by = collections.defaultdict(list)
        for r in rows:
            by[(r["ref"], r["src"])].append(r)
        blank = [r for r in rows if r.get("blank")]
        fp = sum(1 for r in blank if r["lines"] and r["score"] >= a.min_score and len(C.fold(r["lines"])) >= 2)
        for src in sorted({k[1] for k in by}):
            for mode in ("given", "unknown"):
                stats = collections.defaultdict(lambda: collections.Counter())
                cers = collections.defaultdict(list)
                for (ref, s), rs in by.items():
                    if s != src or not mk[ref].get("lines"):
                        continue
                    if mode == "given":
                        rs0 = [r for r in rs if r["rot"] == 0]
                        if not rs0:
                            continue
                        r = pick(rs0, a.min_score)
                    else:
                        if len(rs) < 4:
                            continue
                        r = pick(rs, a.min_score)
                    ag = C.agree(mk[ref]["lines"], r["lines"])
                    ag = "empty" if not r["lines"] else ag
                    oracle = any(C.agree(mk[ref]["lines"], x["lines"]) == "same" for x in (rs if mode == "unknown" else rs0))
                    for g in (C.group(ref), "all"):
                        st = stats[g]
                        st["n"] += 1; st[ag] += 1
                        st["hi_n"] += mk[ref].get("conf") == "high"
                        st["hi_same"] += mk[ref].get("conf") == "high" and ag == "same"
                        st["oracle"] += oracle
                        st["sec"] += sum(x["sec"] for x in rs) / len(rs)
                        cers[g].append(min(1.0, C.cer(mk[ref]["lines"], r["lines"])))
                    detail.setdefault(f"{tag}|{src}|{mode}", {})[ref] = {"lines": r["lines"], "score": r["score"],
                                                                        "rot": r["rot"], "agree": ag}
                for g in ("all",) + C.GROUPS:
                    st = stats[g]
                    if not st["n"]:
                        continue
                    out.append({"tag": tag, "src": src, "mode": mode, "group": g, "n": st["n"], "same": st["same"],
                                "same_close": st["same"] + st["close"], "partial": st["partial"],
                                "differs": st["differs"], "empty": st["empty"],
                                "cer": round(sum(cers[g]) / len(cers[g]), 3), "hi_same": f"{st['hi_same']}/{st['hi_n']}",
                                "oracle": st["oracle"], "sec_read": round(st["sec"] / st["n"], 3),
                                "blank_fp": f"{fp}/{len(blank)}" if g == "all" and blank else ""})
    with open(os.path.join(C.RESULTS, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    json.dump({"rows": out, "detail": detail}, open(os.path.join(C.RESULTS, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    if a.print:
        print(f"{'tag':28} {'src':5} {'mode':7} {'group':17} {'n':>3} {'same':>4} {'s+c':>4} {'part':>4} {'empty':>5} "
              f"{'CER':>5} {'oracle':>6} {'s/read':>6} fp")
        for r in out:
            print(f"{r['tag']:28} {r['src']:5} {r['mode']:7} {r['group']:17} {r['n']:3} {r['same']:4} {r['same_close']:4} "
                  f"{r['partial']:4} {r['empty']:5} {r['cer']:5.2f} {r['oracle']:6} {r['sec_read']:6.2f} {r['blank_fp']}")


if __name__ == "__main__":
    main()
