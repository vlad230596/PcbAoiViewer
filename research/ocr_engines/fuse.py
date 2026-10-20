"""Fusion by voting: several readings of one part (other images and/or other engines) -> one reading.

    python fuse.py <name> <tag>:<src> [<tag>:<src> ...]      -> results/fuse-<name>.jsonl (src "fused")

Per part and rotation, the readings are grouped by their folded text (ocrMarkings.fold); each group's weight is the
sum of its scores; the heaviest group wins and its most confident reading is the output, score = winning weight /
number of voters (so unanimous + confident -> ~1). Empty readings vote for "no marking" and
vote with weight 0.5 each, so a lone hallucination does not beat several empty readings.
"""
import collections
import json
import os
import sys

import common as C


def main():
    name, specs = sys.argv[1], [s.split(":") for s in sys.argv[2:]]
    rows = collections.defaultdict(list)
    for tag, src in specs:
        for l in open(os.path.join(C.RESULTS, tag + ".jsonl"), encoding="utf-8"):
            r = json.loads(l)
            if r["src"] == src:
                rows[(r["ref"], r["rot"])].append(r)
    out = open(os.path.join(C.RESULTS, f"fuse-{name}.jsonl"), "w", encoding="utf-8")
    for (ref, rot), rs in rows.items():
        votes = collections.defaultdict(float)
        best = {}
        for r in rs:
            key = C.fold(r["lines"]) if r["lines"] else ""
            votes[key] += r["score"] if key else 0.5
            if key not in best or r["score"] > best[key]["score"]:
                best[key] = r
        win = max(votes, key=votes.get)
        b = best[win]
        out.write(json.dumps({"ref": ref, "group": b["group"], "src": "fused", "rot": rot,
                              "lines": b["lines"] if win else [], "score": round(votes[win] / len(rs), 4),
                              "sec": round(sum(r["sec"] for r in rs), 3), "blank": b.get("blank", False),
                              "voters": len(rs)}, ensure_ascii=False) + "\n")
    print(f"fuse-{name}: {len(rows)} part x rotation rows from {len(specs)} inputs")


if __name__ == "__main__":
    main()
