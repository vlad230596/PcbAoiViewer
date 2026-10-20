"""Markdown tables for REPORT.md from results/summary.json (run score.py first) -> results/tables.md.

Main table: one row per tag x source; columns = exact (same) with the turn given / unknown, same+close, CER,
exact per group (turn given), false readings on 40 blank parts, s per read from the clean timing pass (time-*).
"""
import json
import os

import common as C

GROUP_SHORT = {"resistor": "R (80)", "ic": "IC (17)", "diode/transistor": "D/Q (5)", "cap/ind/xtal": "C/L/X (14)"}


def main():
    s = json.load(open(os.path.join(C.RESULTS, "summary.json"), encoding="utf-8"))
    rows = s["rows"]
    idx = {(r["tag"], r["src"], r["mode"], r["group"]): r for r in rows}
    timing = {}
    for r in rows:
        if r["tag"].startswith("time-") and r["mode"] == "given" and r["group"] == "all":
            timing[r["tag"][5:]] = r["sec_read"]
    fp = {r["tag"]: r["blank_fp"] for r in rows if r["blank_fp"]}
    out = ["| engine | src | exact, turn given | exact, turn unknown | same+close (given) | CER (given) | "
           + " | ".join(GROUP_SHORT.values()) + " | blank FP |", "|" + "---|" * 11]
    keys = sorted({(r["tag"], r["src"]) for r in rows if not r["tag"].startswith("time-")})
    for tag, src in keys:
        g = idx.get((tag, src, "given", "all"))
        u = idx.get((tag, src, "unknown", "all"))
        if not g:
            continue
        per = [str(idx[(tag, src, "given", k)]["same"]) if (tag, src, "given", k) in idx else "" for k in GROUP_SHORT]
        out.append(f"| {tag} | {src} | **{g['same']}** | {u['same'] if u else '—'} | {g['same_close']} | {g['cer']:.2f} | "
                   + " | ".join(per) + f" | {fp.get(tag, '')} |")
    out += ["", "Timing (24 parts, raw, turn given, 4 CPU threads; GPU = RTX 5080 bf16):", "",
            "| run | s / read |", "|---|---|"]
    out += [f"| {k} | {v:.2f} |" for k, v in sorted(timing.items(), key=lambda x: x[1])]
    open(os.path.join(C.RESULTS, "tables.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
