"""CLI: check the read markings (markings.json, see markingSheets.py) against the BOM in board.json.

    venv/Scripts/python.exe markingCheck.py out/captured_51.5

Per part, into markings.json parts[ref].check = {status, text}:
    missing     BOM says fitted, the place is empty
    populated   BOM says NC, a part is there
    mismatch    resistor code decodes to another value than the BOM's
    ok          resistor code decodes to the BOM value; or an empty NC / optional place
    unread      a code was expected but nothing was read
    info        read text with nothing to check it against yet (ICs, diodes, tantalum caps: see the datasheet)
Prints the findings.
"""
import argparse
import json
import os
import re

from Components import E96, formatSi

EIA96_MULT = {"Z": 1e-3, "Y": 1e-2, "R": 1e-2, "X": 1e-1, "S": 1e-1, "A": 1, "B": 10, "H": 10, "C": 100, "D": 1e3,
              "E": 1e4, "F": 1e5}


def decodeResistor(code):
    """'01C' 10k, '103' 10k, '1002' 10k, '4R7' 4.7, 'R200' 0.2, '0' / '000' 0 -> ohms or None."""
    c = code.strip().upper()
    if re.fullmatch(r"0+", c):
        return 0.0
    m = re.fullmatch(r"(\d\d)([ZYRXSABHCDEF])", c)
    if m and 1 <= int(m.group(1)) <= 96:
        return E96[int(m.group(1)) - 1] * EIA96_MULT[m.group(2)]
    if re.fullmatch(r"\d*R\d*", c) and len(c) >= 2:
        return float(c.replace("R", "."))
    if re.fullmatch(r"\d{3,4}", c):
        return int(c[:-1]) * 10 ** int(c[-1])
    return None


def check(c, line, m):
    fit = c["fit"]
    if m["present"] is False:
        if fit == "fitted":
            return "missing", "BOM: fitted — the place is empty"
        return "ok", f"empty as expected ({fit})"
    if fit == "nc" and m["present"]:
        return "populated", "BOM: NC — a part is fitted"
    if not line:
        return "info", ""
    group = line["cat"]["group"]
    v = line.get("value")
    if group == "Resistors" and v and v.get("marking") is not None:
        if not m["lines"]:
            return ("unread", "no code read") if m["conf"] != "high" else ("info", "no code on the part")
        got = decodeResistor(m["lines"][0])
        if got is None:
            return "unread", f"'{m['lines'][0]}' is not a resistor code"
        want = v["value"]
        if abs(got - want) <= max(1e-9, want * 0.005):
            return "ok", f"{m['lines'][0]} = {formatSi(got, 'Ω')}"
        return "mismatch", f"{m['lines'][0]} = {formatSi(got, 'Ω')}, BOM {line['name']} = {v['text']}"
    if m["lines"]:
        return "info", "check against the datasheet marking"
    return "info", ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("panorama", help="folder with board.json and markings.json")
    a = ap.parse_args()
    b = json.load(open(os.path.join(a.panorama, "board.json"), encoding="utf-8"))
    path = os.path.join(a.panorama, "markings.json")
    mk = json.load(open(path, encoding="utf-8"))
    counts = {}
    rows = []
    for ref, m in mk["parts"].items():
        c = b["components"][ref]
        line = b["lines"][c["line"]] if c["line"] is not None else None
        status, text = check(c, line, m)
        m["check"] = {"status": status, "text": text}
        counts[status] = counts.get(status, 0) + 1
        if status in ("missing", "populated", "mismatch", "unread"):
            rows.append((status, ref, text))
    mk["summary"] = counts
    json.dump(mk, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    for status, ref, text in sorted(rows):
        print(f"  {status:10} {ref:12} {text}")


if __name__ == "__main__":
    main()
