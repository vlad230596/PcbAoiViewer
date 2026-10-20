"""CLI: one page that lines up every reading of every resistor marking — for eyes, and as the labelled dataset view.

    venv/Scripts/python.exe compareReadings.py out/captured_51.5 [--group Resistors]

Row per part: the crop as cut (parts/), the same crop turned upright (`turn` of the reference reading), the
brightened reading image (marking_img/: grey + CLAHE + enlarged + upright), then
    reference   markings.json lines (read by eye / Claude vision; the labels of dataset/resistor_codes)
    BOM         the codes the BOM value may carry (Components.resistorMarking) and the reference's check vs BOM
    methods     Qwen3-VL-2B (vlmMarkings.py), our classifier (readResistorCodes.py, leave-one-out), RapidOCR PP-OCRv3
                (ocrMarkings.py, turn found by itself) and every engine of the OCR survey
                (research/ocr_engines/results/summary.json, best source per engine, reference turn given)
Each reading is coloured against the reference after folding look-alikes (ocrMarkings.agree): same green, close
yellow, partial light green, differs red, nothing grey. A summary table on top counts them per method.
Writes <panorama>/compare/<group>.html (+ compare/img/<ref>_up.jpg); open it through the viewer's http server.
"""
import argparse
import html
import json
import os
import re

import cv2

from ocrMarkings import agree, turn

HERE = os.path.dirname(os.path.abspath(__file__))
SURVEY = os.path.join(HERE, "research", "ocr_engines", "results", "summary.json")
# survey engines worth a column: tag -> label (best source per tag is picked from the summary rows)
SURVEY_TAGS = [
    ("vlm-qwen3vl-4b", "Qwen3-VL-4B"), ("vlm-paddleocrvl", "PaddleOCR-VL"), ("vlm-got", "GOT-OCR 2"),
    ("vlm-florence2-large", "Florence-2-L"), ("rapid3-v5-server", "PP-OCRv5 srv"), ("paddle_", "PaddleOCR 3.7"),
    ("easyocr_-wl+close", "EasyOCR+close"), ("easyocr_-wl", "EasyOCR"), ("tesseract-best-psm11-wl", "Tesseract 5"),
    ("doctr_-nodet+parseq", "docTR PARSeq"), ("vlm-trocr-base", "TrOCR"),
]
COL = {"same": "#2e7d32", "close": "#9e8a00", "partial": "#558b2f", "differs": "#c62828", "ocr-only": "#c62828",
       "none": "#555", "empty": "#555"}
BG = {"same": "#e8f5e9", "close": "#fff8d6", "partial": "#f1f8e9", "differs": "#ffebee", "ocr-only": "#ffebee",
      "none": "#f3f3f3", "empty": "#f3f3f3"}


def natKey(r):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", r)]


def surveyColumns():
    if not os.path.isfile(SURVEY):
        return []
    s = json.load(open(SURVEY, encoding="utf-8"))
    best = {}
    for row in s["rows"]:
        if row["mode"] == "given" and row["group"] == "all" and (row["tag"] not in best or row["same"] > best[row["tag"]][1]):
            best[row["tag"]] = (row["src"], row["same"])
    cols = []
    for tag, label in SURVEY_TAGS:
        if tag in best:
            src = best[tag][0]
            det = s["detail"].get(f"{tag}|{src}|given")
            if det:
                cols.append((f"{label} ({src})", {r: v["lines"] for r, v in det.items()}))
    return cols


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("panorama")
    ap.add_argument("--group", default="Resistors", help="BOM group (Resistors, ICs, Capacitors, ...)")
    a = ap.parse_args()
    pano = a.panorama
    b = json.load(open(os.path.join(pano, "board.json"), encoding="utf-8"))
    mk = json.load(open(os.path.join(pano, "markings.json"), encoding="utf-8"))["parts"]
    crops = b["align"]["crops"]
    outDir = os.path.join(pano, "compare")
    os.makedirs(os.path.join(outDir, "img"), exist_ok=True)

    refs = []
    for ref, c in b["components"].items():
        line = b["lines"][c["line"]] if c["line"] is not None else None
        if ref in mk and ref in crops and line and line["cat"]["group"] == a.group:
            refs.append(ref)
    refs.sort(key=natKey)

    methods = [("Qwen3-VL-2B", {r: mk[r]["vlm"]["lines"] for r in refs if "vlm" in mk[r]}),
               ("our classifier", {r: [mk[r]["resistorCode"]["code"]] if mk[r]["resistorCode"].get("code") else []
                                   for r in refs if "resistorCode" in mk[r]}),
               ("RapidOCR v3 (own turn)", {r: mk[r]["ocr"]["lines"] for r in refs if "ocr" in mk[r]})]
    methods += surveyColumns()
    methods = [(n, d) for n, d in methods if any(r in d for r in refs)]

    stats = {n: {} for n, _ in methods}
    rows = []
    for ref in refs:
        m, c = mk[ref], b["components"][ref]
        line = b["lines"][c["line"]]
        crop = cv2.imread(os.path.join(pano, crops[ref]["file"]))
        name = re.sub(r"[^\w.+-]", "_", ref)
        cv2.imwrite(os.path.join(outDir, "img", f"{name}_up.jpg"), turn(crop, m.get("turn", 0)), [cv2.IMWRITE_JPEG_QUALITY, 92])
        codes = (line.get("value") or {}).get("marking") or {}
        chk = (m.get("check") or {}).get("status", "")
        cells = []
        for n, d in methods:
            if ref not in d:
                cells.append('<td class="na">·</td>')
                continue
            got = d[ref]
            ag = agree(m["lines"], got) if (m["lines"] or got) else "none"
            if not m["lines"] and m.get("present") is False and not got:
                ag = "none"
            stats[n][ag] = stats[n].get(ag, 0) + 1
            cells.append(f'<td style="background:{BG[ag]};color:{COL[ag]}" title="{ag}">'
                         f'{"<br>".join(html.escape(x) for x in got) or "—"}</td>')
        refText = "<br>".join(html.escape(x) for x in m["lines"]) or ("empty" if m.get("present") is False else "—")
        rows.append(
            f'<tr><td class="ref">{html.escape(ref)}<br><small>{html.escape(line["name"])} · {html.escape(line["footprint"])}</small></td>'
            f'<td><img src="../{html.escape(crops[ref]["file"])}"></td>'
            f'<td><img src="img/{name}_up.jpg"><br><small>turn {m.get("turn", 0)}°</small></td>'
            f'<td>{"<img src=../" + html.escape(m["img"]) + ">" if m.get("img") else ""}</td>'
            f'<td class="gt">{refText}<br><small>{html.escape(m.get("conf", ""))}</small></td>'
            f'<td class="bom">{" ".join(html.escape(v) for v in codes.values()) or "—"}'
            f'<br><small class="{chk}">{html.escape(chk)}</small></td>' + "".join(cells) + "</tr>")

    head = "".join(f"<th>{html.escape(n)}</th>" for n, _ in methods)
    summ = "".join(
        f'<tr><td>{html.escape(n)}</td>' + "".join(
            f'<td style="color:{COL[k]}">{stats[n].get(k, 0)}</td>' for k in ("same", "close", "partial", "differs", "ocr-only", "none"))
        + "</tr>" for n, _ in methods)
    page = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(a.group)} readings</title>
<style>
  body {{ font: 13px system-ui, sans-serif; margin: 12px; background: #fafafa; color: #222; }}
  table {{ border-collapse: collapse; }}
  th, td {{ border: 1px solid #ddd; padding: 3px 6px; vertical-align: middle; text-align: center; }}
  th {{ background: #eee; position: sticky; top: 0; z-index: 1; font-weight: 600; }}
  td img {{ height: 64px; display: block; margin: auto; image-rendering: auto; }}
  td.ref {{ text-align: left; font: 13px monospace; white-space: nowrap; }}
  td.gt {{ font: 600 15px monospace; background: #e3f2fd; }}
  td.bom {{ font: 13px monospace; color: #555; }}
  td.na {{ color: #bbb; }}
  td:not(.ref):not(.gt):not(.bom) {{ font: 13px monospace; }}
  small {{ color: #888; font-family: system-ui; font-size: 11px; }}
  small.mismatch, small.missing, small.populated {{ color: #c62828; font-weight: 600; }}
  small.ok {{ color: #2e7d32; }}
  .sum td {{ font: 13px monospace; }}
  label {{ margin-left: 10px; }}
</style></head><body>
<h2>{html.escape(a.group)}: {len(refs)} parts, every reading side by side</h2>
<p>Reference = markings.json (read by eye / Claude vision), the labels of the dataset. Colours: each method vs the
reference after folding O/0/D, I/L/1, S/5, B/8, Z/2, G/6 — <b style="color:{COL['same']}">same</b>,
<b style="color:{COL['close']}">close</b> (one char), <b style="color:{COL['partial']}">partial</b> (some lines),
<b style="color:{COL['differs']}">differs</b>, grey = nothing. Survey engines: best source per engine, reference turn
given; our classifier: leave-one-out (trained without that part). Image size: <input type="range" min="40" max="200"
value="64" oninput="document.querySelectorAll('td img').forEach(i=>i.style.height=this.value+'px')"></p>
<table class="sum"><tr><th>method</th><th>same</th><th>close</th><th>partial</th><th>differs</th><th>ocr-only</th><th>none</th></tr>{summ}</table>
<br>
<table><tr><th>part</th><th>crop as cut</th><th>upright</th><th>brightened (reading image)</th><th>reference</th><th>BOM codes</th>{head}</tr>
{"".join(rows)}
</table></body></html>"""
    out = os.path.join(outDir, f"{re.sub(r'[^A-Za-z]+', '_', a.group).strip('_')}.html")
    open(out, "w", encoding="utf-8").write(page)
    print(f"{out}: {len(refs)} parts, {len(methods)} methods")
    for n, _ in methods:
        st = stats[n]
        print(f"  {n:28} same {st.get('same', 0):3}  close {st.get('close', 0):3}  differs {st.get('differs', 0) + st.get('ocr-only', 0):3}")


if __name__ == "__main__":
    main()
