"""Antilatency.Electronics.Datasheets: BOM line -> digitized datasheet (Markdown) and the parts of it the AOI reads.

The library (an SVN working copy, D:\\Svn\\Antilatency.Electronics.Datasheets) holds one document per
Components/<Top>/<Sub>/<FAMILY>/<doc>.md with a front matter (doc, vendor, family, mpn[], status, digikey, source)
and figures under img/. Every document has the same top sections:

    ## Excursus          **Category:** key · parameters · ...  and a few lines on what the part is
    ## Working summary   pinout, ratings, ordering and package, pitfalls
    ## Full transfer     the whole datasheet page by page, headings end with "(p. N)"

Marking has no fixed section: it sits under headings like "Top marking (p. 2)", "Marking Information",
"Package Marking Information", "Ordering information" (a Marking column), "Marking diagram", "Date Code Key".
`sections` collects them by heading.

Matching is the server's (server.py ds_docs / match_line, data/datasheet-links.json), strongest first:
    manual      linked by hand on the server, key Name|Footprint|LibRef
    mpn         a document MPN equals the BOM Name / LibRef / a Description token
    mpn-prefix  ordering code longer or shorter than a document MPN (AMS1117 -> AMS1117-3.3), >= 5 characters
    family      the document family is a prefix of the part number (2N7002K -> 2N7002KT1G)
"""
import json
import re
import shutil
from pathlib import Path

DEFAULT_ROOT = Path(r"D:\Svn\Antilatency.Electronics.Datasheets")
DEFAULT_LINKS = Path(r"D:\Svn\Antilatency.Electronics.Registry.Server\data\datasheet-links.json")
STATUS_ORDER = {"verified": 0, "coverage-passed": 1, "draft": 2}
MATCH_FIELDS = (("name", "Name"), ("libRef", "LibRef"), ("description", "Description"))
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.\-_/#+]{3,}")

# headings worth showing next to a component on the board
SECTION_RULES = {
    "marking": re.compile(r"marking|date code|top mark|device code|part code", re.I),
    "ordering": re.compile(r"ordering|order code|part number|device options|package option", re.I),
    "package": re.compile(r"package (outline|information|dimension|drawing)|mechanical|outline dimension", re.I),
}
_HEADING = re.compile(r"^(#{2,6})\s+(.+?)\s*$")
_PAGE = re.compile(r"\(pp?\.\s*([\d\s,–-]+)\)\s*$")
_IMG = re.compile(r"!\[[^\]]*\]\(([^)\s]+)")


def frontMatter(text):
    """The library's minimal YAML-ish header: key: value, [a, b] lists, {k: v} dicts."""
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    meta = {}
    for line in text[3:end].strip().splitlines():
        if ":" not in line or line.startswith("#"):
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        if v.startswith("[") and v.endswith("]"):
            v = [x.strip() for x in v[1:-1].split(",") if x.strip()]
        elif v.startswith("{") and v.endswith("}"):
            v = {a.strip(): b.strip() for a, b in (x.split(":", 1) for x in v[1:-1].split(",") if ":" in x)}
        meta[k.strip()] = v
    return meta


def normPn(s):
    """Part number for comparison: upper case, no quotes/markers/spaces."""
    s = re.sub(r"[❓⚠️∅⛔\"'`]", "", str(s)).strip().upper()
    return re.sub(r"\s+", "", s)


def familyPrefix(fam, n):
    """Family name starts the part number: >= 5 characters, or 4 when the part number goes on with '-'
    (ABM8-32.000MHZ-9-D1G-T -> ABM8; a bare 4-character prefix is too loose: ABM8G, AO34…)."""
    if not n.startswith(fam):
        return False
    return len(fam) >= 5 or (len(fam) == 4 and n[4:5] == "-")


def bomLineKey(it):
    return "|".join((it.get("name", ""), it.get("footprint", ""), it.get("libRef", "")))


def bomTokens(it):
    """[(field label, raw token, loose)] — candidate part numbers of a BOM line, strongest first.
    loose=False: a piece of Name/LibRef ("MICRO-MATCH_FOB.04P_215079-4" -> "215079-4") — exact matching only,
    a piece is too short a context for prefix matching ("IDC-10_STM32_JTAG" -> "STM32")."""
    whole, parts, desc = [], [], []
    for key, label in MATCH_FIELDS:
        val = (it.get(key) or "").strip()
        if key == "description":
            desc += [(label, t, True) for t in TOKEN_RE.findall(val)]
        elif val:
            whole.append((label, val, True))
            sub = [t for t in re.split(r"[\s_,;]+", val) if t and t != val]
            parts += [(label, t, False) for t in sub if len(t) >= 5]
    seen, out = set(), []
    for label, t, loose in whole + parts + desc:
        if (label, t) in seen or not re.search(r"\d", t):
            continue
        if not re.search(r"[A-Za-z]", t) and not (loose is False and "-" in t):  # 215079-4 only as a piece
            continue
        seen.add((label, t))
        out.append((label, t, loose))
    return out


def _best(docs):
    return sorted(docs, key=lambda d: (STATUS_ORDER.get(d["status"], 9), d["id"]))


class DatasheetLibrary:
    def __init__(self, root=DEFAULT_ROOT, linksFile=DEFAULT_LINKS):
        self.root = Path(root)
        if not (self.root / "Components").is_dir():
            raise FileNotFoundError(f"datasheet library not found: {self.root}")
        self.links = {}
        if linksFile and Path(linksFile).is_file():
            self.links = json.loads(Path(linksFile).read_text(encoding="utf-8"))
        self.docs = self._readDocs()
        self.byId = {d["id"]: d for d in self.docs}
        self.mpnIndex = {}
        for d in self.docs:
            for m in d["mpnNorm"]:
                self.mpnIndex.setdefault(m, []).append(d)

    def _readDocs(self):
        """A document is an .md 3 levels below Components/ whose front matter has `doc:`."""
        docs = []
        for f in sorted((self.root / "Components").glob("*/*/*/*.md")):
            try:
                with f.open(encoding="utf-8") as fh:
                    head = fh.read(2_000_000)  # the front matter may hold thousands of MPNs (HolyStone-C: 40 KB)
            except OSError:
                continue
            meta = frontMatter(head)
            if not meta.get("doc"):
                continue
            rel = f.relative_to(self.root).as_posix()
            mpn = meta.get("mpn") or []
            mpn = mpn if isinstance(mpn, list) else [mpn]
            dk = [x.strip() for x in (meta.get("digikey") or "").split(">") if x.strip()]
            docs.append({
                "id": rel, "family": meta.get("family") or f.parent.name, "doc": meta.get("doc"),
                "vendor": meta.get("vendor", ""), "revision": str(meta.get("revision", "")).strip('"'),
                "mpn": [m.strip('"') for m in mpn], "mpnNorm": sorted({normPn(m) for m in mpn if normPn(m)}),
                "status": meta.get("status", ""), "category": dk or rel.split("/")[1:3],
                "date": meta.get("date", ""), "source": meta.get("source", ""),
            })
        return docs

    def match(self, it):
        """BOM line -> {method, doc, field, token, mpn, others[]} or None."""
        manual = self.links.get(bomLineKey(it))
        if manual:
            d = self.byId.get(manual["doc"])
            return {"method": "manual", "doc": d["id"] if d else manual["doc"], "missing": d is None,
                    "field": None, "token": None, "mpn": None, "others": []}
        toks = bomTokens(it)
        # 1) exact
        for field, t, _ in toks:
            hit = self.mpnIndex.get(normPn(t))
            if hit:
                ds = _best(hit)
                return {"method": "mpn", "doc": ds[0]["id"], "field": field, "token": t, "mpn": t,
                        "others": [d["id"] for d in ds[1:]]}
        # 2) prefix either way, longest common part wins
        cands = []
        for field, t, loose in toks:
            n = normPn(t)
            if len(n) < 5 or not loose:
                continue
            for m, ds in self.mpnIndex.items():
                if not ds or len(m) < 5 or m == n:
                    continue
                if n.startswith(m) or m.startswith(n):
                    cands.append((min(len(m), len(n)), field, t, m, ds))
        if cands:
            cands.sort(key=lambda c: -c[0])
            _, field, t, m, ds = cands[0]
            ds = _best(ds)
            mpn = next((x for x in ds[0]["mpn"] if normPn(x) == m), m)
            return {"method": "mpn-prefix", "doc": ds[0]["id"], "field": field, "token": t, "mpn": mpn,
                    "others": [d["id"] for d in ds[1:]]}
        # 3) family name as prefix
        for field, t, loose in toks:
            n = normPn(t)
            if not loose:
                continue
            fam = [d for d in self.docs if familyPrefix(normPn(d["family"]), n)]
            if fam:
                ds = _best(fam)
                return {"method": "family", "doc": ds[0]["id"], "field": field, "token": t, "mpn": ds[0]["family"],
                        "others": [d["id"] for d in ds[1:]]}
        return None

    def text(self, docId):
        return (self.root / docId).read_text(encoding="utf-8")

    def excursus(self, docId):
        """-> {category, params: [str], text} from the document's '## Excursus' section."""
        m = re.search(r"^##\s+Excursus\s*$(.*?)(?=^##\s|\Z)", self.text(docId), re.M | re.S)
        if not m:
            return None
        body = m.group(1).strip()
        params = []
        cat = re.match(r"\*\*Category:\*\*\s*(.+?)(?:\n\s*\n|\Z)", body, re.S)
        if cat:
            params = [p.strip() for p in " ".join(cat.group(1).split()).split(" · ") if p.strip()]
            body = body[cat.end():].strip()
        return {"category": params[0] if params else "", "params": params[1:], "text": body}

    def sections(self, docId, rules=SECTION_RULES):
        """{kind: [{title, page, line, text}]} — whole sections (heading to the next one of the same or higher
        level) whose heading matches a rule. Nested matches are dropped: the outer section already holds them."""
        lines = self.text(docId).splitlines()
        heads = []
        for i, ln in enumerate(lines):
            m = _HEADING.match(ln)
            if m:
                heads.append((i, len(m.group(1)), m.group(2)))
        out = {k: [] for k in rules}
        for kind, rx in rules.items():
            taken_until = -1
            for j, (i, lvl, title) in enumerate(heads):
                if i < taken_until or not rx.search(title):
                    continue
                end = next((h[0] for h in heads[j + 1:] if h[1] <= lvl), len(lines))
                pm = _PAGE.search(title)
                out[kind].append({"title": _PAGE.sub("", title).strip(), "page": pm.group(1).strip() if pm else None,
                                  "line": i + 1, "text": "\n".join(lines[i:end]).strip()})
                taken_until = end
        return out

    def bundle(self, docId, outDir):
        """Copies the document's .md and the images it references to outDir/<docId>. -> path of the copy."""
        src = self.root / docId
        dst = Path(outDir) / docId
        dst.parent.mkdir(parents=True, exist_ok=True)
        text = src.read_text(encoding="utf-8")
        dst.write_text(text, encoding="utf-8")
        for rel in sorted(set(_IMG.findall(text))):
            if "://" in rel or rel.startswith("/"):
                continue
            f = (src.parent / rel).resolve()
            if f.is_file() and src.parent.resolve() in f.parents:
                t = dst.parent / rel
                t.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, t)
        return dst
