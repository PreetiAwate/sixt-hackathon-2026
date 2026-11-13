"""Layer 3, the Proof Engine: cheap, deterministic checks that turn "the model said so" into evidence.
  1. evidence matching   - is the value (and its quoted evidence) really in the document? where?
  2. schema validation   - type / enum / pattern / JSON shape
  3. semantic consistency- does the neighbourhood look like the field? plausible format? right kind of clause?
  4. cross-field checks  - duplicated values, overlapping spans, impossible date order
Every check is a feature for the risk controller; a few are hard stops (an ungrounded value is never worth a model call)."""
from __future__ import annotations

import difflib
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date

from .cuad_defs import CLAUSES, HEADER_FIELDS
from .docs import Document
from .extract import Cand
from .retrieval import score as clause_score
from .schema import FieldSpec

_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def squash(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", unicodedata.normalize("NFKC", s).lower())


def words(s: str) -> list:
    return re.findall(r"[0-9a-z]+", unicodedata.normalize("NFKC", s).lower())


class DocIndex:
    """Squashed (alnum-only) view of the evidence text with a map back to original offsets, plus a token trigram index for fuzzy matching."""

    def __init__(self, text: str):
        self.text = text
        sq, idx = [], []
        for i, ch in enumerate(text):
            c = unicodedata.normalize("NFKC", ch).lower()
            for cc in c:
                if cc.isalnum() and cc.isascii():
                    sq.append(cc)
                    idx.append(i)
        self.sq = "".join(sq)
        self.idx = idx
        self.toks = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[0-9a-zA-Z]+", text)]
        self.tw = [t[0].lower() for t in self.toks]
        self._tri: dict | None = None
        self._nums: set | None = None

    @property
    def tri(self) -> dict:
        if self._tri is None:
            d: dict = {}
            for i in range(len(self.tw) - 2):
                d.setdefault((self.tw[i], self.tw[i + 1], self.tw[i + 2]), []).append(i)
            self._tri = d
        return self._tri

    @property
    def nums(self) -> set:
        if self._nums is None:
            out = set()
            for m in re.finditer(r"\(?-?\$?\s?\d[\d,]*(?:\.\d+)?\)?", self.text):
                s = re.sub(r"[^\d.\-]", "", m.group(0).replace(",", ""))
                try:
                    out.add(abs(float(s)))
                except ValueError:
                    pass
            self._nums = out
        return self._nums

    def find(self, value: str) -> tuple[float, int, int, int]:
        """-> (score 0..1, start, end, occurrences). Exact (alnum-squashed) match = 1.0, else best token-window ratio."""
        sv = squash(value)
        if len(sv) >= 2 and self.sq:
            p = self.sq.find(sv)
            if p >= 0:
                occ = self.sq.count(sv) if len(sv) < 40 else 1
                return 1.0, self.idx[p], self.idx[p + len(sv) - 1] + 1, occ
        vt = words(value)
        if len(vt) < 3 or not self.tw:
            return 0.0, -1, -1, 0
        starts = set()
        for k in range(0, min(len(vt) - 2, 6)):
            for pos in self.tri.get(tuple(vt[k:k + 3]), [])[:10]:
                starts.add(max(0, pos - k))
        best = (0.0, -1, -1)
        for s0 in list(starts)[:25]:
            win = self.tw[s0: s0 + len(vt) + 6]
            r = difflib.SequenceMatcher(None, vt, win, autojunk=False).ratio()
            r = r * len(win) / max(len(win), len(vt)) if len(win) > len(vt) else r
            if r > best[0]:
                best = (r, self.toks[s0][1], self.toks[min(s0 + len(vt), len(self.toks)) - 1][2])
        return (best[0] * 0.98, best[1], best[2], 1) if best[1] >= 0 else (0.0, -1, -1, 0)


def parse_date(s: str):
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", s.lower().replace(",", " "))
    y = re.search(r"\b(1[89]\d\d|20\d\d)\b", s)
    if not y:
        return None
    mon = next((v for k, v in _MONTHS.items() if re.search(r"\b" + k, s)), None)
    d = re.search(r"\b([0-2]?\d|3[01])\b(?!\s*(?:19|20)\d\d)", s.replace(y.group(1), " "))
    if mon and d:
        try:
            return date(int(y.group(1)), mon, int(d.group(1)))
        except ValueError:
            return None
    m = re.search(r"\b(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})\b", s)
    if m:
        a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        c += 2000 if c < 100 else 0
        try:
            return date(c, a, b) if a <= 12 else date(c, b, a)
        except ValueError:
            return None
    return None


# name-token -> (regex the value should match)
_PAT_RULES = [
    (("ein",), r"\d{2}-?\d{7}"), (("ssn",), r"\d{3}-?\d{2}-?\d{4}"), (("zip",), r"\d{5}(?:-\d{4})?"),
    (("phone", "telephone", "fax"), r"[\d()+\-.\s]{7,}"), (("email",), r"[^@\s]+@[^@\s]+\.[a-z]+"),
]


@dataclass
class Proof:
    feat: dict = field(default_factory=dict)
    hard: str | None = None
    start: int = -1
    end: int = -1
    page: int | None = None
    window: str = ""
    notes: list = field(default_factory=list)


def _label_words(spec: FieldSpec) -> list:
    return [w for w in re.split(r"[\W_]+", spec.name.lower()) if len(w) >= 3]


def window_for(doc: Document, spec: FieldSpec, start: int, end: int, ev_text: str, width: int = 700) -> str:
    """The excerpt a verifier is shown: the neighbourhood of the located evidence (not the whole document)."""
    t = ev_text
    if start >= 0:
        if doc.source == "cuad":
            return t[max(0, start - 600): min(len(t), end + 400)][:4200]
        return t[max(0, start - width): min(len(t), end + width)]
    if doc.source == "cuad" and spec.name in CLAUSES:
        from .retrieval import top_chunks
        ch = top_chunks(doc, spec.name, 2)
        return "\n...\n".join(c.text for c in ch)[:4200] if ch else t[:2500]
    pages = doc.pages if doc.has_text else doc.ocr_pages
    if pages:
        lw = _label_words(spec)
        best = max(range(len(pages)), key=lambda i: sum(w in pages[i].lower() for w in lw))
        return pages[best][:2800]
    return ""


def check_document(doc: Document, specs: list, cands: dict, schema_root: dict) -> dict:
    """Run all proofs for one document. Returns {field name: Proof}."""
    ev_text = doc.evidence_text
    ek = doc.evidence_kind
    index = DocIndex(ev_text) if ev_text.strip() else None
    out: dict = {}
    vals = {n: c.value for n, c in cands.items() if c.value is not None}
    # document-level helpers for cross-field checks
    value_count: dict = {}
    for s in specs:
        v = vals.get(s.name)
        if v is not None and s.kind in ("text", "number", "integer") and len(str(v)) >= 4:
            value_count[squash(str(v))] = value_count.get(squash(str(v)), 0) + 1
    dates = {n: parse_date(v) for n, v in vals.items() if n in ("Agreement Date", "Effective Date", "Expiration Date") or "date" in n.lower()}
    locs: dict = {}

    for s in specs:
        c = cands[s.name]
        p = Proof()
        f = p.feat
        out[s.name] = p
        if c.value is None:
            continue
        v = c.value
        f["ext_conf"] = c.conf
        f["len_log"] = math.log1p(len(v))
        f["very_long"] = float(len(v) > 1500)
        f["kind_bool"], f["kind_json"] = float(s.kind == "bool"), float(s.kind == "json")
        f["kind_num"] = float(s.kind in ("number", "integer"))
        f["is_cuad"] = float(doc.source == "cuad")
        f["ev_layer"], f["ev_ocr"] = float(ek == "layer"), float(ek == "ocr")

        # --- 2. schema validation
        ok, why = validate(s, v, schema_root)
        f["schema_ok"] = float(ok)
        if not ok:
            p.hard, p.notes = "schema_invalid: " + why, p.notes + [why]

        # --- 1. evidence matching
        g_score, st, en, occ = 0.0, -1, -1, 0
        if s.kind in ("bool",):
            g_score = 1.0 if index else 0.0     # a checkbox has no text to match; proof comes from the verifier/recheck
            f["g_applicable"] = 0.0
        else:
            f["g_applicable"] = 1.0
            if index:
                if s.kind in ("number", "integer"):
                    try:
                        fv = abs(float(v))
                        g_score = 1.0 if (fv in index.nums or round(fv, 2) in index.nums) else 0.0
                        if g_score:
                            _, st, en, occ = index.find(v)
                    except ValueError:
                        g_score = 0.0
                elif s.kind == "json":
                    leaves = [str(x) for x in _leaves(v)][:12]
                    hits = [index.find(x)[0] for x in leaves if len(squash(x)) >= 2]
                    g_score = sum(h >= 0.9 for h in hits) / len(hits) if hits else 0.0
                else:
                    g_score, st, en, occ = index.find(v)
        f["g_score"], f["g_exact"] = g_score, float(g_score >= 0.999)
        f["occ_log"] = math.log1p(min(occ, 20))
        p.start, p.end = st, en
        locs[s.name] = (st, en)
        if st >= 0:
            p.page = doc.page_of(st) if ek == "layer" else None

        # boundary proofs: a value cut mid-word or mid-sentence is a truncated/over-extended extraction even though it is a perfect substring
        f["bound_ok"] = f["sent_aligned"] = 0.5
        if st >= 0 and g_score >= 0.999 and s.kind in ("span", "text") and v:
            before = ev_text[st - 1] if st > 0 else " "
            after = ev_text[en] if en < len(ev_text) else " "
            f["bound_ok"] = float(not (before.isalnum() and v[0].isalnum()) and not (after.isalnum() and v[-1].isalnum()))
            start_ok = st == 0 or bool(re.search(r"(?:^|[\n.;:])\s*$", ev_text[max(0, st - 8): st])) or not before.isalnum() and bool(re.search(r"[\n.;:]\s*$", ev_text[max(0, st - 8): st]))
            end_ok = bool(re.search(r"[.;:!?)\"”']$", v.rstrip())) or en >= len(ev_text) or ev_text[en:en + 2].lstrip(" \t").startswith("\n") or not after.isalnum() and bool(v.rstrip()[-1:] in ".;:!?)")
            f["sent_aligned"] = float(start_ok and end_ok) if s.kind == "span" else float(end_ok or start_ok)
        f["end_punct"] = float(bool(re.search(r"[.;:!?)\"”']$", v.strip())))

        # evidence quote (value + where it was read)
        if c.evidence:
            f["ev_present"] = 1.0
            if index:
                es, _, _, _ = index.find(c.evidence)
                f["ev_found"] = float(es >= 0.9)
                f["ev_has_value"] = float(squash(v) in squash(c.evidence)) if s.kind not in ("json",) else 0.5
            else:
                f["ev_found"] = f["ev_has_value"] = 0.0
            f["page_ok"] = float(c.page is not None and p.page is not None and c.page == p.page) if c.page else 0.5
        else:
            f["ev_present"] = f["ev_found"] = f["ev_has_value"] = 0.0
            f["page_ok"] = 0.5

        # --- 3. semantic consistency
        prox = 0.5
        if st >= 0 and doc.source != "cuad":
            lw = _label_words(s)
            if lw:
                near = ev_text[max(0, st - 260): en + 40].lower()
                prox = sum(w in near for w in lw) / len(lw)
        f["label_prox"] = prox
        low = s.name.lower()
        rule = next((rx for keys, rx in _PAT_RULES if any(k in re.split(r"[\W_]+", low) for k in keys)), None)
        f["pat_rule"] = float(rule is not None)
        f["pat_ok"] = float(bool(rule and re.search(rule, v)))
        if s.kind not in ("bool", "json") and "date" in low and doc.source != "cuad":
            f["pat_rule"], f["pat_ok"] = 1.0, float(parse_date(v) is not None or bool(re.search(r"\d", v)))
        if doc.source == "cuad" and s.name in CLAUSES:
            rx = sum(w * min(len(pt.findall(v)), 2) for pt, w in CLAUSES[s.name]["patterns"])
            f["cue_rx"] = min(rx, 6) / 6
            f["cue_lm"] = math.tanh(clause_score(s.name, v) / 12)
            f["pos_early"] = float(0 <= st < 3500) if s.name in HEADER_FIELDS else 0.5
        else:
            f["cue_rx"] = f["cue_lm"] = 0.5
            f["pos_early"] = 0.5

        # --- 4. cross-field
        f["dup_n"] = float(min(value_count.get(squash(v), 1) - 1, 2)) if s.kind in ("text", "number", "integer") and len(v) >= 4 else 0.0
        f["date_conflict"] = 0.0
        if doc.source == "cuad":
            e, ex = dates.get("Effective Date") or dates.get("Agreement Date"), dates.get("Expiration Date")
            if s.name == "Expiration Date" and e and ex and ex < e:
                f["date_conflict"] = 1.0
        else:
            toks = set(re.split(r"[\W_]+", low))
            if dates.get(s.name) and toks & {"end", "ending", "expiration", "to", "through"}:
                stem = [w for w in toks if w not in ("end", "ending", "expiration", "to", "through", "date")]
                for n2, d2 in dates.items():
                    if d2 and n2 != s.name and set(re.split(r"[\W_]+", n2.lower())) & {"start", "begin", "beginning", "from", "effective"} and any(w in n2.lower() for w in stem) and dates[s.name] < d2:
                        f["date_conflict"] = 1.0
        f["overlap_n"] = 0.0

        # --- hard stops: an answer that is not in a readable document is never booked, and never worth a verifier call
        if p.hard is None and ek == "layer" and s.kind in ("span", "text") and len(squash(v)) >= 4 and g_score < 0.55:
            p.hard = f"not_in_document (best match {g_score:.2f})"
        p.window = window_for(doc, s, st, en, ev_text) if ev_text.strip() else ""

    # overlap between different clauses' spans (CUAD): the same sentence claimed for unrelated clause types is suspicious
    if doc.source == "cuad":
        names = [n for n, (a, b) in locs.items() if a >= 0]
        for n in names:
            a, b = locs[n]
            cnt = 0
            for m in names:
                if m == n:
                    continue
                a2, b2 = locs[m]
                inter = max(0, min(b, b2) - max(a, a2))
                if inter >= 0.5 * min(b - a, b2 - a2) and min(b - a, b2 - a2) > 0:
                    cnt += 1
            out[n].feat["overlap_n"] = float(min(cnt, 3))
    return out


def _leaves(v):
    try:
        x = json.loads(v) if isinstance(v, str) else v
    except ValueError:
        return [v]
    stack, res = [x], []
    while stack:
        y = stack.pop()
        if isinstance(y, dict):
            stack += list(y.values())
        elif isinstance(y, list):
            stack += y
        elif y is not None and not isinstance(y, bool):
            res.append(y)
    return res


def validate(spec: FieldSpec, value: str, root: dict) -> tuple[bool, str]:
    """Schema validation of the proposed value (type, enum, pattern, JSON shape; jsonschema if installed)."""
    try:
        if spec.kind == "bool":
            return value in ("true", "false"), "not a boolean"
        if spec.kind in ("number", "integer"):
            x = float(value)
            if spec.kind == "integer" and not x.is_integer():
                return False, "not an integer"
            lo, hi = spec.raw.get("minimum"), spec.raw.get("maximum")
            if (lo is not None and x < lo) or (hi is not None and x > hi):
                return False, "out of range"
        if spec.kind == "json":
            x = json.loads(value)
            if not isinstance(x, (dict, list)):
                return False, "not an object/array"
            try:
                import jsonschema
                node = dict(spec.raw)
                if "$defs" in root:
                    node["$defs"] = root["$defs"]
                jsonschema.validate(x, node)
            except ImportError:
                pass
            except Exception as e:                       # jsonschema.ValidationError and friends
                if e.__class__.__name__ == "ValidationError":
                    return False, "json shape: " + str(getattr(e, "message", e))[:80]
        if spec.enum and value.strip().lower() not in {str(e).lower() for e in spec.enum}:
            return False, "not in enum"
        if spec.pattern and spec.kind in ("text", "span") and not re.search(spec.pattern, value):
            return False, "pattern mismatch"
    except (ValueError, TypeError):
        return False, "unparseable"
    return True, ""
