"""Layers 1-2: schema-guided LLM extraction that must return value + verbatim evidence + location. Offline rule fallback included."""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .config import Config
from .cuad_defs import CLAUSES, HEADER_FIELDS
from .docs import Document
from .llm import Client, LLMError, parse_json
from .retrieval import build_digest, top_chunks
from .schema import FieldSpec

NULLISH = {"", "null", "none", "n/a", "na", "not found", "not available", "not specified", "unknown", "-", "nil"}


@dataclass
class Cand:
    name: str
    value: str | None = None
    conf: float = 0.0           # the model's own stated confidence (a weak signal; calibrated later)
    evidence: str | None = None
    page: int | None = None
    source: str = "llm"
    error: bool = False         # call failed / unparseable: nothing is known about this field


# ---------------------------------------------------------------- normalisation
def _num_text(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else repr(round(x, 6)).rstrip("0").rstrip(".")


def normalize_value(spec: FieldSpec, v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() in NULLISH:
        return None
    if spec.kind == "bool":
        if isinstance(v, bool):
            return "true" if v else "false"
        s = str(v).strip().lower()
        return {"true": "true", "yes": "true", "checked": "true", "x": "true", "1": "true",
                "false": "false", "no": "false", "unchecked": "false", "0": "false", "": None}.get(s)
    if spec.kind in ("number", "integer"):
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return _num_text(float(v))
        s = str(v).strip()
        neg = s.startswith("(") and s.endswith(")")
        s2 = re.sub(r"[^\d.,\-]", "", s)
        if "," in s2 and "." in s2:
            s2 = s2.replace(",", "")
        elif "," in s2:
            s2 = s2.replace(",", "") if re.fullmatch(r"-?\d{1,3}(,\d{3})+", s2) else s2.replace(",", ".")
        try:
            x = float(s2)
        except ValueError:
            return None
        return _num_text(-abs(x) if neg else x)
    if spec.kind == "json":
        if isinstance(v, (dict, list)):
            return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return str(v)
    if isinstance(v, list) and all(isinstance(x, (str, int, float)) for x in v):
        return " | ".join(" ".join(str(x).split()) for x in v if x is not None and str(x).strip()) or None
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return " ".join(str(v).split()) if spec.kind != "span" else str(v).strip()


def _conf(x) -> float:
    try:
        return min(max(float(x), 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.5


# ---------------------------------------------------------------- prompts
SYSTEM = ("You are a meticulous document-extraction engine for a car-rental company's back office. "
          "Accuracy beats coverage: a wrong value that reaches the database costs 100 times more than leaving the field empty for a human.")
SYSTEM_RECHECK = ("You are an independent second reader auditing an extraction. Read the document yourself from scratch; "
                  "answer only what the text/image proves. Prefer null over a guess.")

CUAD_PROMPT = """Fill in every field below from the contract. Reply with ONE JSON object and nothing else:
{{"<Field>": {{"value": <string or null>, "confidence": <0..1>}}, ...}}

Rules
1. "value" = the clause/sentence(s) that answer the field, copied VERBATIM from the contract (same words and punctuation; never paraphrase, never merge text from distant places). For date fields copy the date wording exactly as written (e.g. "7th day of September, 1999"). If several separate passages qualify, join them with " | ".
2. If the contract does not contain that clause, "value" MUST be null. Most contracts lack most of these clauses. Merely mentioning a topic is not the same as containing the clause. When unsure, use null.
3. "confidence" = your probability that the value is exactly right AND is really this kind of clause (use <0.5 when guessing).

Fields:
{fields}

Contract{note}:
<<<
{text}
>>>"""

FORM_PROMPT = """Extract the fields below from the document. Reply with ONE JSON object and nothing else:
{{"<field>": {{"value": <see type>, "evidence": <verbatim quote (max 150 chars) of what you read, or null>, "page": <page number or null>, "confidence": <0..1>}}, ...}}

Rules
- Copy values exactly as printed (same spelling, hyphens, capitalisation). Never guess, infer, round or compute.
- true/false fields are checkboxes: true only if a mark is clearly printed inside the box; an empty box is false (not null) when the field says so.
- Numbers are JSON numbers without currency symbols or thousands separators.
- If the field is not in the document or you cannot read it, use null. A wrong value costs 100 times more than null.
- "confidence" = your probability that the value is exactly right (use <0.5 when guessing).

Fields (name (type): description):
{fields}
{doc}"""


def _shape(node: dict, defs: dict, depth: int = 0) -> str:
    if not isinstance(node, dict) or depth > 3:
        return "any"
    if "$ref" in node:
        return _shape(defs.get(node["$ref"].split("/")[-1], {}), defs, depth + 1)
    for key in ("anyOf", "oneOf"):
        if key in node:
            opts = [_shape(a, defs, depth + 1) for a in node[key] if a.get("type") != "null"]
            return opts[0] if opts else "any"
    t = node.get("type")
    if t == "array":
        return "[" + _shape(node.get("items", {}), defs, depth + 1) + ", ...]"
    if t == "object" or "properties" in node:
        return "{" + ", ".join(f'"{k}": {_shape(v, defs, depth + 1)}' for k, v in list(node.get("properties", {}).items())[:14]) + "}"
    return t if isinstance(t, str) else "string"


def field_line(spec: FieldSpec, defs: dict) -> str:
    kw = {"bool": "true/false", "number": "number", "integer": "integer", "json": "JSON", "text": "string", "span": "string"}[spec.kind]
    desc = " ".join(spec.description.split())[:650]
    extra = ""
    if spec.kind == "json":
        extra = f"  Shape: {_shape(spec.raw, defs)[:700]}"
    if spec.enum:
        extra += f"  Allowed values: {spec.enum}"
    return f"- {spec.name} ({kw}): {desc}{extra}"


def _page_blocks(doc: Document, specs: list, budget: int) -> str:
    pages = doc.pages if doc.has_text else doc.ocr_pages
    if not pages:
        return ""
    words = set(w for s in specs for w in re.findall(r"[a-z]{4,}", (s.name + " " + s.description[:200]).lower()))
    sel, used = [], 0
    ranked = sorted(range(len(pages)), key=lambda i: (i != 0, -sum(w in pages[i].lower() for w in words)))
    for i in ranked:
        n = len(pages[i]) + 20
        if used + n > budget and sel:
            continue
        sel.append(i)
        used += n
    return "\n".join(f"=== PAGE {i + 1} ===\n{pages[i]}" for i in sorted(sel))


# ---------------------------------------------------------------- LLM extractor
class Extractor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = Client(cfg.extractor, cfg.cache_dir, cfg.timeout)
        self.esc = Client(cfg.escalator, cfg.cache_dir, cfg.timeout)

    # ---- public
    def extract(self, doc: Document, specs: list, defs: dict, second: bool = False) -> dict:
        cl = self.esc if second else self.client
        if doc.source == "cuad":
            return self._cuad(doc, specs, cl, second)
        return self._forms(doc, specs, defs, cl, second)

    # ---- CUAD
    def _cuad(self, doc: Document, specs: list, cl: Client, second: bool) -> dict:
        if not second:
            text, trunc = build_digest(doc, self.cfg.max_context_chars)
            note = " (excerpts in document order; text between excerpts was omitted)" if trunc else ""
            return self._cuad_call(specs, text, note, cl, SYSTEM)
        # second reading: a *different view* of the document per field (its own best chunks), decorrelating retrieval errors
        out: dict = {}
        groups = [specs[i:i + 6] for i in range(0, len(specs), 6)]

        def run(g):
            ids = {0}
            for s in g:
                if s.name in CLAUSES:
                    ids.update(c.id for c in top_chunks(doc, s.name, 6))
            budget, parts, used = self.cfg.max_context_chars, [], 0
            for c in doc.chunks:
                if c.id in ids and (used + len(c.text) < budget or not parts):
                    parts.append(c.text if c.id == 0 else f"[... §{c.id} ...]\n{c.text}")
                    used += len(c.text)
            text = doc.text[:3500] + "\n" + "\n".join(parts[1:]) if len(doc.text) > budget else doc.text
            return self._cuad_call(g, text, " (excerpts)" if len(doc.text) > budget else "", cl, SYSTEM_RECHECK)

        with ThreadPoolExecutor(max(1, min(self.cfg.workers, len(groups)))) as ex:
            for d in ex.map(run, groups):
                out.update(d)
        return out

    def _cuad_call(self, specs: list, text: str, note: str, cl: Client, system: str) -> dict:
        fields = "\n".join(f"- {s.name}: {CLAUSES[s.name]['definition'] if s.name in CLAUSES else s.description[:300]}" for s in specs)
        prompt = CUAD_PROMPT.format(fields=fields, note=note, text=text)
        return self._ask(specs, system, [{"type": "text", "text": prompt}], cl, max_tokens=6000)

    # ---- forms / scans
    def _forms(self, doc: Document, specs: list, defs: dict, cl: Client, second: bool) -> dict:
        n = self.cfg.fields_per_call
        groups = [specs[i:i + n] for i in range(0, len(specs), n)]
        out: dict = {}

        def run(g):
            body = _page_blocks(doc, g, self.cfg.max_context_chars)
            parts: list = []
            want_img = os.environ.get("PG_VISION", "auto") != "0" and ((not doc.has_text) or os.environ.get("PG_VISION") == "1")
            if doc.has_text:
                docblock = "\nDocument text (pdf text layer, page by page):\n<<<\n" + body + "\n>>>"
                if want_img:
                    docblock += "\nThe page images are attached too: use them to decide checkbox state (the text layer does not show marks)."
            else:
                docblock = ("\nThe document is a scan; read the attached page images." +
                            (("\nNoisy OCR text for orientation only (may contain errors):\n<<<\n" + body + "\n>>>") if body else ""))
            parts.append({"type": "text", "text": FORM_PROMPT.format(fields="\n".join(field_line(s, defs) for s in g), doc=docblock)})
            if want_img:
                for pno, url in doc.images():
                    parts.append({"type": "text", "text": f"[Page {pno}]"})
                    parts.append({"type": "image_url", "image_url": {"url": url}})
            return self._ask(g, SYSTEM_RECHECK if second else SYSTEM, parts, cl, max_tokens=4500)

        with ThreadPoolExecutor(max(1, min(self.cfg.workers, len(groups)))) as ex:
            for d in ex.map(run, groups):
                out.update(d)
        return out

    # ---- shared
    def _ask(self, specs: list, system: str, parts: list, cl: Client, max_tokens: int) -> dict:
        content = parts if len(parts) > 1 else parts[0]["text"]
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        try:
            ans = parse_json(cl.chat(msgs, temperature=0.0, max_tokens=max_tokens, json_mode=True).text)
            if not ans:                       # different request, not a cached malformed answer
                retry = msgs + [{"role": "user", "content": "Return only the requested valid JSON object. No commentary or markdown."}]
                ans = parse_json(cl.chat(retry, max_tokens=max_tokens, json_mode=True).text)
        except LLMError:
            ans = None
        out = {}
        for s in specs:
            if ans is None or not isinstance(ans, dict) or not ans:
                out[s.name] = Cand(s.name, error=True, source="llm")
                continue
            if s.name not in ans:
                out[s.name] = Cand(s.name, error=True, source="llm")
                continue
            a = ans[s.name]
            if isinstance(a, dict):
                if "value" not in a:
                    out[s.name] = Cand(s.name, error=True, source="llm")
                    continue
                v, conf, ev, pg = a.get("value"), a.get("confidence", 0.5), a.get("evidence"), a.get("page")
            else:
                v, conf, ev, pg = a, 0.5, None, None
            try:
                pg = int(pg) if pg is not None else None
            except (TypeError, ValueError):
                pg = None
            out[s.name] = Cand(s.name, normalize_value(s, v), _conf(conf), None if ev is None else str(ev)[:400], pg, "llm")
        return out


# ---------------------------------------------------------------- offline rules (no LLM): reference + graceful degradation
_DATE = re.compile(r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+day\s+of\s+[A-Z][a-z]+,?\s+\d{4}|[A-Z][a-z]{2,8}\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}|\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?[A-Z][a-z]{2,8},?\s+\d{4})")
_SENT = re.compile(r"(?<=[.;])\s+(?=[A-Z(\d\"“])")
_TITLE = re.compile(r"\b(AGREEMENT|CONTRACT|LICENSE|LICENCE|MEMORANDUM|ADDENDUM|AMENDMENT|CHARTER|PLAN|TERMS)\b")


class OfflineExtractor:
    """Deterministic rules; deliberately modest. Lets the pipeline (proof engine, calibration, router) run end-to-end without any API key."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg

    def extract(self, doc: Document, specs: list, defs: dict, second: bool = False) -> dict:
        return {s.name: self._one(doc, s) for s in specs}

    def _one(self, doc: Document, s: FieldSpec) -> Cand:
        txt = doc.evidence_text
        if not txt.strip():
            return Cand(s.name, source="offline")
        if doc.source == "cuad":
            return self._cuad(doc, s)
        words = [w for w in re.split(r"[\W_]+", (s.name.replace("_", " ")).lower()) if len(w) > 2]
        if s.kind in ("bool", "json") or not words:
            return Cand(s.name, source="offline")
        for line in txt.splitlines():
            low = line.lower()
            if all(w in low for w in words):
                tail = line[low.rindex(words[-1]) + len(words[-1]):].strip(" :.\t")
                if not tail:
                    continue
                m = re.match(r"[\w$#.,/\-()&' ]{2,80}?(?=\s{3,}|$)", tail)
                v = (m.group(0) if m else tail[:60]).strip()
                v = normalize_value(s, v)
                if v:
                    return Cand(s.name, v, 0.4, line.strip()[:200], None, "offline")
        return Cand(s.name, source="offline")

    def _cuad(self, doc: Document, s: FieldSpec) -> Cand:
        t = doc.text
        n = s.name
        if n == "Document Name":
            for line in t[:1500].splitlines():
                L = line.strip()
                if 4 < len(L) < 90 and L.upper() == L and _TITLE.search(L) and not re.search(r"\d{4}", L):
                    return Cand(n, L, 0.55, L, 1, "offline")
            return Cand(n, source="offline")
        if n == "Agreement Date":
            head = t[:4000]
            for m in re.finditer(r"(?:dated|made|entered\s+into|as\s+of|effective)[^.]{0,80}?(" + _DATE.pattern + ")", head, re.I | re.S):
                return Cand(n, " ".join(m.group(1).split()), 0.6, m.group(0)[:200], 1, "offline")
            m = _DATE.search(head)
            return Cand(n, m.group(0), 0.35, m.group(0), 1, "offline") if m else Cand(n, source="offline")
        if n in HEADER_FIELDS or n not in CLAUSES:
            return Cand(n, source="offline")
        best = None
        for c in top_chunks(doc, n, 3):
            for sent in _SENT.split(c.text):
                sc = sum(w * min(len(p.findall(sent)), 2) for p, w in CLAUSES[n]["patterns"])
                if sc >= 3 and 25 < len(sent) < 900 and (best is None or sc > best[0]):
                    best = (sc, sent)
        if best:
            return Cand(n, " ".join(best[1].split()), min(0.2 + 0.08 * best[0], 0.7), None, None, "offline")
        return Cand(n, source="offline")
