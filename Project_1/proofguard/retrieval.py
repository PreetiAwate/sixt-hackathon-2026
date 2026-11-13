"""Clause retrieval for long contracts: regex cues + log-odds token weights learned from dev gold spans.
Measured recall of the gold span inside the top-4 chunks (held-out public split): regex 72%, learned 73%, both 83% (top-8: 87%)."""
from __future__ import annotations

import collections
import json
import math
import re
from pathlib import Path

from .cuad_defs import CLAUSES, HEADER_FIELDS
from .docs import Document, load_document

MODEL_PATH = Path(__file__).with_name("retrieval_model.json")
_W: dict | None = None


def _feats(s: str) -> set:
    t = re.findall(r"[a-z]{3,}", s.lower())
    return set(t) | {a + "_" + b for a, b in zip(t, t[1:])}


def weights() -> dict:
    global _W
    if _W is None:
        _W = json.loads(MODEL_PATH.read_text(encoding="utf-8")) if MODEL_PATH.exists() else {}
    return _W


def fit(dev_records: list, cache_dir: Path, out: Path = MODEL_PATH) -> dict:
    pos = collections.defaultdict(collections.Counter)
    npos, bg, nbg = collections.Counter(), collections.Counter(), 0
    for r in dev_records:
        if r["source_dataset"] != "cuad":
            continue
        d = load_document(r, cache_dir)
        for c in d.chunks:
            bg.update(_feats(c.text))
            nbg += 1
        for f, v in r["gold"].items():
            if v:
                for sp in r["gold_evidence"].get(f, {}).get("spans", []):
                    pos[f].update(_feats(d.text[sp["start"]:sp["end"]]))
                    npos[f] += 1
    W = {}
    for f in pos:
        ws = {}
        for t, c in pos[f].items():
            if c >= 3:
                w = math.log(((c + 0.5) / (npos[f] + 1)) / ((bg[t] + 0.5) / (nbg + 1)))
                if w > 0.7:
                    ws[t] = round(w, 3)
        W[f] = dict(sorted(ws.items(), key=lambda x: -x[1])[:80])
    out.write_text(json.dumps(W), encoding="utf-8")
    global _W
    _W = W
    return W


def score(field: str, text: str) -> float:
    cl = CLAUSES[field]
    rx = sum(w * min(len(p.findall(text)), 2) for p, w in cl["patterns"])
    wt = weights().get(field)
    lm = sum(wt.get(t, 0.0) for t in _feats(text)) if wt else 0.0
    return rx + lm


def top_chunks(doc: Document, field: str, k: int) -> list:
    sc = sorted(((score(field, c.text), -c.id, c) for c in doc.chunks), key=lambda x: (x[0], x[1]), reverse=True)
    return [x[2] for x in sc[:k] if x[0] > 0]


def build_digest(doc: Document, budget: int, header_chars: int = 3500) -> tuple[str, bool]:
    """Whole contract if it fits the budget; else header + the best chunks for every clause, in document order.
    Returns (text, truncated)."""
    if len(doc.text) <= budget:
        return doc.text, False
    header = doc.text[:header_chars]
    fields = [f for f in CLAUSES if f not in HEADER_FIELDS]
    for k in (8, 6, 4, 3, 2, 1):
        ids = set()
        for f in fields:
            ids.update(c.id for c in top_chunks(doc, f, k))
        chosen = [c for c in doc.chunks if c.id in ids and c.end > header_chars]
        size = len(header) + sum(len(c.text) + 12 for c in chosen)
        if size <= budget or k == 1:
            break
    parts, used = [header], len(header)
    for c in chosen:
        if used + len(c.text) + 12 > budget:
            break
        parts.append(f"\n[... §{c.id} ...]\n{c.text}")
        used += len(c.text) + 12
    return "\n".join(parts), True
