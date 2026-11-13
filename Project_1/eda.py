#!/usr/bin/env python3
"""Reproducible EDA for Project 1. Writes EDA.md with real numbers computed from data/unified/*.jsonl.
    python3 eda.py [--out EDA.md]
Standard library only (pdftotext is used if installed, to count scanned PDFs)."""
from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import statistics as st
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"


def load(name):
    p = DATA / "unified" / f"{name}.jsonl"
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


def pdf_chars(path: Path) -> int:
    if not shutil.which("pdftotext"):
        return -1
    r = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True)
    return len(r.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "EDA.md"))
    a = ap.parse_args()
    dev, pub = load("dev"), load("test_public")
    out = ["# EDA: what the data looks like (computed by `eda.py`, not hand-written)", ""]

    out += ["## 1. Size", "", "| split | docs | CUAD docs | ExtractBench docs | fields | fields with a value | fields absent (gold null) |", "|---|---|---|---|---|---|---|"]
    for nm, rs in (("dev", dev), ("test_public", pub)):
        nf = sum(len(r["gold"]) for r in rs)
        nn = sum(v is None or str(v).strip() == "" for r in rs for v in r["gold"].values())
        nc = sum(r["source_dataset"] == "cuad" for r in rs)
        out.append(f"| {nm} | {len(rs)} | {nc} | {len(rs) - nc} | {nf} | {nf - nn} ({pct(nf - nn, nf)}) | {nn} ({pct(nn, nf)}) |")
    out.append("")
    allr = dev + pub
    nf = sum(len(r["gold"]) for r in allr)
    npres = sum(not (v is None or str(v).strip() == "") for r in allr for v in r["gold"].values())
    out += [f"**Ceiling on savings.** Only {pct(npres, nf)} of all fields have a value. An absent field can never be 'booked' "
            f"(a true null costs one review, saves nothing), so even a perfect system can save at most {pct(npres, nf)} of the manual cost. "
            "Every wrongly accepted value costs 100 reviews, so the realistic goal is *high precision on a modest, safe slice*.", ""]

    # per source
    out += ["## 2. Per source", "", "| source | fields | absent | fields/doc (median / max) |", "|---|---|---|---|"]
    for src in ("cuad", "extractbench"):
        rs = [r for r in allr if r["source_dataset"] == src]
        fl = [len(r["gold"]) for r in rs]
        n = sum(fl)
        nn = sum(v is None or str(v).strip() == "" for r in rs for v in r["gold"].values())
        out.append(f"| {src} | {n} | {pct(nn, n)} | {st.median(fl):.0f} / {max(fl)} |")
    out.append("")

    # CUAD
    cu_path = DATA / "raw/cuad/CUADv1.json"
    cuad = [r for r in allr if r["source_dataset"] == "cuad"]
    if cuad and cu_path.exists():
        raw = json.loads(cu_path.read_text(encoding="utf-8"))["data"]
        L = sorted(sum(len(p["context"]) for p in d["paragraphs"]) for d in raw)
        out += ["## 3. CUAD (contracts, 41 clause types, span answers)", "",
                f"- Contract length (chars): median {st.median(L):,.0f}, p90 {L[int(.9 * len(L))]:,}, max {L[-1]:,}.",
                f"- **{pct(sum(l > 60000 for l in L), len(L))} of contracts are longer than 60,000 characters**, the cut-off used by the starter, "
                "so a naive 'paste the contract into the prompt' extractor silently never sees the end of the document "
                "(where termination, liability and governing-law clauses often sit). A small model with a 32K-token window is worse off. "
                "-> we retrieve candidate paragraphs per clause instead of truncating.", ""]
        cn, cc = collections.Counter(), collections.Counter()
        for r in cuad:
            for f, v in r["gold"].items():
                cc[f] += 1
                cn[f] += v is None
        rows = sorted(((cn[f] / cc[f], f) for f in cc))
        out += ["Absent-rate per clause type (a model that 'finds' a clause here is hallucinating):", "",
                "| clause | absent | clause | absent | clause | absent |", "|---|---|---|---|---|---|"]
        for i in range(0, len(rows), 3):
            chunk = rows[i:i + 3]
            out.append("| " + " | ".join(f"{f} | {100 * p:.0f}%" for p, f in chunk) + " |" * 1)
        out += ["", "Always-present: " + ", ".join(f for p, f in rows if p == 0) + ". Almost-never-present (>=90% absent): "
                + ", ".join(f for p, f in rows if p >= 0.9) + ".", ""]
        spans = [len(v) for r in cuad for v in r["gold"].values() if v]
        out += [f"- Gold answers are long: median {st.median(spans):.0f} chars, p90 {sorted(spans)[int(.9 * len(spans))]} chars; "
                "scoring is token-F1 >= 0.8, so the model must copy the *whole* clause, not a keyword.", ""]

    # ExtractBench
    eb = [r for r in allr if r["source_dataset"] == "extractbench"]
    if eb:
        kinds = collections.Counter()
        for r in eb:
            for f, v in r["gold"].items():
                if v is None:
                    kinds["absent"] += 1
                elif str(v) in ("true", "false"):
                    kinds["checkbox=" + str(v)] += 1
                elif r["schema"]["properties"].get(f, {}).get("x-normalizer") == "json":
                    kinds["nested json (exact match)"] += 1
                elif re.fullmatch(r"-?\d+", str(v)):
                    kinds["integer"] += 1
                elif re.fullmatch(r"-?\d+\.\d+", str(v)):
                    kinds["decimal"] += 1
                else:
                    kinds["text"] += 1
        tot = sum(kinds.values())
        out += ["## 4. ExtractBench (forms and scans)", "", "| value kind | fields | share |", "|---|---|---|"]
        for k, v in kinds.most_common():
            out.append(f"| {k} | {v} | {pct(v, tot)} |")
        out += ["", "- **An empty checkbox is not null**: it is the string `false` "
                f"({kinds['checkbox=false']} such fields vs {kinds['checkbox=true']} `true`). A system that returns null for an unticked box wastes a free, easy saving; "
                "one that reads a scan badly and says `false` for a ticked box books a wrong value.",
                f"- Nested objects/arrays ({kinds['nested json (exact match)']} fields) are scored all-or-nothing, so one wrong cell makes the whole value wrong. "
                "These are the fields we would never automate.", ""]
        if shutil.which("pdftotext"):
            chars = [pdf_chars(DATA / r["pages"][0]) for r in eb if (DATA / r["pages"][0]).exists()]
            scans = sum(c < 200 for c in chars)
            out += [f"- **{scans} of {len(chars)} ExtractBench PDFs ({pct(scans, len(chars))}) have no usable text layer** (scans/photos). "
                    "The text-only starter path cannot read them at all; they need OCR or a vision model, and there is no text to verify an answer against. "
                    f"Median text length of the rest: {st.median([c for c in chars if c >= 200]):,.0f} chars.", ""]
        tg = collections.Counter(t for r in eb for t in r["tags"] if t.startswith(("perception:", "challenge:", "domain:")))
        out += ["Capture-condition / challenge tags (docs):", "", ", ".join(f"`{k}`: {v}" for k, v in tg.most_common(12)), ""]
        lens = [len(json.dumps(r["schema"])) for r in eb]
        out += [f"- Schema size (JSON chars): median {st.median(lens):,.0f}, max {max(lens):,}. Some forms have {max(len(r['gold']) for r in eb)} fields, "
                "which does not fit one reliable prompt for a small model: we chunk the schema.", ""]

    # baseline
    out += ["## 5. Reference points", "",
            "- 'Never answer' saves 0%. 'Most common value' (3_baseline.py) is the bar a real system must clear.",
            "- Break-even precision at 1:100 costs is 100/101 = **99.01%**: a booked field only pays off if the chance it is right exceeds that. "
            "This number, not accuracy, drives the whole design of ProofGuard.", ""]
    Path(a.out).write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
