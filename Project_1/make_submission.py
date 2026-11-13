#!/usr/bin/env python3
"""Renders SUBMISSION.md from measured files (runs/*.results.json, runs/ablation.json, calibrators/*.report.json, *.cost.json).
No number in the output is typed by hand. Re-run after `bash live.sh`.
    python3 make_submission.py [--team "My team"] [--extractor-open open|closed] [--extractor-where "Google AI Studio API"]"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
pct = lambda x, d=1: "n/a" if x is None else f"{100 * x:.{d}f}%"


def load(p):
    p = HERE / p
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default=os.environ.get("TEAM_NAME", "FILL: your team name"))
    ap.add_argument("--extractor-open", default="FILL: open or closed")
    ap.add_argument("--extractor-where", default="")
    a = ap.parse_args()
    from proofguard.config import get_config
    cfg = get_config()
    live = load("runs/pub_llm.results.json")
    rules = load("runs/pub_offline.results.json")
    abl = load("runs/ablation.json")
    cal = load("calibrators/llm.report.json")
    calr = load("calibrators/offline.report.json")
    cost = load("runs/pub_llm.jsonl.cost.json") or {}
    main_res = live or rules
    d = main_res["declared"]
    thr = main_res["threshold"]
    host = urlparse(cfg.extractor.base_url).netloc or "(not set in this environment)"
    where = a.extractor_where or host
    L = []
    if not live:
        L += ["> **STATUS: the live-LLM run has not been executed yet.** The numbers below are the rules-only fallback (no model). Run `bash live.sh`, then `python3 make_submission.py`.", ""]
    L += [f"# Submission: ProofGuard, a structured extractor that checks evidence for its answers", "",
          "## Authors", f"**Team name:** {a.team}", "", "**Project:** Project 1: Teach a machine to say \"I don't know\" (Structured Extraction from Documents)", "",
          "## Problem and approach",
          "A wrong value that reaches the system costs 100 reviews; a review costs 1. A booked field only pays off if it is right more than **99%** of the time, so the product is a *decision system*, not just an extractor. "
          "ProofGuard makes every extracted value prove itself and then decides under the real cost structure:", "",
          "1. **Layers 1-2, LLM extraction**: value + verbatim evidence quote + location, per field. Contracts are not truncated at 60k characters: a retriever (regex cues + token weights learned from dev) picks the best passages per clause. Scans are read from page images (and OCR when available).",
          "2. **Layer 3, Proof Engine** (deterministic): evidence matching (is the value in the document, where, is the quote real), schema validation, semantic consistency (label proximity, format rules, clause cues), cross-field checks (duplicates, overlapping spans, date order) and span-boundary checks (catches values cut mid-word/mid-clause that still match as substrings). An ungrounded value is never worth a model call.",
          "3. **Layer 4, H2O-Lightning-4B + adversarial verification**: the decision model answers three yes/no questions per field in one call (supported / complete / right kind of information), each an uncalibrated verifier signal. A *challenger* LLM then tries to refute the value, and a second reader re-extracts it from a different view of the document: both only for gray-zone fields.",
          "4. **Layer 5, cost-aware risk controller**: a calibrated logistic model over all signals plus a per-field track record (out-of-fold, so no leakage) gives P(correct). The declared threshold maximises the 10th percentile of bootstrap net saving over resampled *documents* (the uncertainty margin), not the point estimate. Fields that stay unreliable inside the accepted slice, and nested JSON fields (all-or-nothing match), are never automated.",
          "5. **Layer 6, adaptive router**: ACCEPT clear cases (1 extraction + 1 one-token check), REVIEW hopeless ones with no further spend, RECHECK only the gray zone. Only accepted values are written to the output; everything else is null (a human reviews it).", "",
          "Assumptions: the simplified costs in the task (review 1, wrong value 100); the dev split is used for all tuning (retrieval weights, calibrator, threshold) and the public split only for reporting; no hidden-test data is used.", "",
          "## Models and why",
          "Only runs backed by saved live evaluation outputs should be described as measured. Role marginals come from the layer ablation below (calibrators refitted on dev with growing signal sets, scored on the public split).", "",
          "| Model | Open-weight or closed | Where it runs | Main metric(s) on held-out data | Cost and time per document | Chosen? |", "|---|---|---|---|---|---|"]
    if rules:
        r = rules["declared"]
        L.append(f"| Rules only (no LLM; same proof engine + controller) | n/a | local CPU | net saving {pct(r['net_saving_pct'])}, accepted {r['accepted']} | 0 USD, no API | fallback: accepts nothing it cannot prove (safe, but no saving) |")
    if live:
        c = live["declared"]
        L.append(f"| Extractor / challenger / second reader: `{cfg.extractor.model or 'FILL'}` | {a.extractor_open} | {where} | end to end: net saving {pct(c['net_saving_pct'])}, precision {pct(c['precision'], 2)}, coverage {pct(c['coverage'])}, {c['hallucinated']} hallucinations booked | {cost.get('tokens_per_doc', 'n/a')} tokens, {cost.get('usd_per_doc', 'n/a')} USD, {cost.get('seconds_per_doc', 'n/a')} s | yes |")
        vm = cfg.verifier.model or "h2o-lightning-4b"
        step = ""
        if abl and len(abl) >= 3:
            step = f"adds {100 * (abl[2]['net_saving_pct'] - abl[1]['net_saving_pct']):+.1f} points of net saving over the proof engine alone"
        L.append(f"| Verifier: `{vm}` ({'H2O decision API' if cfg.verifier.api == 'systemone' else 'chat + logprobs'}) | open-weight (Apache-2.0) | {urlparse(cfg.verifier.base_url).netloc or 'same endpoint as extractor'} | {step or 'see ablation'} | input tokens only, one forward pass per question | yes |")
    else:
        L.append(f"| Extractor / verifier / challenger (LLM) | FILL | FILL | not measured yet: run `bash live.sh` | FILL | pending |")
    L += ["", f"**Declared confidence threshold: `{thr}`** (chosen on dev only: out-of-fold, maximising the 10th percentile of bootstrap net saving; break-even for 1:100 costs is 0.99). "
          "Calibrators are model-specific: if the extractor or verifier changes, re-run `bash live.sh` (one command) to refit and re-declare.", "",
          "## Results",
          f"Held-out split `{main_res['gold']}`: {main_res['docs']} documents, {main_res['fields']} fields. Only {pct(main_res.get('ceiling_pct'), 1)} of fields have a value, so that is the ceiling on saving. "
          + ("" if live else "(rules-only fallback shown; LLM run pending)"), "",
          "| System | Net saving | Precision of booked | Coverage | Hallucinations booked | Fields with no human |", "|---|---|---|---|---|---|",
          f"| **ProofGuard at the declared threshold** | **{pct(d['net_saving_pct'])}** | {pct(d['precision'], 2)} | {pct(d['coverage'])} | {d['hallucinated']} | {pct(d['automation_rate'])} |"]
    b = main_res.get("baselines", {})
    if b:
        L.append(f"| Most-common-value baseline (best possible threshold) | {pct(b['most_common_value']['net_saving_pct_at_best_threshold'])} | | | | |")
        L.append(f"| Most-common-value baseline at our threshold | {pct(b['most_common_value']['net_saving_pct_at_our_threshold'])} | | | | |")
    L += ["| Never answer | 0.0% | | | | 0% |", ""]
    if abl:
        L += ["### What each layer adds (same held-out data, calibrators fitted on dev only)", "", "| Signals used | Accepted | Wrong/hallucinated | Precision | Net saving |", "|---|---|---|---|---|"]
        for r in abl:
            L.append(f"| {r['layers']} | {r['accepted']} | {r['wrong_or_hallucinated']} | {pct(r['precision'], 2)} | {pct(r['net_saving_pct'])} |")
        L.append("")
    L += ["### If a wrong value costs 20x / 100x / 500x (same accepted fields)", "", "| Wrong value costs | Net saving |", "|---|---|"]
    for k in ("20", "100", "500"):
        L.append(f"| {k}x | {pct(main_res['cost_sensitivity_same_decisions'][k])} |")
    bt = main_res.get("best_threshold_by_cost")
    if bt:
        L += ["", "Re-choosing the threshold for each cost (an oracle on this split, for intuition only): " + ", ".join(f"{k}x -> {pct(v['saving_pct'])}" for k, v in bt.items()) + "."]
    L += ["", "### By source and field kind", "", "| Group | Fields | Accepted | Precision | Errors | Net saving |", "|---|---|---|---|---|---|"]
    for grp in ("by_source", "by_kind"):
        for k, v in main_res.get(grp, {}).items():
            L.append(f"| {k} | {v['fields']} | {v['accepted']} | {pct(v['precision'], 2)} | {v['wrong'] + v['hallucinated']} | {pct(v['net_saving_pct'])} |")
    rel = (cal or calr or {}).get("reliability")
    if rel:
        L += ["", "### Calibration: is 0.95 right 95% of the time? (out-of-fold on the tuning data)", "", "| Score bin | n | mean score | actual precision |", "|---|---|---|---|"]
        for r in rel:
            L.append(f"| {r['bin']} | {r['n']} | {'' if r['mean_p'] is None else round(r['mean_p'], 3)} | {'' if r['precision'] is None else round(r['precision'], 3)} |")
    nv = (cal or calr or {}).get("never_automate")
    if nv:
        L += ["", "Fields the audit keeps manual (unreliable even inside the accepted slice): " + ", ".join(f"`{k}`" for k in nv) + ". Nested JSON fields are always manual."]
    L += ["", "**Honest limits.** One wrong booked value costs as much as 100 right ones save, so results on a small split are noisy (we report the threshold's out-of-fold lower bound, not a lucky number). "
          "CUAD gold spans are sometimes whole sentences where the label name suggests a date; scans need a vision-capable model, otherwise they go to a human. "
          "Simulated runs test plumbing only. Offline baseline results must be labeled separately from live model results.", "",
          "## How to run",
          "Python 3.11+. The extractor needs only the standard library (`pdftotext`/`pdftoppm`/`tesseract` are used if installed). Optional: `pip install -r requirements.txt numpy matplotlib` (faster calibration, report slides).", "",
          "```bash", "cd Project_1", "bash setup.sh --accept-licenses     # venv, data, baseline", "cp .env.example .env              # add LLM_* (and VERIFIER_* for H2O-Lightning-4B)",
          "python3 pg.py doctor                # probes every model role", "bash live.sh                        # calibrate on dev, evaluate on public, build slides + this file",
          "bash run_hidden.sh <input.jsonl> <output.jsonl>   # what the organisers run; uses calibrators/llm.json", "python3 -m unittest discover -s tests   # unit + end-to-end tests (simulated model, plumbing only)", "```", "",
          "Environment variables (names only): `LLM_BASE_URL LLM_API_KEY LLM_MODEL`; optional `VERIFIER_BASE_URL VERIFIER_API VERIFIER_API_KEY VERIFIER_MODEL`, `CHALLENGER_*`, `ESCALATE_*`, `PG_WORKERS PG_MIN_GAP_SECONDS PG_MAX_CONTEXT_CHARS PG_VISION PG_JSON_MODE PG_CACHE_DIR`. "
          "Without a key `run_hidden.sh` runs the rules-only mode and books nothing it cannot prove.", "",
          "## Demo", "`submission/slides.pdf` (5 slides) and `submission/report.html` (booked fields with evidence, review reasons, calibration).", "",
          "## Next steps",
          "Fine-tune the verifier on SIXT's own accepted/rejected fields; learn per-field thresholds once shadow-mode data exists; layout-aware OCR zoom and checkbox crops; wire the review queue so a human sees the evidence and the proof notes next to the model's value. "
          "Rollout: shadow mode against human entries first, then auto-book only the fields whose calibration table holds, widening as evidence accumulates; the 1 / 100 costs are a parameter (`--error-cost`) to replace with SIXT's real figures.", "",
          "## Credits", "ExtractBench (Zhang et al., 2026, arXiv:2607.29677) and CUAD (Hendrycks et al., 2021). H2O-Lightning-4B by H2O.ai (Apache-2.0)."]
    text = "\n".join(L) + "\n"
    (HERE / "SUBMISSION.md").write_text(text, encoding="utf-8")
    print("wrote SUBMISSION.md" + ("   !! still contains FILL placeholders: edit them or pass --team/--extractor-open" if "FILL" in text else ""))


if __name__ == "__main__":
    main()
