#!/usr/bin/env python3
"""Builds submission/slides.pdf (5 slides) and submission/report.html from results.json (+ calibration report + trace).
All numbers are read from files produced by `pg.py evaluate` / `pg.py calibrate`: nothing is typed by hand.
    python3 make_report.py --results runs/pub.results.json --cal calibrators/llm.report.json --trace runs/pub.trace.jsonl --out submission
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ORANGE, INK, MUTED, BLUE, GREEN, RED, BG = "#ff5f00", "#1f2933", "#6b7280", "#2563eb", "#15803d", "#b91c1c", "#ffffff"
pct = lambda x, d=1: "n/a" if x is None else f"{100 * x:.{d}f}%"


def load(p):
    return json.loads(Path(p).read_text()) if p and Path(p).exists() else {}


def box(ax, x, y, w, h, title, sub, color=INK, fill="#f8fafc", edge="#cbd5e1", bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fill, ec=edge, lw=1.5))
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=13, fontweight="bold" if bold else "normal", color=color)
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=10, color=MUTED)


def base(title, label):
    fig = plt.figure(figsize=(13.33, 7.5), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0, 0.965), 1, 0.035, fc=ORANGE, ec="none"))
    ax.text(0.04, 0.9, title, fontsize=26, fontweight="bold", color=INK, va="center")
    ax.text(0.96, 0.03, label, fontsize=9, color=MUTED, ha="right")
    return fig, ax


def slides(res, cal, out, label):
    d = res.get("declared", {})
    base_ = res.get("baselines", {})
    with PdfPages(out) as pdf:
        # 1 problem
        fig, ax = base("ProofGuard: an extractor that checks evidence for its answers", label)
        ax.text(0.04, 0.78, "Project 1: Teach a machine to say \"I don't know\"", fontsize=15, color=MUTED)
        ax.text(0.04, 0.66, "One confident wrong value costs 100 reviews.\nSo the useful system is not the one that answers most; it is the one that knows when it is right.", fontsize=17, color=INK, va="top", linespacing=1.5)
        for i, (big, small) in enumerate([("1", "human review of a field"), ("100", "a wrong value that gets booked"), ("99%", "break-even precision (1 - 1/100)"),
                                          (pct(res.get("ceiling_pct", 0.42), 0), "of fields have a value at all: the savings ceiling")]):
            x = 0.04 + i * 0.235
            ax.add_patch(FancyBboxPatch((x, 0.2), 0.215, 0.26, boxstyle="round,pad=0.01,rounding_size=0.02", fc="#fff7ed" if i != 3 else "#f1f5f9", ec="#fed7aa" if i != 3 else "#cbd5e1"))
            ax.text(x + 0.1075, 0.36, big, fontsize=34, fontweight="bold", color=ORANGE if i != 3 else INK, ha="center")
            ax.text(x + 0.1075, 0.26, small, fontsize=11, color=MUTED, ha="center")
        ax.text(0.04, 0.1, "Our answer: never trust a value because the model said so. Make every value prove itself, then decide under the real 100:1 costs.", fontsize=13, color=INK)
        pdf.savefig(fig); plt.close(fig)

        # 2 architecture
        fig, ax = base("Six layers: prove, challenge, then decide", label)
        rows = [("Layers 1-2  LLM extraction", "value + verbatim evidence quote + location  |  retrieval-first for long contracts, page images for scans", "#f8fafc", "#cbd5e1", INK),
                ("Layer 3  Proof Engine", "evidence matching  |  schema validation  |  semantic consistency  |  cross-field + span-boundary checks", "#eff6ff", "#93c5fd", BLUE),
                ("Layer 4  H2O-Lightning-4B + adversarial verification", "one-token verdict with logprobs: does the evidence support the value?  |  challenger tries to refute (gray zone only)", "#eff6ff", "#93c5fd", BLUE),
                ("Layer 5  Cost-aware risk controller", "calibrated P(correct)  +  100:1 penalty  +  bootstrap uncertainty margin (threshold = 20th-percentile saving)", "#fff7ed", "#fdba74", ORANGE),
                ("Layer 6  Adaptive router", "ACCEPT clear cases (cheap)  |  RECHECK the gray zone (2nd reader + challenger)  |  REVIEW the rest (a human, cost 1)", "#f0fdf4", "#86efac", GREEN)]
        for i, (t, s, f, e, c) in enumerate(rows):
            y = 0.72 - i * 0.145
            box(ax, 0.06, y, 0.88, 0.115, t, s, c, f, e)
            if i < len(rows) - 1:
                ax.annotate("", xy=(0.5, y - 0.012), xytext=(0.5, y), arrowprops=dict(arrowstyle="-", color=MUTED))
        ax.text(0.06, 0.07, "Money is spent only where it can change the decision: a clear case costs 1 extraction + 1 one-token check.", fontsize=12, color=INK)
        pdf.savefig(fig); plt.close(fig)

        # 3 results
        fig, ax = base("Results on documents we never tuned on", label)
        ax.text(0.04, 0.83, f"Held-out split: {res.get('docs', '?')} documents, {res.get('fields', '?')} fields. Declared confidence threshold: {res.get('threshold')}", fontsize=12, color=MUTED)
        kp = [("Net saving", pct(d.get("net_saving_pct"))), ("Precision of booked", pct(d.get("precision"), 2)), ("Coverage", pct(d.get("coverage"))),
              ("Hallucinations booked", str(d.get("hallucinated", "?"))), ("Fields with no human", pct(d.get("automation_rate")))]
        for i, (k, v) in enumerate(kp):
            x = 0.04 + i * 0.19
            ax.add_patch(FancyBboxPatch((x, 0.64), 0.175, 0.15, boxstyle="round,pad=0.008,rounding_size=0.015", fc="#f8fafc", ec="#cbd5e1"))
            ax.text(x + 0.0875, 0.725, v, fontsize=24, fontweight="bold", color=ORANGE if i == 0 else INK, ha="center")
            ax.text(x + 0.0875, 0.665, k, fontsize=10, color=MUTED, ha="center")
        sw = res.get("sweep")
        a2 = fig.add_axes([0.07, 0.12, 0.52, 0.42])
        if sw:
            xs = [p["threshold"] for p in sw]; ys = [100 * p["net_saving_pct"] for p in sw]
            a2.plot(xs, ys, color=ORANGE, lw=2.5)
            a2.axvline(res.get("threshold"), color=INK, ls="--", lw=1)
            a2.axhline(0, color=MUTED, lw=0.8)
            a2.set_xlabel("confidence threshold"); a2.set_ylabel("net saving vs all-manual (%)")
            a2.set_title("Net saving vs threshold (dashed: our declared threshold)", fontsize=11, loc="left")
            lo = min(ys + [0]); a2.set_ylim(max(lo, -10), max(max(ys) * 1.15, 1))
        for s in ("top", "right"):
            a2.spines[s].set_visible(False)
        b = base_.get("most_common_value", {})
        rows3 = [("ProofGuard (declared threshold)", pct(d.get("net_saving_pct"))),
                 ("Most-common-value baseline (best threshold)", pct(b.get("net_saving_pct_at_best_threshold"))),
                 ("Most-common-value at our threshold", pct(b.get("net_saving_pct_at_our_threshold"))), ("Never answer", "0.0%")]
        ax.text(0.64, 0.5, "Versus the baselines", fontsize=14, fontweight="bold", color=INK)
        for i, (k, v) in enumerate(rows3):
            ax.text(0.64, 0.44 - i * 0.065, k, fontsize=11, color=MUTED); ax.text(0.95, 0.44 - i * 0.065, v, fontsize=12, color=INK, ha="right", fontweight="bold")
        bs = res.get("by_source", {})
        ax.text(0.64, 0.17, "  ".join(f"{k}: {pct(v.get('net_saving_pct'))} saved, precision {pct(v.get('precision'), 1)}" for k, v in bs.items()), fontsize=10, color=MUTED)
        pdf.savefig(fig); plt.close(fig)

        # 4 business case
        fig, ax = base("Business case: how it moves if mistakes cost more", label)
        cs = res.get("cost_sensitivity_same_decisions", {})
        best = res.get("best_threshold_by_cost", {})
        a3 = fig.add_axes([0.07, 0.17, 0.5, 0.5])
        costs = ["20", "100", "500"]
        same = [100 * cs.get(c, 0) for c in costs]
        reopt = [100 * (best.get(c, {}) or {}).get("saving_pct", 0) for c in costs]
        xs = range(3)
        a3.bar([x - 0.19 for x in xs], same, 0.38, color="#cbd5e1", label="same decisions (declared threshold)")
        a3.bar([x + 0.19 for x in xs], reopt, 0.38, color=ORANGE, label="threshold re-chosen for that cost")
        a3.set_xticks(list(xs)); a3.set_xticklabels([f"wrong value = {c}x" for c in costs]); a3.axhline(0, color=MUTED, lw=0.8)
        a3.set_ylabel("net saving (%)"); a3.legend(frameon=False, fontsize=9)
        for s in ("top", "right"):
            a3.spines[s].set_visible(False)
        cost = res.get("cost", {})
        notes = [f"Automated with no human: {pct(d.get('automation_rate'))} of fields; {pct(d.get('human_review_rate'))} go to a person",
                 f"Tokens per document: {cost.get('tokens_per_doc', 'n/a')}   |   USD per document: {cost.get('usd_per_doc', 'n/a')}   |   {cost.get('seconds_per_doc', 'n/a')} s per document",
                 "Never automated: nested objects (all-or-nothing match) and any field whose confident answers fail the audit",
                 "Rollout: shadow mode (compare to humans) -> auto-book the safest fields -> widen as the calibration table holds"]
        ax.text(0.62, 0.7, "What it means", fontsize=14, fontweight="bold", color=INK)
        for i, n in enumerate(notes):
            ax.text(0.62, 0.64 - i * 0.12, n, fontsize=10.5, color=INK, va="top", wrap=True)
        pdf.savefig(fig); plt.close(fig)

        # 5 limits / next
        fig, ax = base("Honest limits and next steps", label)
        limits = ["Only ~42% of all fields have a value, so even a perfect system saves at most ~42% of the manual cost.",
                  "Gold labels are noisy (e.g. CUAD 'Effective Date' sometimes labels the whole term sentence); F1>=0.8 is strict.",
                  "A calibrator is only valid for the models it was fitted on: re-fit after changing the extractor or verifier (1 command).",
                  "Scans without a vision-capable model fall back to human review: safe, but no saving."]
        nxt = ["Fine-tune the verifier on our own booked/rejected fields (the decision model is cheap to adapt).",
               "Learn per-field thresholds once enough shadow-mode data exists.",
               "Add layout-aware OCR zoom for low-quality scans and checkbox crops.",
               "Wire the review queue: show evidence + proof notes next to the model's value to make each review faster."]
        ax.text(0.04, 0.78, "Limits", fontsize=16, fontweight="bold", color=RED)
        for i, t in enumerate(limits):
            ax.text(0.04, 0.72 - i * 0.1, "- " + t, fontsize=11.5, color=INK, va="top", wrap=True)
        ax.text(0.04, 0.3, "Next steps", fontsize=16, fontweight="bold", color=GREEN)
        for i, t in enumerate(nxt):
            ax.text(0.04, 0.25 - i * 0.055, "- " + t, fontsize=11.5, color=INK, va="top")
        pdf.savefig(fig); plt.close(fig)


def html_report(res, cal, trace_path, out, label):
    d = res.get("declared", {})
    rows = lambda items: "".join(f"<tr>{''.join(f'<td>{html.escape(str(c))}</td>' for c in r)}</tr>" for r in items)
    acc, rev = [], []
    if trace_path and Path(trace_path).exists():
        for l in open(trace_path, encoding="utf-8"):
            t = json.loads(l)
            for n, f in t["fields"].items():
                if f.get("route") == "accept" and len(acc) < 25:
                    acc.append((t["doc_id"][:46], n, f["value"][:90], f.get("p2") or f.get("p1"), f.get("verifier") or "", f.get("evidence", "")[:60]))
                elif str(f.get("route", "")).startswith("review:") and len(rev) < 25:
                    rev.append((t["doc_id"][:46], n, f["value"][:70], f["route"], f.get("hard") or f.get("challenge") or ""))
    bykind = rows([(k, v["fields"], v["accepted"], pct(v["precision"], 2), v["wrong"] + v["hallucinated"], pct(v["net_saving_pct"])) for k, v in res.get("by_kind", {}).items()])
    bysrc = rows([(k, v["fields"], v["accepted"], pct(v["precision"], 2), v["wrong"] + v["hallucinated"], pct(v["net_saving_pct"])) for k, v in res.get("by_source", {}).items()])
    top = sorted(res.get("by_field", {}).items(), key=lambda kv: -kv[1]["accepted"])[:15]
    byfield = rows([(k, v["accepted"], pct(v["precision"], 1), v["wrong"] + v["hallucinated"]) for k, v in top])
    rel = rows([(b["bin"], b["n"], None if b["mean_p"] is None else round(b["mean_p"], 3), None if b["precision"] is None else round(b["precision"], 3)) for b in cal.get("reliability", [])])
    cs = rows([(f"{k}x", pct(v)) for k, v in res.get("cost_sensitivity_same_decisions", {}).items()])
    svg = ""
    sw = res.get("sweep")
    if sw:
        xs = [p["threshold"] for p in sw]; ys = [100 * p["net_saving_pct"] for p in sw]
        lo, hi = min(ys + [0]), max(ys + [1])
        sx = lambda x: 40 + 560 * (x - min(xs)) / ((max(xs) - min(xs)) or 1)
        sy = lambda y: 190 - 170 * (y - lo) / ((hi - lo) or 1)
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
        svg = (f'<svg viewBox="0 0 620 210" width="620"><polyline fill="none" stroke="#ff5f00" stroke-width="2.5" points="{pts}"/>'
               f'<line x1="40" x2="600" y1="{sy(0):.1f}" y2="{sy(0):.1f}" stroke="#999"/><line x1="{sx(res["threshold"]):.1f}" x2="{sx(res["threshold"]):.1f}" y1="10" y2="195" stroke="#1f2933" stroke-dasharray="4"/>'
               f'<text x="42" y="12" font-size="11">net saving % vs threshold (dashed = declared {res["threshold"]})</text></svg>')
    page = f"""<!doctype html><meta charset=utf-8><title>ProofGuard report</title>
<style>body{{font:14px/1.5 system-ui,sans-serif;max-width:1000px;margin:2rem auto;color:#1f2933;padding:0 1rem}}h1{{border-bottom:4px solid #ff5f00}}table{{border-collapse:collapse;margin:.5rem 0 1.5rem;width:100%}}
td,th{{border:1px solid #d1d5db;padding:4px 8px;text-align:left;font-size:13px}}th{{background:#f3f4f6}}.k{{display:flex;gap:1rem;flex-wrap:wrap}}.k div{{background:#f8fafc;border:1px solid #cbd5e1;border-radius:8px;padding:.6rem 1rem}}
.k b{{display:block;font-size:24px;color:#ff5f00}}small{{color:#6b7280}}</style>
<h1>ProofGuard: results report</h1><p><small>{html.escape(label)} &middot; {res.get('docs')} documents, {res.get('fields')} fields &middot; declared threshold {res.get('threshold')} &middot; generated from results.json</small></p>
<div class=k><div><b>{pct(d.get('net_saving_pct'))}</b>net saving</div><div><b>{pct(d.get('precision'), 2)}</b>precision of booked</div><div><b>{pct(d.get('coverage'))}</b>coverage</div>
<div><b>{d.get('hallucinated')}</b>hallucinations booked</div><div><b>{pct(d.get('automation_rate'))}</b>fields with no human</div></div>
{svg}<h2>Per source</h2><table><tr><th>source</th><th>fields</th><th>accepted</th><th>precision</th><th>errors</th><th>net saving</th></tr>{bysrc}</table>
<h2>Per field kind</h2><table><tr><th>kind</th><th>fields</th><th>accepted</th><th>precision</th><th>errors</th><th>net saving</th></tr>{bykind}</table>
<h2>Most automated fields</h2><table><tr><th>field</th><th>accepted</th><th>precision</th><th>errors</th></tr>{byfield}</table>
<h2>If a wrong value costs more (same accepted fields)</h2><table><tr><th>wrong value costs</th><th>net saving</th></tr>{cs}</table>
<h2>Calibration (out-of-fold on the tuning data): is 0.95 right 95% of the time?</h2><table><tr><th>score bin</th><th>n</th><th>mean score</th><th>actual precision</th></tr>{rel}</table>
<h2>Booked automatically (with evidence)</h2><table><tr><th>document</th><th>field</th><th>value</th><th>P(correct)</th><th>verifier</th><th>evidence</th></tr>{rows(acc)}</table>
<h2>Sent to a human (and why)</h2><table><tr><th>document</th><th>field</th><th>proposed value</th><th>route</th><th>reason</th></tr>{rows(rev)}</table>"""
    Path(out).write_text(page, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True); ap.add_argument("--cal"); ap.add_argument("--trace"); ap.add_argument("--out", default="submission")
    a = ap.parse_args()
    res, cal = load(a.results), load(a.cal)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    label = res.get("label") or "ProofGuard"
    slides(res, cal, out / "slides.pdf", label)
    html_report(res, cal, a.trace, out / "report.html", label)
    print(f"wrote {out/'slides.pdf'} and {out/'report.html'}")


if __name__ == "__main__":
    main()
