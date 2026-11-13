"""Evaluation + report helpers: everything the SUBMISSION needs, computed from predictions (never typed by hand)."""
from __future__ import annotations

import collections
import json
from pathlib import Path

from .scoring import S5
from .schema import flatten

HERE = Path(__file__).resolve().parent.parent


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def _common_baseline(train: list, target: list) -> dict:
    """3_baseline.py --mode common, computed in-process: most common training value per (source, schema) field."""
    sig = lambda r: (r["source_dataset"], tuple(sorted(r["schema"]["properties"])))
    counts, n_docs = collections.defaultdict(collections.Counter), collections.Counter()
    for r in train:
        n_docs[sig(r)] += 1
        for f, v in r["gold"].items():
            counts[(sig(r), f)][None if v is None or str(v).strip() == "" else str(v)] += 1
    preds = {}
    for r in target:
        fields = {}
        for f in r["gold"]:
            value, conf = None, 0.0
            c = counts.get((sig(r), f))
            if c:
                top, k = c.most_common(1)[0]
                if top is not None:
                    value, conf = top, k / n_docs[sig(r)]
            fields[f] = {"value": value, "confidence": conf}
        preds[r["doc_id"]] = fields
    return preds


def breakdown(rows: list, thr: float, key) -> dict:
    groups = collections.defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    out = {}
    for k, rs in sorted(groups.items()):
        m = S5.at_threshold(rs, thr)
        out[k] = {"fields": m["n_fields"], "accepted": m["accepted"], "wrong": m["wrong"], "hallucinated": m["hallucinated"],
                  "precision": m["precision"] if m["accepted"] else None, "coverage": m["coverage"],
                  "net_saving_pct": (m["accepted"] - 100 * (m["wrong"] + m["hallucinated"])) / m["n_fields"] if m["n_fields"] else 0.0}
    return out


def evaluate(a) -> dict:
    gold = read_jsonl(a.gold)
    preds = S5.load_preds(a.pred)
    thr = a.threshold if a.threshold is not None else 0.95
    kinds = {}
    for r in gold:
        for s in flatten(r["schema"]):
            kinds[(r["doc_id"], s.name)] = s.kind
    rows = S5.collect(gold, preds)
    for r in rows:
        r["kind"] = kinds.get((r["doc_id"], r["field"]), "text")
    m = S5.at_threshold(rows, thr)
    n = m["n_fields"]
    ceiling = sum(1 for r in rows if r["gold_present"]) / n if n else 0.0
    ns = lambda mm, ec: (mm["accepted"] - ec * (mm["wrong"] + mm["hallucinated"])) / mm["n_fields"]
    res = {
        "label": a.label, "gold": Path(a.gold).name, "docs": len(gold), "fields": n, "threshold": thr, "ceiling_pct": ceiling,
        "declared": {"precision": m["precision"] if m["accepted"] else None, "coverage": m["coverage"], "hallucination_rate": m["hallucination_rate"],
                     "hallucinated": m["hallucinated"], "wrong": m["wrong"], "accepted": m["accepted"], "correct": m["correct"],
                     "automation_rate": m["accepted"] / n, "human_review_rate": 1 - m["accepted"] / n, "net_saving_pct": ns(m, 100)},
        "cost_sensitivity_same_decisions": {str(ec): ns(m, ec) for ec in (1, 5, 20, 100, 200, 500)},
        "by_source": breakdown(rows, thr, lambda r: r["source"]),
        "by_kind": breakdown(rows, thr, lambda r: r["kind"]),
        "by_field": breakdown(rows, thr, lambda r: f"{r['source']}:{r['field']}"),
    }
    # analysis from the full (value + probability) file: threshold sweep and re-optimised threshold per cost ratio
    if a.full_pred and Path(a.full_pred).exists():
        fpreds = S5.load_preds(a.full_pred)
        frows = S5.collect(gold, fpreds)
        curve = S5.risk_coverage(frows)
        res["sweep"] = [{"threshold": p["threshold"], "precision": p["precision"], "coverage": p["coverage"], "accepted": p["accepted"],
                         "net_saving_pct": ns(p, 100)} for p in curve][:: max(1, len(curve) // 80)]
        res["best_threshold_by_cost"] = {str(ec): S5.best_saving(curve, 1.0, float(ec)) for ec in (20, 100, 500)}
        res["coverage_at_precision"] = {str(t): S5.headline(curve, t)["coverage_at_target_precision"] for t in (0.99, 0.95, 0.90)}
    # baselines at their own best threshold (generous to them)
    try:
        train = read_jsonl(HERE / "data/unified/dev.jsonl")
        if Path(a.gold).name != "dev.jsonl":
            bp = _common_baseline(train, gold)
            bc = S5.risk_coverage(S5.collect(gold, bp))
            res["baselines"] = {"never_answer": {"net_saving_pct": 0.0},
                                "most_common_value": {"net_saving_pct_at_best_threshold": S5.best_saving(bc, 1.0, 100.0)["saving_pct"],
                                                      "net_saving_pct_at_our_threshold": ns(S5.at_threshold(S5.collect(gold, bp), thr), 100)}}
    except Exception as e:  # baselines are context, never a reason to fail the evaluation
        res["baselines_error"] = str(e)
    cp = HERE / "calibrators"
    cost = Path(str(a.pred) + ".cost.json")
    if cost.exists():
        res["cost"] = json.loads(cost.read_text())
    Path(a.out).write_text(json.dumps(res, indent=1))
    d = res["declared"]
    print(f"=== {a.label or a.gold} @ threshold {thr}: net saving {d['net_saving_pct']:.2%} | precision "
          f"{('n/a' if d['precision'] is None else format(d['precision'], '.2%'))} | coverage {d['coverage']:.2%} | automated {d['automation_rate']:.2%} of fields | "
          f"hallucinations {d['hallucinated']} | wrong {d['wrong']} | accepted {d['accepted']}")
    return res
