#!/usr/bin/env python3
"""ProofGuard command line.
  python3 pg.py predict   --data data/unified/dev.jsonl --out preds.jsonl [--mode auto|llm|offline] [--limit N] [--resume]
  python3 pg.py calibrate --traces runs/dev_traces.jsonl --gold data/unified/dev.jsonl --out calibrators/llm.json
  python3 pg.py evaluate  --gold data/unified/test_public.jsonl --pred preds.jsonl --full-pred preds.full.jsonl [--threshold T]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from proofguard.config import get_config  # noqa: E402
from proofguard.llm import USAGE  # noqa: E402


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def pick_calibrator(args, cfg, mode):
    from proofguard.risk import Calibrator
    path = Path(args.calibrator) if args.calibrator else HERE / "calibrators" / f"{mode}.json"
    if not path.exists():
        print(f"[proofguard] WARNING: no calibrator at {path}: using UNCALIBRATED all-review mode (threshold 1.0001). "
              f"Run the calibration step (see SUBMISSION.md) before trusting any number.", file=sys.stderr)
        return None
    cal = Calibrator.load(path)
    sig = cal.meta.get("signature")
    if sig != cfg.signature and mode == "llm":
        print(f"[proofguard] WARNING: calibrator was fitted for models '{sig}' but the active models are '{cfg.signature}'. "
              f"Ignoring incompatible calibrator; all-review mode. Recalibrate.", file=sys.stderr)
        return None
    return cal


def cmd_predict(a):
    from proofguard.pipeline import ProofGuard
    cfg = get_config()
    if a.workers:
        cfg.workers = a.workers
    mode = a.mode
    eff_mode = "llm" if (mode == "llm" or (mode == "auto" and cfg.llm_ready)) else "offline"
    cal = None if a.calibrating else pick_calibrator(a, cfg, eff_mode)
    pg = ProofGuard(cfg, mode, cal, full=a.calibrating)
    recs = read_jsonl(a.data)
    if a.sample:
        random.Random(a.seed).shuffle(recs)
    recs = recs[: a.limit or a.sample or None]
    import hashlib
    provenance = {"signature": cfg.signature, "mode": eff_mode, "calibrating": a.calibrating,
                  "data_sha256": hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
                  "doc_ids": [r["doc_id"] for r in recs],
                  "calibrator": None if cal is None else {"meta": cal.meta, "threshold": cal.threshold}}
    manifest = Path(a.out + ".manifest.json")
    if a.resume and Path(a.out).exists():
        if not manifest.exists() or json.loads(manifest.read_text()) != provenance:
            raise SystemExit("Cannot resume: input/config/calibration changed or manifest missing. Use a new output file.")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(provenance, indent=2))
    done = set()
    if a.resume and os.path.exists(a.out):
        done = {json.loads(l)["doc_id"] for l in open(a.out, encoding="utf-8") if l.strip()}
    todo = [r for r in recs if r["doc_id"] not in done]
    print(f"[proofguard] mode={pg.mode} docs={len(todo)} (skipping {len(done)}) threshold={pg.threshold} models={cfg.signature if pg.mode == 'llm' else 'rules'}", file=sys.stderr)
    for p in (a.out, a.full_out, a.trace):
        if p:
            Path(p).parent.mkdir(parents=True, exist_ok=True)
    mode_w = "a" if a.resume else "w"
    fo = open(a.out, mode_w, encoding="utf-8")
    ff = open(a.full_out, mode_w, encoding="utf-8") if a.full_out else None
    ft = open(a.trace, mode_w, encoding="utf-8") if a.trace else None
    t0, n = time.time(), 0

    def on_done(res):
        nonlocal n
        row, trace = res
        full = row.pop("_full", None)
        fo.write(json.dumps(row, ensure_ascii=False) + "\n"); fo.flush()
        if ff and full is not None:
            ff.write(json.dumps({"doc_id": row["doc_id"], "fields": full}, ensure_ascii=False) + "\n"); ff.flush()
        if ft:
            ft.write(json.dumps(trace, ensure_ascii=False) + "\n"); ft.flush()
        n += 1
        c = trace.get("counts", {})
        print(f"[{n}/{len(todo)}] {row['doc_id'][:60]} acc={c.get('accepted', 0)}/{c.get('fields', '?')} {trace.get('seconds', 0)}s" + (f" ERROR {trace['error']}" if trace.get("error") else ""), file=sys.stderr)

    pg.run_many(todo, on_done)
    for f in (fo, ff, ft):
        if f:
            f.close()
    dt = time.time() - t0
    u = USAGE.snapshot()
    cost = {"docs": n, "seconds_total": round(dt, 1), "seconds_per_doc": round(dt / max(n, 1), 2), "models": cfg.signature, "usage": u,
            "usd_total": round(sum(v["usd"] for v in u.values()), 4), "usd_per_doc": round(sum(v["usd"] for v in u.values()) / max(n, 1), 5),
            "tokens_per_doc": round(sum(v["prompt_tokens"] + v["completion_tokens"] for v in u.values()) / max(n, 1))}
    Path(a.out + ".cost.json").write_text(json.dumps(cost, indent=1))
    print(f"[proofguard] done {n} docs in {dt:.0f}s; ~{cost['tokens_per_doc']} tokens/doc; usage {json.dumps(u)}", file=sys.stderr)


def cmd_calibrate(a):
    from proofguard.risk import fit_calibrator, BREAK_EVEN
    from proofguard.schema import flatten
    from proofguard.scoring import values_match, scorer_kind
    cfg = get_config()
    gold = {r["doc_id"]: r for r in read_jsonl(a.gold)}
    manual = tuple(x for x in a.manual_kinds.split(",") if x)
    rows, mode = [], "offline"
    traces = read_jsonl(a.traces)
    if not traces:
        raise SystemExit("No traces to calibrate.")
    if len({t["doc_id"] for t in traces}) != len(traces):
        raise SystemExit("Duplicate documents in calibration traces; use a fresh run.")
    if any(t.get("mode") == "llm" and t.get("signature") != cfg.signature for t in traces):
        raise SystemExit("Traces do not match current code/config. Run fresh development predictions.")
    if all(t.get("error") for t in traces):
        raise SystemExit("Every calibration document failed; fix the provider before calibrating.")
    for t in traces:
        g = gold.get(t["doc_id"])
        if not g:
            raise SystemExit("Trace document missing from gold file: " + t["doc_id"])
        mode = t.get("mode", mode)
        for s in flatten(g["schema"]):
            gv = g["gold"].get(s.name)
            gp = gv is not None and str(gv).strip() != ""
            tf = t["fields"].get(s.name, {})
            val = tf.get("value")
            cand = val is not None and "f1" in tf and s.kind not in manual and not tf.get("hard")
            y = 0
            if cand and gp:
                y = int(values_match(tf["value"], gv, scorer_kind(s.kind)))
            f1 = tf.get("f1", {})
            f2 = dict(f1)
            for k in ("c_ran", "c_contra", "c_amb", "c_sev", "c_alt_diff", "r_ran", "r_agree", "r_sim", "r_null"):
                f2.setdefault(k, 0.0)
            f2.update(tf.get("f2", {}))
            rows.append({"doc": t["doc_id"], "key": f"{g['source_dataset']}:{s.name}", "kind": s.kind, "has_cand": cand, "y": y,
                         "gold_present": gp, "f1": f1, "f2": f2})
    print(f"[calibrate] rows={len(rows)} candidates={sum(r['has_cand'] for r in rows)} correct={sum(r['y'] for r in rows)}", file=sys.stderr)
    cal, diag = fit_calibrator(rows, manual_kinds=manual, error_cost=a.error_cost, quantile=a.quantile)
    cal.meta.update({"signature": cfg.signature if mode == "llm" else "rules", "mode": mode, "fitted": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "docs": len({r["doc"] for r in rows}), "gold_file": os.path.basename(a.gold)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cal.save(Path(a.out))
    # honest out-of-fold report: threshold sweep + reliability table
    oof = [(p2, y) for _, _, p2, y, key, kind in diag["oof"] if key not in diag["never"] and kind not in manual]
    n_total = len(rows)
    sweep = []
    for t in [0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999] + [cal.threshold]:
        acc = [(p, y) for p, y in oof if p >= t]
        err = sum(1 - y for _, y in acc)
        sweep.append({"threshold": t, "accepted": len(acc), "precision": (1 - err / len(acc)) if acc else None,
                      "net_saving_pct": (len(acc) - a.error_cost * err) / n_total})
    bins = []
    for lo, hi in [(0, .5), (.5, .8), (.8, .9), (.9, .95), (.95, .98), (.98, .99), (.99, 1.0001)]:
        b = [(p, y) for p, y in oof if lo <= p < hi]
        bins.append({"bin": f"{lo:.2f}-{min(hi, 1):.2f}", "n": len(b), "mean_p": (sum(p for p, _ in b) / len(b)) if b else None, "precision": (sum(y for _, y in b) / len(b)) if b else None})
    rep = {"threshold": cal.threshold, "oof": cal.meta["oof"], "sweep": sweep, "reliability": bins, "never_automate": diag["never"], "break_even": BREAK_EVEN}
    Path(a.out).with_suffix(".report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("threshold", "oof", "never_automate")}, indent=1))
    print("reliability (out-of-fold):")
    for b in bins:
        print(f"  {b['bin']}: n={b['n']:5d} mean_p={b['mean_p'] and round(b['mean_p'], 3)} precision={b['precision'] and round(b['precision'], 3)}")
    print(f"[calibrate] wrote {a.out}  (declare threshold {cal.threshold})")


def cmd_evaluate(a):
    from proofguard.report import evaluate
    res = evaluate(a)
    print(json.dumps({k: v for k, v in res.items() if k in ("declared", "baselines", "cost_sensitivity")}, indent=1))


def cmd_fill_missing(a):
    """Safety net for run_hidden.sh: any document without an output line gets all-null fields (= human review), so scoring never breaks."""
    from proofguard.schema import flatten
    have = set()
    if os.path.exists(a.out):
        have = {json.loads(l)["doc_id"] for l in open(a.out, encoding="utf-8") if l.strip()}
    n = 0
    with open(a.out, "a", encoding="utf-8") as f:
        for r in read_jsonl(a.data):
            if r["doc_id"] not in have:
                f.write(json.dumps({"doc_id": r["doc_id"], "fields": {s.name: {"value": None, "confidence": 0.0} for s in flatten(r["schema"])}}) + "\n")
                n += 1
    if n:
        print(f"[proofguard] filled {n} missing documents with null (human review)", file=sys.stderr)


def cmd_doctor(a):
    """Probe every configured model role: reachability, latency, logprobs, JSON, vision. Run this first with a new key."""
    from proofguard.llm import Client, LLMError, p_yes, parse_json
    cfg = get_config()
    import base64, struct, zlib
    def pngchunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    raw = (b"\0" + b"\xff\0\0" * 64) * 64
    png = b"\x89PNG\r\n\x1a\n" + pngchunk(b"IHDR", struct.pack(">IIBBBBB",64,64,8,2,0,0,0)) + pngchunk(b"IDAT",zlib.compress(raw)) + pngchunk(b"IEND",b"")
    tiny_png = "data:image/png;base64," + base64.b64encode(png).decode()

    seen = {}
    for role in (cfg.extractor, cfg.verifier, cfg.challenger, cfg.escalator):
        if not role.ready:
            print(f"{role.name:10s} NOT CONFIGURED (set {'LLM' if role.name == 'extractor' else ('ESCALATE' if role.name == 'escalator' else role.name.upper())}_BASE_URL / _API_KEY / _MODEL)")
            continue
        key = (role.base_url, role.model)
        if key in seen:
            print(f"{role.name:10s} same endpoint+model as {seen[key]}"); continue
        seen[key] = role.name
        res = []
        if role.api == "systemone":
            from proofguard.llm import DecisionClient
            from proofguard.verify import QUESTIONS
            try:
                t = time.time()
                pr = DecisionClient(role, Path("/tmp/pg_doctor_cache"), 60).ask(
                    {"field": "Governing Law", "field_meaning": "Which law governs the agreement.", "proposed_value": "This Agreement is governed by the laws of the State of Illinois.",
                     "document_excerpt": "12. Governing Law. This Agreement is governed by the laws of the State of Illinois."}, QUESTIONS)
                res.append(f"decision API ok {time.time() - t:.1f}s  P(supported)={pr.get('supported')} P(complete)={pr.get('complete')} P(right_kind)={pr.get('right_kind')}  (expect ~1.0 for a true claim)")
            except LLMError as e:
                res.append("FAILED: " + str(e)[:1600] + "  (is serve.sh warm? GET /health must return 200)")
            print(f"{role.name:10s} H2O decision API @ {role.base_url}\n           " + "\n           ".join(res))
            continue
        cl = Client(role, Path("/tmp/pg_doctor_cache"), 60)
        try:
            t = time.time()
            r = cl.chat([{"role": "user", "content": "Reply with the single character 1."}], max_tokens=256, logprobs=True, temperature=0.0)
            pyv, how = p_yes(r.top_logprobs, r.text)
            res.append(f"chat ok {time.time() - t:.1f}s")
            res.append(f"logprobs={'yes' if r.top_logprobs else 'NO (verifier will use the text answer)'}  p_yes={pyv and round(pyv, 3)} via {how}")
            r = cl.chat([{"role": "user", "content": 'Reply with JSON only: {"a": 1, "b": null}'}], max_tokens=256)
            res.append("json " + ("ok" if parse_json(r.text).get("a") == 1 else f"UNPARSEABLE -> {r.text[:60]!r} (try PG_JSON_MODE=1)"))
            try:
                r = cl.chat([{"role": "user", "content": [{"type": "text", "text": "What colour is this image? One word."}, {"type": "image_url", "image_url": {"url": tiny_png}}]}], max_tokens=256)
                res.append(f"vision ok ({r.text[:20]!r})")
            except LLMError as e:
                res.append("vision probe FAILED (does not establish lack of vision support): " + str(e)[:1600])
        except LLMError as e:
            res.append("FAILED: " + str(e)[:1600])
        print(f"{role.name:10s} {role.model} @ {role.base_url}\n           " + "\n           ".join(res))
    print(f"context budget {cfg.max_context_chars} chars | workers {cfg.workers} | cache {cfg.cache_dir}")


def cmd_ablate(a):
    """Layer-by-layer contribution, honest: calibrators fitted on dev traces with growing feature sets, evaluated on the public traces."""
    from proofguard.risk import fit_calibrator, S1, S2, Priors
    from proofguard.schema import flatten
    from proofguard.scoring import values_match, scorer_kind

    def rows_of(traces, gold_path, manual=("json",)):
        gold = {r["doc_id"]: r for r in read_jsonl(gold_path)}
        rows = []
        for t in read_jsonl(traces):
            g = gold.get(t["doc_id"])
            if not g or t.get("error"):
                continue
            for s in flatten(g["schema"]):
                gv = g["gold"].get(s.name); gp = gv is not None and str(gv).strip() != ""
                tf = t["fields"].get(s.name, {}); val = tf.get("value")
                cand = val is not None and "f1" in tf and s.kind not in manual and not tf.get("hard")
                y = int(cand and gp and values_match(val, gv, scorer_kind(s.kind)))
                f1 = tf.get("f1", {}); f2 = dict(f1)
                for k in ("c_ran", "c_contra", "c_amb", "c_sev", "c_alt_diff", "r_ran", "r_agree", "r_sim", "r_null"):
                    f2.setdefault(k, 0.0)
                f2.update(tf.get("f2", {}))
                rows.append({"doc": t["doc_id"], "key": f"{g['source_dataset']}:{s.name}", "kind": s.kind, "has_cand": cand, "y": y, "gold_present": gp, "f1": f1, "f2": f2})
        return rows

    dev, pub = rows_of(a.dev_traces, a.dev_gold), rows_of(a.pub_traces, a.pub_gold)
    verif = ["v_avail", "v_logit", "v_p", "v_complete_avail", "v_complete_logit", "v_kind_avail", "v_kind_logit"]
    pri = ["prior_prec_logit", "prior_null", "prior_n_log"]
    proof = [n for n in S1 if n not in verif and n not in pri and n != "ext_conf"]
    sets = [("A  extractor's own confidence + field track record", ["ext_conf"] + pri),
            ("B  + Proof Engine (evidence, schema, semantics, cross-field)", ["ext_conf"] + pri + proof),
            ("C  + H2O-Lightning verifier (one token)", ["ext_conf"] + pri + proof + verif),
            ("D  + adversarial challenger + second reader (full ProofGuard)", S2)]
    out = []
    n_pub = len(pub)
    for label, names in sets:
        n2 = names
        cal, diag = fit_calibrator(dev, names1=[n for n in names if n in S1] or ["ext_conf"], names2=n2, quantile=a.quantile)
        t = cal.threshold
        acc = err = 0
        for r in pub:
            if not r["has_cand"] or cal.never(r["key"], r["kind"]):
                continue
            p = cal.p(2, r["f2"], r["key"])
            if p >= t:
                acc += 1; err += 1 - r["y"]
        gp = sum(r["gold_present"] for r in pub)
        cor = acc - err
        out.append({"layers": label, "threshold": t, "accepted": acc, "wrong_or_hallucinated": err, "precision": (cor / acc) if acc else None,
                    "coverage": cor / gp if gp else 0, "net_saving_pct": (acc - 100 * err) / n_pub, "dev_oof_saving_pct": cal.meta["oof"].get("point_saving_pct")})
        print(f"{label:62s} thr={t:<8} accepted={acc:4d} errors={err:3d} precision={('n/a' if not acc else format(cor / acc, '.3f'))} net saving={100 * (acc - 100 * err) / n_pub:6.2f}%")
    Path(a.out).write_text(json.dumps(out, indent=1))


def cmd_fit_retriever(a):
    from proofguard import retrieval
    W = retrieval.fit(read_jsonl(a.data), get_config().cache_dir)
    print(f"retrieval model: {len(W)} clauses, {sum(len(v) for v in W.values())} weights")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("predict")
    p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--full-out", help="also write every candidate value with its probability (for analysis; not for submission)")
    p.add_argument("--trace", help="per-field trace (features, route, proof notes)")
    p.add_argument("--mode", choices=["auto", "llm", "offline"], default="auto")
    p.add_argument("--calibrator"); p.add_argument("--limit", type=int); p.add_argument("--sample", type=int); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int); p.add_argument("--resume", action="store_true")
    p.add_argument("--calibrating", action="store_true", help="compute every signal for every field (used to fit the calibrator)")
    p.set_defaults(fn=cmd_predict)
    c = sp.add_parser("calibrate")
    c.add_argument("--traces", required=True); c.add_argument("--gold", required=True); c.add_argument("--out", required=True)
    c.add_argument("--manual-kinds", default="json"); c.add_argument("--error-cost", type=float, default=100.0)
    c.add_argument("--quantile", type=float, default=0.1, help="threshold maximises this quantile of bootstrap net saving (lower = more cautious)")
    c.set_defaults(fn=cmd_calibrate)
    e = sp.add_parser("evaluate")
    e.add_argument("--gold", required=True); e.add_argument("--pred", required=True); e.add_argument("--full-pred")
    e.add_argument("--threshold", type=float); e.add_argument("--out", default="results.json"); e.add_argument("--label", default="")
    e.set_defaults(fn=cmd_evaluate)
    m = sp.add_parser("fill-missing"); m.add_argument("--data", required=True); m.add_argument("--out", required=True); m.set_defaults(fn=cmd_fill_missing)
    sp.add_parser("doctor").set_defaults(fn=cmd_doctor)
    b = sp.add_parser("ablate")
    b.add_argument("--dev-traces", required=True); b.add_argument("--dev-gold", default="data/unified/dev.jsonl")
    b.add_argument("--pub-traces", required=True); b.add_argument("--pub-gold", default="data/unified/test_public.jsonl")
    b.add_argument("--out", default="runs/ablation.json"); b.add_argument("--quantile", type=float, default=0.1)
    b.set_defaults(fn=cmd_ablate)
    r = sp.add_parser("fit-retriever"); r.add_argument("--data", default="data/unified/dev.jsonl"); r.set_defaults(fn=cmd_fit_retriever)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
