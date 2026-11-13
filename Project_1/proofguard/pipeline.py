"""ProofGuard pipeline + Layer 6 Adaptive Router.

  extract (L1-2) -> prove (L3) -> verify with H2O (L4) -> risk score (L5) -> route (L6):
     ACCEPT   p >= hi                       no more spend
     REVIEW   p <  lo, hard proof failure,  never-automate field  -> a human (cost 1), no more spend
     RECHECK  lo <= p < hi                  independent second reader (other view / stronger model) + adversarial challenger, then re-score
  Money is spent where it can change the decision, so a clear case costs 1 extraction + 1 one-token verification."""
from __future__ import annotations

import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import Config
from .docs import load_document
from .extract import Cand, Extractor, OfflineExtractor
from .proof import check_document
from .risk import Calibrator, heuristic_p, logit
from .schema import flatten
from .scoring import same, span_f1
from .verify import CResult, Verifier

FALLBACK_THRESHOLD = 1.0001  # No uncalibrated automatic acceptance


class ProofGuard:
    def __init__(self, cfg: Config, mode: str = "auto", calibrator: Calibrator | None = None, full: bool = False):
        self.cfg = cfg
        self.mode = "llm" if (mode == "llm" or (mode == "auto" and cfg.llm_ready)) else "offline"
        if self.mode == "llm" and not cfg.llm_ready:
            raise SystemExit("mode=llm needs LLM_BASE_URL, LLM_API_KEY and LLM_MODEL (see .env.example)")
        self.cal = calibrator
        self.full = full                       # calibration runs: compute every signal for every field, whatever the router would do
        self.extractor = Extractor(cfg) if self.mode == "llm" else OfflineExtractor(cfg)
        self.verifier = Verifier(cfg) if self.mode == "llm" else None
        self.threshold = calibrator.threshold if calibrator else FALLBACK_THRESHOLD
        slack = cfg.gray_hi_slack
        self.hi = 1 - (1 - min(self.threshold, 0.9999)) * slack
        self.lo = cfg.gray_lo

    # ------------------------------------------------------------ scoring helpers
    def _p(self, stage: int, feats: dict, key: str) -> float:
        return self.cal.p(stage, feats, key) if self.cal else heuristic_p(feats)

    def _manual(self, key: str, kind: str) -> bool:
        if self.cal:
            return self.cal.never(key, kind)
        return kind == "json"

    # ------------------------------------------------------------ one document
    def run_record(self, rec: dict, document=None) -> tuple[dict, dict]:
        rec = {k: v for k, v in rec.items() if k not in ("gold", "gold_evidence")}
        t0 = time.time()
        specs = flatten(rec["schema"])
        trace = {"doc_id": rec["doc_id"], "mode": self.mode, "fields": {}, "error": None, "signature": self.cfg.signature}
        try:
            return self._run(rec, specs, trace, t0, document)
        except Exception as e:                                # a broken document must never crash the batch or book a value
            trace["error"] = f"{type(e).__name__}: {e}"
            trace["traceback"] = traceback.format_exc()[-1200:]
            trace["seconds"] = round(time.time() - t0, 2)
            empty = {s.name: {"value": None, "confidence": 0.0} for s in specs}
            return {"doc_id": rec["doc_id"], "fields": empty}, trace

    def _run(self, rec: dict, specs: list, trace: dict, t0: float, document=None) -> tuple[dict, dict]:
        cfg, src = self.cfg, (rec.get("source_dataset") or rec["doc_id"].split(":", 1)[0])
        doc = document or load_document(rec, cfg.cache_dir, cfg.max_pages)
        defs = rec["schema"].get("$defs", {})
        cands = self.extractor.extract(doc, specs, defs)                              # layers 1-2
        proofs = check_document(doc, specs, cands, rec["schema"])                     # layer 3
        by = {s.name: s for s in specs}
        keyof = lambda n: f"{src}:{n}"
        live = [n for n, c in cands.items() if c.value is not None]
        trace["doc"] = {"evidence": doc.evidence_kind, "chars": len(doc.evidence_text), "scan": doc.is_scan}

        # layer 4: verifier on every plausible candidate (hard failures and manual fields are not worth a call)
        manual = {n for n in live if self._manual(keyof(n), by[n].kind)}
        todo = [n for n in live if n not in manual and proofs[n].hard is None]
        todo.sort(key=lambda n: -cands[n].conf)
        todo = todo[: cfg.verify_max_per_doc]
        vres = {}
        if self.verifier and self.verifier.available:
            for n, r in zip(todo, self.verifier.verify_many([(doc, by[n], cands[n], proofs[n].window) for n in todo])):
                vres[n] = r
        f1 = {}
        for n in live:
            f = dict(proofs[n].feat)
            r = vres.get(n)
            f["v_avail"] = float(r is not None and r.p is not None)
            f["v_p"] = r.p if (r and r.p is not None) else 0.5
            f["v_logit"] = logit(r.p) if (r and r.p is not None) else 0.0
            for src_, nm in ((getattr(r, "p_complete", None), "v_complete"), (getattr(r, "p_kind", None), "v_kind")):
                f[nm + "_logit"] = logit(src_) if src_ is not None else 0.0
                f[nm + "_avail"] = float(src_ is not None)
            f1[n] = f
        p1 = {n: self._p(1, f1[n], keyof(n)) for n in live}

        # layer 6: route
        route, gray = {}, []
        for n in live:
            if n in manual:
                route[n] = "review:manual_field"
            elif proofs[n].hard:
                route[n] = "review:" + proofs[n].hard.split(" ")[0].rstrip(":")
            elif p1[n] >= self.hi:
                route[n] = "accept"
            elif p1[n] < self.lo:
                route[n] = "review:low"
            else:
                route[n] = "recheck"
            # calibration runs compute stage-2 signals for everything the router could plausibly send there (verifier did not clearly reject it)
            if route[n] == "recheck" or (self.full and n in todo and (n not in vres or vres[n].p is None or vres[n].p >= 0.2)):
                gray.append(n)

        f2, p2, cres = {n: dict(f1[n]) for n in live}, dict(p1), {}
        if gray and self.mode == "llm":
            second = self.extractor.extract(doc, [by[n] for n in gray], defs, second=True)
            ch = self.verifier.challenge_many([(doc, by[n], cands[n], proofs[n].window, proofs[n].start) for n in gray]) if self.verifier else [CResult()] * len(gray)
            for n, c in zip(gray, ch):
                cres[n] = c
                s2 = second.get(n)
                f = f2[n]
                f.update({"c_ran": float(c.ran), "c_contra": c.contradicted, "c_amb": c.ambiguous, "c_sev": c.severity, "c_alt_diff": c.alt_differs})
                if s2 is not None and not s2.error:
                    f["r_ran"] = 1.0
                    f["r_null"] = float(s2.value is None)
                    f["r_agree"] = float(s2.value is not None and same(by[n].kind, cands[n].value, s2.value))
                    f["r_sim"] = span_f1(cands[n].value, s2.value) if s2.value is not None and by[n].kind in ("span", "text") else f["r_agree"]
                else:
                    f["r_ran"] = f["r_agree"] = f["r_null"] = f["r_sim"] = 0.0
                p2[n] = self._p(2, f, keyof(n))
                if route[n] == "recheck":
                    route[n] = "accept" if (p2[n] >= self.threshold and not proofs[n].hard) else "review:after_recheck"
        elif gray:
            for n in gray:
                if route[n] == "recheck":
                    route[n] = "accept" if p1[n] >= self.threshold else "review:no_second_reader"
        # accepted means: final probability clears the declared threshold
        for n in live:
            pf = p2[n] if n in cres else p1[n]
            if route[n] == "accept" and pf < self.threshold:
                route[n] = "review:below_threshold"

        strict, full_fields, tf = {}, {}, {}
        for s in specs:
            n = s.name
            c = cands[n]
            if c.value is None:
                strict[n] = {"value": None, "confidence": 0.0}
                full_fields[n] = {"value": None, "confidence": 0.0}
                tf[n] = {"route": "no_answer" if not c.error else "error", "value": None, "error": c.error}
                continue
            pf = round(p2[n] if n in cres else p1[n], 5)
            acc = route[n] == "accept"
            strict[n] = {"value": c.value, "confidence": pf} if acc else {"value": None, "confidence": 0.0}
            full_fields[n] = {"value": c.value, "confidence": pf}
            tf[n] = {"route": route[n], "value": c.value, "p1": round(p1[n], 5), "p2": round(p2[n], 5) if n in cres else None,
                     "f1": {k: round(v, 4) for k, v in f1[n].items()}, "evidence": (c.evidence or "")[:200],
                     "hard": proofs[n].hard, "verifier": vres[n].how if n in vres else None,
                     "challenge": cres[n].reason if n in cres else None}
            if n in cres:
                tf[n]["f2"] = {k: round(v, 4) for k, v in f2[n].items() if k.startswith(("c_", "r_"))}
        trace["fields"] = tf
        trace["field_errors"] = sum(bool(c.error) for c in cands.values())
        if cands and trace["field_errors"] == len(cands):
            trace["error"] = "All extraction fields failed (provider error or invalid model response)."
        trace["seconds"] = round(time.time() - t0, 2)
        trace["counts"] = {"fields": len(specs), "candidates": len(live), "accepted": sum(r == "accept" for r in route.values()),
                           "rechecked": len(cres), "verified": len(vres)}
        return {"doc_id": rec["doc_id"], "fields": strict, "_full": full_fields}, trace

    # ------------------------------------------------------------ many documents
    def run_many(self, recs: list, on_done=None) -> list:
        def one(rec):
            return self.run_record(rec)
        out = []
        with ThreadPoolExecutor(max(1, self.cfg.workers)) as ex:
            for res in ex.map(one, recs):
                out.append(res)
                if on_done:
                    on_done(res)
        return out
