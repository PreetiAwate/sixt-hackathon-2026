"""Bounded recovery with fresh proof and verification of final candidates.
Recovery selection uses a fixed heuristic, independent of fitted acceptance calibration.
"""
import os
import time
from .docs import load_document
from .proof import check_document
from .risk import heuristic_p, logit


def run_slim(pg, rec, specs, trace, t0, document=None):
    cfg = pg.cfg
    doc = document or load_document(rec, cfg.cache_dir, cfg.max_pages)
    defs = rec['schema'].get('$defs', {})
    by = {s.name: s for s in specs}
    source = rec.get('source_dataset', 'unknown')
    candidates = pg.extractor.extract(doc, specs, defs)
    original = {n: c.value for n, c in candidates.items()}

    def assess():
        proofs = check_document(doc, specs, candidates, rec['schema'])
        jobs = [s.name for s in specs if candidates[s.name].value is not None
                and s.kind != 'json' and not proofs[s.name].hard]
        jobs = jobs[:cfg.verify_max_per_doc]
        verified = {}
        if pg.verifier and pg.verifier.available:
            verified = dict(zip(jobs, pg.verifier.verify_many([
                (doc, by[n], candidates[n], proofs[n].window) for n in jobs])))
        features = {}
        for s in specs:
            f = dict(proofs[s.name].feat)
            v = verified.get(s.name)
            available = v is not None and v.p is not None
            f.update(v_avail=float(available), v_p=v.p if available else 0.5,
                     v_logit=logit(v.p) if available else 0.0)
            for attr, key in [('p_complete', 'v_complete'), ('p_kind', 'v_kind')]:
                p = getattr(v, attr, None)
                f[key + '_avail'] = float(p is not None)
                f[key + '_logit'] = logit(p) if p is not None else 0.0
            features[s.name] = f
        return proofs, verified, features

    proofs, verified, features = assess()
    # Fixed gate avoids changing recovery distribution after acceptance calibration.
    limit = max(0, int(os.environ.get('PG_RECOVERY_MAX_FIELDS', '6')))
    recovery = [s.name for s in specs if s.kind != 'json'
                and candidates[s.name].value is not None
                and not candidates[s.name].error
                and not proofs[s.name].hard
                and 0.2 <= heuristic_p(features[s.name]) < 0.9][:limit]
    changed = []
    if recovery and pg.mode == 'llm':
        second = pg.extractor.extract(doc, [by[n] for n in recovery], defs, second=True)
        for n in recovery:
            c = second.get(n)
            if c is not None and not c.error:
                candidates[n] = c  # A null reread explicitly withdraws the candidate.
                changed.append(n)
        if changed:
            # Cross-field checks and verifier must see the replacement, never old evidence.
            proofs, verified, features = assess()

    strict, full, details = {}, {}, {}
    accepted = 0
    for s in specs:
        n = s.name
        c, p, f = candidates[n], proofs[n], features[n]
        confidence = round(pg._p(1, f, source + ':' + n), 5) if c.value is not None else 0.0
        v = verified.get(n)
        allow = (c.value is not None and not c.error and not p.hard
                 and not pg._manual(source + ':' + n, s.kind)
                 and v is not None and v.p is not None
                 and confidence >= pg.threshold)
        accepted += int(allow)
        strict[n] = {'value': c.value if allow else None, 'confidence': confidence if allow else 0.0}
        full[n] = {'value': c.value, 'confidence': confidence}
        details[n] = {'value': c.value, 'original_value': original[n], 'p1': confidence,
                      'f1': f, 'hard': p.hard, 'error': c.error,
                      'evidence': c.evidence or '', 'recovered': n in changed,
                      'verifier': v.how if v else None,
                      'route': 'accept' if allow else 'review'}
    trace.update(fields=details, seconds=round(time.time()-t0, 2),
                 field_errors=sum(bool(c.error) for c in candidates.values()),
                 counts={'fields': len(specs), 'accepted': accepted,
                         'rechecked': len(recovery), 'verified': len(verified)})
    if candidates and all(c.error for c in candidates.values()):
        trace['error'] = 'All extraction fields failed.'
    return {'doc_id': rec['doc_id'], 'fields': strict, '_full': full}, trace
