"""Layer 5, the Cost-Aware Risk Controller.
Turns every signal about a field (proof checks, H2O verifier probability, challenger verdict, second-reader agreement, per-field track record)
into ONE calibrated probability that the value is correct, then decides under the real cost structure:
    Accept has expected cost 100*(1-p); review costs 1. Break-even is p=0.99.
The margin is not a fudge factor: the threshold is the one that maximises the 20th-percentile of net saving over bootstrap resamples of
*documents* (out-of-fold), so a lucky small sample cannot talk us into automating a 98%-precise slice."""
from __future__ import annotations

import json
import math
import random
import zlib
from pathlib import Path

BREAK_EVEN = 0.99

S1 = ["ext_conf", "g_score", "g_exact", "g_applicable", "ev_layer", "ev_ocr", "occ_log", "ev_present", "ev_found", "ev_has_value", "page_ok", "bound_ok", "sent_aligned", "end_punct",
      "schema_ok", "pat_rule", "pat_ok", "label_prox", "cue_rx", "cue_lm", "pos_early", "len_log", "very_long", "dup_n", "overlap_n",
      "date_conflict", "is_cuad", "kind_bool", "kind_json", "kind_num", "prior_prec_logit", "prior_null", "prior_n_log", "v_avail", "v_logit", "v_p", "v_complete_avail", "v_complete_logit", "v_kind_avail", "v_kind_logit"]
S2 = S1 + ["c_ran", "c_contra", "c_amb", "c_sev", "c_alt_diff", "r_ran", "r_agree", "r_sim", "r_null"]
DEFAULTS = {"bound_ok": 0.5, "sent_aligned": 0.5, "ext_conf": 0.5, "g_score": 0.0, "label_prox": 0.5, "cue_rx": 0.5, "cue_lm": 0.5, "pos_early": 0.5, "page_ok": 0.5, "v_p": 0.5}


def logit(p: float, c: float = 6.0) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return max(-c, min(c, math.log(p / (1 - p))))


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))


def fold_of(doc_id: str, k: int = 5) -> int:
    return zlib.crc32(doc_id.encode()) % k


# ---------------------------------------------------------------- per-field track record (target encoding)
class Priors:
    """Smoothed per-field precision/absence rates learned from the calibration documents."""

    def __init__(self, stats: dict | None = None):
        self.s = stats or {}      # key -> [N, Ncand, Kcorrect, Nnull]

    @staticmethod
    def build(rows: list) -> "Priors":
        s: dict = {}
        for r in rows:
            a = s.setdefault(r["key"], [0, 0, 0, 0])
            a[0] += 1
            a[1] += int(r["has_cand"])
            a[2] += int(r["has_cand"] and r["y"] == 1)
            a[3] += int(not r["gold_present"])
        return Priors(s)

    def feats(self, key: str) -> dict:
        tot = [sum(v[i] for v in self.s.values()) for i in range(4)] or [0, 0, 0, 0]
        p0 = (tot[2] + 1) / (tot[1] + 2) if tot[1] else 0.5
        q0 = (tot[3] + 1) / (tot[0] + 2) if tot[0] else 0.5
        n, nc, k, nn = self.s.get(key, [0, 0, 0, 0])
        prec = (k + 8 * p0) / (nc + 8)
        return {"prior_prec_logit": logit(prec), "prior_null": (nn + 8 * q0) / (n + 8), "prior_n_log": math.log1p(nc)}


# ---------------------------------------------------------------- logistic regression (numpy if present, else stdlib)
def _solve(A: list, b: list) -> list:
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[piv] = M[piv], M[i]
        d = M[i][i] or 1e-9
        for j in range(i, n + 1):
            M[i][j] /= d
        for r in range(n):
            if r != i and M[r][i]:
                f = M[r][i]
                for j in range(i, n + 1):
                    M[r][j] -= f * M[i][j]
    return [M[i][n] for i in range(n)]


def fit_lr(X: list, y: list, l2: float = 2.0, iters: int = 30) -> tuple[list, float]:
    """L2-regularised logistic regression by Newton/IRLS. X rows must already be standardised. Returns (weights, bias)."""
    n, d = len(X), len(X[0])
    try:
        import numpy as np
        Xb = np.hstack([np.asarray(X, float), np.ones((n, 1))])
        yy = np.asarray(y, float)
        reg = np.eye(d + 1) * l2
        reg[-1, -1] = 0
        beta = np.zeros(d + 1)
        for _ in range(iters):
            p = 1 / (1 + np.exp(-np.clip(Xb @ beta, -30, 30)))
            g = Xb.T @ (p - yy) + reg @ beta
            H = (Xb * (p * (1 - p) + 1e-6)[:, None]).T @ Xb + reg
            step = np.linalg.solve(H, g)
            beta -= step
            if np.abs(step).max() < 1e-6:
                break
        return list(map(float, beta[:-1])), float(beta[-1])
    except ImportError:
        pass
    beta = [0.0] * (d + 1)
    for _ in range(min(iters, 12)):
        H = [[(l2 if i == j and i < d else 0.0) for j in range(d + 1)] for i in range(d + 1)]
        g = [(l2 * beta[i] if i < d else 0.0) for i in range(d + 1)]
        for xi, yi in zip(X, y):
            xe = xi + [1.0]
            p = sigmoid(sum(a * b for a, b in zip(xe, beta)))
            w = p * (1 - p) + 1e-6
            for i in range(d + 1):
                g[i] += (p - yi) * xe[i]
                wi = w * xe[i]
                Hi = H[i]
                for j in range(i, d + 1):
                    Hi[j] += wi * xe[j]
        for i in range(d + 1):
            for j in range(i):
                H[i][j] = H[j][i]
        step = _solve(H, g)
        beta = [a - b for a, b in zip(beta, step)]
        if max(abs(s) for s in step) < 1e-5:
            break
    return beta[:-1], beta[-1]


class Stage:
    def __init__(self, names: list, mean=None, std=None, w=None, b=0.0):
        self.names, self.mean, self.std, self.w, self.b = names, mean or [0.0] * len(names), std or [1.0] * len(names), w or [0.0] * len(names), b

    def vec(self, f: dict) -> list:
        return [f.get(n, DEFAULTS.get(n, 0.0)) for n in self.names]

    def predict(self, f: dict) -> float:
        z = self.b + sum(w * (x - m) / s for w, x, m, s in zip(self.w, self.vec(f), self.mean, self.std))
        return sigmoid(z)

    @staticmethod
    def fit(names: list, feats: list, y: list, l2: float = 2.0) -> "Stage":
        X = [[f.get(n, DEFAULTS.get(n, 0.0)) for n in names] for f in feats]
        d = len(names)
        mean = [sum(r[j] for r in X) / len(X) for j in range(d)]
        std = [max(math.sqrt(sum((r[j] - mean[j]) ** 2 for r in X) / len(X)), 1e-6) for j in range(d)]
        Z = [[(r[j] - mean[j]) / std[j] for j in range(d)] for r in X]
        w, b = fit_lr(Z, y, l2)
        return Stage(names, mean, std, w, b)

    def to_json(self) -> dict:
        return {"names": self.names, "mean": self.mean, "std": self.std, "w": self.w, "b": self.b}


# ---------------------------------------------------------------- thresholds under 100:1
def saving(rows_scores: list, t: float, review: float = 1.0, error: float = 100.0) -> float:
    """rows_scores: [(p, y)]; accepted rows are those with p >= t. Net saving in review units."""
    return sum(review - error * (1 - y) for p, y in rows_scores if p >= t)


def choose_threshold(scored: list, n_fields_total: int, error_cost: float = 100.0, boots: int = 300, q: float = 0.2, seed: int = 7) -> dict:
    """scored: [(doc, p, y)] for candidate fields (out-of-fold). Picks the threshold maximising the q-quantile of bootstrap net saving."""
    if not scored:
        return {"threshold": 1.0, "p20_saving_pct": 0.0, "point_saving_pct": 0.0}
    docs = sorted({d for d, _, _ in scored})
    by_doc = {d: [] for d in docs}
    for d, p, y in scored:
        by_doc[d].append((p, y))
    ps = sorted((p for _, p, _ in scored), reverse=True)
    grid = sorted({round(ps[min(len(ps) - 1, int(i * len(ps) / 120))], 5) for i in range(120)} | {round(BREAK_EVEN, 5)}, reverse=True)
    grid = [t for t in grid if t >= 0.5] or [1.0]
    per = {t: {d: saving(by_doc[d], t, 1.0, error_cost) for d in docs} for t in grid}
    rng = random.Random(seed)
    samples = [[rng.randrange(len(docs)) for _ in docs] for _ in range(boots)]
    best = None
    for t in grid:
        vec = [per[t][d] for d in docs]
        tot = sorted(sum(vec[i] for i in s) for s in samples)
        qv = tot[int(q * (boots - 1))]
        pt = sum(vec)
        cand = (qv, pt, t)
        if best is None or cand > best:
            best = cand
    qv, pt, t = best
    if qv <= 0:                      # automation does not pay off with 80% confidence: accept nothing
        t = 1.0001
    acc = [(p, y) for _, p, y in scored if p >= t]
    return {"threshold": round(t, 5), "p20_saving_pct": max(qv, 0.0) / n_fields_total, "point_saving_pct": saving(scored_flat(scored), t, 1.0, error_cost) / n_fields_total,
            "accepted_cand": len(acc), "oof_precision": (sum(y for _, y in acc) / len(acc)) if acc else None}


def scored_flat(scored: list) -> list:
    return [(p, y) for _, p, y in scored]


# ---------------------------------------------------------------- calibrator
class Calibrator:
    def __init__(self, s1: Stage, s2: Stage, priors: Priors, threshold: float, meta: dict | None = None):
        self.s1, self.s2, self.priors, self.threshold, self.meta = s1, s2, priors, threshold, meta or {}

    def p(self, stage: int, feats: dict, key: str) -> float:
        f = dict(feats)
        f.update(self.priors.feats(key))
        return (self.s1 if stage == 1 else self.s2).predict(f)

    def never(self, key: str, kind: str) -> bool:
        return key in set(self.meta.get("never_automate", [])) or kind in set(self.meta.get("manual_kinds", []))

    def save(self, path: Path):
        Path(path).write_text(json.dumps({"s1": self.s1.to_json(), "s2": self.s2.to_json(), "priors": self.priors.s, "threshold": self.threshold, "meta": self.meta}, indent=1), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "Calibrator":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return Calibrator(Stage(**d["s1"]), Stage(**d["s2"]), Priors(d["priors"]), d["threshold"], d.get("meta"))


def heuristic_p(f: dict) -> float:
    """Uncalibrated, conservative fallback used only when no calibrator exists for the active models. Never trust it above ~0.9."""
    p = f.get("ext_conf", 0.5) * (0.5 + 0.5 * f.get("g_score", 0.0)) * (1.0 if f.get("schema_ok", 1.0) else 0.3)
    if f.get("v_avail"):
        p *= f.get("v_p", 0.5)
    if f.get("c_contra"):
        p *= 0.3
    if f.get("r_ran") and not f.get("r_agree"):
        p *= 0.4
    return min(p, 0.9)


def fit_calibrator(rows: list, folds: int = 5, l2: float = 2.0, error_cost: float = 100.0, manual_kinds: tuple = ("json",), log=print,
                   names1: list | None = None, names2: list | None = None, quantile: float = 0.1) -> tuple[Calibrator, dict]:
    """rows: one per (doc, field): {doc, key, kind, has_cand, y, gold_present, f1, f2}. Fits both stages on all rows, and produces an
    honest out-of-fold estimate (nested priors, folds by document) to choose the threshold and to report expected performance."""
    n_total = len(rows)
    fold = [fold_of(r["doc"], folds) for r in rows]

    def with_priors(row, pri):
        a = dict(row["f1"]); a.update(pri.feats(row["key"]))
        b = dict(row["f2"]); b.update(pri.feats(row["key"]))
        return a, b

    def fit_on(train_idx: list, test_idx: list | None):
        # training rows get leave-one-fold-out priors, so the model never learns from a prior that contains its own label
        tf = sorted({fold[i] for i in train_idx})
        pri_in = {j: Priors.build([rows[i] for i in train_idx if fold[i] != j]) for j in tf}
        F1, F2, Y = [], [], []
        for i in train_idx:
            r = rows[i]
            if not r["has_cand"]:
                continue
            a, b = with_priors(r, pri_in[fold[i]])
            F1.append(a); F2.append(b); Y.append(r["y"])
        if len(Y) < 30 or len(set(Y)) < 2:
            raise ValueError("not enough labelled candidates to calibrate (need at least 30, with both right and wrong answers)")
        s1, s2 = Stage.fit(names1 or S1, F1, Y, l2), Stage.fit(names2 or S2, F2, Y, l2)
        return s1, s2, Priors.build([rows[i] for i in train_idx])

    oof = []
    for k in range(folds):
        tr = [i for i in range(n_total) if fold[i] != k]
        te = [i for i in range(n_total) if fold[i] == k and rows[i]["has_cand"]]
        if not te:
            continue
        try:
            s1, s2, pri = fit_on(tr, te)
        except ValueError:
            continue
        for i in te:
            r = rows[i]
            a, b = with_priors(r, pri)
            oof.append((r["doc"], s1.predict(a), s2.predict(b), r["y"], r["key"], r["kind"]))
    s1, s2, pri = fit_on(list(range(n_total)), None)
    # never-automate: fields that stay unreliable even inside the slice we would accept (two passes: pick a threshold, audit each field, re-pick)
    manual = set(manual_kinds)
    elig0 = [(d, p2, y, key) for d, _, p2, y, key, kind in oof if kind not in manual]
    audit_t = 0.9          # below any plausible final threshold: a field whose "confident" answers are <97% right there is mis-calibrated
    acc: dict = {}
    for d, p, y, key in elig0:
        if p >= audit_t:
            acc.setdefault(key, []).append(y)
    never = sorted(k for k, ys in acc.items() if len(ys) >= 10 and sum(ys) / len(ys) < 0.97)
    eligible = [(d, p, y) for d, p, y, key in elig0 if key not in never]
    info = choose_threshold(eligible, n_total, error_cost, q=quantile)
    meta = {"never_automate": never, "manual_kinds": list(manual), "n_rows": n_total, "n_candidates": sum(r["has_cand"] for r in rows),
            "oof": info, "break_even": BREAK_EVEN, "folds": folds, "error_cost": error_cost}
    cal = Calibrator(s1, s2, pri, info["threshold"], meta)
    diag = {"oof": oof, "info": info, "never": never}
    return cal, diag
