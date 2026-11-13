"""Thin wrapper over the organisers' 5_score.py so ProofGuard compares values exactly the way the hidden test does."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_p = Path(__file__).resolve().parent.parent / "5_score.py"
_spec = importlib.util.spec_from_file_location("score5", _p)
S5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S5)

values_match = S5.values_match
span_f1 = S5.span_f1


def scorer_kind(field_kind: str) -> str:
    return {"span": "span", "json": "json"}.get(field_kind, "string")


def same(spec_kind: str, a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return S5.values_match(a, b, scorer_kind(spec_kind))
