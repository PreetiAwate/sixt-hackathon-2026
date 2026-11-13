"""Layer 4: adversarial verification.
    * Verifier (Microsoft Decision-1, legacy H2O SystemOne, or chat model): "Does this excerpt prove the value is right for this field?"
        Typed decision APIs return a probability directly; chat models use first-token logprobs or a text fallback.
  * Challenger (any LLM, run only for gray-zone fields): actively tries to refute the value - contradictions, ambiguity, truncated clause, wrong field.
Both see the *evidence neighbourhood*, not the whole document, so a 4B model is enough and a call costs a few hundred tokens."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .config import Config
from .cuad_defs import CLAUSES
from .docs import Document
from .extract import Cand
from .llm import Client, DecisionClient, LLMError, p_yes, parse_json
from .retrieval import top_chunks
from .schema import FieldSpec
from .scoring import span_f1

VERIFY_SYSTEM = "You are a strict verification judge for document extraction. Reply with a single character: 1 or 0."
VERIFY_PROMPT = """FIELD: {name}
FIELD MEANING: {meaning}
PROPOSED VALUE: {value}

DOCUMENT EXCERPT:
<<<
{window}
>>>

Is the proposed value exactly the correct value of this field according to the excerpt? It must really be this kind of information (a value that only touches the topic, is incomplete, over-long, or belongs to another field is wrong).
Answer 1 for yes, 0 for no."""

CHALLENGE_SYSTEM = "You are an adversarial auditor. Your job is to find reasons a proposed extraction is WRONG. Be skeptical but fair."
CHALLENGE_PROMPT = """FIELD: {name}
FIELD MEANING: {meaning}
PROPOSED VALUE: {value}

DOCUMENT CONTEXT:
<<<
{window}
>>>

Try to refute the proposed value. Look for: another passage that contradicts or supersedes it (amendment, exception, different party/date); several competing candidate values (ambiguity); a clause that continues beyond the proposed text; the value really belonging to a different field; something not actually stated.
Reply with ONE JSON object and nothing else:
{{"verdict": "holds" | "contradicted" | "ambiguous" | "not_stated", "alternative": <better verbatim value or null>, "severity": <0..1 probability the proposed value is wrong>, "reason": "<one sentence>"}}"""


@dataclass
class VResult:
    p: float | None = None          # P(excerpt proves the value is exactly right for this field)
    how: str = "none"
    p_complete: float | None = None  # P(value is complete: not cut off, not padded)         (decision API only)
    p_kind: float | None = None      # P(value is the kind of information this field asks for) (decision API only)


QUESTIONS = {
    "supported": {"type": "noul", "instructions": "Does the document excerpt prove that the proposed value is exactly the correct value of this field?",
                  "criteria": {"true": "yes, the excerpt shows this value for this field", "false": "no: absent, different, or belongs to another field"}},
    "complete": {"type": "noul", "instructions": "Is the proposed value complete: not cut off, and not padded with unrelated text?",
                 "criteria": {"true": "yes, complete", "false": "no, truncated or over-long"}},
    "right_kind": {"type": "noul", "instructions": "Is the proposed value the kind of information this field asks for, rather than a different field or a passage that merely mentions the topic?",
                   "criteria": {"true": "yes, the right kind of information", "false": "no, wrong kind"}},
}


@dataclass
class CResult:
    ran: bool = False
    contradicted: float = 0.0
    ambiguous: float = 0.0
    severity: float = 0.0
    alt_differs: float = 0.0
    reason: str = ""


def meaning(spec: FieldSpec) -> str:
    if spec.name in CLAUSES:
        return CLAUSES[spec.name]["definition"]
    return " ".join(spec.description.split())[:600] or spec.name


class Verifier:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.v = None
        self.decision = None
        if cfg.verifier.ready:
            if cfg.verifier.api in ("systemone", "foundry"):
                self.decision = DecisionClient(cfg.verifier, cfg.cache_dir, cfg.timeout)
                self.v = self.decision
            else:
                self.v = Client(cfg.verifier, cfg.cache_dir, cfg.timeout)
        self.c = Client(cfg.challenger, cfg.cache_dir, cfg.timeout) if cfg.challenger.ready else None
        self.vision = os.environ.get("PG_VISION", "auto")

    @property
    def available(self) -> bool:
        return self.v is not None

    def _images(self, doc: Document, spec: FieldSpec, cand: Cand) -> list:
        if not (doc.is_scan or (self.vision == "1" and spec.kind == "bool")):
            return []
        imgs = doc.images()
        if cand.page:
            sel = [x for x in imgs if x[0] == cand.page]
            if sel:
                return sel[:1]
        return imgs[:1]

    def verify(self, doc: Document, spec: FieldSpec, cand: Cand, window: str) -> VResult:
        if self.v is None or cand.value is None:
            return VResult()
        if self.decision is not None:
            return self._verify_decision(doc, spec, cand, window)
        text = VERIFY_PROMPT.format(name=spec.name, meaning=meaning(spec), value=cand.value[:1800], window=window[:4200] or "(no readable text)")
        imgs = self._images(doc, spec, cand)
        content: object = text
        if imgs:
            content = [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": {"url": u}} for _, u in imgs]
        msgs = [{"role": "system", "content": VERIFY_SYSTEM}, {"role": "user", "content": content}]
        try:
            r = self.v.chat(msgs, temperature=0.0, max_tokens=3, logprobs=True)
        except LLMError:
            return VResult()
        p, how = p_yes(r.top_logprobs, r.text)
        return VResult(p, how)

    def _verify_decision(self, doc: Document, spec: FieldSpec, cand: Cand, window: str) -> VResult:
        state = {"field": spec.name, "field_meaning": meaning(spec), "proposed_value": cand.value[:1800],
                 "document_excerpt": window[:4200] or "(no readable text)"}
        imgs = [u for _, u in self._images(doc, spec, cand)][:4] if self.cfg.verifier.api == "systemone" else []
        try:
            pr = self.decision.ask(state, QUESTIONS, imgs or None)
        except LLMError:
            return VResult()
        if "supported" not in pr:
            return VResult()
        return VResult(pr["supported"], self.cfg.verifier.api, pr.get("complete"), pr.get("right_kind"))

    def challenge(self, doc: Document, spec: FieldSpec, cand: Cand, window: str, start: int) -> CResult:
        if self.c is None or cand.value is None:
            return CResult()
        ctx = window
        if doc.source == "cuad" and spec.name in CLAUSES:
            extra = [c.text for c in top_chunks(doc, spec.name, 3) if c.text[:80] not in window]
            ctx = window + "\n--- other passages that may relate to this field ---\n" + "\n...\n".join(extra[:2])
        elif doc.has_text and start >= 0:
            t = doc.text
            ctx = t[max(0, start - 2500): start + 2500]
        text = CHALLENGE_PROMPT.format(name=spec.name, meaning=meaning(spec), value=cand.value[:1800], window=ctx[:7000] or "(no readable text)")
        imgs = self._images(doc, spec, cand)
        content: object = text
        if imgs:
            content = [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": {"url": u}} for _, u in imgs]
        msgs = [{"role": "system", "content": CHALLENGE_SYSTEM}, {"role": "user", "content": content}]
        try:
            ans = parse_json(self.c.chat(msgs, temperature=0.0, max_tokens=300, json_mode=True).text)
        except LLMError:
            return CResult()
        if not isinstance(ans, dict) or "verdict" not in ans:
            return CResult()
        verdict = str(ans.get("verdict", "")).lower()
        try:
            sev = min(max(float(ans.get("severity", 0.5)), 0.0), 1.0)
        except (TypeError, ValueError):
            sev = 0.5
        alt = ans.get("alternative")
        differs = 0.0
        if alt and str(alt).strip().lower() not in ("null", "none", ""):
            differs = float(span_f1(str(alt), cand.value) < 0.8)
        return CResult(True, float(verdict in ("contradicted", "not_stated")), float(verdict == "ambiguous"), sev, differs, str(ans.get("reason", ""))[:200])

    def verify_many(self, jobs: list) -> list:
        """jobs: [(doc, spec, cand, window)] -> [VResult]"""
        if not jobs:
            return []
        with ThreadPoolExecutor(max(1, self.cfg.workers * 2)) as ex:
            return list(ex.map(lambda j: self.verify(*j), jobs))

    def challenge_many(self, jobs: list) -> list:
        if not jobs:
            return []
        with ThreadPoolExecutor(max(1, self.cfg.workers)) as ex:
            return list(ex.map(lambda j: self.challenge(*j), jobs))
