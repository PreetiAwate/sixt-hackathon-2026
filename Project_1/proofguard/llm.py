"""OpenAI-compatible chat client: disk cache, retries/backoff (free tiers rate-limit), logprobs, token accounting."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .config import Role


class LLMError(RuntimeError):
    pass


@dataclass
class Reply:
    text: str = ""
    top_logprobs: list = field(default_factory=list)   # first generated token: [(token, logprob), ...]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    cached: bool = False


class Usage:
    """Thread-safe per-role totals (tokens, calls, seconds, USD)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.t: dict[str, dict] = {}

    def add(self, role: Role, r: Reply):
        with self.lock:
            d = self.t.setdefault(role.name, {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0, "usd": 0.0})
            d["calls"] += 1
            d["cached"] += int(r.cached)
            if not r.cached:
                d["prompt_tokens"] += r.prompt_tokens
                d["completion_tokens"] += r.completion_tokens
                d["seconds"] += r.seconds
                d["usd"] += (r.prompt_tokens * role.price_in + r.completion_tokens * role.price_out) / 1e6

    def snapshot(self) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.t))


USAGE = Usage()
_THINK = re.compile(r"<think>.*?</think>", re.S)


class Client:
    def __init__(self, role: Role, cache_dir: Path, timeout: float = 180.0):
        self.role, self.timeout = role, timeout
        self.cache_dir = cache_dir
        self.no_logprobs = False
        self.min_gap = float(os.environ.get("PG_MIN_GAP_SECONDS", "0"))   # set e.g. 2 for strict free-tier RPM limits
        self._last = 0.0
        self._lock = threading.Lock()

    def _key(self, payload: dict) -> Path:
        h = hashlib.sha256((self.role.base_url + "\n" + json.dumps(payload, sort_keys=True)).encode()).hexdigest()
        return self.cache_dir / h[:2] / f"{h}.json"

    def chat(self, messages: list, *, temperature: float = 0.0, max_tokens: int | None = None,
             logprobs: bool = False, seed: int | None = None, json_mode: bool = False) -> Reply:
        payload = {"model": self.role.model, "messages": messages}
        # This gateway/model rejects temperature and requires >=16 output tokens.
        if max_tokens:
            payload["max_tokens"] = max(16, max_tokens)
        if seed is not None:
            payload["seed"] = seed
        if json_mode and os.environ.get("PG_JSON_MODE", "0") == "1":
            payload["response_format"] = {"type": "json_object"}
        want_lp = logprobs and not self.no_logprobs
        if want_lp:
            payload["logprobs"] = True
            payload["top_logprobs"] = 10
        payload.update(self.role.extra_body)
        path = self._key(payload)
        if path.exists():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                r = Reply(d["text"], [tuple(x) for x in d["top"]], d["pt"], d["ct"], d["s"], cached=True)
                USAGE.add(self.role, r)
                return r
            except (ValueError, KeyError):
                pass
        r = self._post(payload, want_lp)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"text": r.text, "top": r.top_logprobs, "pt": r.prompt_tokens, "ct": r.completion_tokens, "s": r.seconds}), encoding="utf-8")
        except OSError:
            pass
        USAGE.add(self.role, r)
        return r

    def _post(self, payload: dict, want_lp: bool) -> Reply:
        url = self.role.base_url.rstrip("/") + "/chat/completions"
        delay, last_err = 2.0, None
        for attempt in range(max(1, int(os.environ.get("PG_MAX_RETRIES", "3")))):
            if self.min_gap:
                with self._lock:
                    wait = self._last + self.min_gap - time.time()
                    if wait > 0:
                        time.sleep(wait)
                    self._last = time.time()
            req = urllib.request.Request(url, json.dumps(payload).encode(), {
                "Content-Type": "application/json", "Authorization": f"Bearer {self.role.api_key}"})
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    d = json.load(resp)
                return self._parse(d, time.time() - t0)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")
                if self.role.api_key:
                    body = body.replace(self.role.api_key, "[REDACTED]")
                body = body[:1600]
                last_err = f"HTTP {e.code}: {body}"
                if e.code == 400 and "logprob" in body.lower() and "logprobs" in payload:
                    self.no_logprobs = True                      # provider has no logprobs: fall back to text answers
                    payload = {k: v for k, v in payload.items() if k not in ("logprobs", "top_logprobs")}
                    continue
                if e.code == 400 and "response_format" in payload:
                    payload = {k: v for k, v in payload.items() if k != "response_format"}
                    continue
                if e.code in (429, 500, 502, 503, 504):
                    ra = e.headers.get("Retry-After") if e.headers else None
                    time.sleep(min(float(ra), 60) if ra and ra.replace(".", "").isdigit() else delay)
                    delay = min(delay * 2, 45)
                    continue
                raise LLMError(last_err)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(delay)
                delay = min(delay * 2, 45)
        raise LLMError(f"giving up after retries: {last_err}")

    @staticmethod
    def _parse(d: dict, secs: float) -> Reply:
        try:
            ch = d["choices"][0]
            text = ch["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"unexpected response: {str(d)[:300]}")
        text = _THINK.sub("", text).strip()
        top = []
        try:
            content = ch.get("logprobs", {}).get("content") or []
            if content:
                first = content[0]
                top = [(t["token"], float(t["logprob"])) for t in first.get("top_logprobs", [])]
                if not top:
                    top = [(first["token"], float(first["logprob"]))]
        except (KeyError, TypeError, ValueError, AttributeError):
            top = []
        u = d.get("usage") or {}
        return Reply(text, top, int(u.get("prompt_tokens", 0) or 0), int(u.get("completion_tokens", 0) or 0), secs)


def parse_json(reply: str):
    """Tolerant JSON extraction: code fences, prose around the object, trailing commas, truncated output."""
    s = (reply or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*", "", s).rstrip("`").strip()
    for cand in (s, s[s.find("{"): s.rfind("}") + 1] if "{" in s else ""):
        if not cand:
            continue
        for fix in (cand, re.sub(r",\s*([}\]])", r"\1", cand)):
            try:
                return json.loads(fix)
            except ValueError:
                pass
    # truncated object: close it at the last complete top-level key
    i = s.find("{")
    if i >= 0:
        body = s[i:]
        for cut in range(len(body), 0, -1):
            if body[cut - 1] == "}":
                try:
                    return json.loads(re.sub(r",\s*([}\]])", r"\1", body[:cut] + "}"))
                except ValueError:
                    continue
    return {}


def p_yes(top: list, text: str) -> tuple[float | None, str]:
    """Probability that the one-token verdict is YES/1, from first-token logprobs, else from the text. Returns (p, how)."""
    yes = ("1", "yes", "true", "y")
    no = ("0", "no", "false", "n")
    py = pn = 0.0
    for tok, lp in top:
        t = tok.strip().lower().lstrip("▁Ġ").strip()
        if t in yes:
            py += math.exp(lp)
        elif t in no:
            pn += math.exp(lp)
    if py + pn > 1e-9:
        return py / (py + pn), "logprob"
    t = (text or "").strip().lower()
    if t[:1] in ("1", "y", "t") or t.startswith("yes"):
        return 0.9, "text"
    if t[:1] in ("0", "n", "f") or t.startswith("no"):
        return 0.1, "text"
    m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%?", t)
    if m:
        v = float(m.group(1))
        return min(max(v / 100 if v > 1 else v, 0.0), 1.0), "verbalized"
    return None, "none"


class DecisionClient:
    """Client for Microsoft Foundry Decision-1 or legacy H2O SystemOne typed decisions."""

    def __init__(self, role: Role, cache_dir: Path, timeout: float = 120.0):
        self.role, self.timeout, self.cache_dir = role, timeout, cache_dir

    def _url(self) -> str:
        b = self.role.base_url.rstrip("/")
        if self.role.api == "foundry":
            suffix = "/providers/microsoft/v1/systemone"
            return b if b.endswith(suffix) else b + suffix
        return b if b.endswith("/systemone") else (b[:-3] if b.endswith("/v1") else b) + "/v1/systemone"

    def ask(self, state: dict, questions: dict, images: list | None = None) -> dict:
        payload = {"state": state, "questions": questions}
        if self.role.api == "foundry" or (
            self.role.api == "systemone" and self.role.model and "h2o" not in self.role.model.lower()
        ):
            payload["model"] = self.role.model
        if images:
            payload["images"] = images
        h = hashlib.sha256((self._url() + json.dumps(payload, sort_keys=True)).encode()).hexdigest()
        path = self.cache_dir / "s1" / h[:2] / f"{h}.json"
        if path.exists():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                USAGE.add(self.role, Reply("", [], d["pt"], d["ct"], d["s"], cached=True))
                return d["p"]
            except (ValueError, KeyError):
                pass
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.role.api_key:
            if self.role.api == "foundry":
                headers["api-key"] = self.role.api_key
            else:
                headers["Authorization"] = f"Bearer {self.role.api_key}"
        delay, last = 2.0, None
        for _ in range(7):
            t0 = time.time()
            try:
                req = urllib.request.Request(self._url(), json.dumps(payload).encode(), headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    d = json.load(r)
                probs = {}
                for key, answer in d["answers"].items():
                    if not isinstance(answer, dict):
                        continue
                    value = answer.get("probability", answer.get("noul"))
                    if value is None:
                        continue
                    probability = float(value)
                    if math.isfinite(probability) and 0.0 <= probability <= 1.0:
                        probs[key] = probability
                u = d.get("usage") or {}
                rep = Reply("", [], int(u.get("input_tokens", 0) or 0), int(u.get("output_tokens", 0) or 0), time.time() - t0)
                USAGE.add(self.role, rep)
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({"p": probs, "pt": rep.prompt_tokens, "ct": rep.completion_tokens, "s": rep.seconds}), encoding="utf-8")
                except OSError:
                    pass
                return probs
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}"
                if e.code in (429, 500, 502, 503, 504):        # 503 = still warming up
                    time.sleep(delay); delay = min(delay * 2, 30); continue
                raise LLMError(last)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, KeyError, ValueError) as e:
                last = f"{type(e).__name__}: {e}"
                time.sleep(delay); delay = min(delay * 2, 30)
        raise LLMError(f"decision API giving up: {last}")
