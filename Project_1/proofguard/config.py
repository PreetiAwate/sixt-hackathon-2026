"""Configuration: .env loading and model roles. Standard library only.

Roles (each is any OpenAI-compatible /chat/completions endpoint; every role falls back to LLM_*):
  extractor  LLM_BASE_URL / LLM_API_KEY / LLM_MODEL               layers 1-2
    verifier   VERIFIER_BASE_URL / VERIFIER_API_KEY / VERIFIER_MODEL   layer 4 (Microsoft Foundry Decision-1 or OpenAI-compatible chat)
  challenger CHALLENGER_*                                            layer 4 (adversarial pass, run selectively)
  escalator  ESCALATE_*                                              layer 6 (stronger second reader for the gray zone)
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def load_env(path: Path | None = None) -> None:
    """Minimal .env reader (KEY=VALUE, '#' comments). Real environment variables win."""
    p = path or HERE / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if v and not os.environ.get(k.strip()):
            os.environ[k.strip()] = v


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except ValueError:
        return default


@dataclass
class Role:
    name: str
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    price_in: float = 0.0     # USD per 1M input tokens (0 = free tier / self-hosted)
    price_out: float = 0.0
    extra_body: dict = field(default_factory=dict)  # provider-specific params, e.g. {"chat_template_kwargs": {"enable_thinking": false}}
    api: str = "chat"                               # "chat" = OpenAI /chat/completions | "systemone" = legacy H2O | "foundry" = Microsoft Decision API

    @property
    def ready(self) -> bool:
        # Legacy H2O SystemOne has a fixed model; Microsoft Foundry requires a deployment name.
        return bool(self.base_url and (self.model or self.api == "systemone"))

    @property
    def signature(self) -> str:
        default_model = "h2o-lightning-4b" if self.api == "systemone" else "microsoft-decision-1"
        suffix = "@systemone" if self.api == "systemone" else ("@foundry" if self.api == "foundry" else "")
        return f"{self.name}={self.model or default_model}{suffix}" if self.ready else f"{self.name}=none"


def _role(name: str, prefix: str, fallback: "Role | None") -> Role:
    g = lambda k: os.environ.get(f"{prefix}_{k}", "").strip()
    base, key, model = g("BASE_URL"), g("API_KEY"), g("MODEL")
    if not (base or key or model) and fallback is not None:
        base, key, model = fallback.base_url, fallback.api_key, fallback.model
        pin, pout, extra = fallback.price_in, fallback.price_out, fallback.extra_body
    else:
        pin, pout, extra = _f(f"{prefix}_PRICE_IN", 0.0), _f(f"{prefix}_PRICE_OUT", 0.0), {}
    eb = os.environ.get(f"{prefix}_EXTRA_BODY", "").strip()
    if eb:
        try:
            extra = json.loads(eb)
        except ValueError:
            pass
    api = os.environ.get(f"{prefix}_API", "").strip().lower() or ("systemone" if "systemone" in base or ":8741" in base else "chat")
    if fallback is not None and not (os.environ.get(f"{prefix}_BASE_URL") or os.environ.get(f"{prefix}_API_KEY")):
        api = fallback.api
    return Role(name, base, key, model, pin, pout, extra, api)


@dataclass
class Config:
    extractor: Role
    verifier: Role
    challenger: Role
    escalator: Role
    max_context_chars: int
    workers: int
    verify_max_per_doc: int
    max_pages: int
    cache_dir: Path
    timeout: float
    fields_per_call: int
    gray_lo: float
    gray_hi_slack: float

    @property
    def llm_ready(self) -> bool:
        return self.extractor.ready

    @property
    def signature(self) -> str:
        roles = [(r.name, r.base_url, r.model, r.api, r.extra_body) for r in
                 (self.extractor, self.verifier, self.challenger, self.escalator)]
        sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (HERE / "proofguard").glob("*") if p.suffix in (".py", ".json")}
        knobs = {k: os.environ.get(k, "") for k in
                 ("PG_PIPELINE", "PG_RECOVERY_MAX_FIELDS", "PG_VISION", "PG_JSON_MODE", "PG_MAX_CONTEXT_CHARS", "PG_MAX_PAGES",
                  "PG_FIELDS_PER_CALL", "PG_VERIFY_MAX_PER_DOC", "PG_GRAY_LO", "PG_GRAY_HI_SLACK")}
        digest = hashlib.sha256(json.dumps([roles, sources, knobs], sort_keys=True).encode()).hexdigest()[:20]
        return "|".join(r.signature for r in (self.extractor, self.verifier, self.challenger, self.escalator)) + "|build=" + digest


def get_config() -> Config:
    load_env()
    ext = _role("extractor", "LLM", None)
    ext.price_in, ext.price_out = _f("LLM_PRICE_IN", 0.0), _f("LLM_PRICE_OUT", 0.0)
    return Config(
        extractor=ext,
        verifier=_role("verifier", "VERIFIER", ext),
        challenger=_role("challenger", "CHALLENGER", ext),
        escalator=_role("escalator", "ESCALATE", ext),
        max_context_chars=_i("PG_MAX_CONTEXT_CHARS", 40000),
        workers=_i("PG_WORKERS", 4),
        verify_max_per_doc=_i("PG_VERIFY_MAX_PER_DOC", 60),
        max_pages=_i("PG_MAX_PAGES", 4),
        cache_dir=Path(os.environ.get("PG_CACHE_DIR", str(HERE / ".pg_cache"))),
        timeout=_f("PG_TIMEOUT", 180.0),
        fields_per_call=_i("PG_FIELDS_PER_CALL", 18),
        gray_lo=_f("PG_GRAY_LO", 0.45),
        gray_hi_slack=_f("PG_GRAY_HI_SLACK", 0.5),
    )
