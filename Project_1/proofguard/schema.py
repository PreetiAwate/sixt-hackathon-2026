"""Schema -> list of FieldSpec (what to extract, how it is compared, what shape it must have)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FieldSpec:
    name: str
    kind: str                 # span | text | bool | number | integer | json
    description: str = ""
    enum: list | None = None
    pattern: str | None = None
    nullable: bool = True
    raw: dict = field(default_factory=dict)

    @property
    def key(self) -> str:      # stable identity for learned per-field priors
        return self.name


def _types(spec: dict) -> list:
    t = spec.get("type")
    out = []
    if isinstance(t, list):
        out += t
    elif t:
        out.append(t)
    for a in spec.get("anyOf", []) + spec.get("oneOf", []):
        at = a.get("type")
        out += at if isinstance(at, list) else [at or ("object" if "$ref" in a else None)]
    return [x for x in out if x]


def flatten(schema: dict) -> list[FieldSpec]:
    specs = []
    for name, sp in schema.get("properties", {}).items():
        types = [t for t in _types(sp) if t != "null"]
        norm = sp.get("x-normalizer")
        if norm == "span":
            kind = "span"
        elif norm == "json" or any(t in ("object", "array") for t in types) or "$ref" in sp:
            kind = "json"
        elif "boolean" in types:
            kind = "bool"
        elif "integer" in types and "number" not in types:
            kind = "integer"
        elif "number" in types:
            kind = "number"
        else:
            kind = "text"
        desc = sp.get("description") or sp.get("title") or ""
        enum = sp.get("enum")
        if enum is None:
            for a in sp.get("anyOf", []):
                if a.get("enum"):
                    enum = [e for e in a["enum"] if e is not None]
        specs.append(FieldSpec(name, kind, str(desc), enum, sp.get("pattern"), "null" in _types(sp) or "type" not in sp, sp))
    return specs
