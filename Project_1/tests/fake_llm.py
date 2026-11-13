#!/usr/bin/env python3
"""A fake OpenAI-compatible /chat/completions server for PLUMBING TESTS ONLY.
It knows the gold answers for the documents it is given and behaves like a noisy, over-confident model. Numbers produced with it are
simulations and must never be reported as results. Usage: python3 tests/fake_llm.py --gold data/unified/dev.jsonl --port 8765 [--limit 40]"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from proofguard.docs import load_document  # noqa: E402
from proofguard.schema import flatten  # noqa: E402
from proofguard.scoring import same  # noqa: E402


class Oracle:
    def __init__(self, gold_path: str, limit: int | None, cache: Path, seed: int | None = None):
        recs = [json.loads(l) for l in Path(gold_path).read_text(encoding="utf-8").splitlines() if l.strip()]
        if seed is not None:                       # same selection as `pg.py predict --sample N --seed S`
            random.Random(seed).shuffle(recs)
        self.recs = recs[:limit]
        self.docs = {}
        for r in self.recs:
            d = load_document(r, cache, want_ocr=False)
            self.docs[r["doc_id"]] = (r, d, {s.name: s for s in flatten(r["schema"])})
        self.lock = threading.Lock()
        self.calls = {"extract": 0, "verify": 0, "challenge": 0, "other": 0}
        self.decision_requests = []

    def find_doc(self, snippet: str):
        sn = re.sub(r"\s+", " ", snippet).strip()[:100]
        for did, (r, d, sp) in self.docs.items():
            if sn and sn in re.sub(r"\s+", " ", d.text[:200000]):
                return did
        return None

    def rnd(self, *parts) -> random.Random:
        return random.Random(int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:12], 16))


def make_server(oracle: Oracle, port: int):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.endswith("/systemone"):
                return self.systemone(body)
            msgs = body["messages"]
            user = msgs[-1]["content"]
            system = msgs[0]["content"]
            utext = user if isinstance(user, str) else next(p["text"] for p in user if p["type"] == "text")
            text, top = "{}", None
            if "Reply with a single character" in system:
                text, top = self.verify(utext)
            elif "adversarial auditor" in system:
                text = self.challenge(utext)
            elif "FIELD:" not in utext and ("Fill in every field" in utext or "Extract the fields" in utext):
                text = self.extract(utext, "second reader" in system)
            else:
                oracle.calls["other"] += 1
            choice = {"message": {"role": "assistant", "content": text}}
            if top is not None:
                choice["logprobs"] = {"content": [{"token": top[0][0], "logprob": top[0][1], "top_logprobs": [{"token": t, "logprob": lp} for t, lp in top]}]}
            out = {"choices": [choice], "usage": {"prompt_tokens": len(utext) // 4, "completion_tokens": len(text) // 4}}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def systemone(self, body):
            """Simulated typed decision API for H2O and Microsoft Foundry plumbing tests."""
            oracle.decision_requests.append({"path": self.path, "body": body,
                                             "api_key": self.headers.get("api-key"),
                                             "authorization": self.headers.get("Authorization")})
            oracle.calls["verify"] += 1
            st = body["state"]
            name, val, win = st["field"], st["proposed_value"], st["document_excerpt"]
            did = oracle.find_doc(win.strip()[:100]) or oracle.find_doc(win.strip()[-100:])
            ok = bool(did) and self.truth(did, name, val)
            rr = oracle.rnd(did, name, "v", val[:30])
            clamp = lambda x: min(0.999, max(0.005, x))
            sup = clamp(rr.gauss(0.93, 0.07)) if ok else clamp(rr.gauss(0.22, 0.18))
            ans = {"supported": sup, "complete": clamp(sup + rr.gauss(0.03, 0.08)), "right_kind": clamp(0.5 * sup + 0.5 * (0.9 if ok else rr.gauss(0.5, 0.25)))}
            decision_answers = {}
            for key in body["questions"]:
                if key not in ans:
                    continue
                probability = round(ans[key], 4)
                if "decision" in str(body.get("model", "")).lower():
                    decision_answers[key] = {"type": "noul", "probability": probability}
                else:
                    decision_answers[key] = {"type": "noul", "noul": probability}
            out = {"model": body.get("model", "fake-h2o"), "answers": decision_answers,
                   "usage": {"input_tokens": len(json.dumps(st)) // 4, "output_tokens": 1, "latency_ms": 5}}
            data = json.dumps(out).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers()
            self.wfile.write(data)

        # ---- behaviours
        def extract(self, u: str, second: bool) -> str:
            oracle.calls["extract"] += 1
            m = re.search(r"<<<\n(.*)", u, re.S)
            body = m.group(1) if m else u
            body = re.sub(r"=== PAGE \d+ ===\n", "", body)
            did = oracle.find_doc(body.strip()[:150])
            names = re.findall(r"^- (.+?)(?: \(\w[\w/]*\))?:", u.split("Fields", 1)[-1], re.M)
            if not did:
                return json.dumps({n: {"value": None, "confidence": 0.1} for n in names})
            r, d, specs = oracle.docs[did]
            out = {}
            for n in names:
                if n not in specs:
                    continue
                g = r["gold"].get(n)
                v, conf = self.noisy(did, n, g, d, 0)
                if second:
                    rr = oracle.rnd(did, n, "b")
                    if rr.random() > 0.7:
                        v, conf = self.noisy(did, n, g, d, 1)
                out[n] = {"value": json.loads(v) if (v and specs[n].kind == "json") else v, "evidence": (v or "")[:120] or None, "page": 1 if v else None, "confidence": conf}
                if specs[n].kind == "bool" and v is not None:
                    out[n]["value"] = (v == "true")
            return json.dumps(out)

        def noisy(self, did, n, g, d, attempt):
            rr = oracle.rnd(did, n, "x", attempt)
            u = rr.random()
            if g is not None and str(g).strip():
                if u < 0.80:
                    return str(g), round(min(0.99, max(0.5, rr.gauss(0.93, 0.06))), 2)
                if u < 0.90:
                    s = str(g)
                    return (s[: max(3, len(s) // 3)] if len(s) > 10 else s + "x"), round(min(0.99, max(0.4, rr.gauss(0.88, 0.1))), 2)
                return None, 0.2
            if u < 0.10:
                sents = re.split(r"(?<=[.;])\s+", d.text[2000:30000])
                sents = [s for s in sents if 40 < len(s) < 300]
                v = " ".join(rr.choice(sents).split()) if sents else None
                return v, round(min(0.97, max(0.4, rr.gauss(0.8, 0.12))), 2)
            return None, 0.1

        def truth(self, did, n, value):
            r, d, specs = oracle.docs[did]
            g = r["gold"].get(n)
            return g is not None and same(specs[n].kind, value, str(g))

        def verify(self, u: str):
            oracle.calls["verify"] += 1
            name = re.search(r"^FIELD: (.+)$", u, re.M).group(1)
            val = re.search(r"PROPOSED VALUE: (.*?)\n\nDOCUMENT EXCERPT", u, re.S).group(1)
            win = re.search(r"<<<\n(.*?)\n>>>", u, re.S).group(1)
            did = oracle.find_doc(win.strip()[:100]) or oracle.find_doc(win.strip()[-100:])
            if not did:
                p = 0.5
            else:
                ok = self.truth(did, name, val)
                rr = oracle.rnd(did, name, "v", val[:30])
                p = min(0.999, max(0.01, rr.gauss(0.93, 0.07))) if ok else min(0.99, max(0.005, rr.gauss(0.22, 0.18)))
            return "1" if p > 0.5 else "0", [("1", math.log(p)), ("0", math.log(1 - p))] if p > 0.5 else [("0", math.log(1 - p)), ("1", math.log(p))]

        def challenge(self, u: str) -> str:
            oracle.calls["challenge"] += 1
            name = re.search(r"^FIELD: (.+)$", u, re.M).group(1)
            val = re.search(r"PROPOSED VALUE: (.*?)\n\nDOCUMENT CONTEXT", u, re.S).group(1)
            win = re.search(r"<<<\n(.*?)\n>>>", u, re.S).group(1)
            did = oracle.find_doc(win.strip()[:100])
            ok = bool(did) and self.truth(did, name, val)
            rr = oracle.rnd(did, name, "c", val[:30])
            if ok:
                v = "holds" if rr.random() < 0.85 else "ambiguous"
            else:
                v = rr.choice(["contradicted", "contradicted", "ambiguous", "holds"])
            sev = round(rr.uniform(0.0, 0.25) if v == "holds" else rr.uniform(0.5, 0.95), 2)
            return json.dumps({"verdict": v, "alternative": None, "severity": sev, "reason": "simulated"})

    return ThreadingHTTPServer(("127.0.0.1", port), H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(HERE / "data/unified/dev.jsonl"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    o = Oracle(a.gold, a.limit, HERE / ".pg_cache_fake", a.seed)
    srv = make_server(o, a.port)
    print(f"fake LLM on http://127.0.0.1:{a.port}/v1 with {len(o.docs)} docs (SIMULATION, plumbing tests only)", file=sys.stderr)
    srv.serve_forever()


if __name__ == "__main__":
    main()
