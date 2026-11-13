"""Unit + end-to-end tests. Run:  python3 -m unittest discover -s tests -v
The end-to-end tests use tests/fake_llm.py (a simulated model, plumbing only) and are skipped if data/ has not been downloaded."""
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))

from proofguard.docs import Document, make_chunks  # noqa: E402
from proofguard.extract import Cand, normalize_value  # noqa: E402
from proofguard.llm import p_yes, parse_json  # noqa: E402
from proofguard.proof import DocIndex, check_document, parse_date, validate  # noqa: E402
from proofguard.risk import BREAK_EVEN, Priors, Stage, choose_threshold, fit_calibrator, fit_lr, logit  # noqa: E402
from proofguard.schema import FieldSpec, flatten  # noqa: E402

HAVE_DATA = (HERE / "data/unified/test_public.jsonl").exists()


class TestParsing(unittest.TestCase):
    def test_parse_json_variants(self):
        self.assertEqual(parse_json('```json\n{"a": {"value": 1}}\n```')["a"]["value"], 1)
        self.assertEqual(parse_json('Sure! {"a": 1,} done')["a"], 1)
        self.assertEqual(parse_json('{"a": 1, "b": {"x": 2}, "c": "unfinished'), {"a": 1, "b": {"x": 2}})
        self.assertEqual(parse_json("no json here"), {})

    def test_missing_field_from_partial_model_object_is_an_error(self):
        from proofguard.config import Config
        from proofguard.extract import Extractor
        from proofguard.schema import FieldSpec

        class FakeClient:
            def chat(self, *args, **kwargs):
                return type("Reply", (), {"text": '{"first": {"value": "present"}}'})()

        cfg = Config(None, None, None, None, 1000, 1, 10, 4, Path("."), 1, 1, 0.45, 0.5)
        extractor = Extractor.__new__(Extractor)
        specs = [FieldSpec("first", "text"), FieldSpec("second", "text")]
        result = extractor._ask(specs, "system", [{"text": "prompt"}], FakeClient(), 100)
        self.assertFalse(result["first"].error)
        self.assertTrue(result["second"].error)

    def test_p_yes(self):
        p, how = p_yes([("1", -0.05), ("0", -3.0)], "1")
        self.assertEqual(how, "logprob")
        self.assertGreater(p, 0.9)
        p, how = p_yes([], "No")
        self.assertEqual(how, "text")
        self.assertLess(p, 0.2)
        self.assertEqual(p_yes([], "???")[0], None)

    def test_normalize(self):
        b, n, i = FieldSpec("b", "bool"), FieldSpec("n", "number"), FieldSpec("i", "integer")
        self.assertEqual(normalize_value(b, True), "true")
        self.assertEqual(normalize_value(b, "No"), "false")
        self.assertEqual(normalize_value(n, "$1,234.50"), "1234.5")
        self.assertEqual(normalize_value(n, 33.3333), "33.3333")
        self.assertEqual(normalize_value(i, 12.0), "12")
        self.assertEqual(normalize_value(n, "(500)"), "-500")
        self.assertIsNone(normalize_value(FieldSpec("t", "text"), "N/A"))
        self.assertEqual(normalize_value(FieldSpec("j", "json"), {"b": 1, "a": 2}), '{"a":2,"b":1}')


class TestProof(unittest.TestCase):
    TEXT = ("THIS DISTRIBUTOR AGREEMENT is made this 7th day of September, 1999 between Acme Corp. and Beta LLC.\n\n"
            "12. Governing Law. This Agreement is to be construed according to the laws\n of the State of Illinois.\n\nTotal due: $1,234.50 EIN 26-3839121")

    def test_find_exact_fuzzy_absent(self):
        ix = DocIndex(self.TEXT)
        s, a, b, _ = ix.find("construed according to the laws of the State of Illinois.")
        self.assertEqual(s, 1.0)
        self.assertIn("Illinois", self.TEXT[a:b])
        s, *_ = ix.find("This Agreement is to be construed under the laws of the State of Illinois.")
        self.assertTrue(0.5 < s < 1.0)
        self.assertEqual(ix.find("completely different sentence about penguins swimming north")[0], 0.0)

    def test_dates(self):
        self.assertEqual(str(parse_date("7th day of September, 1999")), "1999-09-07")
        self.assertEqual(str(parse_date("March 27, 2020.")), "2020-03-27")
        self.assertIsNone(parse_date("no date"))

    def _check(self, specs, vals):
        doc = Document("x", "extractbench", text=self.TEXT * 2, pages=[self.TEXT, self.TEXT])
        cands = {s.name: Cand(s.name, vals.get(s.name), 0.9) for s in specs}
        return check_document(doc, specs, cands, {"properties": {}})

    def test_grounding_and_hard_stop(self):
        specs = [FieldSpec("ein", "text"), FieldSpec("invented", "text"), FieldSpec("total", "number"), FieldSpec("cut", "text")]
        pr = self._check(specs, {"ein": "26-3839121", "invented": "Zanzibar Holdings International", "total": "1234.5", "cut": "Beta LL"})
        self.assertEqual(pr["ein"].feat["g_exact"], 1.0)
        self.assertIsNone(pr["ein"].hard)
        self.assertIsNotNone(pr["invented"].hard)                  # not in the document: never worth a model call
        self.assertEqual(pr["total"].feat["g_score"], 1.0)         # number found as 1,234.50
        self.assertEqual(pr["cut"].feat["bound_ok"], 0.0)          # 'Beta LL' is cut mid-word: truncation caught even though it is a substring

    def test_schema_validation(self):
        enum = FieldSpec("kind", "text", enum=["A", "B"])
        self.assertFalse(validate(enum, "C", {})[0])
        self.assertTrue(validate(enum, "a", {})[0])
        self.assertFalse(validate(FieldSpec("i", "integer"), "1.5", {})[0])
        self.assertFalse(validate(FieldSpec("b", "bool"), "maybe", {})[0])
        self.assertFalse(validate(FieldSpec("j", "json"), "[1", {})[0])

    def test_chunks_cover_text(self):
        t = "HEADING\n\n" + ("Sentence one is here. " * 120) + "\n\nLast paragraph."
        ch = make_chunks(t)
        self.assertTrue(all(t[c.start:c.end] == c.text for c in ch))
        self.assertGreaterEqual(len(ch), 2)

    def test_flatten(self):
        sch = {"properties": {"a": {"anyOf": [{"type": "string"}, {"type": "null"}]}, "b": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
                              "c": {"type": "string", "x-normalizer": "span"}, "d": {"anyOf": [{"type": "array"}, {"type": "null"}]}}}
        self.assertEqual([s.kind for s in flatten(sch)], ["text", "bool", "span", "json"])


class TestRisk(unittest.TestCase):
    def test_lr_learns_sign(self):
        X = [[x / 10.0] for x in range(-10, 11)] * 5
        y = [1 if r[0] > 0 else 0 for r in X]
        w, b = fit_lr(X, y, 0.1)
        self.assertGreater(w[0], 1.0)

    def test_threshold_refuses_when_errors_dominate(self):
        scored = [("d%d" % (i % 50), 0.995, 1 if i % 20 else 0) for i in range(400)]     # 95% precision at 'very confident': 5 errors per 100 accepted
        self.assertGreater(choose_threshold(scored, 400)["threshold"], 1.0)
        good = [("d%d" % (i % 50), 0.99, 1) for i in range(400)] + [("d1", 0.3, 0)] * 100
        info = choose_threshold(good, 500)
        self.assertLessEqual(info["threshold"], 0.99)
        self.assertGreater(info["p20_saving_pct"], 0.5)

    def test_break_even(self):
        self.assertAlmostEqual(BREAK_EVEN, 0.99)

    def test_priors_smoothing(self):
        rows = [{"key": "k", "has_cand": True, "y": 1, "gold_present": True}] * 3
        f = Priors.build(rows).feats("k")
        self.assertLess(f["prior_prec_logit"], 6)                  # 3/3 correct must not mean certainty
        self.assertEqual(Priors().feats("unseen")["prior_n_log"], 0.0)

    def test_calibrator_quarantines_noisy_field(self):
        import random
        rnd = random.Random(3)
        rows = []
        for d in range(160):
            for j in range(10):
                y = int(rnd.random() < 0.7)
                v = min(max(rnd.gauss(0.93 if y else 0.3, 0.07), 0.01), 0.99)
                if j % 5 == 0:
                    v, y = rnd.random(), int(rnd.random() < 0.8)
                f = {"ext_conf": 0.9, "g_score": 1.0, "v_avail": 1.0, "v_p": v, "v_logit": logit(v), "schema_ok": 1.0}
                rows.append({"doc": f"d{d}", "key": f"k{j % 5}", "kind": "text", "has_cand": True, "y": y, "gold_present": True, "f1": f, "f2": dict(f)})
        cal, _ = fit_calibrator(rows)
        self.assertIn("k0", cal.meta["never_automate"])
        self.assertGreater(cal.meta["oof"]["oof_precision"], 0.98)


@unittest.skipUnless(HAVE_DATA, "data/ not downloaded")
class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fake_llm import Oracle, make_server
        cls.recs = [json.loads(l) for l in (HERE / "data/unified/test_public.jsonl").read_text().splitlines()]
        cls.recs = [r for r in cls.recs if r["source_dataset"] == "cuad"][:2]
        path = Path(tempfile.mkdtemp()) / "gold.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in cls.recs) + "\n")
        cls.tmp = path.parent
        cls.oracle = Oracle(str(path), 2, cls.tmp / "c")
        cls.srv = make_server(cls.oracle, 8791)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        os.environ.update(LLM_BASE_URL="http://127.0.0.1:8791/v1", LLM_API_KEY="x", LLM_MODEL="fake", PG_CACHE_DIR=str(cls.tmp / "cache"),
                          VERIFIER_BASE_URL="http://127.0.0.1:8791", VERIFIER_API="systemone")

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def run_pg(self, **kw):
        from proofguard.config import get_config
        from proofguard.pipeline import ProofGuard
        cfg = get_config()
        cfg.workers = 2
        pg = ProofGuard(cfg, "llm", None, **kw)
        return [pg.run_record(r) for r in self.recs]

    def test_contract_every_doc_every_field_and_strict_output(self):
        for (row, trace), rec in zip(self.run_pg(), self.recs):
            self.assertEqual(row["doc_id"], rec["doc_id"])
            self.assertEqual(set(row["fields"]), set(rec["gold"]))
            for n, f in row["fields"].items():
                self.assertEqual(set(f) - {"value", "confidence"}, set())
                if f["value"] is not None:                           # only routed-to-ACCEPT values are ever emitted
                    self.assertEqual(trace["fields"][n]["route"], "accept")
                    self.assertTrue(0.0 <= f["confidence"] <= 1.0)
            self.assertIsNone(trace["error"])

    def test_second_run_is_served_from_cache(self):
        from proofguard.llm import USAGE
        self.run_pg()
        before = USAGE.snapshot().get("extractor", {}).get("cached", 0)
        self.run_pg()
        self.assertGreater(USAGE.snapshot()["extractor"]["cached"], before)

    def test_h2o_decision_api_features(self):
        (row, trace), = self.run_pg()[:1]
        verified = [f for f in trace["fields"].values() if f.get("verifier")]
        self.assertTrue(verified)
        self.assertTrue(all(f["verifier"] == "systemone" for f in verified))
        self.assertTrue(all(f["f1"]["v_complete_avail"] == 1.0 for f in verified))

    def test_microsoft_decision_systemone_gateway(self):
        from proofguard.config import Role
        from proofguard.llm import DecisionClient
        from proofguard.verify import QUESTIONS

        role = Role("verifier", "http://127.0.0.1:8791", "test-key", "microsoft-decision-1", api="systemone")
        client = DecisionClient(role, self.tmp / "decision-cache", 10)
        probabilities = client.ask(
            {"field": "Governing Law", "proposed_value": "Illinois", "document_excerpt": "Governing law: Illinois."},
            QUESTIONS,
        )

        self.assertIn("supported", probabilities)
        request = next(r for r in reversed(self.oracle.decision_requests)
                       if r["body"].get("model") == "microsoft-decision-1")
        self.assertEqual(request["path"], "/v1/systemone")
        self.assertIsNone(request["api_key"])
        self.assertEqual(request["authorization"], "Bearer test-key")

    def test_microsoft_foundry_decision_api_contract(self):
        names = ("VERIFIER_API", "VERIFIER_BASE_URL", "VERIFIER_API_KEY", "VERIFIER_MODEL")
        previous = {name: os.environ.get(name) for name in names}
        os.environ.update(VERIFIER_API="foundry", VERIFIER_BASE_URL="http://127.0.0.1:8791",
                          VERIFIER_API_KEY="foundry-test-key", VERIFIER_MODEL="decision-test-deployment")
        try:
            (_, trace), = self.run_pg()[:1]
            verified = [f for f in trace["fields"].values() if f.get("verifier")]
            self.assertTrue(verified)
            self.assertTrue(all(f["verifier"] == "foundry" for f in verified))
            request = next(r for r in reversed(self.oracle.decision_requests)
                           if r["path"] == "/providers/microsoft/v1/systemone")
            self.assertEqual(request["body"]["model"], "decision-test-deployment")
            self.assertEqual(request["api_key"], "foundry-test-key")
            self.assertIsNone(request["authorization"])
            self.assertIn("supported", request["body"]["questions"])
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_calibrating_mode_records_stage2_signals(self):
        (row, trace), = self.run_pg(full=True)[:1]
        stage2 = [f for f in trace["fields"].values() if "f2" in f]
        self.assertTrue(stage2 or not any(f.get("p1") for f in trace["fields"].values()))


if __name__ == "__main__":
    unittest.main()
