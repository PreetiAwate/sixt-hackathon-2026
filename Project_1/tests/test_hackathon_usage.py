import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from proofguard.config import get_config
from proofguard.pipeline import ProofGuard, FALLBACK_THRESHOLD
from proofguard.docs import Document
from proofguard.extract import Cand
from proofguard.submission_config import load_submission

class StarterParserTests(unittest.TestCase):
    def test_invalid_json_is_not_an_empty_answer(self):
        import baseline_llm_compat
        with self.assertRaises(ValueError):
            baseline_llm_compat.parse_json("not JSON")

    def test_missing_required_field_is_processing_failure(self):
        import baseline_llm_compat
        rec = {'doc_id':'test', 'source_dataset':'cuad',
               'schema':{'properties':{'expected':{'type':'string'}}},
               'pages':['raw/cuad/CUADv1.json']}
        with patch.object(baseline_llm_compat, 'build_content', return_value=([], 'text')):
            with patch.object(baseline_llm_compat, 'call_llm', return_value='{"other": null}'):
                with self.assertRaisesRegex(ValueError, 'omitted required field'):
                    baseline_llm_compat.extract(rec)

    def test_annotated_field_name_is_supported(self):
        import baseline_llm_compat
        rec = {'doc_id':'test', 'source_dataset':'cuad',
               'schema':{'properties':{'clause':{'type':'string','x-normalizer':'span'}}},
               'pages':['raw/cuad/CUADv1.json']}
        response = '{"clause [span]": {"value": "The clause text.", "confidence": 0.8}}'
        with patch.object(baseline_llm_compat, 'build_content', return_value=([], 'The clause text.')):
            with patch.object(baseline_llm_compat, 'call_llm', return_value=response):
                result = baseline_llm_compat.extract(rec)
        self.assertEqual(result['fields']['clause']['value'], 'The clause text.')

class UsageTests(unittest.TestCase):
    def test_uncalibrated_never_accepts(self):
        self.assertGreater(FALLBACK_THRESHOLD,1)
        with patch.dict('os.environ',{'LLM_BASE_URL':'','LLM_MODEL':''},clear=True):
            cfg=get_config()
            pg=ProofGuard(cfg,'offline')
            text='Invoice number: INV-12345. '+('Example text. '*30)
            rec={'doc_id':'upload:test','schema':{'properties':{'invoice_number':{'type':'string'}}},'gold':{'invoice_number':'SECRET'}}
            doc=Document('upload:test','upload',text=text,pages=[text])
            row,trace=pg.run_record(rec,document=doc)
            self.assertTrue(all(f['value'] is None for f in row['fields'].values()))
            self.assertNotIn('SECRET',json.dumps(trace))
    def test_long_candidate_not_truncated_in_trace(self):
        cfg=get_config();pg=ProofGuard(cfg,'offline')
        value='Long sentence about the agreement. '*20
        c=Cand(name="clause",value=value,conf=.99,evidence=value)
        with patch.object(pg.extractor,'extract',return_value={'clause':c}):
            row,trace=pg.run_record({'doc_id':'upload:test','schema':{'properties':{'clause':{'type':'string'}}}}, document=Document('upload:test','upload',text=value,pages=[value]))
        self.assertIsNone(trace['error'])
        self.assertEqual(trace['fields']['clause']['value'],value)
    def test_unconfigured_submission_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'submission_config.json').write_text('{"calibrator":"missing.json","threshold":null}')
            with patch('proofguard.submission_config.HERE',p):
                with self.assertRaises(OSError):load_submission()
