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
