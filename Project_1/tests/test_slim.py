import os, unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
from proofguard.config import get_config
from proofguard.docs import Document
from proofguard.extract import Cand
from proofguard.pipeline import ProofGuard
from proofguard.verify import VResult

class SlimTests(unittest.TestCase):
    def case(self, value, error=False, hard=False):
        pg=ProofGuard(get_config(),'offline');pg.mode='llm';pg.extractor=Mock();pg.verifier=Mock()
        pg.extractor.extract.side_effect=[{'id':Cand('id','OLD',.8)}, {'id':Cand('id',value,.8,error=error)}]
        pg.verifier.available=True;seen=[]
        def verify(jobs):
            seen.extend(j[2].value for j in jobs)
            return [VResult(.8,'text') for j in jobs]
        pg.verifier.verify_many.side_effect=verify
        def proof(doc,specs,cands,schema):
            return {'id':NS(hard='invalid' if hard and cands['id'].value=='NEW' else None,window='source',feat={'ext_conf':.8,'g_score':1,'schema_ok':1})}
        rec={'doc_id':'upload:x','schema':{'properties':{'id':{'type':'string'}}}}
        with patch.dict(os.environ,{'PG_PIPELINE':'slim'}),patch('proofguard.slim.check_document',side_effect=proof):
            row,t=pg.run_record(rec,document=Document('upload:x','upload',text='source'))
        self.assertIsNone(t['error']);self.assertIsNone(row['fields']['id']['value']);pg.verifier.challenge_many.assert_not_called()
        return row,t,seen
    def test_replacement_reverified(self):
        r,t,s=self.case('NEW');self.assertEqual(r['_full']['id']['value'],'NEW');self.assertEqual(s,['OLD','NEW'])
    def test_invalid_replacement_blocked(self):
        r,t,s=self.case('NEW',hard=True);self.assertEqual(t['fields']['id']['hard'],'invalid');self.assertEqual(s,['OLD'])
    def test_null_withdraws(self):
        r,t,s=self.case(None);self.assertIsNone(r['_full']['id']['value'])
    def test_error_retains_original(self):
        r,t,s=self.case(None,error=True);self.assertEqual(r['_full']['id']['value'],'OLD')
    def test_signature(self):
        with patch.dict(os.environ,{'PG_PIPELINE':'full'}): a=get_config().signature
        with patch.dict(os.environ,{'PG_PIPELINE':'slim'}): b=get_config().signature
        self.assertNotEqual(a,b)
