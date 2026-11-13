import sys
import unittest
import json
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from backend.app import app, normalize

class AppTests(unittest.TestCase):
    def setUp(self): self.client=TestClient(app)
    def test_frontend(self):
        self.assertEqual(self.client.get('/').status_code,200)
        self.assertEqual(self.client.get('/assets/app.js').status_code,200)
    def test_schema_rejection(self):
        r=self.client.post('/api/extract',files={'file':('a.txt',b'hello')},data={'schema':'{}'})
        self.assertEqual(r.status_code,422)
    def test_bad_extension(self):
        r=self.client.post('/api/extract',files={'file':('a.exe',b'hello')},data={'pipeline':'baseline','schema':'{"properties":{"id":{}}}'})
        self.assertEqual(r.status_code,415)
    def test_evidence_and_nan(self):
        fields=normalize({'a':{'value':'1','confidence':float('nan')},'b':{'value':'2','confidence':1,'evidence':'invented'}},{'properties':{'a':{},'b':{}}},.98,'real source')
        self.assertTrue(all(f['status']=='review' for f in fields))
        self.assertEqual(fields[0]['confidence'],0)
    @patch('backend.app.extract_document')
    def test_success(self, mock):
        mock.return_value=({'id':{'value':'42','confidence':.99,'evidence':'ID: 42'}},'test-model',{},'ID: 42')
        r=self.client.post('/api/extract',files={'file':('a.txt',b'ID: 42')},data={'pipeline':'baseline','schema':json.dumps({'properties':{'id':{'type':'string'}}})})
        self.assertEqual(r.status_code,200)
        self.assertEqual(r.json()['fields'][0]['status'],'candidate')
    @patch('backend.app.extract_document')
    def test_failure_is_error(self,mock):
        from backend.engine import EngineError
        mock.side_effect=EngineError('Provider unavailable')
        r=self.client.post('/api/extract',files={'file':('a.txt',b'x')},data={'pipeline':'baseline','schema':'{"properties":{"id":{}}}'})
        self.assertEqual(r.status_code,502)
        self.assertNotIn('fields',r.json())
if __name__=='__main__': unittest.main()

class SharedEngineTests(unittest.TestCase):
    @patch('backend.proofguard_adapter.ProofGuard')
    @patch('backend.proofguard_adapter.get_config')
    def test_gui_uses_shared_pipeline_and_requires_review(self, config, pipeline):
        from types import SimpleNamespace
        config.return_value=SimpleNamespace(llm_ready=True,max_pages=8,extractor=SimpleNamespace(model='mock'))
        pipeline.return_value.run_record.return_value=(
            {'fields':{'id':{'value':None}},'_full':{'id':{'value':'42','confidence':.99}}},
            {'fields':{'id':{'evidence':'ID: 42','route':'review:below_threshold'}},'seconds':1,'error':None})
        r=TestClient(app).post('/api/extract',files={'file':('test.txt',b'ID: 42')},data={'schema':'{"properties":{"id":{"type":"string"}}}'})
        self.assertEqual(r.status_code,200)
        self.assertEqual(r.json()['pipeline'],'proofguard')
        self.assertEqual(r.json()['fields'][0]['status'],'review')
        self.assertEqual(r.json()['fields'][0]['value'],'42')
        record=pipeline.return_value.run_record.call_args.args[0]
        self.assertNotIn('gold',record)
        doc=pipeline.return_value.run_record.call_args.kwargs['document']
        self.assertFalse(doc.path.exists())  # temporary upload removed after request
