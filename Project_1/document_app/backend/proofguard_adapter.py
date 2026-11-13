"""GUI adapter to exactly the pipeline used by pg.py; uploads remain human-reviewed."""
import sys
import tempfile
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path: sys.path.insert(0,str(PROJECT))
from proofguard.config import get_config
from proofguard.docs import Document, _pdf_pages, _ocr
from proofguard.pipeline import ProofGuard
from .engine import EngineError


def analyze(data, filename, schema):
    cfg=get_config()
    if not cfg.llm_ready: raise EngineError('Configure Project_1/.env before live analysis.')
    with tempfile.TemporaryDirectory(prefix='proofguard-upload-') as tmp:
        root=Path(tmp); path=root/Path(filename).name; path.write_bytes(data)
        # Keep uploaded documents and provider replies out of the persistent benchmark cache.
        cfg.cache_dir=root/'cache'
        doc=Document('upload:'+filename,'upload',path=path,cache_dir=cfg.cache_dir,max_pages=cfg.max_pages)
        if path.suffix.lower()=='.txt':
            doc.text=data.decode('utf-8-sig');doc.pages=[doc.text]
        elif path.suffix.lower()=='.pdf':
            import fitz
            with fitz.open(stream=data,filetype='pdf') as pdf:
                if pdf.needs_pass: raise EngineError('Password-protected PDF.')
                if len(pdf)>cfg.max_pages:
                    raise EngineError(f'Upload has {len(pdf)} pages; PG_MAX_PAGES={cfg.max_pages}. Split it or increase the limit before restarting.')
            doc.pages=_pdf_pages(path);doc.text='\n\f'.join(doc.pages)
            if not doc.has_text: doc.ocr_pages=_ocr(path,cfg.max_pages,cfg.cache_dir)
        # Upload schemas/domains were not part of benchmark calibration. Always require review.
        row,trace=ProofGuard(cfg,'llm',None).run_record({'doc_id':doc.doc_id,'source_dataset':'upload','schema':schema},document=doc)
        if trace.get('error'): raise EngineError('ProofGuard failed to process this document: '+trace['error'])
        fields=[]
        for name in schema['properties']:
            info=trace['fields'].get(name,{})
            raw=row.get('_full',{}).get(name,{})
            fields.append({'name':name,'value':raw.get('value'),'confidence':raw.get('confidence',0),'evidence':info.get('evidence',''),'page':None,'status':'review',
                'reason':'Human review required for uploaded documents. '+str(info.get('hard') or info.get('route','no answer')),
                'route':info.get('route'), 'signals':info.get('f1',{})})
        return {'filename':filename,'mode':'live','pipeline':'proofguard','model':cfg.extractor.model,'seconds':trace['seconds'],'usage':{},'threshold':None,'fields':fields}
