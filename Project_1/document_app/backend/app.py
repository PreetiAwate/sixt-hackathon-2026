import json
import math
import time
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from .engine import extract_document, EngineError

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT.parent / '.env')
load_dotenv(ROOT / '.env')
app = FastAPI(title='Document Desk', version='1.0.0')
app.mount('/assets', StaticFiles(directory=ROOT/'frontend'), name='assets')

@app.get('/')
def index():
    return FileResponse(ROOT/'frontend/index.html')

@app.get('/api/health')
def health():
    import os
    return {'status':'ok','configured':all(os.getenv(k) for k in ('LLM_BASE_URL','LLM_API_KEY','LLM_MODEL')), 'model':os.getenv('LLM_MODEL','Not configured')}


def normalize(fields, schema, threshold, source_text=None):
    output = []
    for name in schema['properties']:
        field = fields.get(name, {})
        if not isinstance(field, dict):
            field = {'value':field}
        value = field.get('value')
        try:
            confidence = float(field.get('confidence',0))
            if not math.isfinite(confidence): confidence = 0
        except (ValueError, TypeError): confidence = 0
        confidence = max(0,min(confidence,1))
        evidence = str(field.get('evidence') or '')
        reason = str(field.get('reason') or '')
        supported = bool(evidence.strip())
        if source_text is not None and evidence:
            supported = ' '.join(evidence.lower().split()) in ' '.join(source_text.lower().split())
        status = 'candidate' if value is not None and confidence >= threshold and supported else 'review'
        if not supported: reason = 'Supporting evidence is missing or could not be located. ' + reason
        output.append({'name':name,'value':value,'confidence':confidence,'evidence':evidence,'page':field.get('page'),'reason':reason,'status':status})
    return output

@app.post('/api/extract')
def extract(file: UploadFile = File(...), schema: str = Form(...), threshold: float = Form(0.98), pipeline: str = Form("proofguard")):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise HTTPException(422,'Threshold must be between 0 and 1.')
    try:
        parsed = json.loads(schema)
        props = parsed['properties']
        if not isinstance(props,dict) or not 1 <= len(props) <= 80: raise ValueError()
        if not all(isinstance(v,dict) for v in props.values()): raise ValueError()
    except (ValueError, KeyError, TypeError):
        raise HTTPException(422,'Schema must have a properties object with 1–80 field definitions.')
    filename = Path(file.filename or 'document').name
    if Path(filename).suffix.lower() not in ('.pdf','.png','.jpg','.jpeg','.txt'):
        raise HTTPException(415,'Upload a PDF, PNG, JPG, or UTF-8 TXT file.')
    data = file.file.read(15*1024*1024+1)
    if not data or len(data)>15*1024*1024:
        raise HTTPException(413,'Upload a nonempty file no larger than 15 MB.')
    if pipeline not in ("proofguard", "baseline"):
        raise HTTPException(422, "Choose proofguard or baseline.")
    if pipeline == "proofguard":
        from .proofguard_adapter import analyze
        try:
            return analyze(data, filename, parsed)
        except EngineError as error:
            raise HTTPException(502, str(error))
        except Exception:
            raise HTTPException(422, "Document processing failed; inspect local server logs and document validity.")
    start = time.monotonic()
    try:
        fields, model, usage, text = extract_document(data,filename,parsed)
    except EngineError as error:
        raise HTTPException(502,str(error))
    except Exception:
        raise HTTPException(422,'Document could not be read. Check that it is a valid, supported file.')
    return {'filename':filename,'mode':'live','model':model,'seconds':round(time.monotonic()-start,2),'usage':usage,'threshold':threshold,'fields':normalize(fields,parsed,threshold,text)}
