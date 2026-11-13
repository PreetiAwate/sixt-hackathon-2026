"""Direct no-cache gateway probe; never changes active model settings."""
import argparse,json,urllib.request,urllib.error
from proofguard.config import get_config

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',default='microsoft-decision-1');ap.add_argument('--temperature',type=float);a=ap.parse_args()
    role=get_config().extractor
    cases=[('supported','Contract ID: DEMO-123.','DEMO-123',True),('wrong','Contract ID: DEMO-123.','DEMO-999',False),('absent','Supplier: Example Rentals.','DEMO-123',False)]
    for name,doc,value,expected in cases:
        body={'model':a.model,'messages':[{'role':'user','content':f'Check contract_id={value} against document: {doc}\nReturn JSON only with supported (boolean) and reason. Missing evidence means false.'}]}
        if a.temperature is not None: body['temperature']=a.temperature
        req=urllib.request.Request(role.base_url.rstrip('/')+'/chat/completions',json.dumps(body).encode(),{'Authorization':'Bearer '+role.api_key,'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=60) as r: response=json.load(r)
        except urllib.error.HTTPError as e:
            detail=e.read().decode(errors='replace')
            if role.api_key: detail=detail.replace(role.api_key,'[REDACTED]')
            print('HTTP',e.code,detail[:1500]);raise SystemExit('Probe failed; active GPT configuration unchanged. Ask organizer for deployment route/request example.')
        except (urllib.error.URLError,TimeoutError) as e: raise SystemExit(type(e).__name__)
        text=response.get('choices',[{}])[0].get('message',{}).get('content')
        print(name,repr(text))
        try: parsed=json.loads(text)
        except (ValueError,TypeError): raise SystemExit('Not strict JSON; adapter not verified')
        if not isinstance(parsed,dict) or parsed.get('supported') is not expected: raise SystemExit('Decision check failed')
    print('Basic chat/JSON probe passed. This is NOT calibration or SystemOne validation.')
if __name__=='__main__':main()
