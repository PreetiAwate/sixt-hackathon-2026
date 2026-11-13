"""Run after live dev calibration. Does not select a threshold on public test."""
import json
from pathlib import Path
from proofguard.config import HERE, get_config
from proofguard.risk import Calibrator
p=HERE/'calibrators/llm.json'
c=Calibrator.load(p)
if c.meta.get('signature') != get_config().signature:
    raise SystemExit('Calibrator does not match current code/configuration; recalibrate first.')
(HERE/'submission_config.json').write_text(json.dumps({'calibrator':'calibrators/llm.json','threshold':c.threshold},indent=2)+'\n')
print('Declared threshold:',c.threshold)
