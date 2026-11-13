"""Strict submission calibration loader, shared by preflight and hidden runner."""
import json
import math
from pathlib import Path
from .config import HERE, get_config
from .risk import Calibrator

def load_submission():
    cfg = get_config()
    manifest = json.loads((HERE/'submission_config.json').read_text())
    path = (HERE / manifest['calibrator']).resolve()
    if not path.is_relative_to(HERE.resolve()):
        raise ValueError('Calibrator must be inside Project_1')
    cal = Calibrator.load(path)
    if cal.meta.get('signature') != cfg.signature:
        raise ValueError('Calibrator is stale or belongs to different settings. Recalibrate.')
    threshold = float(manifest['threshold'])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1.0001:
        raise ValueError('Invalid declared threshold')
    if abs(cal.threshold-threshold)>1e-8:
        raise ValueError('Declared threshold differs from fitted calibrator')
    if not cfg.llm_ready:
        raise ValueError('LLM configuration missing')
    return cfg, cal, path
