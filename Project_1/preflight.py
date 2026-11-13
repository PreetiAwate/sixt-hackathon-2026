#!/usr/bin/env python3
from proofguard.submission_config import load_submission
if __name__ == '__main__':
    try:
        cfg,cal,path = load_submission()
    except (OSError,ValueError,KeyError,TypeError) as e:
        raise SystemExit('NOT SUBMISSION READY: '+str(e))
    print(f'Calibration matches. Declared threshold: {cal.threshold}')
    if cal.threshold > 1:
        print('WARNING: all-review policy; zero automation saving.')
