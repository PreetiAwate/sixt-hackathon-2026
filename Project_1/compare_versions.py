"""Compare common-value, LLM starter, full, and slim on held-out dev documents."""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from proofguard.config import get_config, HERE


def read(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def write(p, records):
    Path(p).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/unified/dev.jsonl', help='Labeled development set; public test is refused')
    ap.add_argument('--outdir', required=True, help='New directory; existing directories are refused')
    ap.add_argument('--sample', type=int, default=40, help='Held-out dev document count (default: 40)')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--threshold', type=float, default=0.95, help='Diagnostic scoring threshold, not a calibrated policy')
    a = ap.parse_args()
    os.chdir(HERE)
    cfg = get_config()
    if not cfg.llm_ready:
        raise SystemExit('Configure the extractor in .env first')
    if not cfg.verifier.ready:
        raise SystemExit('Configure the intended verifier in .env first')
    if 'test_public' in Path(a.data).name.lower():
        raise SystemExit('Public test is reserved for reporting after freeze; compare on dev only')

    records = read(a.data)
    if not records or any('gold' not in r for r in records):
        raise SystemExit('Need labeled development records')
    if len({r['doc_id'] for r in records}) != len(records):
        raise SystemExit('Duplicate IDs')
    if not 1 <= a.sample < len(records):
        raise SystemExit(f'--sample must be between 1 and {len(records) - 1} to reserve baseline training data')

    rng = random.Random(a.seed)
    chosen = set(r['doc_id'] for r in rng.sample(records, a.sample))
    heldout = [r for r in records if r['doc_id'] in chosen]
    train = [r for r in records if r['doc_id'] not in chosen]
    out = Path(a.outdir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    gold = out / 'gold.jsonl'
    inputs = out / 'inputs.jsonl'
    baseline_target = out / 'baseline_target.jsonl'
    train_path = out / 'baseline_train.jsonl'
    write(gold, heldout)
    write(inputs, [{k: v for k, v in r.items() if k not in ('gold', 'gold_evidence')} for r in heldout])
    # 3_baseline.py only uses target gold keys to enumerate fields; dummy null labels avoid leakage.
    write(baseline_target, [
        {**{k: v for k, v in r.items() if k not in ('gold', 'gold_evidence')},
         'gold': {name: None for name in r['schema']['properties']}}
        for r in heldout
    ])
    write(train_path, train)

    env = dict(os.environ)
    roles = {'LLM': cfg.extractor, 'VERIFIER': cfg.verifier,
             'CHALLENGER': cfg.challenger, 'ESCALATE': cfg.escalator}
    for prefix, role in roles.items():
        for key, value in {
            'BASE_URL': role.base_url, 'API_KEY': role.api_key, 'MODEL': role.model,
            'API': role.api, 'EXTRA_BODY': json.dumps(role.extra_body),
            'PRICE_IN': str(role.price_in), 'PRICE_OUT': str(role.price_out),
        }.items():
            env[f'{prefix}_{key}'] = value
    env['PG_WORKERS'] = '1'

    role_manifest = {
        prefix.lower(): {'host': __import__('urllib.parse', fromlist=['urlparse']).urlparse(role.base_url).netloc,
                         'model': role.model, 'api': role.api, 'key_present': bool(role.api_key)}
        for prefix, role in roles.items()
    }
    manifest = {
        'dataset': str(a.data), 'sample_docs': len(heldout), 'baseline_train_docs': len(train),
        'seed': a.seed, 'threshold': a.threshold, 'roles': role_manifest,
        'doc_ids': [r['doc_id'] for r in heldout],
        'input_sha256': hashlib.sha256(inputs.read_bytes()).hexdigest(),
        'note': 'Dev-only, uncalibrated comparison. Baseline train excludes scored documents. Scores at threshold are diagnostic. ProofGuard uses configured verifier; no pipeline layers changed.',
        'files': {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (HERE / 'proofguard').glob('*.py')},
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    common = out / 'common.jsonl'
    commands = [
        ('common', [sys.executable, '3_baseline.py', '--train', str(train_path),
                    '--target', str(baseline_target), '--out', str(common), '--mode', 'common']),
        ('starter', [sys.executable, 'baseline_llm_compat.py', '--data', str(inputs),
                     '--out', str(out / 'starter.jsonl'), '--mode', 'llm']),
    ]
    for version in ('full', 'slim'):
        commands.append((version, [sys.executable, 'pg.py', 'predict', '--data', str(inputs),
                                   '--mode', 'llm', '--workers', '1', '--calibrating',
                                   '--out', str(out / (version + '.strict.jsonl')),
                                   '--full-out', str(out / (version + '.jsonl')),
                                   '--trace', str(out / (version + '.trace.jsonl'))]))

    elapsed = {}
    for version, cmd in commands:
        env['PG_PIPELINE'] = version if version in ('full', 'slim') else 'full'
        env['PG_CACHE_DIR'] = str(out / ('cache_' + version))
        start = time.monotonic()
        with (out / (version + '.log')).open('w') as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        elapsed[version] = round(time.monotonic() - start, 2)
        print(version, 'finished:', elapsed[version], 'seconds', flush=True)
        with (out / (version + '.score.txt')).open('w') as report:
            subprocess.run([sys.executable, '5_score.py', '--gold', str(gold), '--pred',
                            str(out / (version + '.jsonl')), '--threshold', str(a.threshold)],
                           stdout=report, check=True)
    (out / 'timing.json').write_text(json.dumps(elapsed, indent=2))
    print('Inspect logs and traces for errors/evidence. This is dev-only and uncalibrated:', out)

if __name__=='__main__': main()
