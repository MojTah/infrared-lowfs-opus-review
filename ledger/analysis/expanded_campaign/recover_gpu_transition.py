"""Explicit recovery after interrupted policy handoff; never touch campaign arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history',required=True)
    args=parser.parse_args()
    project=Path(__file__).resolve().parents[3]
    run=project/'benchmark/runs/keck-benchmark-05'
    history=(run/'policy-history'/args.history).resolve()
    if not history.is_relative_to(run/'policy-history'): raise ValueError('Require this run policy-history directory')
    if (run/'active-owner.json').exists(): raise RuntimeError('Verify owner termination before recovery')
    manifest=json.loads((history/'recovery-manifest.json').read_text())
    source=project/'ledger/analysis/expanded_campaign/campaign.py'
    if Path(manifest['controller_path']).resolve()!=source: raise ValueError('Recovery controller identity changed')
    if set(manifest['backups']) != {'campaign.py.txt','execution-policy.json','driver-admission.json','generation-state.json'}:
        raise ValueError('Incomplete recovery backup set')
    for name,digest in manifest['backups'].items():
        if name not in ('campaign.py.txt','execution-policy.json','driver-admission.json','generation-state.json'):
            raise ValueError('Unexpected recovery file')
        if hashlib.sha256((history/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Recovery backup changed')
    retained=history/f'recovery-attempt-{time.time_ns()}';retained.mkdir()
    for name in ('execution-policy.json','driver-admission.json','generation-state.json'):
        if (run/name).exists(): shutil.copyfile(run/name,retained/name)
        partial=(run/name).with_suffix('.json.partial')
        if partial.exists(): shutil.move(str(partial),str(retained/('interrupted-'+name+'.txt')))
        shutil.copyfile(history/name,run/name)
    shutil.copyfile(source,retained/'campaign.py.txt')
    shutil.copyfile(history/'campaign.py.txt',source)
    print('Prior CPU controller/policy restored; run campaign.py check before resuming. Arrays retained.')
