"""Production full-parent equality against saved TRAIN prefix/continuation."""
import json
from pathlib import Path
import sys
import threading
import time

PROJECT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT))
from benchmark import cli,core
import campaign as c
import blocks
import numpy as np

if __name__=='__main__':
    started=time.perf_counter()
    run=PROJECT/'benchmark/runs/keck-benchmark-05'
    output=run/'t001-full-block-resources.json'
    if output.exists(): raise FileExistsError('Preserve complete-parent canary')
    _,policy,record,config,state,_=c.check(run)
    if c.spent_cpu(run,policy,state)>c.CPU_LIMIT-1000: raise TimeoutError('Canary CPU allowance absent')
    result={'stage':'full-block-canary','status':'FAIL','generator_policy_sha256':state['generator_policy_sha256'],
            'source_sha256':core.file_hash(__file__),'scope':'production1024-pupil TRAIN parent; discard arrays; no held-out access'}
    try:
        with c.execution_owner(run):
            blocks.init(config,record,threading.Event())
            arrays,metadata=blocks.generate_block(('train',0,0,20,time.time()+300))
            saved=sorted((s for s in state['shards'] if s['parent_id']=='train-0000'),key=lambda s:s['acquisition_start'])
            assert [(s['acquisition_start'],s['acquisition_stop']) for s in saved]==[(0,5),(5,20)]
            parts={name:[] for name in arrays}
            for shard in saved:
                with np.load(run/shard['path'],allow_pickle=False) as prior:
                    for name in parts: parts[name].append(prior[name].copy())
            for name in arrays: np.testing.assert_array_equal(arrays[name],np.concatenate(parts[name]))
            result.update(status='PASS',rows=metadata['rows'],checks=['bitwise full20 equals native5 plus replayed15','three flux replicas','both designs','continuous phase and exact stream transition'])
    finally:
        result.update(cpu_seconds=time.process_time(),elapsed_s=time.perf_counter()-started)
        core.save_json(output,result)
    print(json.dumps(result),flush=True)
