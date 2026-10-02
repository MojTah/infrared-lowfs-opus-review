"""Reproduce one existing TRAIN prefix with the frozen production resolution."""
import json
from pathlib import Path
import sys
import threading
import time

PROJECT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT))
from benchmark import cli,core
import blocks
import numpy as np

if __name__=='__main__':
    start=time.perf_counter()
    run=PROJECT/'benchmark/runs/keck-benchmark-03'
    record,config=cli.admitted(run)
    state=json.loads((run/'generation-state.json').read_text())
    shard=next(s for s in state['shards'] if s['parent_id']=='train-0000')
    assert core.file_hash(run/shard['path'])==shard['sha256']
    blocks.init(config,record,threading.Event())
    arrays,metadata=blocks.generate_block(('train',0,0,5,time.time()+180))
    with np.load(run/shard['path'],allow_pickle=False) as original:
        for name in arrays: np.testing.assert_array_equal(arrays[name],original[name])
    result={'status':'PASS','scope':'one TRAIN prefix; actual1024 pupil/6pixel/5wavelength/10time samples',
            'source_hash':core.source_hash(),'measurement_hash':record['measurement_hash'],
            'prefix_sha256':shard['sha256'],'cpu_seconds':time.process_time(),'elapsed_s':time.perf_counter()-start}
    assert result['cpu_seconds']<200
    core.save_json(run/'t001-native-prefix-reproduction.json',result)
    print(json.dumps(result),flush=True)
