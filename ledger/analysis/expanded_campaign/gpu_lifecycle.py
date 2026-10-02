"""Interrupt and resume real bounded GPU phase transport without campaign writes."""
from pathlib import Path
import sys
import time
PROJECT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT))
from benchmark import core
import campaign as c
import gpu_campaign as g
import numpy as np
import psutil

if __name__=='__main__':
    run=PROJECT/'benchmark/runs/keck-benchmark-05'; output=run/'t001-gpu-lifecycle-resources.json'
    if output.exists(): raise FileExistsError('Preserve lifecycle evidence')
    _,policy,record,config,state,_=c.check(run)
    owner=psutil.Process(); start=time.perf_counter(); max_cpu=0.; ticks=0
    result={'stage':'GPU-lifecycle','status':'FAIL','source_hash':core.source_hash(),
            'backend_sources':{p.resolve().relative_to(PROJECT).as_posix():c.sha(p) for p in
                               [Path(g.__file__),Path(g.gpu_packets.__file__),
                                Path(__file__).with_name('gpu_backend.py'),Path(__file__).with_name('gpu_mft.py')]}}
    def guard(interrupt=False):
        global max_cpu,ticks
        family=[owner]+owner.children(recursive=True)
        max_cpu=max(max_cpu,sum(sum(p.cpu_times()[:2]) for p in family if p.is_running()))
        if max_cpu>350 or time.perf_counter()-start>350: raise TimeoutError('Lifecycle400CPU/400wall bound')
        if interrupt:
            ticks+=1
            if ticks==16: raise TimeoutError('INTENTIONAL_STREAM_STOP')
    def forbidden(*args): raise AssertionError('Interrupted partial parent must not be published')
    try:
        try:
            g.stream(config,record,[('train',0,0,20,time.time()+180),('train',1,0,20,time.time()+180)],
                     2,180,forbidden,lambda:guard(True),run/'gpu-cache')
        except TimeoutError as error: assert str(error)=='INTENTIONAL_STREAM_STOP',error
        else: raise AssertionError('Intentional stop not observed')
        assert not any(p.is_running() for p in owner.children(recursive=True))
        interrupted_cpu=max_cpu
        completed=[]
        def accept(arrays,metadata):
            saved=sorted((s for s in state['shards'] if s['parent_id']=='train-0000'),key=lambda s:s['acquisition_start'])
            for name in arrays:
                parts=[]
                for shard in saved:
                    with np.load(run/shard['path'],allow_pickle=False) as prior: parts.append(prior[name].copy())
                np.testing.assert_array_equal(arrays[name],np.concatenate(parts))
            completed.append(metadata['parent_id'])
        report=g.stream(config,record,[('train',0,0,20,time.time()+180)],1,180,accept,guard,run/'gpu-cache')
        assert completed==['train-0000'] and not report['fallbacks']
        assert not any(p.is_running() for p in owner.children(recursive=True))
        result.update(status='PASS',checks=['real partial packet stream interrupted','owned producers exited',
                                           'same TRAIN parent resumed to bitwise full20 arrays','no campaign arrays written'])
        # Separate process families retire between the two stages: charge both peaks.
        max_cpu+=interrupted_cpu
    except Exception as error:
        result.update(error=repr(error)); raise
    finally:
        result.update(cpu_seconds=max_cpu+100,elapsed_s=time.perf_counter()-start)
        core.save_json(output,result); print(result,flush=True)
