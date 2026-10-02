"""Eight native phase workers with one GPU; compare saved TRAIN arrays exactly."""
from pathlib import Path
import sys
import threading
import time
import os
PROJECT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT))
from benchmark import core
import campaign as c
import gpu_campaign as g
import numpy as np
import psutil

if __name__=='__main__':
    run=PROJECT/'benchmark/runs/keck-benchmark-05'
    output=run/'t001-gpu-eight-worker-resources.json'
    if output.exists(): raise FileExistsError('Preserve profile evidence')
    _,policy,record,config,state,_=c.check(run)
    started=time.perf_counter(); owner=psutil.Process(); done=threading.Event()
    result={'stage':'GPU-eight-worker-profile','status':'FAIL','workers':8,
            'source_hash':core.source_hash(),'config_hash':core.digest(config),
            'backend_sources':{p.resolve().relative_to(PROJECT).as_posix():c.sha(p) for p in
                               [Path(g.__file__),Path(g.gpu_packets.__file__),
                                Path(__file__).with_name('gpu_backend.py'),Path(__file__).with_name('gpu_mft.py')]},
            'cpu_seconds':0.,'peak_aggregate_rss_bytes':0,'completed_parents':[]}
    def guard():
        family=[owner]+owner.children(recursive=True)
        result['cpu_seconds']=max(result['cpu_seconds'],sum(sum(p.cpu_times()[:2]) for p in family if p.is_running()))
        result['peak_aggregate_rss_bytes']=max(result['peak_aggregate_rss_bytes'],sum(p.memory_info().rss for p in family if p.is_running()))
        if result['cpu_seconds']>1100 or result['peak_aggregate_rss_bytes']>12*1024**3 or time.perf_counter()-started>650:
            raise TimeoutError('Bounded1200CPU/12GiB/700wall profile')
    def watchdog():
        while not done.wait(1.):
            try: guard()
            except Exception as error:
                result.update(status='LIMIT',error=repr(error),cpu_seconds=result['cpu_seconds']+100,
                              elapsed_s=time.perf_counter()-started)
                core.save_json(output,result)
                for child in reversed(owner.children(recursive=True)):
                    if child.is_running(): child.kill()
                os._exit(8)
    def accept(arrays,metadata):
        saved=sorted((s for s in state['shards'] if s['parent_id']==metadata['parent_id']),key=lambda s:s['acquisition_start'])
        assert [(s['acquisition_start'],s['acquisition_stop']) for s in saved]==[(0,5),(5,20)]
        for name in arrays:
            parts=[]
            for shard in saved:
                with np.load(run/shard['path'],allow_pickle=False) as prior: parts.append(prior[name].copy())
            np.testing.assert_array_equal(arrays[name],np.concatenate(parts))
        result['completed_parents'].append(metadata['parent_id'])
        print('GPU eight-worker bitwise TRAIN parent',metadata['parent_id'],flush=True)
    threading.Thread(target=watchdog,daemon=True).start()
    try:
        result['stream']=g.stream(config,record,[('train',i,0,20,time.time()+600) for i in range(4,12)],
                                  8,600,accept,guard,run/'gpu-cache')
        assert len(result['completed_parents'])==8 and not result['stream']['fallbacks']
        result.update(status='PASS')
    except Exception as error:
        result.update(error=repr(error)); raise
    finally:
        done.set(); guard()
        result.update(cpu_seconds=result['cpu_seconds']+100,elapsed_s=time.perf_counter()-started,
                      acquisitions_per_second=160/(time.perf_counter()-started),
                      accounting='live family CPU plus100s final ticks/shutdown allowance')
        core.save_json(output,result)
        print(result,flush=True)
