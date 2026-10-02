"""One FP64 GPU rendering owner fed by bounded native CPU phase producers."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
from queue import Empty
import sys
import time

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT))
import campaign as c
from benchmark import cli, core
import blocks
import gpu_packets
from gpu_mft import cuda_backend
from gpu_backend import attach_device_phase, probabilities_with_fallback
import numpy as np
import psutil


def stream(config, record, jobs, workers, max_seconds, accept, guard, cache):
    """Bound phase transport to two cubes, plus one active cube per producer."""
    if not 1 <= workers <= 14 or max_seconds <= 0:
        raise ValueError('One to fourteen CPU phase workers and bounded deadline required')
    started = time.perf_counter()
    context = multiprocessing.get_context('spawn')
    stop = context.Event(); packets = context.Queue(maxsize=2)
    try:
        cp = cuda_backend(cache)
        model = core.OpticalModel(config,basis=record['basis'])
        extension = attach_device_phase(model,cp,64)
        pool = ProcessPoolExecutor(max_workers=workers,mp_context=context,
                                  initializer=gpu_packets.init,initargs=(config,record,stop,packets))
    except BaseException:
        packets.close(); packets.cancel_join_thread()
        raise
    pending, parents = {}, {}
    iterator = iter(jobs); clean = False
    fallbacks = []; fallback_reason = None
    def submit():
        job = next(iterator,None)
        if job is not None:
            pending[f'{job[0]}-{job[1]:04d}'] = pool.submit(gpu_packets.prepare_parent,job)
    try:
        for _ in range(workers): submit()
        while pending:
            guard()
            if time.perf_counter()-started > max_seconds:
                raise TimeoutError('GPU stream wall deadline')
            for future in pending.values():
                if future.done() and future.exception() is not None: future.result()
            try: packet = packets.get(timeout=1.)
            except Empty: continue
            parent = packet['parent_id']; kind = packet['type']
            if parent not in pending: raise ValueError('Unknown packet parent')
            if kind == 'start':
                if parent in parents: raise ValueError('Duplicate start packet')
                metadata = {k:v for k,v in packet.items() if k!='type'}
                parents[parent] = {'metadata':metadata,'rows':[],
                                   'next':metadata['acquisition_start'],
                                   'noise':np.random.default_rng(metadata['noise_seed'])}
            elif kind == 'acquisition':
                item = parents[parent]; metadata = item['metadata']; index = packet['index']
                if index != item['next'] or packet['block_id'] != metadata['block_id']:
                    raise ValueError('Acquisition packet order/identity changed')
                if metadata['acquisition_start']==0 and index==5:
                    item['noise'] = np.random.default_rng(metadata['continuation_noise_seed'])
                args = (packet['truth_nm'],packet['residual_cube_nm'],packet['centroid_mas'],packet['diversity_scale'])
                if extension is None:
                    from gpu_backend import validate_probabilities
                    values = model.probabilities(*args); validate_probabilities(values)
                else:
                    values, fallback = probabilities_with_fallback(model,extension,*args)
                    if fallback:
                        fallbacks.append({'parent':parent,'index':index,'reason':fallback})
                        fallback_reason = fallback
                        extension = None
                single,pair = values
                for flux in config['flux_e']:
                    item['rows'].append({
                        'images_single':core.detector_read(single*flux,item['noise'],config['background_e']+config['dark_e'],config['read_noise_e']).astype('float32'),
                        'images_pair':core.detector_read(pair*flux/2,item['noise'],(config['background_e']+config['dark_e'])/2,config['read_noise_e']).astype('float32'),
                        'labels_nm':packet['truth_nm'],'flux_e':float(flux)})
                item['next'] += 1
            elif kind == 'finish':
                item = parents.pop(parent)
                metadata = {k:v for k,v in packet.items() if k!='type'}
                if item['next'] != metadata['acquisition_stop']:
                    raise ValueError('Incomplete parent packet stream')
                pending.pop(parent).result()
                arrays = {k:np.array([row[k] for row in item['rows']]) for k in item['rows'][0]}
                blocks._validate_arrays(arrays,metadata['acquisition_stop']-metadata['acquisition_start'],np.asarray(config['flux_e'],float))
                metadata['optical_backend'] = ('resident_phase_fp64_native_hcipy_mft' if extension is not None
                                               else 'native_cpu_after_optical_fallback')
                metadata['cpu_optical_fallbacks'] = [f for f in fallbacks if f['parent']==parent]
                metadata['persistent_cpu_fallback_reason'] = fallback_reason
                accept(arrays,metadata)
                submit()
            else: raise ValueError('Unknown packet type')
        clean = True
    finally:
        owned = psutil.Process().children(recursive=True)
        stop.set()
        for future in pending.values(): future.cancel()
        pool.shutdown(wait=False,cancel_futures=True)
        # Stop queue feeder threads safely; pending unsaved cubes are discarded.
        _,alive = psutil.wait_procs(owned,timeout=15)
        for process in reversed(alive):
            if process.is_running(): process.kill()
        _,alive = psutil.wait_procs(alive,timeout=5)
        if alive: raise RuntimeError('GPU phase-worker shutdown unverified')
        packets.close(); packets.cancel_join_thread()
        cp.cuda.get_current_stream().synchronize()
        cp.get_default_memory_pool().free_all_blocks()
    return {'status':'PASS' if clean else 'STOPPED','fallbacks':fallbacks,
            'wall_s':time.perf_counter()-started}


def generate(run,max_seconds):
    run,policy,record,config,state,groups = c.check(run)
    c.admission(run,policy)
    if policy.get('backend')!='resident_phase_fp64_native_hcipy_mft':
        raise ValueError('Admitted GPU execution policy required')
    if (run/'dataset.json').exists(): raise FileExistsError('Preserve complete dataset')
    deadline = time.time()+max_seconds
    jobs = []
    for split in cli.SPLITS:
        for parent in range(config['parent_counts'][split]):
            indices = groups.get(f'{split}-{parent:04d}',set())
            if indices==set(range(20)): continue
            if indices and indices!=set(range(5)): raise ValueError('Unsupported retained coverage')
            start,count = (5,15) if indices else (0,20)
            jobs.append((split,parent,start,count,deadline))
    prior_cpu = c.spent_cpu(run,policy,state); previous_cpu = state['cpu_seconds_upper']
    previous_wall = state['elapsed_s']; start = time.perf_counter(); cpu_upper = 0.
    workers = policy['workers']; owner = psutil.Process(); last_cpu = {}; retired_cpu = 0.
    def resources():
        nonlocal retired_cpu,cpu_upper,last_cpu
        current = {p.pid:sum(p.cpu_times()[:2]) for p in [owner]+owner.children(recursive=True) if p.is_running()}
        retired_cpu += sum(value for pid,value in last_cpu.items() if pid not in current)
        last_cpu = current; cpu_upper = retired_cpu+sum(current.values())
        rss = sum(p.memory_info().rss for p in [owner]+owner.children(recursive=True) if p.is_running())
        if rss > policy['ram_ceiling_bytes']: raise MemoryError('Aggregate 16 GiB ceiling')
        if prior_cpu+cpu_upper > c.CPU_LIMIT-policy['downstream_cpu_reserve_seconds']-(workers+1)*35:
            raise TimeoutError('Cumulative CPU ceiling/downstream reserve approached')
        if (run/'stop-request.json').exists(): raise TimeoutError('Requested safe stop')
    def checkpoint(status):
        state.update(status=status,cpu_seconds_upper=previous_cpu+cpu_upper,
                     elapsed_s=previous_wall+time.perf_counter()-start)
        core.save_json(run/'generation-state.json',state,overwrite=True)
    def accept(arrays,metadata):
        resources()
        directory = 'continuation' if metadata['acquisition_start']==5 else 'full'
        path = run/'data'/directory/f"{metadata['parent_id']}.npz"; path.parent.mkdir(parents=True,exist_ok=True)
        temporary = path.with_suffix('.partial.npz')
        if path.exists() or temporary.exists(): raise FileExistsError('Preserve unindexed/partial block')
        np.savez_compressed(temporary,**arrays); os.replace(temporary,path)
        metadata['backend_sources'] = policy['backend_sources']
        state['shards'].append({**metadata,'path':path.relative_to(run).as_posix(),'sha256':c.sha(path)})
        checkpoint('IN_PROGRESS')
        print('GPU acquisitions',sum(s['rows']//3 for s in state['shards']),'/ 12400',
              'cpu_s_upper',round(previous_cpu+cpu_upper),'elapsed_s',round(time.perf_counter()-start),flush=True)
        retained = sum(p.stat().st_size for folder in (run,c.OLD) for p in folder.rglob('*') if p.is_file())
        if retained > policy['data_ceiling_bytes']: raise MemoryError('Aggregate retained 4 GiB ceiling')
    with c.execution_owner(run) as lease:
        lease['safe_to_release'] = False; completed = False
        try:
            result = stream(config,record,jobs,workers,max_seconds,accept,resources,run/'gpu-cache')
            completed = True
        finally:
            survivors = owner.children(recursive=True)
            if any(p.is_running() for p in survivors):
                raise RuntimeError('Owned GPU worker cleanup unverified; lease retained')
            lease['safe_to_release'] = True
            # Account for final worker ticks and bounded shutdown not sampled live.
            cpu_upper += 20*(workers+1)
            checkpoint('COMPLETE' if completed else 'STOPPED_AT_SAFE_PARENT_BOUNDARY')
    c.coverage(state['shards'],config,True)
    core.save_json(run/'dataset.json',state)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--max-seconds',type=float,default=30000)
    args = parser.parse_args()
    print(generate(args.run,args.max_seconds),flush=True)
