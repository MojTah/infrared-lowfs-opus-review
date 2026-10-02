"""Expanded grouped campaign; frozen engine and new generator policy are separate."""
import argparse
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from benchmark import cli, core
import psutil
import numpy as np
import blocks

OLD = PROJECT / 'benchmark/runs/keck-benchmark-03'
CPU_LIMIT = 48 * 3600
GENERATOR_FILES = [Path(__file__).resolve(), Path(blocks.__file__).resolve(), PROJECT/'ledger/analysis/T001-run-stage.ps1']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def checked_path(run):
    run = Path(run).resolve()
    if not run.is_relative_to(PROJECT / 'benchmark/runs') or run == OLD:
        raise ValueError('Require a separate run under benchmark/runs')
    return run


@contextmanager
def execution_owner(run):
    path = run / 'active-owner.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump({'pid':os.getpid(), 'started_utc':cli.now()}, stream)
    lease = {'safe_to_release':True}
    try:
        yield lease
    finally:
        if lease['safe_to_release']: path.unlink()


def coverage(shards, config, complete=False):
    expected = {f'{split}-{i:04d}' for split in cli.SPLITS for i in range(config['parent_counts'][split])}
    groups, block_ids, paths = {}, set(), set()
    for shard in shards:
        parent = shard['parent_id']
        if parent not in expected or not parent.startswith(shard['split'] + '-'):
            raise ValueError('Parent/split allocation changed')
        start, end = shard['acquisition_start'], shard['acquisition_stop']
        if (start, end) not in ((0, 5), (5, 20), (0, 20)) or shard['rows'] != (end-start)*3:
            raise ValueError('Block acquisition/flux coverage changed')
        split_index = cli.SPLITS.index(shard['split'])
        parent_index = int(parent.rsplit('-',1)[1])
        for tag, name in ((613,'phase_seed'),(449,'calibration_seed'),(991,'label_seed'),(227,'noise_seed')):
            entropy = [config['seed'],split_index,parent_index,tag]
            if start == 5 and tag in (991,227): entropy.append(5)
            if shard[name] != int(np.random.SeedSequence(entropy).generate_state(1)[0]):
                raise ValueError('Parent sampling seed changed')
        if end == 20:
            for tag, name in ((991,'continuation_label_seed'),(227,'continuation_noise_seed')):
                expected_seed = int(np.random.SeedSequence([config['seed'],split_index,parent_index,tag,5]).generate_state(1)[0])
                if shard[name] != expected_seed: raise ValueError('Continuation seed changed')
        key = (parent, start, end)
        if key in block_ids or shard['path'] in paths:
            raise ValueError('Duplicate block identity/path')
        block_ids.add(key); paths.add(shard['path'])
        indices = groups.setdefault(parent, set())
        if indices.intersection(range(start, end)):
            raise ValueError('Overlapping acquisition indices')
        indices.update(range(start, end))
    if complete and (set(groups) != expected or any(v != set(range(20)) for v in groups.values())):
        raise ValueError('Require exactly 620 grouped parents and all 20 indices')
    return groups


def admission(run, policy):
    evidence = read(run/'driver-admission.json')
    if evidence['status'] != 'PASS' or evidence['generator_policy_sha256'] != sha(run/'execution-policy.json'):
        raise ValueError('New driver admission does not match frozen policy')
    return evidence


def spent_cpu(run, policy, state):
    downstream = sum(read(path)['cpu_seconds'] for path in run.glob('t001-*-resources.json'))
    return policy['base_cpu_seconds_upper'] + state['cpu_seconds_upper'] + downstream


def check(run, require_dataset=False):
    run = checked_path(run)
    policy = read(run / 'execution-policy.json')
    record, config = cli.admitted(run)
    if policy['target_acquisitions'] != 20 or policy['cpu_ceiling_seconds'] != CPU_LIMIT:
        raise ValueError('Approved target/resource identity changed')
    if (policy['workers'] not in (4,8) or policy['ram_ceiling_bytes'] != 16*1024**3
            or policy['data_ceiling_bytes'] != 4*1024**3 or policy['initial_training_gpu_occupancy_seconds'] != 7200):
        raise ValueError('Approved execution ceilings changed')
    if policy['measurement_hash'] != record['measurement_hash'] or policy['engine_source_hash'] != core.source_hash():
        raise ValueError('Frozen measurement/engine changed')
    expected_paths = {p.relative_to(PROJECT).as_posix() for p in GENERATOR_FILES}
    if set(policy['generator_sources']) != expected_paths:
        raise ValueError('Expanded generator source list changed')
    for path, expected in policy['generator_sources'].items():
        if sha(PROJECT / path) != expected:
            raise ValueError('Expanded generator source changed')
    if policy.get('backend') is not None:
        if policy['backend'] != 'resident_phase_fp64_native_hcipy_mft':
            raise ValueError('Unrecognized optical backend')
        expected_backend = {f'ledger/analysis/expanded_campaign/{name}' for name in
                            ('gpu_campaign.py','gpu_packets.py','gpu_backend.py','gpu_mft.py')}
        if set(policy['backend_sources']) != expected_backend:
            raise ValueError('GPU backend source list changed')
        for path, expected in policy['backend_sources'].items():
            if sha(PROJECT/path) != expected: raise ValueError('Admitted GPU backend source changed')
        if set(policy['gpu_evidence']) != {'t001-gpu-admission-resources.json',
                                          't001-gpu-eight-worker-resources.json','t001-gpu-lifecycle-resources.json',
                                          't001-gpu-lifecycle-input-binding.json'}:
            raise ValueError('GPU admission evidence list changed')
        for name, expected in policy['gpu_evidence'].items():
            if sha(run/name) != expected or read(run/name)['status'] != 'PASS':
                raise ValueError('GPU admission evidence changed')
    for name, expected in policy['inherited_files'].items():
        if sha(run / name) != expected:
            raise ValueError('Inherited physics/configuration evidence changed')
    state = read(run / ('dataset.json' if require_dataset else 'generation-state.json'))
    if state['generator_policy_sha256'] != sha(run / 'execution-policy.json'):
        raise ValueError('Generator policy changed')
    if (state['source_hash'] != record['source_hash'] or state['config_hash'] != record['config_hash']
            or state['measurement_hash'] != record['measurement_hash'] or state['basis'] != record['basis']
            or state['mode_order'] != core.MODES or state['acquisitions_per_parent'] != 20):
        raise ValueError('Dataset physics/header identity changed')
    for shard in state['shards']:
        path = (run / shard['path']).resolve()
        if not path.is_relative_to(run) or sha(path) != shard['sha256']:
            raise ValueError('Expanded shard identity changed')
    if list(run.rglob('*.partial*')):
        raise ValueError('Inspect retained partial evidence before resume')
    indexed = {s['path'] for s in state['shards']}
    if {p.relative_to(run).as_posix() for p in (run/'data').rglob('*.npz')} != indexed:
        raise ValueError('Unindexed campaign shards')
    if policy.get('backend') is not None:
        for shard in state['shards']:
            old_hash = policy['pre_gpu_shards'].get(shard['path'])
            if old_hash is not None:
                if shard['sha256'] != old_hash: raise ValueError('Pre-GPU shard changed')
            elif (shard.get('backend_sources') != policy['backend_sources'] or shard.get('optical_backend') not in
                  ('resident_phase_fp64_native_hcipy_mft','native_cpu_after_optical_fallback')):
                raise ValueError('GPU shard backend provenance changed')
    groups = coverage(state['shards'], config, require_dataset)
    if require_dataset:
        if state['status'] != 'COMPLETE':
            raise ValueError('Complete admitted expanded dataset required')
        admission(run,policy)
    return run, policy, record, config, state, groups


def prepare(run, workers):
    run = checked_path(run)
    generator_sources = {p.relative_to(PROJECT).as_posix():sha(p) for p in GENERATOR_FILES}
    record, config = cli.admitted(OLD)
    old_state = read(OLD / 'generation-state.json')
    if old_state['status'] != 'STOPPED_AT_SAFE_PARENT_BOUNDARY' or workers not in (4, 8):
        raise ValueError('Stopped predecessor and measured worker choice required')
    trials = [read(OLD/f't001-workers-{n}.json') for n in (4, 8)]
    selected = next(t for t in trials if t['workers'] == workers)
    if selected['status'] != 'PASS' or selected['peak_aggregate_rss_bytes'] > config['budgets']['ram_bytes']:
        raise ValueError('Worker/memory admission absent')
    if selected['source_hash'] != core.source_hash() or selected['config_hash'] != core.digest(config):
        raise ValueError('Worker canary identity changed')
    # Prior uncounted development is an allowance, never described as measured CPU.
    base_cpu = old_state['cpu_seconds'] + record['cpu_seconds'] + read(OLD/'canary.json')['cpu_seconds']
    base_cpu += 5000 + 120 + 120 + 200 + sum(t['cpu_seconds_upper'] for t in trials)
    remaining = 20*sum(config['parent_counts'].values()) - sum(s['rows']//3 for s in old_state['shards'])
    replay_estimate = len(old_state['shards'])*5*2.4
    projected = base_cpu + remaining*selected['cpu_seconds_per_acquisition_upper'] + replay_estimate
    if projected >= CPU_LIMIT - 7200:
        raise RuntimeError(f'20 acquisitions cannot fit with 7200 CPU-seconds downstream reserve: projected {projected:.1f}')
    for old_shard in old_state['shards']:
        if old_shard['rows'] != 15 or sha(OLD/old_shard['path']) != old_shard['sha256']:
            raise ValueError('Predecessor prefix changed')
    run.mkdir(exist_ok=False)
    inherited = {}
    for name in ('config.json','readiness.json','canary.json','hardware.json'):
        shutil.copyfile(OLD/name, run/name); inherited[name] = sha(run/name)
    policy = {'schema_version':1, 'target_acquisitions':20, 'cpu_ceiling_seconds':CPU_LIMIT,
              'initial_training_gpu_occupancy_seconds':7200, 'ram_ceiling_bytes':16*1024**3,
              'data_ceiling_bytes':4*1024**3, 'workers':workers,
              'engine_source_hash':core.source_hash(), 'measurement_hash':record['measurement_hash'],
              'engine_config_hash':core.digest(config), 'inherited_files':inherited,
              'generator_sources':generator_sources,
              'predecessor_run':str(OLD), 'base_cpu_seconds_upper':base_cpu,
              'preliminary_cpu_allowances_seconds':{'prior_uncounted_development':5000,'GPU_trials':120,'sampler_development_checks':120,'native_prefix_reproduction':200},
              'projected_cpu_seconds_upper':projected, 'downstream_cpu_reserve_seconds':7200,
              'worker_canaries':trials, 'legacy_config_role':'immutable original engine/measurement bundle; this policy overrides campaign count, CPU ceiling and worker scheduling',
              'sampling':'original streams indices0..4; new SeedSequence tag5 streams indices5..19; same native atmosphere/calibration parent',
              'gpu_generation':'not adopted: measured complete-acquisition slowdown in FP64; exploratory FP32 gain too small for production complexity',
              'admission_status':'NEW DRIVER PENDING; inherited gates apply only to unchanged engine/measurement'}
    core.save_json(run/'execution-policy.json',policy)
    shards = []
    for old_shard in old_state['shards']:
        if old_shard['rows'] != 15 or sha(OLD/old_shard['path']) != old_shard['sha256']:
            raise ValueError('Predecessor prefix changed')
        target = run/'data/prefix'/f"{old_shard['parent_id']}.npz"; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(OLD/old_shard['path'],target)
        shards.append({**old_shard,'path':target.relative_to(run).as_posix(),
                       'acquisition_start':0,'acquisition_stop':5,
                       'reuse_origin':{'run':str(OLD),'path':old_shard['path'],'sha256':old_shard['sha256']}})
    state = {'schema_version':1,'source_hash':core.source_hash(),'config_hash':core.digest(config),
             'measurement_hash':record['measurement_hash'],'basis':record['basis'],'mode_order':core.MODES,
             'label_units':'nm OPD RMS','acquisitions_per_parent':20,'shards':shards,
             'generator_policy_sha256':sha(run/'execution-policy.json'),'created_utc':cli.now(),
             'cpu_seconds_upper':0.,'elapsed_s':0.,'status':'PREPARED_NOT_ADMITTED'}
    core.save_json(run/'generation-state.json',state)
    check(run)
    return {'status':'PREPARED','prefixes_preserved':len(shards),'projected_cpu_seconds_upper':projected}


def generate_owned(run, max_seconds, canary_blocks, lease):
    start = time.perf_counter()
    run, policy, record, config, state, groups = check(run)
    if policy.get('backend') is not None:
        raise ValueError('Use the admitted GPU generation entrypoint')
    if (run/'dataset.json').exists():
        raise FileExistsError('Preserve complete dataset')
    if canary_blocks:
        if not 1 <= canary_blocks <= 8 or max_seconds > 600:
            raise ValueError('Bound driver admission to eight blocks and 600 seconds')
    else: admission(run,policy)
    jobs = []
    for split in cli.SPLITS:
        for parent in range(config['parent_counts'][split]):
            indices = groups.get(f'{split}-{parent:04d}',set())
            if indices == set(range(20)): continue
            begin, count = (5,15) if indices == set(range(5)) else (0,20)
            if indices and indices != set(range(5)): raise ValueError('Unsupported retained coverage')
            jobs.append((split,parent,begin,count,time.time()+max_seconds))
    if canary_blocks: jobs = jobs[:canary_blocks]
    workers = policy['workers']; previous_cpu = state['cpu_seconds_upper']; previous_wall = state['elapsed_s']
    prior_total_cpu = spent_cpu(run,policy,state)
    cpu_used, stopped, pending = 0., True, {}
    context = multiprocessing.get_context('spawn'); stop = context.Event()
    lease['safe_to_release'] = False
    pool = ProcessPoolExecutor(max_workers=workers,mp_context=context,initializer=blocks.init,initargs=(config,record,stop))
    iterator = iter(jobs)
    def checkpoint(status):
        state.update(status=status,cpu_seconds_upper=previous_cpu+cpu_used,elapsed_s=previous_wall+time.perf_counter()-start)
        core.save_json(run/'generation-state.json',state,overwrite=True)
    def submit():
        job = next(iterator,None)
        if job is not None: pending[pool.submit(blocks.generate_block,job)] = job
    try:
        for _ in range(workers): submit()
        while pending:
            children = [p for p in psutil.Process().children(recursive=True) if p.is_running()]
            cpu_used = sum(psutil.Process().cpu_times()[:2])+sum(sum(p.cpu_times()[:2]) for p in children)
            memory = psutil.Process().memory_info().rss+sum(p.memory_info().rss for p in children)
            if memory > policy['ram_ceiling_bytes']: raise MemoryError('Aggregate 16 GiB ceiling')
            if prior_total_cpu+cpu_used > CPU_LIMIT-policy['downstream_cpu_reserve_seconds']-(workers+1)*35:
                raise TimeoutError('Cumulative CPU ceiling/downstream reserve approached')
            if time.perf_counter()-start > max_seconds or (run/'stop-request.json').exists():
                raise TimeoutError('Bounded deadline or requested safe stop')
            done,_ = wait(pending,timeout=1,return_when=FIRST_COMPLETED)
            for future in done:
                arrays, metadata = future.result(); pending.pop(future)
                directory = 'continuation' if metadata['acquisition_start']==5 else 'full'
                path = run/'data'/directory/f"{metadata['parent_id']}.npz"; path.parent.mkdir(parents=True,exist_ok=True)
                temporary = path.with_suffix('.partial.npz')
                if path.exists() or temporary.exists(): raise FileExistsError('Preserve unindexed/partial block')
                np.savez_compressed(temporary,**arrays); os.replace(temporary,path)
                state['shards'].append({**metadata,'path':path.relative_to(run).as_posix(),'sha256':sha(path)})
                checkpoint('IN_PROGRESS')
                total = sum(s['rows']//3 for s in state['shards'])
                print('expanded acquisitions',total,'/ 12400','cpu_s_upper',round(previous_cpu+cpu_used),'elapsed_s',round(time.perf_counter()-start),flush=True)
                retained = sum(p.stat().st_size for folder in (run,OLD) for p in folder.rglob('*') if p.is_file())
                if retained > policy['data_ceiling_bytes']: raise MemoryError('Aggregate retained 4 GiB ceiling')
                submit()
        stopped = False
    finally:
        shutdown = time.perf_counter()
        if stopped:
            stop.set()
            for future in pending: future.cancel()
            owned = psutil.Process().children(recursive=True)
            pool.shutdown(wait=False,cancel_futures=True)
            _,alive = psutil.wait_procs(owned,timeout=30)
            for process in reversed(alive):
                if process.is_running(): process.kill()
            _,alive = psutil.wait_procs(alive,timeout=5)
            if alive: raise RuntimeError('Owned worker stop unverified')
        else: pool.shutdown(wait=True,cancel_futures=True)
        lease['safe_to_release'] = True
        cpu_used += (1+time.perf_counter()-shutdown)*(workers+1)
        checkpoint('STOPPED_AT_SAFE_PARENT_BOUNDARY' if stopped or canary_blocks else 'COMPLETE')
    if not canary_blocks:
        coverage(state['shards'],config,complete=True)
        core.save_json(run/'dataset.json',state)
    return {'status':state['status'],'rows':sum(s['rows'] for s in state['shards']),'cpu_seconds_upper':state['cpu_seconds_upper']}


def generate(run, max_seconds, canary_blocks=0):
    with execution_owner(checked_path(run)) as lease:
        return generate_owned(run,max_seconds,canary_blocks,lease)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','generate','check'))
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--max-seconds',type=float,default=40000)
    parser.add_argument('--canary-blocks',type=int,default=0)
    parser.add_argument('--require-dataset',action='store_true')
    args = parser.parse_args()
    if args.max_seconds <= 0 or not np.isfinite(args.max_seconds): parser.error('Positive finite deadline required')
    if args.action=='prepare': result=prepare(args.run,args.workers)
    elif args.action=='generate': result=generate(args.run,args.max_seconds,args.canary_blocks)
    else:
        _,policy,_,_,state,_=check(args.run,args.require_dataset)
        result={'status':'PASS','rows':sum(s['rows'] for s in state['shards']),'policy_sha256':state['generator_policy_sha256'],'cpu_seconds_upper':spent_cpu(args.run,policy,state)}
    print(json.dumps(result),flush=True)
