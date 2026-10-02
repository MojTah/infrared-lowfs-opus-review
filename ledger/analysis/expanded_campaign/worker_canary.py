"""Measured worker admission using the unchanged, admitted HCIPy generator."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
import numpy as np
import psutil
from benchmark.cli import admitted, init_worker, generate_parent
from benchmark.core import source_hash, digest, save_json


def run(workers, output, max_seconds):
    if workers not in (4, 8) or max_seconds <= 0:
        raise ValueError('Require four/eight workers and a positive deadline')
    output = Path(output).resolve()
    if not output.is_relative_to(PROJECT / 'benchmark' / 'runs') or output.exists():
        raise ValueError('Require a new evidence file under benchmark/runs')
    old = PROJECT / 'benchmark/runs/keck-benchmark-03'
    record, config = admitted(old)
    if psutil.cpu_count(logical=False) < workers:
        raise RuntimeError('Insufficient physical cores')
    if psutil.virtual_memory().available < config['budgets']['ram_bytes']:
        raise MemoryError('Insufficient current RAM headroom')
    start = time.perf_counter()
    deadline = time.time() + max_seconds
    # Developer identities cannot enter any of the 620 planned parent splits.
    jobs = [('shifted' if i % 3 == 2 else 'train', 10000 + i,
             13 if i < 4 else 12, deadline) for i in range(8)]
    context = multiprocessing.get_context('spawn')
    stop = context.Event()
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=context,
                              initializer=init_worker, initargs=(config, record, stop))
    pending = {pool.submit(generate_parent, job): job for job in jobs}
    metadata, peak, cpu, failed = [], 0, 0., True
    try:
        while pending:
            children = psutil.Process().children(recursive=True)
            cpu = sum(psutil.Process().cpu_times()[:2]) + sum(sum(p.cpu_times()[:2]) for p in children)
            memory = psutil.Process().memory_info().rss + sum(p.memory_info().rss for p in children)
            peak = max(peak, memory)
            if memory > config['budgets']['ram_bytes']:
                raise MemoryError('Aggregate 16 GiB admission ceiling')
            if time.perf_counter() - start > max_seconds:
                raise TimeoutError('Worker canary deadline')
            done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                arrays, parent = future.result()
                assert parent['rows'] == job[2] * len(config['flux_e'])
                assert all(np.isfinite(v).all() for v in arrays.values())
                metadata.append(parent)
                print('worker_canary', workers, len(metadata), '/ 8',
                      'elapsed_s', round(time.perf_counter()-start),
                      'cpu_s', round(cpu), 'peak_GiB', round(peak/1024**3, 2), flush=True)
        failed = False
    finally:
        shutdown = time.perf_counter()
        if failed:
            stop.set()
            for future in pending:
                future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        cpu += (time.perf_counter() - shutdown) * (workers + 1)
    result = {'status': 'PASS', 'workers': workers, 'acquisitions': 100,
              'wall_seconds': time.perf_counter()-start, 'cpu_seconds_upper': cpu,
              'peak_aggregate_rss_bytes': peak, 'source_hash': source_hash(),
              'config_hash': digest(config), 'measurement_hash': record['measurement_hash'],
              'canary_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'cpu_accounting': 'owner/child lifetime CPU sampled at one-second yields; conservative shutdown allowance',
              'parent_records': metadata, 'scope': 'developer throughput/memory admission; no campaign shards'}
    result['cpu_seconds_per_acquisition_upper'] = cpu / 100
    save_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=900)
    args = parser.parse_args()
    result = run(args.workers, args.output, args.max_seconds)
    print(json.dumps({k: result[k] for k in ('status','workers','wall_seconds','cpu_seconds_upper','cpu_seconds_per_acquisition_upper','peak_aggregate_rss_bytes')}), flush=True)
