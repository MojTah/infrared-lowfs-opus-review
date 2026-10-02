"""Select execution resources without changing the checked optical measurement."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import importlib
import math
import os
import platform

import psutil

GIB = 1024**3
OWNER_BYTES = GIB
WORKER_BYTES = 5*GIB//2
CEILINGS = {'cpu_seconds': 57600, 'gpu_seconds': 7200,
            'ram_bytes': 16*GIB, 'data_bytes': 4*GIB}


def _gpu():
    """Prove CUDA arithmetic on device zero; a driver name alone is insufficient."""
    report = {'usable': False, 'device_index': 0, 'torch_version': None,
              'cuda_runtime_version': None, 'name': None, 'total_bytes': None,
              'free_bytes': None, 'probe': None, 'fallback_reason': None}
    try:
        torch = importlib.import_module('torch')
        report.update(torch_version=str(torch.__version__),
                      cuda_runtime_version=torch.version.cuda)
        if not torch.cuda.is_available():
            report['fallback_reason'] = 'PyTorch reports CUDA unavailable; use CPU training.'
            return report
        device = torch.device('cuda:0')
        properties = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        report.update(name=properties.name, total_bytes=int(total), free_bytes=int(free))
        values = torch.tensor([1., 2., 3.], dtype=torch.float32, device=device)
        actual = float((values*values).sum().item())
        torch.cuda.synchronize(device)
        if actual != 14.:
            raise RuntimeError(f'float32 CUDA arithmetic returned {actual}; expected 14')
        report['probe'] = {'dtype': 'float32', 'sum_squares': actual, 'synchronized': True}
        del values
        free, total = torch.cuda.mem_get_info(device)
        report.update(total_bytes=int(total), free_bytes=int(free))
        if free < 128*1024**2:
            report['fallback_reason'] = 'Less than 128 MiB GPU headroom after the CUDA probe; use CPU training.'
        else:
            report['usable'] = True
    except Exception as error:
        report['fallback_reason'] = f'CUDA verification failed ({type(error).__name__}: {error}); use CPU training.'
    return report


def configure(config):
    """Return a config copy and PASS/RESOURCE_LIMIT report; persist both before a run."""
    chosen = deepcopy(config)
    reasons = []
    requested_workers = chosen['generation_workers']
    if isinstance(requested_workers, bool) or not isinstance(requested_workers, int) or requested_workers < 1:
        raise ValueError('generation_workers must be a positive integer')
    for key, ceiling in CEILINGS.items():
        requested = chosen['budgets'][key]
        if isinstance(requested, bool) or not isinstance(requested, (int, float)) or not math.isfinite(requested) or requested < 1:
            raise ValueError(f'positive finite resource budget required: {key}')
        chosen['budgets'][key] = min(int(requested), ceiling)
        if requested > ceiling:
            reasons.append(f'{key} capped at benchmark ceiling {ceiling}.')
    memory = psutil.virtual_memory()
    physical = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    # Unknown physical cores use one worker rather than assuming logical cores are physical.
    cores = physical or 1
    ram_limit = min(chosen['budgets']['ram_bytes'], int(.8*memory.available))
    chosen['budgets']['ram_bytes'] = ram_limit
    fitting_workers = max(0, (ram_limit-OWNER_BYTES)//WORKER_BYTES)
    workers = min(requested_workers, 8, cores, fitting_workers)
    chosen['generation_workers'] = max(1, workers)
    reasons.append(f'RAM admission uses 80% of available RAM, a 1 GiB owner and 2.5 GiB per HCIPy worker: {ram_limit} bytes.')
    reasons.append(f'Generation uses {max(1, workers)} workers, bounded by requested {requested_workers}, physical cores {cores}, eight workers and RAM.')
    gpu = _gpu()
    batch = 64 if gpu['usable'] and gpu['free_bytes'] < 256*1024**2 else 128
    chosen['compute'] = {'training_device': 'cuda' if gpu['usable'] else 'cpu',
                         'training_batch_size': batch}
    reasons.append('The fixed 32x32 MLP uses batch 128; batch 64 is reserved for 128–256 MiB verified GPU headroom.' if gpu['usable'] else gpu['fallback_reason'])
    report = {'schema_version': 1, 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'PASS' if workers >= 1 else 'RESOURCE_LIMIT',
              'platform': platform.platform(), 'physical_cpu_cores': physical,
              'logical_cpu_cores': logical, 'ram_total_bytes': int(memory.total),
              'ram_available_bytes': int(memory.available), 'ram_limit_bytes': ram_limit,
              'worker_memory_allowance_bytes': WORKER_BYTES, 'owner_memory_allowance_bytes': OWNER_BYTES,
              'gpu': gpu, 'chosen_workers': max(1, workers), 'compute': dict(chosen['compute']),
              'reasons': reasons}
    if workers < 1:
        reasons.append('Insufficient admitted RAM for the checked 1024-pixel pupil; stop without lowering optical resolution.')
    return chosen, report
