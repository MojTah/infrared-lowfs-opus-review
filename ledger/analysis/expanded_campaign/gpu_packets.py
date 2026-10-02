"""CPU-only native phase packets for one external rendering/detector owner."""
from queue import Full
import time

# Native import establishes one-thread numerical settings before NumPy loads.
from benchmark import cli, core
from ledger.analysis.expanded_campaign import blocks
import numpy as np

_QUEUE = None


def init(config, record, stop_event, queue):
    """Initialize the unchanged native CPU model and a bounded shared queue."""
    global _QUEUE
    capacity = getattr(queue, 'maxsize', getattr(queue, '_maxsize', 0))
    if not callable(getattr(queue, 'put', None)) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError('bounded queue with positive integer capacity required')
    blocks.init(config, record, stop_event)
    _QUEUE = queue


def _put(packet, deadline):
    while True:
        blocks._check_stop(deadline)
        try:
            _QUEUE.put(packet, timeout=1.)
            return
        except Full:
            pass


def prepare_parent(job):
    """Emit start, acquisition and finish dictionaries; return metadata only.

    Start/finish metadata are flat dictionary fields. Residual cubes retain
    float64 native phases plus the constant parent calibration error. This
    producer never renders, draws detector noise, saves files or uses a GPU.
    """
    if not isinstance(job, (tuple, list)) or len(job) != 5:
        raise ValueError('job must be (split, parent, start, count, deadline)')
    split, parent, start, count, deadline = job
    if split not in cli.SPLITS:
        raise ValueError('unknown split')
    for name, value in (('parent', parent), ('start', start), ('count', count)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'nonnegative integer {name} required')
    if (start, count) not in ((0, 20), (5, 15)):
        raise ValueError('only full parent (0, 20) or continuation (5, 15) admitted')
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not np.isfinite(deadline) or deadline <= 0:
        raise ValueError('positive finite epoch deadline required')
    if _QUEUE is None or cli._WORKER_MODEL is None or cli._WORKER_STOP is None:
        raise RuntimeError('initialize packet worker before preparation')
    blocks._check_stop(deadline)
    cpu_start = time.process_time()
    model = cli._WORKER_MODEL
    config = model.config
    split_index = cli.SPLITS.index(split)

    def seed(tag, continuation=False):
        entropy = [config['seed'], split_index, parent, tag]
        if continuation:
            entropy.append(5)
        return int(np.random.SeedSequence(entropy).generate_state(1)[0])

    label_seed, noise_seed = seed(991, start == 5), seed(227, start == 5)
    continuation_label_seed, continuation_noise_seed = seed(991, True), seed(227, True)
    phase_seed, calibration_seed = seed(613), seed(449)
    shifted = split == 'shifted'
    rms = config['shifted_residual_nm'] if shifted else config['residual_rms_nm'][parent % 2]
    gain = cli._WORKER_RECORD['residual_gain']['shifted_per_nm' if shifted else 'nominal_per_nm'] * rms
    sequence = core.ResidualSequence(model, phase_seed, gain, shifted)
    static_error = model.calibration_error(calibration_seed)
    labels = np.random.default_rng(label_seed)
    parent_id = f'{split}-{parent:04d}'
    block_id = f'{parent_id}-a{start:02d}-{start + count:02d}'
    metadata = {'split': split, 'parent_id': parent_id, 'block_id': block_id,
                'acquisition_start': start, 'acquisition_stop': start + count, 'rows': count * 3,
                'phase_seed': phase_seed, 'calibration_seed': calibration_seed,
                'label_seed': label_seed, 'noise_seed': noise_seed,
                'continuation_label_seed': continuation_label_seed,
                'continuation_noise_seed': continuation_noise_seed,
                'calibration_error_rms_nm': float(np.sqrt(model.weights @ static_error.ravel() ** 2)),
                'residual_target_ensemble_rms_nm': rms,
                'condition': 'joint_residual_and_diversity_shift' if shifted else 'nominal'}
    _put({'type': 'start', **metadata}, deadline)
    for index in range(start):
        blocks._check_stop(deadline)
        sequence.acquisition(index)
    for index in range(start, start + count):
        blocks._check_stop(deadline)
        if start == 0 and index == 5:
            labels = np.random.default_rng(continuation_label_seed)
        truth = labels.uniform(-core.LIMITS, core.LIMITS)
        centroid = labels.uniform(-20, 20, 2)
        residual = sequence.acquisition(index) + static_error
        if (residual.shape != (config['phase_steps'], model.n, model.n)
                or residual.dtype != np.dtype('float64') or not np.isfinite(residual).all()
                or truth.shape != (4,) or not np.isfinite(truth).all()
                or centroid.shape != (2,) or not np.isfinite(centroid).all()):
            raise ValueError('finite matching native phase cube, labels and centroid required')
        if cli.psutil.Process().memory_info().rss > config['budgets']['ram_bytes'] / config['generation_workers']:
            raise MemoryError('per-worker RAM admission exceeded')
        _put({'type': 'acquisition', 'parent_id': parent_id, 'block_id': block_id, 'index': index,
              'residual_cube_nm': residual, 'truth_nm': truth, 'centroid_mas': centroid,
              'diversity_scale': 1.1 if shifted else 1.}, deadline)
    metadata['worker_cpu_seconds'] = time.process_time() - cpu_start
    _put({'type': 'finish', **metadata}, deadline)
    return metadata
