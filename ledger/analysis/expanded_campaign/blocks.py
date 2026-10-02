"""Native five-acquisition prefixes and independently labelled 15-acquisition continuations."""
import time

# Import the native engine first: it sets the one-thread numerical environment.
from benchmark import cli, core
import numpy as np


def init(config, record, stop_event):
    """Delegate model construction and frozen engine identity to the native worker."""
    core.validate_config(config)
    flux = np.asarray(config['flux_e'], dtype=float)
    residual = np.asarray(config['residual_rms_nm'], dtype=float)
    positive = np.r_[flux, residual, config['shifted_residual_nm'],
                     record['residual_gain']['nominal_per_nm'],
                     record['residual_gain']['shifted_per_nm'], config['budgets']['ram_bytes']]
    if (flux.shape != (3,) or len(set(flux)) != 3 or residual.shape != (2,)
            or not np.isfinite(positive).all() or np.any(positive <= 0)):
        raise ValueError('three distinct positive finite fluxes and positive finite residual/gain/RAM required')
    seed = config['seed']
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('nonnegative integer configuration seed required')
    if not callable(getattr(stop_event, 'is_set', None)):
        raise ValueError('stop event required')
    cli.init_worker(config, record, stop_event)


def _check_stop(deadline):
    if cli._WORKER_STOP.is_set() or time.time() > deadline:
        raise TimeoutError('block generation deadline or stop event')


def _validate_arrays(arrays, count, flux):
    rows = count * 3
    shapes = {'images_single': (rows, 1, 16, 16), 'images_pair': (rows, 2, 16, 16),
              'labels_nm': (rows, 4), 'flux_e': (rows,)}
    if set(arrays) != set(shapes):
        raise ValueError('unexpected block arrays')
    for name, shape in shapes.items():
        values = arrays[name]
        dtype = np.dtype('float32' if name.startswith('images_') else 'float64')
        if values.shape != shape or values.dtype != dtype or not np.isfinite(values).all():
            raise ValueError(f'invalid {name} block shape, dtype or values')
    if not np.array_equal(arrays['flux_e'].reshape(count, 3), np.tile(flux, (count, 1))):
        raise ValueError('flux triplet coverage changed')
    labels = arrays['labels_nm'].reshape(count, 3, 4)
    if not np.array_equal(labels, np.repeat(labels[:, :1], 3, axis=1)):
        raise ValueError('flux replicas must share four physical labels')
    if np.any(np.abs(labels) > core.LIMITS):
        raise ValueError('modal label limits exceeded')


def generate_block(job):
    """Return arrays and metadata; the execution owner alone writes campaign artifacts.

    Production coverage is [0, 20) or [0, 5) plus [5, 20). Phase/calibration seeds
    always match the native parent. Continuation label/noise SeedSequences add
    the fixed start tag 5; replay advances native atmosphere without propagation.
    """
    if not isinstance(job, (tuple, list)) or len(job) != 5:
        raise ValueError('job must be (split, parent, start, count, deadline)')
    split, parent, start, count, deadline = job
    if split not in cli.SPLITS:
        raise ValueError('unknown split')
    for name, value in (('parent', parent), ('start', start), ('count', count)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'nonnegative integer {name} required')
    if (start, count) not in ((0, 5), (5, 15), (0, 20)):
        raise ValueError('only prefix (0, 5), continuation (5, 15) or full parent (0, 20) admitted')
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not np.isfinite(deadline) or deadline <= 0:
        raise ValueError('positive finite epoch deadline required')
    if cli._WORKER_MODEL is None or cli._WORKER_RECORD is None or cli._WORKER_STOP is None:
        raise RuntimeError('initialize native worker before generation')
    _check_stop(deadline)
    cpu_start = time.process_time()
    model = cli._WORKER_MODEL
    config = model.config
    if (start, count) == (0, 5):
        arrays, metadata = cli.generate_parent((split, parent, count, deadline))
    else:
        split_index = cli.SPLITS.index(split)

        def seed(tag, continuation=False):
            entropy = [config['seed'], split_index, parent, tag]
            if continuation:
                entropy.append(5)
            return int(np.random.SeedSequence(entropy).generate_state(1)[0])

        label_seed, noise_seed = seed(991, start == 5), seed(227, start == 5)
        continuation_label_seed, continuation_noise_seed = seed(991, True), seed(227, True)
        phase_seed, calibration_seed = seed(613), seed(449)
        labels = np.random.default_rng(label_seed)
        noise = np.random.default_rng(noise_seed)
        shifted = split == 'shifted'
        rms = config['shifted_residual_nm'] if shifted else config['residual_rms_nm'][parent % 2]
        gain = cli._WORKER_RECORD['residual_gain']['shifted_per_nm' if shifted else 'nominal_per_nm'] * rms
        sequence = core.ResidualSequence(model, phase_seed, gain, shifted)
        static_error = model.calibration_error(calibration_seed)
        for index in range(start):
            _check_stop(deadline)
            sequence.acquisition(index)
        rows = []
        for index in range(start, start + count):
            _check_stop(deadline)
            if start == 0 and index == 5:
                labels = np.random.default_rng(continuation_label_seed)
                noise = np.random.default_rng(continuation_noise_seed)
            truth = labels.uniform(-core.LIMITS, core.LIMITS)
            centroid = labels.uniform(-20, 20, 2)
            single, pair = model.probabilities(truth, sequence.acquisition(index) + static_error,
                                               centroid, 1.1 if shifted else 1.)
            for flux in config['flux_e']:
                rows.append({'images_single': core.detector_read(single * flux, noise, config['background_e'] + config['dark_e'], config['read_noise_e']).astype('float32'),
                             'images_pair': core.detector_read(pair * flux / 2, noise, (config['background_e'] + config['dark_e']) / 2, config['read_noise_e']).astype('float32'),
                             'labels_nm': truth, 'flux_e': float(flux)})
            if cli.psutil.Process().memory_info().rss > config['budgets']['ram_bytes'] / config['generation_workers']:
                raise MemoryError('per-worker RAM admission exceeded')
        arrays = {name: np.array([row[name] for row in rows]) for name in rows[0]}
        metadata = {'split': split, 'parent_id': f'{split}-{parent:04d}', 'rows': len(rows),
                    'phase_seed': phase_seed, 'label_seed': label_seed, 'noise_seed': noise_seed,
                    'continuation_label_seed': continuation_label_seed,
                    'continuation_noise_seed': continuation_noise_seed,
                    'calibration_seed': calibration_seed,
                    'calibration_error_rms_nm': float(np.sqrt(model.weights @ static_error.ravel() ** 2)),
                    'residual_target_ensemble_rms_nm': rms,
                    'condition': 'joint_residual_and_diversity_shift' if shifted else 'nominal'}
    # Native float flux configuration is preserved; no resampling or rewriting.
    _validate_arrays(arrays, count, np.asarray(config['flux_e'], dtype=float))
    _check_stop(deadline)
    metadata.update({'block_id': f"{metadata['parent_id']}-a{start:02d}-{start + count:02d}",
                     'acquisition_start': start, 'acquisition_stop': start + count,
                     'worker_cpu_seconds': time.process_time() - cpu_start})
    return arrays, metadata
