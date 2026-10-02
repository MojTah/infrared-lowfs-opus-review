"""Plain low-resolution development checks; never produces campaign parents or files."""
import json
from pathlib import Path
import sys
import threading
import time
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from benchmark import cli, core
from ledger.analysis.expanded_campaign import blocks
import numpy as np


def rejects(call, error=ValueError):
    try:
        call()
    except error:
        return
    raise AssertionError(f'expected {error.__name__}')


def run():
    wall = time.perf_counter()
    cpu = time.process_time()
    source = core.source_hash()
    assert source.startswith('648c0b'), source
    config = json.loads(json.dumps(core.DEFAULT_CONFIG))
    config.update(pupil_n=64, basis_reference_n=64, pixel_order=2,
                  wavelength_nodes=2, phase_steps=2, generation_workers=1)
    config['atmosphere']['screen_n'] = 32
    # Synthetic development-only normalization, not a science calibration.
    record = {'source_hash': source, 'basis': core.OpticalModel(config).basis,
              'residual_gain': {'nominal_per_nm': .001, 'shifted_per_nm': .001}}
    stop = threading.Event()
    blocks.init(config, record, stop)
    deadline = time.time() + 150
    parent = 10000  # Deliberately outside all planned campaign parent identities.
    prefix, pm = blocks.generate_block(('train', parent, 0, 5, deadline))
    native, nm = cli.generate_parent(('train', parent, 5, deadline))
    for name in prefix:
        np.testing.assert_array_equal(prefix[name], native[name])
    for name in nm:
        if name != 'worker_cpu_seconds':
            assert pm[name] == nm[name], name
    assert (pm['acquisition_start'], pm['acquisition_stop']) == (0, 5)
    assert pm['block_id'] == 'train-10000-a00-05'
    continuation, cm = blocks.generate_block(('train', parent, 5, 15, deadline))
    repeated, repeated_metadata = blocks.generate_block(('train', parent, 5, 15, deadline))
    for name in continuation:
        np.testing.assert_array_equal(continuation[name], repeated[name])
    assert cm['phase_seed'] == pm['phase_seed']
    assert cm['calibration_seed'] == pm['calibration_seed']
    assert cm['calibration_error_rms_nm'] == pm['calibration_error_rms_nm']
    for tag, name in ((991, 'label_seed'), (227, 'noise_seed')):
        expected = int(np.random.SeedSequence([config['seed'], 0, parent, tag, 5]).generate_state(1)[0])
        assert cm[name] == expected and cm[name] != pm[name]
    assert (cm['acquisition_start'], cm['acquisition_stop']) == (5, 20)
    assert cm['block_id'] == 'train-10000-a05-20'
    full, fm = blocks.generate_block(('train', parent, 0, 20, deadline))
    for name in full:
        np.testing.assert_array_equal(full[name][:15], prefix[name])
        np.testing.assert_array_equal(full[name][15:], continuation[name])
    assert (fm['acquisition_start'], fm['acquisition_stop'], fm['rows']) == (0, 20, 60)
    assert fm['label_seed'] == pm['label_seed'] and fm['noise_seed'] == pm['noise_seed']
    assert fm['continuation_label_seed'] == cm['label_seed']
    assert fm['continuation_noise_seed'] == cm['noise_seed']
    shifted_prefix, spm = blocks.generate_block(('shifted', parent + 1, 0, 5, deadline))
    shifted_cont, scm = blocks.generate_block(('shifted', parent + 1, 5, 15, deadline))
    shifted_full, sfm = blocks.generate_block(('shifted', parent + 1, 0, 20, deadline))
    for name in shifted_full:
        np.testing.assert_array_equal(shifted_full[name][:15], shifted_prefix[name])
        np.testing.assert_array_equal(shifted_full[name][15:], shifted_cont[name])
    assert sfm['condition'] == 'joint_residual_and_diversity_shift'
    assert sfm['residual_target_ensemble_rms_nm'] == config['shifted_residual_nm']
    assert spm['phase_seed'] == scm['phase_seed'] == sfm['phase_seed']
    assert sfm['phase_seed'] != fm['phase_seed']
    assert pm['rows'] + cm['rows'] == 60
    assert list(range(pm['acquisition_start'], pm['acquisition_stop'])) + list(range(cm['acquisition_start'], cm['acquisition_stop'])) == list(range(20))
    for name in prefix:
        merged = np.concatenate((prefix[name], continuation[name]))
        assert len(merged) == 60 and np.isfinite(merged).all()
    first_truth = np.random.default_rng(cm['label_seed']).uniform(-core.LIMITS, core.LIMITS)
    np.testing.assert_array_equal(continuation['labels_nm'][0], first_truth)
    model = cli._WORKER_MODEL
    gain = record['residual_gain']['nominal_per_nm'] * config['residual_rms_nm'][parent % 2]
    sequential = core.ResidualSequence(model, pm['phase_seed'], gain)
    original_acquisition = core.ResidualSequence.acquisition
    for index in range(5):
        sequential.acquisition(index)
    expected_phase = sequential.acquisition(5)
    phases = []

    def capture(sequence, index, steps=None):
        value = original_acquisition(sequence, index, steps)
        if index == 5:
            phases.append(value.copy())
        return value

    with patch.object(core.ResidualSequence, 'acquisition', capture):
        blocks.generate_block(('train', parent, 5, 15, deadline))
    assert len(phases) == 1
    np.testing.assert_array_equal(phases[0], expected_phase)
    # Prove a stop requested during replay prevents all detector propagation.
    replayed = []

    def stop_replay(sequence, index, steps=None):
        replayed.append(index)
        value = original_acquisition(sequence, index, steps)
        if index == 1:
            stop.set()
        return value

    with patch.object(core.ResidualSequence, 'acquisition', stop_replay), patch.object(model, 'probabilities', side_effect=AssertionError('replay propagated')):
        rejects(lambda: blocks.generate_block(('train', parent, 5, 15, deadline)), TimeoutError)
    assert replayed == [0, 1]
    rejects(lambda: blocks.generate_block(('train', parent, 0, 5, deadline)), TimeoutError)
    stop.clear()
    rejects(lambda: blocks.generate_block(('train', parent, 0, 5, time.time() - 1)), TimeoutError)
    for job in (('unknown', parent, 0, 5, deadline), ('train', -1, 0, 5, deadline),
                ('train', parent, 1, 5, deadline), ('train', parent, 5, 14, deadline),
                ('train', True, 0, 5, deadline), ('train', parent, 0, 5, float('inf')),
                ('train', parent, 0, 5, float('nan')), ('train', parent, 0, 5, 0)):
        rejects(lambda job=job: blocks.generate_block(job))
    invalid = json.loads(json.dumps(config))
    invalid['flux_e'][1] = float('nan')
    rejects(lambda: blocks.init(invalid, record, stop))
    damaged = {name: value.copy() for name, value in prefix.items()}
    damaged['labels_nm'][1, 0] += 1
    rejects(lambda: blocks._validate_arrays(damaged, 5, config['flux_e']))
    assert core.source_hash() == source
    elapsed_cpu = time.process_time() - cpu
    elapsed_wall = time.perf_counter() - wall
    assert elapsed_cpu < 120 and elapsed_wall < 180
    print(json.dumps({'status': 'PASS', 'scope': '64-pupil development only; synthetic gain; no files',
                      'checks': ['native prefix equality', 'deterministic continuation', 'fixed seed tags',
                                 'continuous full parent equals prefix plus replayed continuation',
                                 '20-acquisition grouped coverage', 'native phase continuity',
                                 'stop during replay', 'invalid inputs and arrays', 'engine hash unchanged'],
                      'source_hash': source, 'cpu_seconds': elapsed_cpu, 'wall_seconds': elapsed_wall}))


if __name__ == '__main__':
    run()
