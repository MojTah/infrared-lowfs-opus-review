"""Mocked adapter flow plus tiny genuine construction/exclusive-save checks."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

START = time.perf_counter()
CANARY = None
PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from ledger.analysis.expanded_campaign import pilot_compat as p
import numpy as np


def rejects(call, error=ValueError):
    try:
        call()
    except error:
        return
    raise AssertionError(f'expected {error.__name__}')


def guard():
    if time.process_time() > 25 or time.perf_counter() - START > 75:
        raise TimeoutError('Stop before 30 CPU-second/90 wall-second development ceiling')


def run():
    global CANARY
    guard()
    source = p.core.source_hash()
    assert source == '648c0bd875c818ed1c06d246213c283b9a2448b194c0ce663b26176550a342cb'
    config = copy.deepcopy(p.core.DEFAULT_CONFIG)
    config.update(parent_counts={'large_telescope_pilot': 5}, residual_rms_nm=[100.],
                  pupil='tmt', diameter_m=30.,
                  pixel_mas=config['band_um'][0] * 1e-6 / (2 * 30. * p.core.MAS_RAD),
                  flux_e=[10000.], pupil_n=64, basis_reference_n=64,
                  pixel_order=2, wavelength_nodes=2)
    before = copy.deepcopy(config)
    effective = p._effective_config(config)
    assert config == before
    assert effective['parent_counts'] == {split: 5 for split in p.cli.SPLITS}
    assert {k: v for k, v in effective.items() if k != 'parent_counts'} == {k: v for k, v in config.items() if k != 'parent_counts'}
    assert effective['residual_rms_nm'] == [100.]
    for counts in (None, {}, {'large_telescope_pilot': True}, {'large_telescope_pilot': 0},
                   {'large_telescope_pilot': -1}, {'large_telescope_pilot': 5.},
                   {'large_telescope_pilot': 5, 'test': 1}):
        invalid = copy.deepcopy(config)
        invalid['parent_counts'] = counts
        rejects(lambda invalid=invalid: p._effective_config(invalid))
    # Synthetic canary artifacts are retained under ignored benchmark/runs.
    directory = Path(tempfile.mkdtemp(prefix='pilot-compat-development-', dir=PROJECT / 'benchmark/runs'))
    CANARY = directory
    guard()
    model = p.core.OpticalModel(effective)
    assert model.n == 64 and model.config['parent_counts'] == effective['parent_counts']
    assert model.config['residual_rms_nm'] == [100.]
    del model
    guard()
    called = []

    def original(config_received, start, seconds, ram):
        called.append((copy.deepcopy(config_received), start, seconds, ram))
        return None, {'synthetic': True}, {'status': 'LIMIT'}

    def simulated(run_dir, max_seconds, count):
        assert count == 5 and max_seconds == 7
        p.pilots._converged_model(config, 1., 7., 16 * 1024 ** 3)
        return {'status': 'PARTIAL', 'cases': []}

    with patch.object(p.pilots, '_converged_model', original), patch.object(p.pilots, 'simulate', side_effect=simulated), patch.object(p.pilots, 'save_json') as save:
        report = p.run(directory, max_seconds=7, count=5)
        assert p.pilots._converged_model is original
        assert called[0][0] == effective and called[0][1:] == (1., 7., 16 * 1024 ** 3)
        adaptation = report['compatibility_adapter']['parent_counts_adaptations'][0]
        assert adaptation['requested_parent_counts'] == config['parent_counts']
        assert adaptation['effective_parent_counts'] == effective['parent_counts']
        assert report['compatibility_adapter']['source_sha256'] == p.core.file_hash(p.__file__)
        save.assert_called_once_with(directory / 'large_telescopes.json', report)
    def fail_convergence(*args):
        raise RuntimeError('synthetic delegated convergence failure')

    with patch.object(p.pilots, '_converged_model', fail_convergence), patch.object(p.pilots, 'simulate', side_effect=simulated):
        rejects(lambda: p.run(directory, max_seconds=7, count=5), RuntimeError)
        assert p.pilots._converged_model is fail_convergence
    guard()
    # Exercise real native admission refusal without writing the summary yet.
    with patch.object(p.pilots, '_admission', side_effect=ValueError('synthetic missing freeze')) as admission, patch.object(p.pilots, 'save_json') as save:
        refused = p.run(directory, max_seconds=7, count=5)
        admission.assert_called_once_with(directory)
        assert refused['status'] == 'NOT_ADMITTED'
        assert not (directory / 'large_telescopes').exists()
        save.assert_called_once_with(directory / 'large_telescopes.json', refused)
    # Permission-dependent native acquisition/checksum and exclusive-summary canaries.
    # Retain every file on success/failure; the execution owner handles cleanup.
    guard()
    acquisition = directory / 'synthetic-acquisition.npz'
    values = {'images_single': np.zeros((1, 1, 16, 16), dtype='float32'),
              'images_pair': np.zeros((1, 2, 16, 16), dtype='float32'),
              'labels_nm': np.zeros((1, 4)), 'flux_e': np.array([10000.])}
    checksum = p.pilots._save_acquisition(acquisition, values)
    assert checksum == p.core.file_hash(acquisition)
    rejects(lambda: p.pilots._save_acquisition(acquisition, values), FileExistsError)
    assert p.core.file_hash(acquisition) == checksum
    with patch.object(p.pilots, '_admission', side_effect=ValueError('synthetic missing freeze')):
        p.run(directory, max_seconds=7, count=5)
    summary = directory / 'large_telescopes.json'
    saved = p.core.file_hash(summary)
    assert json.loads(summary.read_text())['status'] == 'NOT_ADMITTED'
    rejects(lambda: p.run(directory), FileExistsError)
    assert p.core.file_hash(summary) == saved
    assert p.core.source_hash() == source
    cpu, wall = time.process_time(), time.perf_counter() - START
    assert cpu < 30 and wall < 90
    print(json.dumps({'status': 'PASS', 'scope': 'mocked convergence/admission plus genuine 64-pupil TMT constructor and tiny synthetic native saves; no production pilot',
                      'cpu_seconds_root_process': cpu, 'wall_seconds': wall, 'source_hash': source,
                      'adapter_sha256': p.core.file_hash(p.__file__), 'retained_canary': str(directory),
                      'canary_bytes': sum(path.stat().st_size for path in directory.iterdir())}))


if __name__ == '__main__':
    try:
        run()
    except Exception as error:
        print(json.dumps({'status': 'FAIL', 'error': type(error).__name__, 'reason': str(error),
                          'cpu_seconds_root_process': time.process_time(),
                          'wall_seconds': time.perf_counter() - START,
                          'retained_canary': None if CANARY is None else str(CANARY)}), flush=True)
        raise
