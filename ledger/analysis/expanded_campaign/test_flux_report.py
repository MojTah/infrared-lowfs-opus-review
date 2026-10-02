"""Synthetic statistics and mocked freeze guards only; no model or science artifact reads."""
import copy
from pathlib import Path
import sys
import time
from unittest.mock import patch

START = time.perf_counter()
PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from ledger.analysis.expanded_campaign import flux_report as f
import numpy as np


def rejects(call):
    try:
        call()
    except (ValueError, RuntimeError):
        return
    raise AssertionError('invalid metadata/statistics accepted')


def run():
    fluxes = np.array([1000., 10000., 100000.])
    flux = np.tile(fluxes[[0, 1, 2, 0, 1]], 2)
    parents = np.repeat(['p1', 'p2'], 5)
    truth = np.zeros((10, 4))
    errors = np.tile([1., 2., 3., 1., 2.], 2)[:, None] * np.ones((1, 4))
    metrics = f.flux_metrics(errors, truth, parents, flux, fluxes, [2.] * 4)
    standard = metrics['equal_third_standardized']
    np.testing.assert_allclose(standard['coefficient_bias_nm'], 2.)
    np.testing.assert_allclose(standard['coefficient_rmse_nm'], np.sqrt(14 / 3))
    np.testing.assert_allclose(standard['interval_coverage'], 2 / 3)
    assert [m['flux_rows'] for m in metrics['by_flux'].values()] == [4, 4, 2]
    assert not np.isclose(np.mean(errors[:, 0]), standard['coefficient_bias_nm'][0])
    # Unequal row counts per parent cannot change equal-parent metrics.
    repeated = np.r_[np.arange(5), np.tile(np.arange(5, 10), 3)]
    unequal = f.flux_metrics(errors[repeated], truth[repeated], parents[repeated], flux[repeated], fluxes, [2.] * 4)
    assert unequal['equal_third_standardized'] == standard
    rejects(lambda: f.flux_metrics(errors[:2], truth[:2], parents[:2], flux[:2], fluxes, [2.] * 4))
    basis = {'target_indices': [2, 3, 4, 5], 'order': ['x_tilt', 'y_tilt', *f.learn.MODES],
             'units': f.learn.COEFFICIENT_UNIT, 'coordinates': 'synthetic', 'metric': 'synthetic',
             'transform': np.eye(6).tolist(), 'mu': [0.] * 6}
    identity = {'source_hash': 'source', 'config_hash': 'config', 'measurement_hash': 'measurement', 'basis': basis}
    shards = [{'split': s, 'parent_id': s + '-1', 'path': s + '.npz', 'sha256': 'a' * 64,
               'rows': 6, 'acquisition_start': 0, 'acquisition_stop': 2} for s in ('train', 'validation', 'calibration', 'test')]
    manifest = {'schema_version': 1, **identity, 'shards': shards}
    frozen = {'schema_version': 1, **identity, 'modal_order': f.learn.MODES,
              'coefficient_unit': f.learn.COEFFICIENT_UNIT, 'label_scale_nm': f.learn.LABEL_SCALE.tolist(),
              'basis_sha256': f.learn._basis_hash(basis), 'dataset_manifest_sha256': 'dataset',
              'training_parents': ['train-1'], 'validation_parents': ['validation-1'],
              'calibration_parents': ['calibration-1'], 'artifacts': {}, 'models': {}}
    for design in ('single', 'pair'):
        checkpoint, ridge = design + '.pt', design + '.npz'
        frozen['artifacts'].update({checkpoint: 'artifact', ridge: 'artifact'})
        frozen['models'][design] = {'selected_seed': 11, 'selected_checkpoint': checkpoint,
                                   'seeds': [{'seed': 11, 'checkpoint': checkpoint, 'calibration_half_width_nm': [2.] * 4}],
                                   'ridge': {'checkpoint': ridge, 'calibration_half_width_nm': [2.] * 4}}
    mock_run = PROJECT / 'benchmark/runs/mock-flux-never-created'

    def sha(path):
        return 'dataset' if Path(path).name == 'dataset.json' else 'frozen' if Path(path).name == 'frozen.json' else 'artifact'

    with patch.object(Path, 'is_file', return_value=True), patch.object(f, '_read', side_effect=lambda path: frozen), patch.object(f.learn, '_manifest', return_value=manifest), patch.object(f.learn, '_sha', side_effect=sha), patch.object(f.cli, 'admitted', return_value=(identity, {'flux_e': fluxes.tolist()})), patch.object(f.learn, '_load_split', side_effect=AssertionError('held-out loader called during guard')), patch.object(np, 'load', side_effect=AssertionError('NPZ accessed during metadata guard')):
        f._guard(mock_run, None)
        for key, bad in (('measurement_hash', 'changed'), ('basis_sha256', 'changed'), ('dataset_manifest_sha256', 'changed')):
            old = frozen[key]
            frozen[key] = bad
            rejects(lambda: f._guard(mock_run, None))
            frozen[key] = old
        frozen['artifacts']['single.pt'] = 'changed'
        rejects(lambda: f._guard(mock_run, None))
        frozen['artifacts']['single.pt'] = 'artifact'
        shards[-1]['parent_id'] = 'train-1'
        rejects(lambda: f._guard(mock_run, None))
        shards[-1]['parent_id'] = 'test-1'
        with patch.object(Path, 'is_file', return_value=False):
            rejects(lambda: f._guard(mock_run, None))
        oopao = {**identity, 'schema_version': 1, 'frozen_sha256': 'frozen', 'oopao_revision': f.OOPAO_REVISION,
                 'shards': [{'split': 'oopao', 'parent_id': 'independent-1', 'path': 'independent.npz', 'acquisitions': 5}]}
        external_path = mock_run / 'oopao/oopao_dataset.json'
        with patch.object(f.learn, '_manifest', side_effect=lambda path: manifest if Path(path).name == 'dataset.json' else oopao):
            f._guard(mock_run, external_path)
            oopao['frozen_sha256'] = 'changed'
            rejects(lambda: f._guard(mock_run, external_path))
            oopao['frozen_sha256'] = 'frozen'
            oopao['source_hash'] = 'changed'
            rejects(lambda: f._guard(mock_run, external_path))
    counts, _ = f._semantics(manifest, 'test')
    assert counts['physical_acquisitions'] == 2 and counts['flux_rows'] == 6
    external = {'shards': [{'split': 'oopao', 'parent_id': 'p1', 'path': 'p1.npz', 'acquisitions': 5}]}
    counts, _ = f._semantics(external, 'oopao')
    assert counts['physical_acquisitions'] == counts['flux_rows'] == 5
    overlap = copy.deepcopy(manifest)
    overlap['shards'].append(copy.deepcopy(shards[-1]))
    rejects(lambda: f._semantics(overlap, 'test'))
    assert time.process_time() < 20 and time.perf_counter() - START < 120
    print({'status': 'PASS', 'scope': 'synthetic predictions and mocked metadata only',
           'cpu_seconds_process': time.process_time(), 'wall_seconds': time.perf_counter() - START,
           'reporter_sha256': f.core.file_hash(f.__file__)})


if __name__ == '__main__':
    run()
