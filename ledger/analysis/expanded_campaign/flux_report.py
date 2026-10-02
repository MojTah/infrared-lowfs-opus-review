"""Post-freeze flux-specific errors and equal-third standardized reporting; never fits."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from benchmark import cli, core, learning as learn
from benchmark.cross_validate import OOPAO_REVISION
import numpy as np

VERSION = 1


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _semantics(manifest, split):
    """Validate physical grouping from metadata before any held-out NPZ load."""
    groups, kinds, rows = {}, {}, 0
    for shard in (s for s in manifest['shards'] if s['split'] == split):
        if split == 'oopao':
            start, stop = 0, shard['acquisitions']
            kind = 'one_flux_per_acquisition'
            n = stop
        else:
            start = shard.get('acquisition_start', 0)
            stop = shard.get('acquisition_stop', manifest.get('acquisitions_per_parent'))
            kind = 'three_flux_replicas_per_acquisition'
            n = shard['rows']
        if (any(isinstance(v, bool) or not isinstance(v, int) for v in (start, stop, n))
                or start < 0 or stop <= start or n != (stop - start) * (1 if split == 'oopao' else 3)):
            raise ValueError('Invalid acquisition/flux row metadata')
        indices = groups.setdefault(shard['parent_id'], set())
        if indices.intersection(range(start, stop)):
            raise ValueError('Overlapping physical acquisitions')
        indices.update(range(start, stop))
        kinds[shard['path']] = (kind, start, stop, n)
        rows += n
    if not groups:
        raise ValueError('No held-out parents')
    return {'physical_acquisitions': sum(map(len, groups.values())), 'flux_rows': rows,
            'parents': len(groups), 'designs_per_flux_row': 2}, kinds


def _guard(run, manifest_path):
    """All frozen/model/metadata checks precede loading held-out arrays."""
    run = Path(run).resolve()
    frozen_path = run / 'models/frozen.json'
    if not frozen_path.is_file():
        raise RuntimeError('Flux reporting requires models/frozen.json before test access')
    frozen = _read(frozen_path)
    frozen_sha = learn._sha(frozen_path)
    if (frozen.get('schema_version') != 1 or frozen.get('modal_order') != learn.MODES
            or frozen.get('coefficient_unit') != learn.COEFFICIENT_UNIT
            or frozen.get('label_scale_nm') != learn.LABEL_SCALE.tolist()
            or learn._basis_hash(frozen['basis']) != frozen['basis_sha256']):
        raise ValueError('Frozen modal, scale, units or basis identity changed')
    artifacts = frozen['artifacts']
    if not artifacts:
        raise ValueError('Frozen artifact checksums required')
    for name, checksum in artifacts.items():
        path = (run / 'models' / name).resolve()
        if not path.is_relative_to(run / 'models') or learn._sha(path) != checksum:
            raise ValueError('Frozen artifact identity changed')
    for design in ('single', 'pair'):
        spec = frozen['models'][design]
        selected = [s for s in spec['seeds'] if s['seed'] == spec['selected_seed']]
        if len(selected) != 1 or selected[0]['checkpoint'] != spec['selected_checkpoint']:
            raise ValueError('Frozen selected checkpoint identity changed')
        for item in spec['seeds'] + [spec['ridge']]:
            width = np.asarray(item['calibration_half_width_nm'])
            if item['checkpoint'] not in artifacts or width.shape != (4,) or not np.isfinite(width).all() or np.any(width < 0):
                raise ValueError('Unsealed checkpoint or invalid frozen intervals')
    training_path = run / 'dataset.json'
    if learn._sha(training_path) != frozen['dataset_manifest_sha256']:
        raise ValueError('Training manifest changed after freeze')
    training = learn._manifest(training_path)
    record, config = cli.admitted(run)
    for key in ('source_hash', 'config_hash', 'measurement_hash'):
        if frozen[key] != training[key] or frozen[key] != record[key]:
            raise ValueError(f'Frozen/native training {key} mismatch')
    if learn._basis_hash(training['basis']) != frozen['basis_sha256'] or learn._basis_hash(record['basis']) != frozen['basis_sha256']:
        raise ValueError('Native/training basis mismatch')
    if training.get('generator_policy_sha256'):
        policy_path = run / 'execution-policy.json'
        if learn._sha(policy_path) != training['generator_policy_sha256']:
            raise ValueError('Expanded generation policy changed')
        policy = _read(policy_path)
        for sources in ('generator_sources', 'backend_sources'):
            for name, checksum in policy.get(sources, {}).items():
                path = (PROJECT / name).resolve()
                if not path.is_relative_to(PROJECT) or learn._sha(path) != checksum:
                    raise ValueError('Expanded generator source changed')
    known = set()
    for split in ('train', 'validation', 'calibration'):
        parents = set(frozen['training_parents' if split == 'train' else f'{split}_parents'])
        if not parents or parents != {s['parent_id'] for s in training['shards'] if s['split'] == split} or known.intersection(parents):
            raise ValueError('Frozen fitting/calibration parent identity changed')
        known.update(parents)
    manifest_path = Path(manifest_path or training_path).resolve()
    manifest = training if manifest_path == training_path else learn._manifest(manifest_path)
    for key in ('source_hash', 'config_hash', 'measurement_hash'):
        if manifest[key] != frozen[key]:
            raise ValueError(f'Evaluation {key} mismatch')
    if learn._basis_hash(manifest['basis']) != frozen['basis_sha256']:
        raise ValueError('Evaluation basis mismatch')
    splits = [s for s in ('test', 'shifted', 'oopao') if any(r['split'] == s for r in manifest['shards'])]
    if not splits or any(s['parent_id'] in known for s in manifest['shards'] if s['split'] in splits):
        raise ValueError('Missing held-out split or fitting/calibration parent overlap')
    if 'oopao' in splits and (manifest.get('frozen_sha256') != frozen_sha or manifest.get('oopao_revision') != OOPAO_REVISION):
        raise ValueError('OOPAO freeze/source pin mismatch')
    semantics = {split: _semantics(manifest, split) for split in splits}
    flux = np.asarray(config['flux_e'], dtype=float)
    if flux.shape != (3,) or len(set(flux)) != 3 or not np.isfinite(flux).all() or np.any(flux <= 0):
        raise ValueError('Three distinct positive declared fluxes required')
    return run, frozen, manifest_path, manifest, semantics, flux, frozen_sha


def _load_flux(path, manifest, split, dataset, declared, kinds):
    parts, offset = [], 0
    for shard in (s for s in manifest['shards'] if s['split'] == split):
        target = (path.parent / shard['path']).resolve()
        if learn._sha(target) != shard['sha256']:
            raise ValueError('Shard changed before flux load')
        with np.load(target, allow_pickle=False) as arrays:
            flux = np.asarray(arrays['flux_e'], dtype=float)
            kind, start, stop, rows = kinds[shard['path']]
            if flux.shape != (rows,) or not np.isfinite(flux).all():
                raise ValueError('Flux row coverage mismatch')
            expected = np.resize(declared, rows)
            if not np.array_equal(flux, expected):
                raise ValueError('Declared flux sampling/order changed')
            if kind == 'three_flux_replicas_per_acquisition':
                labels = dataset['labels'][offset:offset + rows].reshape(stop - start, 3, 4)
                if not np.array_equal(labels, np.repeat(labels[:, :1], 3, axis=1)):
                    raise ValueError('Physical acquisition labels differ across flux replicas')
            elif not np.array_equal(arrays['acquisition_index'], np.arange(stop)):
                raise ValueError('OOPAO physical acquisition indices changed')
            parts.append(flux)
            offset += rows
    if offset != len(dataset['labels']):
        raise ValueError('Native/flux loader row alignment changed')
    return np.concatenate(parts)


def flux_metrics(prediction, truth, parents, flux, declared, half_width):
    """Equal parent weights within flux; equal thirds across the three fluxes."""
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    parents, flux, width = np.asarray(parents), np.asarray(flux), np.asarray(half_width)
    if (truth.shape != prediction.shape or truth.shape != (len(parents), 4) or flux.shape != (len(parents),)
            or not np.isfinite(truth).all() or not np.isfinite(flux).all()
            or width.shape != (4,) or not np.isfinite(width).all() or np.any(width < 0)
            or len(declared) != 3 or not np.isin(flux, declared).all()):
        raise ValueError('Invalid prediction/truth/parent/flux/interval alignment')
    parent_ids = np.unique(parents)

    def summarize(grouped, failure):
        mean = grouped.mean(axis=0)
        return {'coefficient_bias_nm': mean[:4].tolist(), 'coefficient_rmse_nm': np.sqrt(mean[4:8]).tolist(),
                'reconstructed_wavefront_rmse_nm': float(np.sqrt(mean[8])),
                'interval_coverage': mean[9:13].tolist(), 'interval_width_nm': (2 * width).tolist(),
                'failure_fraction': float(failure)}

    result, groups, failures = {}, [], []
    for value in declared:
        mask = flux == value
        if not np.array_equal(np.unique(parents[mask]), parent_ids):
            raise ValueError('Every physical parent must cover each declared flux')
        grouped, finite = learn._group_statistics(prediction[mask], truth[mask], parents[mask], width)
        failure = np.sum(~finite * learn._parent_weights(parents[mask]))
        result[format(float(value), '.17g')] = {**summarize(grouped, failure), 'flux_e': float(value),
                                               'flux_rows': int(mask.sum()), 'parents': len(parent_ids)}
        groups.append(grouped)
        failures.append(failure)
    return {'by_flux': result, 'equal_third_standardized': summarize(np.mean(groups, axis=0), np.mean(failures)),
            'weighting': 'equal parent weight within each flux; equal one-third flux weights; RMSE from averaged squared errors'}


def report(run, manifest_path=None):
    run, frozen, path, manifest, semantics, declared, frozen_sha = _guard(run, manifest_path)
    learn.torch.set_num_threads(1)
    device = learn.execution_device(frozen['device'])
    result = {'schema_version': 1, 'reporter_version': VERSION, 'reporter_source_sha256': core.file_hash(__file__),
              'frozen_sha256': frozen_sha, 'evaluation_manifest_sha256': learn._sha(path),
              'source_hash': frozen['source_hash'], 'config_hash': frozen['config_hash'],
              'measurement_hash': frozen['measurement_hash'], 'basis_sha256': frozen['basis_sha256'],
              'modal_order': learn.MODES, 'coefficient_unit': learn.COEFFICIENT_UNIT,
              'declared_flux_e': declared.tolist(), 'interval_method': frozen['interval_method'], 'splits': {}}
    for split, (counts, kinds) in semantics.items():
        dataset = learn._load_split(path, manifest, split)
        flux = _load_flux(path, manifest, split, dataset, declared, kinds)
        designs = {}
        for design in ('single', 'pair'):
            specification = frozen['models'][design]
            seeds = {}
            for seed in specification['seeds']:
                model, preprocess = learn._load_checkpoint(run / 'models' / seed['checkpoint'], device)
                prediction = learn._predict_network(model, dataset[design], preprocess, device)
                seeds[str(seed['seed'])] = flux_metrics(prediction, dataset['labels'], dataset['parents'], flux,
                                                       declared, seed['calibration_half_width_nm'])
            with np.load(run / 'models' / specification['ridge']['checkpoint'], allow_pickle=False) as ridge:
                preprocess = {'mean': ridge['feature_mean'], 'std': ridge['feature_std'], 'asinh_scale_e': float(ridge['asinh_scale_e'])}
                prediction = (learn._features(dataset[design], preprocess) @ ridge['weights'] + ridge['intercept']) * learn.LABEL_SCALE
            designs[design] = {'selected_seed': specification['selected_seed'], 'learned': seeds[str(specification['selected_seed'])],
                               'seed_results': seeds, 'ridge': flux_metrics(prediction, dataset['labels'], dataset['parents'], flux,
                                                                          declared, specification['ridge']['calibration_half_width_nm'])}
        result['splits'][split] = {'counts': counts, 'designs': designs}
    if learn._sha(run / 'models/frozen.json') != frozen_sha or learn._sha(path) != result['evaluation_manifest_sha256']:
        raise ValueError('Frozen/evaluation metadata changed during reporting')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing report')
    core.save_json(args.output, report(args.run, args.manifest))
