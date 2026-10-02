"""Invocation-local parent-count compatibility for native conditional telescope pilots."""
import argparse
import copy
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
# Set native one-thread numerical settings before importing telescope code.
from benchmark import core, cli, large_telescopes as pilots


def _effective_config(config):
    counts = config.get('parent_counts')
    if not isinstance(counts, dict) or set(counts) != {'large_telescope_pilot'}:
        raise ValueError('Require exact large_telescope_pilot count mapping')
    count = counts['large_telescope_pilot']
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('Pilot count must be a positive integer')
    effective = copy.deepcopy(config)
    effective['parent_counts'] = {split: count for split in cli.SPLITS}
    return effective


def run(run_dir, max_seconds=1800, count=5):
    """Use native admission/simulation/save; change only constructor count metadata.

    One scientific invocation owns this temporary module binding. The native
    pilot loop receives the original count, with optics and seeds unchanged.
    """
    run_dir = Path(run_dir).resolve()
    output = run_dir / 'large_telescopes.json'
    if output.exists():
        raise FileExistsError('Preserve existing large-telescope summary')
    original = pilots._converged_model
    adaptations = []

    def converged(config, start, seconds, ram_bytes):
        effective = _effective_config(config)
        metadata = {'case': config.get('case'),
                    'requested_parent_counts': copy.deepcopy(config['parent_counts']),
                    'effective_parent_counts': copy.deepcopy(effective['parent_counts']),
                    'scope': 'OpticalModel validation metadata only; native pilot-loop count unchanged'}
        adaptations.append(metadata)
        return original(effective, start, seconds, ram_bytes)

    pilots._converged_model = converged
    try:
        result = pilots.simulate(run_dir, max_seconds=max_seconds, count=count)
    finally:
        pilots._converged_model = original
    result['compatibility_adapter'] = {'schema_version': 1, 'source_sha256': core.file_hash(__file__),
                                       'only_config_field_changed': 'parent_counts',
                                       'parent_counts_adaptations': adaptations, 'binding_restored': True}
    pilots.save_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--max-seconds', type=float, default=1800)
    parser.add_argument('--count', type=int, default=5)
    args = parser.parse_args()
    result = run(args.run, args.max_seconds, args.count)
    print({'status': result['status'], 'summary': str(args.run.resolve() / 'large_telescopes.json')})
    raise SystemExit(0 if result['status'] == 'PASS' else 2)
