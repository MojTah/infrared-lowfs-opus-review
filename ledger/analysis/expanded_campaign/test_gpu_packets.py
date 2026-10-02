"""Low-resolution CPU reconstruction and bounded-queue stop checks; no output files."""
import json
from pathlib import Path
from queue import Queue
import sys
import threading
import time
from unittest.mock import patch

START_WALL = time.perf_counter()
PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from benchmark import cli, core
from ledger.analysis.expanded_campaign import blocks, gpu_packets
import numpy as np


def rejects(call, error=ValueError):
    try:
        call()
    except error:
        return
    raise AssertionError(f'expected {error.__name__}')


def reconstruct(packets, model):
    first, last = packets[0], packets[-1]
    assert first['type'] == 'start' and last['type'] == 'finish'
    assert len(packets) == first['acquisition_stop'] - first['acquisition_start'] + 2
    noise = np.random.default_rng(first['noise_seed'])
    rows = []
    for index, packet in zip(range(first['acquisition_start'], first['acquisition_stop']), packets[1:-1]):
        assert packet['type'] == 'acquisition' and packet['index'] == index
        assert packet['parent_id'] == first['parent_id'] and packet['block_id'] == first['block_id']
        if index == 5 and first['acquisition_start'] == 0:
            noise = np.random.default_rng(first['continuation_noise_seed'])
        single, pair = model.probabilities(packet['truth_nm'], packet['residual_cube_nm'],
                                           packet['centroid_mas'], packet['diversity_scale'])
        c = model.config
        for flux in c['flux_e']:
            rows.append({'images_single': core.detector_read(single * flux, noise, c['background_e'] + c['dark_e'], c['read_noise_e']).astype('float32'),
                         'images_pair': core.detector_read(pair * flux / 2, noise, (c['background_e'] + c['dark_e']) / 2, c['read_noise_e']).astype('float32'),
                         'labels_nm': packet['truth_nm'], 'flux_e': float(flux)})
    arrays = {name: np.array([row[name] for row in rows]) for name in rows[0]}
    blocks._validate_arrays(arrays, len(rows) // 3, model.config['flux_e'])
    return arrays


def run():
    cpu_start = time.process_time()
    source = core.source_hash()
    assert source.startswith('648c0b'), source
    config = json.loads(json.dumps(core.DEFAULT_CONFIG))
    config.update(pupil_n=64, basis_reference_n=64, pixel_order=2,
                  wavelength_nodes=2, phase_steps=2, generation_workers=1)
    config['atmosphere']['screen_n'] = 32
    record = {'source_hash': source, 'basis': core.OpticalModel(config).basis,
              'residual_gain': {'nominal_per_nm': .001, 'shifted_per_nm': .001}}
    stop = threading.Event()
    deadline = time.time() + 120
    cases = []
    for split in ('train', 'shifted'):
        for start, count in ((0, 20), (5, 15)):
            queue = Queue(maxsize=count + 2)
            gpu_packets.init(config, record, stop, queue)
            job = (split, 10000, start, count, deadline)
            with patch.object(core, 'detector_read', side_effect=AssertionError('producer drew detector noise')), patch.object(cli._WORKER_MODEL, 'probabilities', side_effect=AssertionError('producer propagated')):
                metadata = gpu_packets.prepare_parent(job)
            packets = [queue.get_nowait() for _ in range(count + 2)]
            assert queue.empty()
            rebuilt = reconstruct(packets, cli._WORKER_MODEL)
            expected, expected_metadata = blocks.generate_block(job)
            for name in rebuilt:
                np.testing.assert_array_equal(rebuilt[name], expected[name])
            for name in expected_metadata:
                if name != 'worker_cpu_seconds':
                    assert metadata[name] == expected_metadata[name], name
            assert {name: value for name, value in packets[-1].items() if name != 'type'} == metadata
            cases.append({'split': split, 'start': start, 'count': count, 'rows': metadata['rows']})
    attempted = threading.Event()

    class SignalledQueue(Queue):
        def put(self, item, block=True, timeout=None):
            attempted.set()
            return super().put(item, block, timeout)

    full = SignalledQueue(maxsize=1)
    full.put_nowait('existing')
    attempted.clear()
    gpu_packets.init(config, record, stop, full)
    errors = []

    def blocked_producer():
        try:
            gpu_packets.prepare_parent(('train', 10001, 0, 20, deadline))
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=blocked_producer)
    thread.start()
    assert attempted.wait(3), 'producer never reached the full queue'
    stop_started = time.perf_counter()
    stop.set()
    thread.join(3)
    assert not thread.is_alive() and len(errors) == 1 and isinstance(errors[0], TimeoutError)
    assert time.perf_counter() - stop_started < 2
    assert full.get_nowait() == 'existing' and full.empty()
    stop.clear()
    queue = Queue(maxsize=22)
    gpu_packets.init(config, record, stop, queue)
    for job in (('unknown', 10000, 0, 20, deadline), ('train', -1, 0, 20, deadline),
                ('train', 10000, 0, 5, deadline), ('train', 10000, 5, 14, deadline),
                ('train', True, 0, 20, deadline), ('train', 10000, 0, 20, float('nan'))):
        rejects(lambda job=job: gpu_packets.prepare_parent(job))
    assert queue.empty()
    rejects(lambda: gpu_packets.init(config, record, stop, Queue()))
    with patch.object(core.ResidualSequence, 'acquisition', return_value=np.full((2, 64, 64), np.nan)):
        rejects(lambda: gpu_packets.prepare_parent(('train', 10000, 0, 20, deadline)))
    assert queue.get_nowait()['type'] == 'start' and queue.empty()
    assert core.source_hash() == source
    cpu = time.process_time()
    wall = time.perf_counter() - START_WALL
    assert cpu < 30 and wall < 180
    print(json.dumps({'status': 'PASS', 'scope': '64-pupil development; synthetic gains; no files or GPU',
                      'cases': cases, 'checks': ['exact native reconstruction', 'producer never renders/counts',
                                               'stop while queue full', 'invalid jobs/queue/phases', 'engine unchanged'],
                      'source_hash': source, 'cpu_seconds_body': cpu - cpu_start,
                      'cpu_seconds_process': cpu, 'wall_seconds': wall}))


if __name__ == '__main__':
    run()
