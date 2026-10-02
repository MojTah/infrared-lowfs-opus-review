"""Development-only low-resolution process/seed/cancellation canary."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import time
import numpy as np
from core import DEFAULT_CONFIG,OpticalModel,source_hash
from cli import init_worker,generate_parent

def run():
    config=json.loads(json.dumps(DEFAULT_CONFIG))
    config.update(pupil_n=64,basis_reference_n=64,pixel_order=2,wavelength_nodes=2,phase_steps=2,generation_workers=2)
    model=OpticalModel(config)
    record={'basis':model.basis,'source_hash':source_hash(),
            'residual_gain':{'nominal_per_nm':.01,'shifted_per_nm':.01}}
    context=multiprocessing.get_context('spawn');stop=context.Event()
    init_worker(config,record,stop)
    job=('train',0,2,time.time()+180)
    reference,metadata=generate_parent(job)
    with ProcessPoolExecutor(max_workers=2,mp_context=context,initializer=init_worker,initargs=(config,record,stop)) as pool:
        for future in [pool.submit(generate_parent,job) for _ in range(2)]:
            parallel,observed=future.result(timeout=120)
            for key in reference:np.testing.assert_array_equal(parallel[key],reference[key])
            assert observed['calibration_seed']==metadata['calibration_seed']
        stop.set()
        try:pool.submit(generate_parent,job).result(timeout=15)
        except TimeoutError:pass
        else:raise AssertionError('cancellation ignored')
    assert 0<=metadata['calibration_error_rms_nm']<=20
    print('PASS: serial/two-worker arrays identical, grouped flux replicas, static calibration seed, cooperative cancellation; low-resolution development only')

if __name__=='__main__':run()
