"""Bounded GPU optics/transport/full-parent checks; TRAIN references only."""
import json
from pathlib import Path
import sys
import threading
import time
import os

PROJECT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT))
from benchmark import cli,core
import campaign as c
import gpu_campaign as g
from gpu_backend import attach_device_phase,validate_probabilities,probabilities_with_fallback
from gpu_mft import cuda_backend
import numpy as np
import psutil

def main():
    run=PROJECT/'benchmark/runs/keck-benchmark-05'
    output=run/'t001-gpu-admission-resources.json'
    if output.exists(): raise FileExistsError('Preserve prior admission evidence')
    _,policy,record,config,state,_=c.check(run)
    started=time.perf_counter(); process=psutil.Process(); done=threading.Event()
    metrics={'cpu_seconds':0.,'peak_aggregate_rss_bytes':0,'status':'FAIL'}
    result={'stage':'GPU-admission','scope':'TRAIN references and synthetic optics only; arrays discarded',
            'source_hash':core.source_hash(),'config_hash':core.digest(config),
            'backend_sources':{p.resolve().relative_to(PROJECT).as_posix():c.sha(p) for p in
                               [Path(__file__),Path(g.__file__),Path(g.gpu_packets.__file__),
                                Path(__file__).with_name('gpu_backend.py'),Path(__file__).with_name('gpu_mft.py')]},
            'checks':[],'metrics':metrics}
    def resources():
        family=[process]+process.children(recursive=True)
        cpu=sum(sum(p.cpu_times()[:2]) for p in family if p.is_running())
        rss=sum(p.memory_info().rss for p in family if p.is_running())
        metrics['cpu_seconds']=max(metrics['cpu_seconds'],cpu)
        metrics['peak_aggregate_rss_bytes']=max(metrics['peak_aggregate_rss_bytes'],rss)
        if cpu>950 or rss>8*1024**3 or time.perf_counter()-started>780:
            raise TimeoutError('Bounded admission 1000CPU/8GiB/800wall guard')
    def watchdog():
        while not done.wait(1.):
            try: resources()
            except Exception as error:
                result.update(status='LIMIT',error=repr(error),cpu_seconds=metrics['cpu_seconds']+30,
                              elapsed_s=time.perf_counter()-started)
                core.save_json(output,result)
                for child in reversed(process.children(recursive=True)):
                    if child.is_running(): child.kill()
                os._exit(8)
    threading.Thread(target=watchdog,daemon=True).start()
    try:
        cp=cuda_backend(run/'gpu-cache')
        # New guards use a small synthetic model; native refinement evidence remains inherited.
        small={**config,'pupil_n':64}
        model=core.OpticalModel(small,basis=record['basis'])
        extension=attach_device_phase(model,cp,64)
        residual=np.zeros((10,64,64))
        try: model.probabilities([0]*4,residual.astype(complex))
        except ValueError: pass
        else: raise AssertionError('Complex residual accepted')
        try: attach_device_phase(model,cp,64)
        except ValueError: pass
        else: raise AssertionError('Repeated attachment accepted')
        model.static_nm[0,0]+=1
        try: model.image([0]*4)
        except ValueError: pass
        else: raise AssertionError('Mutated model accepted')
        model.static_nm[0,0]-=1
        reference=extension['native_probabilities']
        # Native probabilities dispatch through image: detach BOTH methods for reference/fallback.
        extension['detach'](); expected=model.probabilities([73,-42,61,-27],residual,(7,-11),1.1)
        extension=attach_device_phase(model,cp,64)
        actual=model.probabilities([73,-42,61,-27],residual,(7,-11),1.1)
        for a,b in zip(actual,expected): np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-15)
        def fail(*args): raise FloatingPointError('injected optical failure')
        model.probabilities=fail
        values,fallback=probabilities_with_fallback(model,extension,[73,-42,61,-27],residual,(7,-11),1.1)
        assert fallback=='FloatingPointError' and model.image==extension['native_image']
        for a,b in zip(values,expected): np.testing.assert_array_equal(a,b)
        for invalid in (np.full((2,16,16),np.nan),np.full((2,16,16),-1.)):
            try: validate_probabilities((np.zeros((1,16,16)),invalid))
            except FloatingPointError: pass
            else: raise AssertionError('Invalid probability accepted')
        result['checks']+=['complex input rejected','repeated attachment rejected','model mutation rejected',
                           'both methods restored before optical-only fallback','probability guards before RNG']
        # Equal-weight exposure averaging, with different phases in both pair halves.
        exposure_errors=[]
        for steps in (10,20):
            ramp=np.arange(steps)[:,None,None]*np.sin(np.arange(64)[None,None,:]/7.)*3.
            phases=np.broadcast_to(ramp,(steps,64,64)).copy()
            expected=model.probabilities([73,-42,61,-27],phases,(7,-11),1.1)
            attached=attach_device_phase(model,cp,64)
            actual=model.probabilities([73,-42,61,-27],phases,(7,-11),1.1)
            for a,b in zip(actual,expected): np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-15)
            exposure_errors.append(max(float(np.abs(a-b).sum()/np.abs(b).sum()) for a,b in zip(actual,expected)))
            attached['detach']()
        result['exposure_10_20_relative_l1']=exposure_errors
        # Signed noiseless recovery uses GPU values with the unchanged native derivatives.
        from scipy.optimize import least_squares
        attached=attach_device_phase(model,cp,64)
        recovery=[]
        for truth in ([73.,42.,-61.,27.],[-73.,-42.,61.,-27.]):
            target=model.image(truth,[0,200,0,0])
            fit=least_squares(lambda x:(model.image(x,[0,200,0,0])-target).ravel(),[0.]*4,
                              jac=lambda x:model.image(x,[0,200,0,0],derivatives=True)[1].reshape(4,-1).T,
                              xtol=1e-12,gtol=1e-13,ftol=1e-12,max_nfev=100)
            error=float(np.max(np.abs(fit.x-truth))); assert error<0.1,error
            recovery.append(error)
        attached['detach']()
        result['signed_recovery_max_error_nm']=recovery
        result['checks']+=['ten/twenty-frame exposure integration','signed noiseless GPU recovery with native Jacobian']
        # Refined image integration and signed/mixed forward agreement at distinct sampling settings.
        optical=[]
        for n,pixel,wavelength in ((64,4,3),(128,6,5),(config['pupil_n'],config['pixel_order'],config['wavelength_nodes'])):
            cfg={**config,'pupil_n':n,'pixel_order':pixel,'wavelength_nodes':wavelength}
            native=core.OpticalModel(cfg,basis=record['basis'])
            cases=[([73.,42.,-61.,27.],[0.,200.,0.,0.],(7.,-11.)),
                   ([-73.,-42.,61.,-27.],[200.,0.,0.,0.],(-7.,11.))]
            references=[native.image(c0,d0,centroid_mas=xy) for c0,d0,xy in cases]
            attach_device_phase(native,cp,64)
            errors=[]
            for (c0,d0,xy),expected in zip(cases,references):
                actual=native.image(c0,d0,centroid_mas=xy)
                error=float(np.abs(actual-expected).sum()/np.abs(expected).sum())
                assert error<1e-12
                assert abs(float(actual.sum()-expected.sum()))<1e-12
                errors.append(error)
            optical.append({'pupil_n':n,'pixel_order':pixel,'wavelength_samples':wavelength,'relative_l1':errors})
            resources()
        result['optical_agreement']=optical
        result['checks']+=['signed/mixed/centroid/diversity forward agreement','absolute crop flux',
                           'three pupil/pixel/wavelength sampling profiles']
        # Actual native ten-frame atmosphere, constant calibration and full RNG stream transition.
        completed=[]; canary_start=time.perf_counter(); jobs=[('train',i,0,20,time.time()+600) for i in range(4)]
        def accept(arrays,metadata):
            saved=sorted((s for s in state['shards'] if s['parent_id']==metadata['parent_id']),key=lambda s:s['acquisition_start'])
            assert [(s['acquisition_start'],s['acquisition_stop']) for s in saved]==[(0,5),(5,20)]
            for name in arrays:
                parts=[]
                for shard in saved:
                    with np.load(run/shard['path'],allow_pickle=False) as prior: parts.append(prior[name].copy())
                np.testing.assert_array_equal(arrays[name],np.concatenate(parts))
            completed.append(metadata['parent_id'])
            print('GPU canary bitwise TRAIN parent',metadata['parent_id'],flush=True)
        report=g.stream(config,record,jobs,4,600,accept,resources,run/'gpu-cache')
        assert len(completed)==4 and not report['fallbacks']
        result.update(real_parent_canary=report,completed_parents=completed,
                      full_parent_acquisitions_per_second=80/(time.perf_counter()-canary_start))
        result['checks']+=['four production full20 TRAIN parents bitwise match saved native5+15',
                           'native atmosphere/calibration/labels/three fluxes/detector unchanged',
                           'bounded real multiprocessing transport and verified worker teardown']
        result.update(status='PASS')
    except Exception as error:
        result.update(status='FAIL',error=repr(error))
        raise
    finally:
        done.set(); resources()
        result.update(cpu_seconds=metrics['cpu_seconds']+100,elapsed_s=time.perf_counter()-started,
                      metrics=metrics,accounting='measured live process-family CPU plus100s final ticks/shutdown/import allowance')
        core.save_json(output,result)
        print(json.dumps(result),flush=True)

if __name__=='__main__': main()
