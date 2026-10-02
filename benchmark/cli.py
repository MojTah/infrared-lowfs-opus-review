"""One execution path: validate -> canary -> generate -> train -> evaluate -> OOPAO."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from datetime import datetime, timezone
import importlib.metadata
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
if __package__:
    from .core import (DEFAULT_CONFIG, ROOT, MODES, LIMITS, OpticalModel, ResidualSequence,
                       calibrate_residual_gain, detector_read, digest, file_hash, source_hash,
                       save_json, physical_fit, linearized_fit, HCIPY_RUNTIME)
else:
    from core import (DEFAULT_CONFIG, ROOT, MODES, LIMITS, OpticalModel, ResidualSequence,
                      calibrate_residual_gain, detector_read, digest, file_hash, source_hash,
                      save_json, physical_fit, linearized_fit, HCIPY_RUNTIME)
import numpy as np
import psutil
SPLITS=('train','validation','calibration','test','shifted')

def now():
    return datetime.now(timezone.utc).isoformat()

def load_config(path):
    return json.loads(Path(path).read_text()) if path else json.loads(json.dumps(DEFAULT_CONFIG))

def run_guard(start,seconds):
    if time.perf_counter()-start>seconds:
        raise TimeoutError('bounded run time reached')
    if psutil.Process().memory_info().rss>DEFAULT_CONFIG['budgets']['ram_bytes']:
        raise MemoryError('16GiB RAM ceiling')

def detector_checks():
    records=[]; rng=np.random.default_rng(971328); count=60000
    for mu in (0.,.05,1.,100.):
        for reads in (1,2):
            for noise in (0.,5.):
                total=sum(detector_read(np.full((count,1),mu/reads),rng,5.01/reads,noise) for _ in range(reads))
                expectation=mu+5.01; variance=expectation+reads*noise**2
                sem=np.sqrt(variance/count)
                sev=np.sqrt((expectation+2*count/(count-1)*variance**2)/count)
                mean=float(total.mean()); var=float(total.var(ddof=1))
                records.append({'source_e':mu,'reads':reads,'noise_e':noise,
                                'mean':mean,'variance':var,'expected_mean':expectation,'expected_variance':variance,
                                'pass':abs(mean-expectation)<8*sem and abs(var-variance)<8*sev})
    negatives=detector_read(np.zeros((10000,1)),rng,0.,5.)
    invalid=0
    for source in (np.array([[-1.]]),np.array([[np.nan]]),np.array([[1+0j]])):
        try: detector_read(source,rng,0,5)
        except ValueError: invalid+=1
    return {'records':records,'negative_reads_retained':bool((negatives<0).any()),'invalid_rejections':invalid,
            'pass':all(r['pass'] for r in records) and (negatives<0).any().item() and invalid==3}

def observability(model, states, start, max_seconds):
    """Finite signed/mixed-mode search; nuisance profiles use only observable counts."""
    rng=np.random.default_rng(193287)
    bank=np.vstack([np.zeros(4),states,-np.asarray(states),rng.uniform(-LIMITS,LIMITS,(16,4))])
    records=[]; response=[]
    for design in ('single','pair'):
        ds=[np.array([0.,200.,0.,0.])] if design=='single' else [np.zeros(4),np.array([200.,0.,0.,0.])]
        photons=100000/len(ds)
        for a in states:
            images=[];columns=[];nuisances=[]
            for d in ds:
                p,j=model.image(a,d,centroid_mas=(3.,-7.),derivatives='nuisance')
                y=photons*p+5.01/len(ds);root=np.sqrt(y+25.)
                images.append(y)
                columns.append(photons*j[:4].reshape(4,-1).T/root.ravel()[:,None])
                nuisances.append(np.c_[photons*j[4:].reshape(2,-1).T,p.ravel(),np.ones(256)]/root.ravel()[:,None])
            modal=np.vstack(columns)
            nuisance=np.zeros((256*len(ds),2+2*len(ds)))
            for ch,n in enumerate(nuisances):
                nuisance[ch*256:(ch+1)*256,:2]=n[:,:2]
                nuisance[ch*256:(ch+1)*256,2+2*ch:4+2*ch]=n[:,2:]
            q=np.linalg.qr(nuisance)[0]; projected=modal-q@(q.T@modal)
            sv=np.linalg.svd(projected,compute_uv=False)
            response.append({'design':design,'truth_nm':a.tolist(),'noise_weighted_singular_values_per_nm':sv.tolist(),
                             'condition':float(sv[0]/sv[-1]),'profiled':'centroid, flux and background'})
            trials=[]
            for b in bank:
                separation=float(np.linalg.norm(b-a))
                if separation<20.:continue
                centroid=np.zeros(2)
                for iteration in range(3):
                    jacobians=[];differences=[];distance=0.
                    for d,y in zip(ds,images):
                        p,j=model.image(b,d,centroid_mas=centroid,derivatives='centroid')
                        root=np.sqrt(y.ravel()+25.);nuis=np.c_[p.ravel(),np.ones(256)]/root[:,None]
                        flux,bg=np.linalg.lstsq(nuis,y.ravel()/root,rcond=None)[0];flux=max(flux,0.)
                        r=(y-flux*p-bg).ravel()/root;qn=np.linalg.qr(nuis)[0]
                        jac=flux*j.reshape(2,-1).T/root[:,None]
                        jacobians.append(jac-qn@(qn.T@jac));differences.append(r-qn@(qn.T@r))
                        distance+=float(r@r)
                    if iteration<2:
                        step=np.linalg.lstsq(np.vstack(jacobians),np.concatenate(differences),rcond=None)[0]
                        centroid=np.clip(centroid+np.clip(step,-10.,10.),-30.,30.)
                trials.append({'candidate_nm':b.tolist(),'wavefront_separation_nm':separation,
                               'profiled_distance_sigma':float(np.sqrt(distance)),'centroid_mas':centroid.tolist(),
                               'ambiguous_at_100k_e':bool(distance<16.)})
                run_guard(start,max_seconds)
            trials.sort(key=lambda x:x['profiled_distance_sigma'])
            records.append({'design':design,'truth_nm':a.tolist(),'searched_candidates':len(trials),'nearest':trials[:3],
                            'ambiguous_candidates':[t for t in trials if t['ambiguous_at_100k_e']]})
    return {'response':response,'finite_search':records,'bank_nm':bank.tolist(),
            'limitation':'finite candidate search with two centroid updates; no proof of global uniqueness or low-flux distinguishability'}

def gate(run_dir, config, max_seconds=1800):
    run_dir=Path(run_dir); run_dir.mkdir(parents=True,exist_ok=True)
    if (run_dir/'readiness.json').exists(): raise FileExistsError('readiness already exists; use a fresh run')
    start=time.perf_counter();cpu_start=time.process_time(); records={}; passes=[]
    print('Physics gate: fixed actual-pupil basis and independent propagation',flush=True)
    model=OpticalModel(config)
    states=[np.array([73.,42.,-61.,27.]),np.array([-137.,-44.,39.,-50.]),np.array([149.,99.,-99.,97.])]
    diversity=np.array([0.,200.,0.,0.])
    independent=[]
    for a in states[:2]:
        p=model.image(a,diversity,centroid_mas=(3.,-7.))
        reference=model.independent_image(a,diversity,centroid_mas=(3.,-7.))
        error=float(abs(p-reference).sum()/reference.sum()); independent.append(error)
        run_guard(start,max_seconds)
    records['independent_propagation_l1']=independent
    passes.append(max(independent)<1e-10)
    print('Physics gate: pupil resolution 256/512/1024 and fixed basis',flush=True)
    convergence=[]; chosen=None; refined=None
    for n in (config['pupil_n'],config['pupil_n']*2):
        coarse=OpticalModel(config,pupil_n=n,basis=model.basis)
        fine=OpticalModel(config,pupil_n=n*2,basis=model.basis)
        errors=[]
        for a in states:
            pc=coarse.image(a,diversity); pf=fine.image(a,diversity)
            errors.append(float(abs(pc-pf).sum()/pf.sum()))
        convergence.append({'coarse_n':n,'fine_n':n*2,'l1':errors})
        print('  pupil',n,'->',n*2,'maxL1',max(errors),flush=True)
        run_guard(start,max_seconds)
        if max(errors)<config['thresholds']['image_l1']:
            chosen=coarse; refined=fine; break
    records['pupil_refinement']=convergence; passes.append(chosen is not None)
    if chosen is None:
        records['status']='FAIL'; records['blocked']='pupil boundary convergence; no dataset or training admitted'
        records['config_hash']=digest(config); records['source_hash']=source_hash();records['elapsed_s']=time.perf_counter()-start
        save_json(run_dir/'readiness.json',records); return records
    model=chosen; config['pupil_n']=model.n
    p=model.image(states[0],diversity)
    pixel=OpticalModel(config,pixel_order=config['pixel_order']*2,basis=model.basis)
    spectral=OpticalModel(config,wavelength_nodes=config['wavelength_nodes']*2,basis=model.basis)
    pixel_error=float(abs(p-pixel.image(states[0],diversity)).sum()/p.sum())
    spectral_error=float(abs(p-spectral.image(states[0],diversity)).sum()/p.sum())
    records['pixel_refinement_l1']=pixel_error;records['spectral_refinement_l1']=spectral_error
    passes.extend([pixel_error<1e-3,spectral_error<1e-3])
    refined=OpticalModel(config,pupil_n=model.n*2,pixel_order=model.pixel_order*2,
                         wavelength_nodes=config['wavelength_nodes']*2,basis=model.basis)
    joint_refinement=[]
    for a in states:
        for d in (diversity,np.zeros(4),np.array([200.,0.,0.,0.])):
            coarse_image=model.image(a,d,centroid_mas=(3.,-7.))
            fine_image=refined.image(a,d,centroid_mas=(3.,-7.))
            error=float(abs(coarse_image-fine_image).sum()/fine_image.sum())
            joint_refinement.append({'truth_nm':a.tolist(),'diversity_nm':d.tolist(),'relative_l1':error})
            run_guard(start,max_seconds)
    records['joint_pupil_pixel_spectral_refinement']=joint_refinement
    passes.append(max(r['relative_l1'] for r in joint_refinement)<config['thresholds']['image_l1'])
    print('Physics gate: modal derivatives and independent noiseless recovery',flush=True)
    p,j=model.image(states[0],diversity,derivatives=True);derivative=[]
    for k in range(4):
        da=np.zeros(4);da[k]=.125
        finite=(model.image(states[0]+da,diversity)-model.image(states[0]-da,diversity))/.25
        derivative.append(float(np.linalg.norm(finite-j[k])/np.linalg.norm(j[k])))
    records['derivative_relative_l2']=derivative;passes.append(max(derivative)<1e-5)
    recoveries=[];blind=[]
    for design in ('single','pair'):
        for a in states:
            ds=[diversity] if design=='single' else [np.zeros(4),np.array([200.,0.,0.,0.])]
            truth=np.array([refined.image(a,d,centroid_mas=(3.,-7.)) for d in ds])*100000/len(ds)+5.01/len(ds)
            result=physical_fit(model,truth,design,initial=a*.8,max_nfev=45)
            error=np.array(result['coeff_nm'])-a
            recoveries.append({'design':design,'truth_nm':a.tolist(),'error_nm':error.tolist(),'status':result['status']})
            passes.append(np.max(abs(error))<config['thresholds']['coefficient_refinement_nm'])
            captured=physical_fit(model,truth,design,initial=None,max_nfev=45)
            captured_error=np.array(captured['coeff_nm'])-a
            blind.append({'design':design,'truth_nm':a.tolist(),'error_nm':captured_error.tolist(),'status':captured['status'],
                          'initialization':'three fixed starts, no truth'})
            passes.append(np.max(abs(captured_error))<config['thresholds']['coefficient_refinement_nm'])
            run_guard(start,max_seconds)
    records['noiseless_reference_recovery']=recoveries;records['blind_signed_recovery']=blind
    errors=np.array([r['error_nm'] for r in recoveries])
    rmse=np.sqrt(np.mean(errors**2,axis=0));records['noiseless_rmse_nm']=rmse.tolist();passes.append(max(rmse)<2.)
    print('Physics gate: finite mixed/sign search after nuisance profiling',flush=True)
    records['observability']=observability(model,states,start,max_seconds)
    passes.append(all(r['noise_weighted_singular_values_per_nm'][-1]>1e-6 for r in records['observability']['response']))
    # Negative control outside the declared capture range: exact even-phase conjugacy.
    saved_ncpa=model.static_nm.copy();model.static_nm[:]=0
    a=states[0];conjugate=-a-2*diversity
    p=model.image(a,diversity);q=model.image(conjugate,diversity)
    records['outside_range_conjugacy_l1']=float(abs(p-q).sum()/p.sum())
    records['outside_range_conjugate_nm']=conjugate.tolist();model.static_nm=saved_ncpa
    records['observability_limit']='single-PSF signs depend on calibrated diversity and bounded capture range; no global uniqueness claim'
    print('Physics gate: electron moments, temporal integration and residual statistics',flush=True)
    records['detector']=detector_checks();passes.append(records['detector']['pass'])
    gain,cal=calibrate_residual_gain(model);shift_gain,shiftcal=calibrate_residual_gain(model,True)
    records['residual_gain']={'nominal_per_nm':gain,'shifted_per_nm':shift_gain,'nominal':cal,'shifted':shiftcal}
    sequence=ResidualSequence(model,917381,gain*200.)
    normal=sequence.acquisition(0);fine_time=sequence.acquisition(0,steps=config['phase_steps']*2)
    sn,pn=model.probabilities(states[0],normal);sf,pf=model.probabilities(states[0],fine_time)
    temporal=[float(abs(x-y).sum()/y.sum()) for x,y in zip((sn,pn),(sf,pf))]
    records['temporal_refinement_l1']=temporal;passes.append(max(temporal)<1e-3)
    first=normal[0].ravel();second=normal[1].ravel()
    norms=[float(np.sqrt(model.weights@r.ravel()**2)) for r in normal]
    corr=float((model.weights*first)@second/np.sqrt((model.weights@first**2)*(model.weights@second**2)))
    columns=np.column_stack([np.ones(len(first)),model.q_all])
    projection=float(np.max(abs(columns.T@(model.weights*first))))
    records['residual_statistics']={'sample_rms_nm':norms,'adjacent_weighted_correlation':corr,
                                  'max_controlled_projection_nm':projection,'not_measured_AO':True}
    passes.extend([projection<1e-6,np.isfinite(corr) and 0<corr<=1+1e-12])
    print('Physics gate: identical high-order screens across pupil/pixel/spectral refinement',flush=True)
    disturbed=[]
    for shifted,rms,seed in ((False,200.,718731),(True,300.,882371)):
        factor=(shift_gain if shifted else gain)*rms
        coarse_residual=ResidualSequence(model,seed,factor,shifted).at(.017)+model.calibration_error(seed+71)
        fine_residual=ResidualSequence(refined,seed,factor,shifted).at(.017)+refined.calibration_error(seed+71)
        for d in (diversity,np.zeros(4),np.array([200.,0.,0.,0.])):
            p=model.image(states[0],d,coarse_residual,centroid_mas=(3.,-7.))
            q=refined.image(states[0],d,fine_residual,centroid_mas=(3.,-7.))
            l1=float(abs(p-q).sum()/q.sum())
            disturbed.append({'residual_nm':rms,'shifted_filter':shifted,'seed':seed,'relative_l1':l1,'diversity_nm':d.tolist()})
            run_guard(start,max_seconds)
    records['disturbed_joint_refinement']=disturbed
    passes.append(max(r['relative_l1'] for r in disturbed)<config['thresholds']['image_l1'])
    records.update({'schema_version':1,'status':'PASS' if all(passes) else 'FAIL','checks_pass':all(passes),
                    'source_hash':source_hash(),'config_hash':digest(config),'measurement_hash':model.measurement_hash,
                    'basis':model.basis,'config':config,'created_utc':now(),'elapsed_s':time.perf_counter()-start,
                    'cpu_seconds':time.process_time()-cpu_start,
                    'hcipy_configuration':HCIPY_RUNTIME,'user_overrides_disabled':True,'blas_threads':1,
                    'pid':os.getpid(),'current_rss_bytes':psutil.Process().memory_info().rss,
                    'peak_rss_bytes':getattr(psutil.Process().memory_info(),'peak_wset',None),
                    'versions':{p:importlib.metadata.version(p) for p in ('hcipy','numpy','scipy','torch')},
                    'limitations':['source-backed Keck pupil/atmosphere with explicitly filtered AO surrogate',
                        'same-grid direct propagation is independent numerics; actual-pupil convergence tested separately',
                        'scalar phase-only propagation; no measured detector or telescope controller',
                        'recoverability controls finite, not proof of global uniqueness']})
    save_json(run_dir/'config.json',config)
    save_json(run_dir/'readiness.json',records)
    print('Physics readiness',records['status'],'elapsed',records['elapsed_s'],flush=True)
    return records

def admitted(run_dir):
    run_dir=Path(run_dir)
    record=json.loads((run_dir/'readiness.json').read_text())
    config=json.loads((run_dir/'config.json').read_text())
    if record['status']!='PASS' or record['source_hash']!=source_hash() or record['config_hash']!=digest(config):
        raise ValueError('physics readiness is absent, failed, or source/configuration changed')
    return record,config

def canary(run_dir,count=100,max_seconds=1800):
    start=time.perf_counter();cpu_start=time.process_time();record,config=admitted(run_dir)
    model=OpticalModel(config,basis=record['basis']);rng=np.random.default_rng(238914)
    gain=record['residual_gain']['nominal_per_nm']*200.
    sequence=ResidualSequence(model,117891,gain)
    static_error=model.calibration_error(128781)
    for i in range(count):
        coefficients=rng.uniform(-LIMITS,LIMITS)
        acquisition=model.acquire(coefficients,sequence.acquisition(i)+static_error,rng,10000)
        if not all(np.isfinite(acquisition[k]).all() for k in ('images_single','images_pair')):
            raise FloatingPointError('nonfinite canary output')
        if i%10==0:print('canary',i,'/',count,'elapsed',round(time.perf_counter()-start,2),flush=True)
        run_guard(start,max_seconds)
    elapsed=time.perf_counter()-start
    cpu_seconds=time.process_time()-cpu_start;seconds_per=cpu_seconds/count
    parents=sum(config['parent_counts'].values())
    # Reserve 20% CPU budget for independent validation and classical comparisons.
    choice=next((n for n in (20,10,5) if seconds_per*parents*n<config['budgets']['cpu_seconds']*.8),None)
    result={'status':'PASS' if choice else 'RESOURCE_LIMIT','acquisitions':count,'elapsed_s':elapsed,
            'seconds_per_acquisition':seconds_per,'chosen_acquisitions_per_parent':choice,
            'cpu_seconds':cpu_seconds,'wall_seconds_per_acquisition':elapsed/count,
            'generation_workers':config['generation_workers'],
            'projected_generation_wall_s':None if choice is None else elapsed/count*parents*choice/config['generation_workers'],
            'projected_generator_cpu_seconds':None if choice is None else seconds_per*parents*choice,
            'source_hash':source_hash(),'config_hash':digest(config),'measurement_hash':model.measurement_hash,
            'includes_single_and_pair':True,'flux_variants_reuse_noiseless_exposure':'generation draws independent counts, no extra propagation',
            'switching_readout_overhead_s':None,'no_real_sensor_latency_claim':True}
    save_json(Path(run_dir)/'canary.json',result)
    return result

_WORKER_MODEL=None
_WORKER_RECORD=None
_WORKER_STOP=None

def init_worker(config,record,stop_event):
    global _WORKER_MODEL,_WORKER_RECORD,_WORKER_STOP
    if source_hash()!=record['source_hash']:raise ValueError('worker source identity changed')
    _WORKER_MODEL=OpticalModel(config,basis=record['basis']);_WORKER_RECORD=record;_WORKER_STOP=stop_event

def generate_parent(job):
    """Independent seeded parent; only the execution owner writes output files."""
    split,parent,acquisitions,deadline=job
    model=_WORKER_MODEL;record=_WORKER_RECORD;config=model.config
    cpu_start=time.process_time()
    split_index=SPLITS.index(split)
    def seed(tag):
        return int(np.random.SeedSequence([config['seed'],split_index,parent,tag]).generate_state(1)[0])
    label_seed,noise_seed,phase_seed,calibration_seed=(seed(k) for k in (991,227,613,449))
    labels=np.random.default_rng(label_seed);noise=np.random.default_rng(noise_seed)
    shifted=split=='shifted';rms=config['shifted_residual_nm'] if shifted else config['residual_rms_nm'][parent%2]
    gain=record['residual_gain']['shifted_per_nm' if shifted else 'nominal_per_nm']*rms
    sequence=ResidualSequence(model,phase_seed,gain,shifted)
    static_error=model.calibration_error(calibration_seed)
    rows=[]
    for index in range(acquisitions):
        if _WORKER_STOP.is_set() or time.time()>deadline:raise TimeoutError('parent generation deadline')
        truth=labels.uniform(-LIMITS,LIMITS);centroid=labels.uniform(-20,20,2)
        single,pair=model.probabilities(truth,sequence.acquisition(index)+static_error,centroid,1.1 if shifted else 1.)
        for flux in config['flux_e']:
            rows.append({'images_single':detector_read(single*flux,noise,config['background_e']+config['dark_e'],config['read_noise_e']).astype('float32'),
                         'images_pair':detector_read(pair*flux/2,noise,(config['background_e']+config['dark_e'])/2,config['read_noise_e']).astype('float32'),
                         'labels_nm':truth,'flux_e':flux})
        if psutil.Process().memory_info().rss>config['budgets']['ram_bytes']/config['generation_workers']:
            raise MemoryError('per-worker RAM admission exceeded')
    arrays={k:np.array([row[k] for row in rows]) for k in rows[0]}
    metadata={'split':split,'parent_id':f'{split}-{parent:04d}','rows':len(rows),
              'phase_seed':phase_seed,'label_seed':label_seed,'noise_seed':noise_seed,'calibration_seed':calibration_seed,
              'calibration_error_rms_nm':float(np.sqrt(model.weights@static_error.ravel()**2)),
              'residual_target_ensemble_rms_nm':rms,'condition':'joint_residual_and_diversity_shift' if shifted else 'nominal',
              'worker_cpu_seconds':time.process_time()-cpu_start}
    return arrays,metadata

def generate(run_dir,max_seconds=57600):
    start=time.perf_counter();cpu_start=time.process_time();run_dir=Path(run_dir);record,config=admitted(run_dir)
    canary_record=json.loads((run_dir/'canary.json').read_text())
    if canary_record['status']!='PASS' or canary_record['source_hash']!=source_hash() or canary_record['config_hash']!=digest(config) or canary_record['acquisitions']<100:
        raise ValueError('100-acquisition canary identity not admitted')
    acquisitions=canary_record['chosen_acquisitions_per_parent']
    dataset={'schema_version':1,'source_hash':source_hash(),'config_hash':digest(config),'measurement_hash':record['measurement_hash'],
             'basis':record['basis'],'mode_order':MODES,'label_units':'nm OPD RMS',
             'acquisitions_per_parent':acquisitions,'shards':[],'created_utc':now(),'known_system':config['case'],
             'truth_access':'offline labels only; no residual/nuisance truth in inference inputs'}
    if (run_dir/'dataset.json').exists():raise FileExistsError('completed dataset exists')
    state_path=run_dir/'generation-state.json'
    if state_path.exists():
        old=json.loads(state_path.read_text())
        if old['source_hash']!=dataset['source_hash'] or old['config_hash']!=dataset['config_hash']:raise ValueError('resume identity changed')
        dataset['shards']=old['shards']
        dataset['cpu_seconds']=old.get('cpu_seconds',0.)
        dataset['elapsed_s']=old.get('elapsed_s',0.)
    complete={r['parent_id']:r for r in dataset['shards']}
    jobs=[]
    for split in SPLITS:
        count=config['parent_counts'][split]
        for parent in range(count):
            key=f'{split}-{parent:04d}'
            if key in complete:
                if file_hash(run_dir/complete[key]['path'])!=complete[key]['sha256']:raise ValueError('resume shard hash changed')
            else: jobs.append((split,parent,acquisitions,time.time()+max_seconds))
    # ponytail: four persistent independent parents; one writer owns every shard/state mutation.
    workers=config['generation_workers'];previous_cpu=dataset.get('cpu_seconds',0.);previous_wall=dataset.get('elapsed_s',0.)
    cpu_used=0.;pending={};job_iter=iter(jobs);stopped=True
    def checkpoint(status):
        dataset['shards'].sort(key=lambda r:(SPLITS.index(r['split']),r['parent_id']))
        save_json(state_path,{**dataset,'status':status,'cpu_seconds':previous_cpu+cpu_used,
                             'elapsed_s':previous_wall+time.perf_counter()-start},overwrite=state_path.exists())
    context=multiprocessing.get_context('spawn');stop_event=context.Event()
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=context,initializer=init_worker,initargs=(config,record,stop_event))
    try:
        def submit():
            job=next(job_iter,None)
            if job is not None: pending[pool.submit(generate_parent,job)]=job
        for _ in range(workers): submit()
        while pending:
            children=psutil.Process().children(recursive=True)
            cpu_used=time.process_time()-cpu_start+sum(p.cpu_times().user+p.cpu_times().system for p in children)
            if previous_cpu+cpu_used+record.get('cpu_seconds',0)+canary_record.get('cpu_seconds',0)>config['budgets']['cpu_seconds']-(workers*35+30):
                raise TimeoutError('aggregate 16 CPU-hour ceiling approached')
            if psutil.Process().memory_info().rss+sum(p.memory_info().rss for p in children)>config['budgets']['ram_bytes']:
                raise MemoryError('aggregate 16GiB RAM ceiling')
            run_guard(start,max_seconds)
            done,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
            for future in done:
                arrays,metadata=future.result();pending.pop(future)
                shard=run_dir/'data'/f"{metadata['parent_id']}.npz";shard.parent.mkdir(exist_ok=True)
                temporary=shard.with_suffix('.partial.npz')
                if shard.exists() or temporary.exists():raise FileExistsError('unindexed or partial shard; inspect before resume')
                np.savez_compressed(temporary,**arrays);os.replace(temporary,shard)
                dataset['shards'].append({**metadata,'path':shard.relative_to(run_dir).as_posix(),'sha256':file_hash(shard)})
                checkpoint('IN_PROGRESS')
                print('generation',len(dataset['shards']),'/',sum(config['parent_counts'].values()),
                      'elapsed',round(time.perf_counter()-start),'cpu_s',round(previous_cpu+cpu_used),flush=True)
                retained=sum(p.stat().st_size for p in (run_dir/'data').glob('*.npz'))
                if retained>config['budgets']['data_bytes']:raise MemoryError('4GiB retained data ceiling')
                submit()
        stopped=False
    finally:
        stop_started=time.perf_counter()
        if stopped:
            stop_event.set()
            for future in pending: future.cancel()
            owned=psutil.Process().children(recursive=True)
            pool.shutdown(wait=False,cancel_futures=True)
            _,alive=psutil.wait_procs(owned,timeout=30)
            if alive:
                # Only this owner's read-only workers are killed; completed shards remain intact.
                for process in reversed(alive):
                    if process.is_running():process.kill()
                _,alive=psutil.wait_procs(alive,timeout=5)
                if alive:raise RuntimeError('owned worker stop could not be verified')
        else:
            pool.shutdown(wait=True,cancel_futures=True)
        if stopped:
            cpu_used+=(time.perf_counter()-stop_started)*(workers+1)
            checkpoint('STOPPED_AT_SAFE_PARENT_BOUNDARY')
    dataset['elapsed_s']=previous_wall+time.perf_counter()-start;dataset['cpu_seconds']=previous_cpu+cpu_used;dataset['status']='COMPLETE'
    save_json(run_dir/'dataset.json',dataset)
    return {'status':'COMPLETE','parents':len(dataset['shards']),'rows':sum(s['rows'] for s in dataset['shards']),
            'elapsed_s':dataset['elapsed_s'],'data_bytes':sum(p.stat().st_size for p in (run_dir/'data').glob('*.npz'))}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['configure','gate','canary','generate','train','evaluate','compare','challenges','oopao','telescopes'])
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--max-seconds',type=float)
    parser.add_argument('--count',type=int,default=100)
    args=parser.parse_args()
    if args.max_seconds is not None and (not np.isfinite(args.max_seconds) or args.max_seconds<=0):parser.error('positive finite deadline required')
    path=args.run.resolve()
    if not path.is_relative_to(ROOT/'runs'):parser.error('run must be inside benchmark/runs')
    started=time.perf_counter()
    try:
        if args.stage in ('configure','gate'):
            hardware=importlib.import_module(f'{__package__}.hardware' if __package__ else 'hardware')
            config,report=hardware.configure(load_config(args.config))
            report.update({'source_hash':source_hash(),'config_hash':digest(config),'created_utc':now()})
            save_json(path/'hardware.json',report)
            if report['status']!='PASS':result=report
            elif args.stage=='configure':
                save_json(path/'auto-config.json',config)
                result=report
            else:result=gate(path,config,args.max_seconds or 2400)
        elif args.stage=='canary':result=canary(path,args.count,args.max_seconds or 1800)
        elif args.stage=='generate':result=generate(path,args.max_seconds or 57600)
        elif args.stage=='train':
            admitted(path)
            learning=importlib.import_module(f'{__package__}.learning' if __package__ else 'learning')
            result=learning.train(path,max_gpu_seconds=args.max_seconds or 7200)
        elif args.stage=='evaluate':
            admitted(path)
            learning=importlib.import_module(f'{__package__}.learning' if __package__ else 'learning')
            result=learning.evaluate(path,timing_calls=1000)
            save_json(path/'evaluation.json',result)
        elif args.stage in ('compare','challenges'):
            admitted(path)
            comparisons=importlib.import_module(f'{__package__}.comparisons' if __package__ else 'comparisons')
            result=(comparisons.compare if args.stage=='compare' else comparisons.challenges)(path,max_seconds=args.max_seconds or 1800,parent_count=args.count)
        elif args.stage=='telescopes':
            large_telescopes=importlib.import_module(f'{__package__}.large_telescopes' if __package__ else 'large_telescopes')
            result=large_telescopes.simulate(path,max_seconds=args.max_seconds or 1800,count=args.count)
            save_json(path/'large_telescopes.json',result)
        else:
            cross_validate=importlib.import_module(f'{__package__}.cross_validate' if __package__ else 'cross_validate')
            result=cross_validate.validate(path,max_seconds=args.max_seconds or 1800,parent_count=args.count,acquisitions=5)
        print(json.dumps({'stage':args.stage,'status':result.get('status','COMPLETE'),
                          'artifact_directory':str(path),'elapsed_s':time.perf_counter()-started},allow_nan=False),flush=True)
        return 0 if result.get('status') not in ('FAIL','RESOURCE_LIMIT','PARTIAL') else 2
    except (ValueError,FileExistsError,TimeoutError,MemoryError,RuntimeError) as error:
        print(json.dumps({'status':'ERROR','stage':args.stage,'type':type(error).__name__,'message':str(error),'elapsed_s':time.perf_counter()-started}),flush=True)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
