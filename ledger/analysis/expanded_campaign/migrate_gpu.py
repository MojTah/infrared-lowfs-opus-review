"""Stopped-owner GPU policy handoff; preserve every existing array and old identity."""
import importlib
from contextlib import contextmanager
from pathlib import Path
import shutil
import sys
PROJECT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT))
import campaign as c
from benchmark import core,cli

GPU_NAMES=('gpu_campaign.py','gpu_packets.py','gpu_backend.py','gpu_mft.py')

@contextmanager
def transition(run,history,source):
    """Rollback only this owned controller/policy transition; retain attempted files."""
    try:
        yield
    except BaseException:
        shutil.copyfile(source,history/'failed-campaign.py.txt')
        shutil.copyfile(history/'campaign.py.txt',source)
        for name in ('execution-policy.json','driver-admission.json','generation-state.json'):
            if (run/name).exists(): shutil.copyfile(run/name,history/('failed-'+name))
            partial=(run/name).with_suffix('.json.partial')
            if partial.exists(): shutil.move(str(partial),str(history/('interrupted-'+name+'.txt')))
            shutil.copyfile(history/name,run/name)
        raise

if __name__=='__main__':
    run=PROJECT/'benchmark/runs/keck-benchmark-05'
    if (run/'active-owner.json').exists(): raise RuntimeError('Verified stopped production owner required')
    _,policy,record,config,state,groups=c.check(run)
    if state['status']!='STOPPED_AT_SAFE_PARENT_BOUNDARY': raise ValueError('Stopped checkpoint required')
    sources={f'ledger/analysis/expanded_campaign/{name}':c.sha(Path(__file__).with_name(name)) for name in GPU_NAMES}
    names=['t001-gpu-admission-resources.json','t001-gpu-lifecycle-resources.json']
    evidence={name:c.read(run/name) for name in names}
    profile=c.read(run/'t001-gpu-eight-worker-resources.json')
    for name,item in {**evidence,'eight-worker-profile':profile}.items():
        if item['status']!='PASS' or item['source_hash']!=core.source_hash():
            raise ValueError('GPU admission/profile/lifecycle must PASS with frozen engine')
        if name!='t001-gpu-lifecycle-resources.json' and item['config_hash']!=core.digest(config):
            raise ValueError('GPU admission/profile configuration changed')
        if any(item['backend_sources'].get(path)!=digest for path,digest in sources.items()):
            raise ValueError('Admitted GPU backend changed')
    speed4=evidence[names[0]]['full_parent_acquisitions_per_second']
    speed8=profile['acquisitions_per_second']
    workers=8 if speed8>speed4*1.1 and profile['peak_aggregate_rss_bytes']<policy['ram_ceiling_bytes'] else 4
    allowances={'renewed_GPU_staging_parent':120,'training_preparation_parent':60,
                'GPU_packet_development_parent':30,'GPU_controller_development':200}
    remaining=12400-sum(s['rows']//3 for s in state['shards'])
    sample=evidence[names[0]] if workers==4 else profile
    sample_count=80 if workers==4 else 160
    replay=sum(v==set(range(5)) for v in groups.values())*5*3.5
    projected=c.spent_cpu(run,policy,state)+sum(allowances.values())+remaining*sample['cpu_seconds']/sample_count+replay
    if projected>=c.CPU_LIMIT-policy['downstream_cpu_reserve_seconds']:
        raise TimeoutError('GPU projection does not fit approved CPU ceiling')
    names.append('t001-gpu-eight-worker-resources.json')
    attempt=1
    while (run/f'policy-history/native-cpu-{attempt:03d}').exists(): attempt+=1
    history=run/f'policy-history/native-cpu-{attempt:03d}'; history.mkdir(parents=True,exist_ok=False)
    for name in ('execution-policy.json','driver-admission.json','generation-state.json'):
        shutil.copyfile(run/name,history/name)
    source=Path(c.__file__).resolve(); shutil.copyfile(source,history/'campaign.py.txt')
    core.save_json(history/'recovery-manifest.json',{'controller_path':str(source),
                   'backups':{name:c.sha(history/name) for name in
                              ('campaign.py.txt','execution-policy.json','driver-admission.json','generation-state.json')}})
    binding=run/'t001-gpu-lifecycle-input-binding.json'
    if binding.exists(): shutil.copyfile(binding,history/'prior-lifecycle-input-binding.json')
    core.save_json(binding,{'status':'PASS',
                   'evidence_level':'unchanged input/source binding, separate from original runtime result',
                   'original_lifecycle_result_sha256':c.sha(run/'t001-gpu-lifecycle-resources.json'),
                   'lifecycle_script_sha256':c.sha(Path(__file__).with_name('gpu_lifecycle.py')),
                   'config_hash':core.digest(config),'config_file_sha256':c.sha(run/'config.json'),
                   'inherited_policy_sha256':c.sha(run/'execution-policy.json'),
                   'basis':'Executed lifecycle script obtains its config from c.check(run). Immutable inherited files and policy remained unchanged before/after execution.'},overwrite=True)
    names.append('t001-gpu-lifecycle-input-binding.json')
    old_sha=c.sha(run/'execution-policy.json')
    text=source.read_text(encoding='utf-8')
    needle="    for name, expected in policy['inherited_files'].items():"
    insertion='''    if policy.get('backend') is not None:
        if policy['backend'] != 'resident_phase_fp64_native_hcipy_mft':
            raise ValueError('Unrecognized optical backend')
        expected_backend = {f'ledger/analysis/expanded_campaign/{name}' for name in
                            ('gpu_campaign.py','gpu_packets.py','gpu_backend.py','gpu_mft.py')}
        if set(policy['backend_sources']) != expected_backend:
            raise ValueError('GPU backend source list changed')
        for path, expected in policy['backend_sources'].items():
            if sha(PROJECT/path) != expected: raise ValueError('Admitted GPU backend source changed')
        if set(policy['gpu_evidence']) != {'t001-gpu-admission-resources.json',
                                          't001-gpu-eight-worker-resources.json','t001-gpu-lifecycle-resources.json',
                                          't001-gpu-lifecycle-input-binding.json'}:
            raise ValueError('GPU admission evidence list changed')
        for name, expected in policy['gpu_evidence'].items():
            if sha(run/name) != expected or read(run/name)['status'] != 'PASS':
                raise ValueError('GPU admission evidence changed')
'''
    if text.count(needle)!=1 or "Unrecognized optical backend" in text:
        raise ValueError('Controller patch identity changed')
    text=text.replace(needle,insertion+needle)
    needle="    groups = coverage(state['shards'], config, require_dataset)"
    insertion='''    if policy.get('backend') is not None:
        for shard in state['shards']:
            old_hash = policy['pre_gpu_shards'].get(shard['path'])
            if old_hash is not None:
                if shard['sha256'] != old_hash: raise ValueError('Pre-GPU shard changed')
            elif (shard.get('backend_sources') != policy['backend_sources'] or shard.get('optical_backend') not in
                  ('resident_phase_fp64_native_hcipy_mft','native_cpu_after_optical_fallback')):
                raise ValueError('GPU shard backend provenance changed')
'''
    if text.count(needle)!=1: raise ValueError('Coverage insertion identity changed')
    text=text.replace(needle,insertion+needle)
    needle="    run, policy, record, config, state, groups = check(run)\n    if (run/'dataset.json').exists():"
    replacement="    run, policy, record, config, state, groups = check(run)\n    if policy.get('backend') is not None:\n        raise ValueError('Use the admitted GPU generation entrypoint')\n    if (run/'dataset.json').exists():"
    if text.count(needle)!=1: raise ValueError('Entrypoint insertion identity changed')
    with transition(run,history,source):
        source.write_text(text.replace(needle,replacement),encoding='utf-8',newline='\n')
        c=importlib.reload(c)
        policy.update(workers=workers,backend='resident_phase_fp64_native_hcipy_mft',
                  backend_sources=sources,gpu_evidence={name:c.sha(run/name) for name in names},
                  pre_gpu_shards={s['path']:s['sha256'] for s in state['shards']},
                  predecessor_policy_sha256=old_sha,
                  generator_sources={p.relative_to(PROJECT).as_posix():c.sha(p) for p in c.GENERATOR_FILES},
                  gpu_generation='ADMITTED: one FP64 resident-phase renderer; bounded native CPU phase queue',
                  admission_status='PASS: GPU optics, four/eight-worker TRAIN equality and real interruption/resume',
                  driver_launch='ledger/analysis/expanded_campaign/gpu_campaign.py')
    # Parent allowances include shell/import/partial checks, without recharging measured resource JSON files.
        policy['base_cpu_seconds_upper']+=sum(allowances.values())
        policy['preliminary_cpu_allowances_seconds'].update(allowances)
        policy['projected_cpu_seconds_upper']=projected
        core.save_json(run/'execution-policy.json',policy,overwrite=True)
        state['generator_policy_sha256']=c.sha(run/'execution-policy.json')
        state['backend_transition_utc']=cli.now()
        core.save_json(run/'generation-state.json',state,overwrite=True)
        core.save_json(run/'driver-admission.json',{'status':'PASS','generator_policy_sha256':state['generator_policy_sha256'],
                   'backend_sources':sources,'gpu_evidence':policy['gpu_evidence'],'prior_policy_sha256':old_sha,
                   'scope':'inherited native measurement/engine plus separately verified FP64 GPU controller'},overwrite=True)
        c.check(run)
    print({'status':'PASS','workers':workers,'acquisitions_preserved':12400-remaining,
           'projected_total_cpu_hours':policy['projected_cpu_seconds_upper']/3600,
           'policy_sha256':state['generator_policy_sha256']},flush=True)
