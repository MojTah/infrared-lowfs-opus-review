"""Small manifest/provenance checks; no scientific arrays or held-out files."""
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import campaign as c


def rejects(call):
    try: call()
    except (ValueError,FileExistsError): return
    raise AssertionError('invalid manifest/admission accepted')


def metadata(config,split,index,start,end):
    item = {'split':split,'parent_id':f'{split}-{index:04d}',
            'acquisition_start':start,'acquisition_stop':end,'rows':(end-start)*3,
            'path':f'data/{split}-{index:04d}-{start}.npz'}
    for tag,name in ((613,'phase_seed'),(449,'calibration_seed'),(991,'label_seed'),(227,'noise_seed')):
        entropy = [config['seed'],c.cli.SPLITS.index(split),index,tag]
        if start == 5 and tag in (991,227): entropy.append(5)
        item[name] = int(c.np.random.SeedSequence(entropy).generate_state(1)[0])
    if end == 20:
        for tag,name in ((991,'continuation_label_seed'),(227,'continuation_noise_seed')):
            item[name] = int(c.np.random.SeedSequence([config['seed'],c.cli.SPLITS.index(split),index,tag,5]).generate_state(1)[0])
    return item


def enter_owner(run):
    with c.execution_owner(run): pass


def run():
    assert all(p.is_relative_to(c.PROJECT) for p in c.GENERATOR_FILES)
    config = copy.deepcopy(c.core.DEFAULT_CONFIG)
    shards = [metadata(config,s,i,0,20) for s in c.cli.SPLITS for i in range(config['parent_counts'][s])]
    assert len(c.coverage(shards,config,True)) == 620
    assert sum(s['rows'] for s in shards) == 37200
    shards[:1] = [metadata(config,'train',0,0,5),metadata(config,'train',0,5,20)]
    c.coverage(shards,config,True)
    rejects(lambda:c.coverage(shards[:-1],config,True))
    rejects(lambda:c.coverage(shards+[shards[0]],config))
    for field,value in (('rows',44),('phase_seed',0),('split','test'),('acquisition_start',1)):
        changed=copy.deepcopy(shards);changed[0][field]=value
        rejects(lambda:c.coverage(changed,config))
    with tempfile.TemporaryDirectory(dir=c.PROJECT/'benchmark/runs') as folder:
        project=Path(folder).resolve();run=project/'benchmark/runs/development';run.mkdir(parents=True)
        sources=[project/'campaign.py',project/'blocks.py']
        for source in sources: source.write_text('development source')
        prefix=metadata(config,'train',0,0,5)
        data=run/prefix['path'];data.parent.mkdir();data.write_bytes(b'checksum development fixture')
        prefix['sha256']=c.sha(data)
        record={'source_hash':c.core.source_hash(),'config_hash':c.core.digest(config),'measurement_hash':'development','basis':{'development':True}}
        policy={'target_acquisitions':20,'cpu_ceiling_seconds':c.CPU_LIMIT,'workers':4,
                'ram_ceiling_bytes':16*1024**3,'data_ceiling_bytes':4*1024**3,
                'initial_training_gpu_occupancy_seconds':7200,'engine_source_hash':record['source_hash'],
                'measurement_hash':record['measurement_hash'],'inherited_files':{},
                'generator_sources':{p.name:c.sha(p) for p in sources}}
        c.core.save_json(run/'execution-policy.json',policy)
        state={**record,'schema_version':1,'mode_order':c.core.MODES,'acquisitions_per_parent':20,
               'shards':[prefix],'generator_policy_sha256':c.sha(run/'execution-policy.json')}
        c.core.save_json(run/'generation-state.json',state)
        with patch.object(c,'PROJECT',project),patch.object(c,'GENERATOR_FILES',sources),patch.object(c.cli,'admitted',return_value=(record,config)):
            c.check(run)
            with c.execution_owner(run): rejects(lambda:enter_owner(run))
            assert not (run/'active-owner.json').exists()
            data.write_bytes(b'changed');rejects(lambda:c.check(run));data.write_bytes(b'checksum development fixture')
            sources[0].write_text('changed');rejects(lambda:c.check(run));sources[0].write_text('development source')
            c.core.save_json(run/'driver-admission.json',{'status':'PASS','generator_policy_sha256':'different'})
            rejects(lambda:c.admission(run,policy))
            c.core.save_json(run/'driver-admission.json',{'status':'PASS','generator_policy_sha256':c.sha(run/'execution-policy.json')},overwrite=True)
            c.admission(run,policy)
            if 'Unrecognized optical backend' in Path(c.__file__).read_text():
                gpu_folder=project/'ledger/analysis/expanded_campaign';gpu_folder.mkdir(parents=True)
                gpu_sources={}
                for name in ('gpu_campaign.py','gpu_packets.py','gpu_backend.py','gpu_mft.py'):
                    file=gpu_folder/name;file.write_text('development GPU source')
                    gpu_sources[file.relative_to(project).as_posix()]=c.sha(file)
                gpu_evidence={}
                for name in ('t001-gpu-admission-resources.json','t001-gpu-eight-worker-resources.json',
                             't001-gpu-lifecycle-resources.json','t001-gpu-lifecycle-input-binding.json'):
                    c.core.save_json(run/name,{'status':'PASS'})
                    gpu_evidence[name]=c.sha(run/name)
                policy.update(backend='resident_phase_fp64_native_hcipy_mft',backend_sources=gpu_sources,
                              gpu_evidence=gpu_evidence,pre_gpu_shards={prefix['path']:prefix['sha256']})
                c.core.save_json(run/'execution-policy.json',policy,overwrite=True)
                state['generator_policy_sha256']=c.sha(run/'execution-policy.json')
                c.core.save_json(run/'generation-state.json',state,overwrite=True)
                c.check(run)
                gpu_file=gpu_folder/'gpu_backend.py';gpu_file.write_text('changed GPU source')
                rejects(lambda:c.check(run));gpu_file.write_text('development GPU source')
                c.core.save_json(run/'t001-gpu-lifecycle-resources.json',{'status':'FAIL'},overwrite=True)
                rejects(lambda:c.check(run))
                c.core.save_json(run/'t001-gpu-lifecycle-resources.json',{'status':'PASS'},overwrite=True)
                c.check(run)
            partial=run/'data/unindexed.partial.npz';partial.write_bytes(b'partial')
            rejects(lambda:c.check(run))
            with c.execution_owner(run) as lease: lease['safe_to_release']=False
            assert (run/'active-owner.json').exists()
    print(json.dumps({'status':'PASS','checks':['620 grouped parents/37200 rows','mixed prefix/full coverage','missing/duplicate/seed/row rejection','source/shard/policy binding','exclusive owner and unverified cleanup retention','partial rejection'],'scope':'development metadata only; no scientific arrays'}))


if __name__=='__main__': run()
