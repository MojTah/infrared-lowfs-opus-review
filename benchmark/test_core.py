"""Focused executable checks of physical count and evidence boundaries."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from core import detector_read, save_json, validate_config, DEFAULT_CONFIG, source_hash
from cli import detector_checks

def run():
    assert len(source_hash())==64
    with patch('core.importlib.metadata.version',return_value='changed'):
        try: source_hash()
        except ValueError as error: assert 'runtime dependency changed' in str(error)
        else: raise AssertionError('changed installed dependency accepted')
    record=detector_checks()
    assert record['pass'], record
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        target=Path(directory)/'moments.json'
        save_json(target,record)
        assert json.loads(target.read_text())['pass'] is True
        try: save_json(target,record)
        except FileExistsError: pass
        else: raise AssertionError('evidence was overwritten')
    a=detector_read(np.zeros((1000,16,16)),np.random.default_rng(17),0,5)
    b=detector_read(np.zeros((1000,16,16)),np.random.default_rng(17),0,5)
    np.testing.assert_array_equal(a,b)
    assert (a<0).any()
    invalid=json.loads(json.dumps(DEFAULT_CONFIG));invalid['phase_steps']=3
    try: validate_config(invalid)
    except ValueError: pass
    else: raise AssertionError('odd paired integration accepted')
    print('PASS: detector moments, signed reads, reproducibility, JSON NumPy scalars, immutable evidence, paired contract')

if __name__=='__main__': run()
