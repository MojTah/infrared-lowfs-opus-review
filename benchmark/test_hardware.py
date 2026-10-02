"""Resource admission and CUDA execution checks use mock hardware, never a real GPU."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hardware import GIB, _gpu, configure


def config():
    return {'generation_workers': 4, 'pupil_n': 1024, 'phase_steps': 10,
            'parent_counts': {'train': 300}, 'budgets': {'cpu_seconds': 57600,
            'gpu_seconds': 7200, 'ram_bytes': 16*GIB, 'data_bytes': 4*GIB}}


class Tensor:
    def __mul__(self, other):
        return self

    def sum(self):
        return self

    def item(self):
        return 14.


def torch_mock(available=True, free=8*GIB):
    cuda = Mock()
    cuda.is_available.return_value = available
    cuda.get_device_properties.return_value = SimpleNamespace(name='Mock GPU')
    cuda.mem_get_info.return_value = (free, 8*GIB)
    return SimpleNamespace(__version__='test', version=SimpleNamespace(cuda='test'),
                           device=lambda name: name, float32='float32',
                           tensor=Mock(return_value=Tensor()), cuda=cuda)


def run():
    cpu = torch_mock(available=False)
    gpu = torch_mock()
    with patch('hardware.importlib.import_module', return_value=cpu):
        assert not _gpu()['usable']
        cpu.tensor.assert_not_called()
    with patch('hardware.importlib.import_module', side_effect=ImportError('torch missing')):
        assert 'ImportError' in _gpu()['fallback_reason']
    with patch('hardware.importlib.import_module', return_value=gpu):
        result = _gpu()
        assert result['usable'] and result['probe']['sum_squares'] == 14.
        gpu.cuda.synchronize.assert_called_once_with('cuda:0')
        assert gpu.tensor.call_args.kwargs == {'dtype': 'float32', 'device': 'cuda:0'}
    broken = torch_mock()
    broken.cuda.synchronize.side_effect = RuntimeError('driver failure')
    with patch('hardware.importlib.import_module', return_value=broken):
        assert 'driver failure' in _gpu()['fallback_reason']
    source = config()
    original = deepcopy(source)
    with patch('hardware.psutil.virtual_memory', return_value=SimpleNamespace(total=32*GIB, available=20*GIB)), patch('hardware.psutil.cpu_count', side_effect=lambda logical: 16 if logical else 8), patch('hardware.importlib.import_module', return_value=torch_mock()):
        chosen, report = configure(source)
    assert source == original and chosen['pupil_n'] == 1024 and chosen['parent_counts'] == source['parent_counts']
    assert report['status'] == 'PASS' and chosen['generation_workers'] == 4
    assert chosen['compute'] == {'training_device': 'cuda', 'training_batch_size': 128}
    limited = config()
    limited['generation_workers'] = 100
    limited['budgets'] = {key: value*10 for key, value in limited['budgets'].items()}
    with patch('hardware.psutil.virtual_memory', return_value=SimpleNamespace(total=8*GIB, available=8*GIB)), patch('hardware.psutil.cpu_count', return_value=2), patch('hardware.importlib.import_module', return_value=cpu):
        chosen, report = configure(limited)
    assert chosen['generation_workers'] == 2 and chosen['budgets']['ram_bytes'] == int(.8*8*GIB)
    assert chosen['budgets']['cpu_seconds'] == 57600 and chosen['budgets']['gpu_seconds'] == 7200 and chosen['budgets']['data_bytes'] == 4*GIB
    assert chosen['compute']['training_device'] == 'cpu' and report['gpu']['fallback_reason']
    with patch('hardware.psutil.virtual_memory', return_value=SimpleNamespace(total=4*GIB, available=3*GIB)), patch('hardware.psutil.cpu_count', return_value=2), patch('hardware.importlib.import_module', return_value=cpu):
        chosen, report = configure(config())
    assert report['status'] == 'RESOURCE_LIMIT' and chosen['pupil_n'] == 1024
    with patch('hardware.psutil.virtual_memory', return_value=SimpleNamespace(total=32*GIB, available=20*GIB)), patch('hardware.psutil.cpu_count', return_value=None), patch('hardware.importlib.import_module', return_value=torch_mock(free=200*1024**2)):
        chosen, report = configure(config())
    assert chosen['generation_workers'] == 1 and chosen['compute']['training_batch_size'] == 64
    with patch('hardware.importlib.import_module', return_value=torch_mock(free=100*1024**2)):
        assert not _gpu()['usable']
    print('PASS: verified CUDA selection, missing/broken CUDA CPU fallback, RAM stop, CPU/RAM/budget caps, immutable physics and low-headroom batch')


if __name__ == '__main__':
    run()
