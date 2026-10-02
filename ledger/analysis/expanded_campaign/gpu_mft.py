"""Staged FP64 CuPy adapter for HCIPy 0.7.1's native MFT matrices.

Assign to one OpticalModel.propagator only. Does not replace atmosphere,
OPD construction, wavelength/time/pixel integration or detector statistics.
"""
import importlib.metadata
import os
from pathlib import Path
import numpy as np
import hcipy as hp

_DLL_HANDLES = []


def cuda_backend(cache_directory):
    """Use installed NVIDIA wheels; caller supplies its own writable cache."""
    os.environ['CUPY_CACHE_DIR'] = str(cache_directory)
    os.environ['CUPY_CACHE_IN_MEMORY'] = '1'
    if os.name == 'nt':
        locations = [Path(importlib.metadata.distribution(name).locate_file(relative)).resolve()
                     for name, relative in (
                         ('nvidia-cuda-runtime-cu12', 'nvidia/cuda_runtime'),
                         ('nvidia-cuda-nvrtc-cu12', 'nvidia/cuda_nvrtc'),
                         ('nvidia-cufft-cu12', 'nvidia/cufft'))]
        os.environ['CUDA_PATH'] = str(locations[0])
        bins = [p / 'bin' for p in locations]
        # PyTorch already ships cuBLAS on this host; no package installation.
        bins.append(Path(importlib.metadata.distribution('torch').locate_file('torch/lib')).resolve())
        if not all(p.is_dir() for p in bins):
            raise RuntimeError('Missing installed NVIDIA DLL directory')
        _DLL_HANDLES.extend(os.add_dll_directory(str(p)) for p in bins)
        os.environ['PATH'] = os.pathsep.join(map(str, bins)) + os.pathsep + os.environ.get('PATH', '')
    import cupy as cp
    cp.get_default_memory_pool().set_limit(size=4*1024**3)
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError('No CUDA device; retain native HCIPy CPU propagator')
    assert np.array_equal(cp.asnumpy(cp.asarray([1., 2.], dtype=cp.float64)*2), [2., 4.])
    cp.cuda.get_current_stream().synchronize()
    return cp


class CuPyFraunhofer:
    """Forward scalar MFT only, using native grids, matrices, weights and norm."""
    def __init__(self, native, cp, precision=64):
        if hp.__version__ != '0.7.1':
            raise RuntimeError('Reviewed HCIPy 0.7.1 is required')
        if precision not in (32,64):
            raise ValueError('Explicit32 or64 precision required')
        self.native, self.cp, self.cache = native, cp, {}
        self.dtype = cp.complex128 if precision==64 else cp.complex64
        self.weight_dtype = cp.float64 if precision==64 else cp.float32

    def __call__(self, wavefront):
        cp = self.cp
        if wavefront.electric_field.ndim != 1:
            raise ValueError('Only scalar pupil fields are admitted')
        key = (id(wavefront.grid), float(wavefront.wavelength))
        if key not in self.cache:
            instance = self.native.get_instance_data(wavefront.grid, None, wavefront.wavelength)
            transform = instance.fourier_transform
            if not isinstance(transform, hp.MatrixFourierTransform) or transform.ndim != 2:
                raise TypeError('Native CPU fallback required: supported 2D MFT only')
            transform._compute_matrices(np.dtype('complex128'))
            self.cache[key] = (cp.asarray(transform.M1,dtype=self.dtype), cp.asarray(transform.M2,dtype=self.dtype),
                               cp.asarray(transform.weights_input,dtype=self.weight_dtype), transform.shape_input,
                               instance.norm_factor, instance.output_grid)
        m1, m2, weights, shape, norm, grid = self.cache[key]
        pupil = cp.asarray(np.asarray(wavefront.electric_field), dtype=self.dtype)
        # Same order as HCIPy: intermediate=f@M2, result=M1@intermediate.
        weighted = (pupil*weights).reshape(shape)
        output = (m1 @ (weighted @ m2)).ravel()*norm
        return hp.Wavefront(hp.Field(cp.asnumpy(output), grid), wavefront.wavelength,
                            wavefront.input_stokes_vector)
