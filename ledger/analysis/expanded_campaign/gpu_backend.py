"""Instance-local device phase extension of the previously checked HCIPy adapter.

Native pupil/basis, MFT, integration and acquisition definitions are retained.
Only phase/forward-image arithmetic is moved to resident CUDA arrays.
"""
import types
import hcipy as hp
import numpy as np
from gpu_mft import CuPyFraunhofer


def attach_device_phase(model, cp, precision=64):
    """Attach one immutable optical model; restore both methods before fallback."""
    if getattr(model, '_device_phase_attachment', None) is not None:
        raise ValueError('Device phase already attached')
    native_image, native_probabilities = model.image, model.probabilities
    captured = {name: np.asarray(getattr(model,name)).copy() for name in
                ('modes_nm_units','static_nm','pixel_weights','wavelengths','spectral_weights')}
    identity = (model.n, model.pixel_order, model.pupil_power_m2,
                model.config['diversity_nm'], id(model.pupil_grid), id(model.propagator))
    coordinates = [np.asarray(model.pupil_grid.x).copy(), np.asarray(model.pupil_grid.y).copy()]

    def unchanged():
        current = (model.n, model.pixel_order, model.pupil_power_m2,
                   model.config['diversity_nm'], id(model.pupil_grid), id(model.propagator))
        if (current != identity or any(not np.array_equal(getattr(model,k),v) for k,v in captured.items())
                or not np.array_equal(model.pupil_grid.x, coordinates[0])
                or not np.array_equal(model.pupil_grid.y, coordinates[1])):
            raise ValueError('Attached optical model changed; detach and attach a fresh model')

    def detach():
        model.image, model.probabilities = native_image, native_probabilities
        model._device_phase_attachment = None
    adapter = CuPyFraunhofer(model.propagator,cp,precision)
    for lam in model.wavelengths:
        # Populate the established native-matrix adapter's cache without changing it.
        adapter(hp.Wavefront(hp.Field(np.ones(model.n**2,dtype=complex),model.pupil_grid),lam))
    modes = cp.asarray(model.modes_nm_units,dtype=cp.float64)
    static = cp.asarray(model.static_nm,dtype=cp.float64)
    x = cp.asarray(np.asarray(model.pupil_grid.x),dtype=cp.float64).reshape(model.n,model.n)
    y = cp.asarray(np.asarray(model.pupil_grid.y),dtype=cp.float64).reshape(model.n,model.n)
    weights = cp.asarray(model.pixel_weights,dtype=cp.float64)
    mas_rad=np.pi/(180*3600*1000)

    def fields(coeff,diversity=None,residual=None,centroid=(0.,0.)):
        c=np.asarray(coeff)
        d=np.zeros(4) if diversity is None else np.asarray(diversity,float)
        if c.shape!=(4,) or np.iscomplexobj(c) or not np.isfinite(c).all() or d.shape!=(4,) or not np.isfinite(d).all():
            raise ValueError('Four finite real coefficients/diversity required')
        if np.asarray(centroid).shape!=(2,) or not np.isfinite(centroid).all():
            raise ValueError('Two finite centroid values required')
        opd=cp.einsum('k,kij->ij',cp.asarray(c+d,dtype=cp.float64),modes)+static
        if residual is not None:
            if residual.shape!=(model.n,model.n) or residual.dtype.kind!='f':
                raise ValueError('Matching finite residual map required')
            if not bool(cp.isfinite(cp.asarray(residual)).all()):
                raise ValueError('Nonfinite residual OPD')
            opd+=cp.asarray(residual,dtype=cp.float64)
        opd=opd*1e-9+mas_rad*(float(centroid[0])*x+float(centroid[1])*y)
        for lam in model.wavelengths:
            m1,m2,input_weights,shape,norm,_=adapter.cache[(id(model.pupil_grid),float(lam))]
            pupil=cp.exp(2j*np.pi*opd/lam).astype(adapter.dtype,copy=False)
            weighted=(pupil.ravel()*input_weights).reshape(shape)
            yield (m1@(weighted@m2))*norm

    def image_device(coeff,diversity=None,residual=None,centroid=(0.,0.)):
        result=cp.zeros((16,16),dtype=cp.float64)
        for electric,sw in zip(fields(coeff,diversity,residual,centroid),model.spectral_weights):
            # Small focal-plane conversion keeps intensity/integration in float64.
            e=electric.astype(cp.complex128,copy=False)
            density=(e.real*e.real+e.imag*e.imag)/model.pupil_power_m2
            result+=sw*cp.einsum('a,b,iajb->ij',weights,weights,density.reshape(16,model.pixel_order,16,model.pixel_order))
        return result

    def image(self,coeff_nm,diversity_nm=None,residual_nm=None,centroid_mas=(0.,0.),derivatives=False):
        unchanged()
        if derivatives:
            return native_image(coeff_nm,diversity_nm,residual_nm,centroid_mas,derivatives)
        result=cp.asnumpy(image_device(coeff_nm,diversity_nm,residual_nm,centroid_mas))
        if not np.isfinite(result).all() or np.any(result < -1e-15):
            raise FloatingPointError('Invalid propagated GPU intensity')
        return result

    def probabilities(self,coeff_nm,residual_sequence_nm,centroid_mas=(0.,0.),diversity_scale=1.):
        unchanged()
        r=np.asarray(residual_sequence_nm)
        if r.ndim!=3 or r.shape[1:]!=(model.n,model.n) or len(r)%2 or len(r)<2 or np.iscomplexobj(r) or not np.isfinite(r).all():
            raise ValueError('Finite even residual time sequence required')
        resident=cp.asarray(r,dtype=cp.float64)
        single=cp.zeros((1,16,16),dtype=cp.float64);pair=cp.zeros((2,16,16),dtype=cp.float64)
        n=len(r)
        for i,residual in enumerate(resident):
            single[0]+=image_device(coeff_nm,[0,self.config['diversity_nm']*diversity_scale,0,0],residual,centroid_mas)/n
            channel=int(i>=n//2)
            pair[channel]+=image_device(coeff_nm,[channel*self.config['diversity_nm']*diversity_scale,0,0,0],residual,centroid_mas)/(n//2)
        result = cp.asnumpy(single),cp.asnumpy(pair)
        validate_probabilities(result)
        return result

    model.image=types.MethodType(image,model)
    model.probabilities=types.MethodType(probabilities,model)
    extension = {'native_image':native_image,'native_probabilities':native_probabilities,
                 'complex_fields':fields,'precision':precision,'adapter':adapter,'detach':detach}
    model._device_phase_attachment = extension
    return extension


def validate_probabilities(values):
    if len(values) != 2:
        raise ValueError('Both optical designs required before detector draws')
    for value, shape in zip(values, ((1,16,16),(2,16,16))):
        if value.shape != shape or np.iscomplexobj(value) or not np.isfinite(value).all() or np.any(value < 0):
            raise FloatingPointError('Invalid optical probabilities before detector draws')


def probabilities_with_fallback(model, extension, *args):
    """Retry optical arithmetic only; this function never consumes detector RNG."""
    try:
        values = model.probabilities(*args)
        validate_probabilities(values)
        return values, False
    except (FloatingPointError, MemoryError,
            extension['adapter'].cp.cuda.memory.OutOfMemoryError) as error:
        extension['detach']()
        values = model.probabilities(*args)
        validate_probabilities(values)
        return values, type(error).__name__
