"""HCIPy four-mode Keck PSFs. Physical truth is separate from estimator inputs."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import hcipy as hp
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

hp.Configuration().reset(enable_user_overrides=False)
hp.Configuration().fourier.fft.method=['numpy']
HCIPY_RUNTIME=str(hp.Configuration())

ROOT = Path(__file__).resolve().parent
MAS_RAD = np.pi / (180 * 3600 * 1000)
MODES = ['focus', 'astig_cos', 'astig_sin', 'spherical']
LIMITS = np.array([150., 100., 100., 100.])
DEFAULT_CONFIG = {
    'schema_version': 1, 'case': 'published_Keck_TRICK_historical_AO_surrogate',
    'diameter_m': 10.949, 'pupil': 'keck', 'pixel_mas': 50., 'roi': 16, 'band_um': [1.5, 1.8],
    'diversity_nm': 200., 'exposure_s': .01, 'phase_steps': 10,
    'read_noise_e': 5., 'background_e': 5., 'dark_e': .01,
    'flux_e': [1000., 10000., 100000.], 'ncpa_nm': 50., 'calibration_error_rms_nm': 20.,
    'pupil_n': 1024, 'pixel_order': 6, 'wavelength_nodes': 5,
    'pupil_registration_diameter': [0., 0.],
    'basis_reference_n': 1024, 'boundary_supersampling': 4,
    'parent_counts': {'train': 300, 'validation': 60, 'calibration': 60, 'test': 100, 'shifted': 100},
    'seed': 20260930, 'generation_workers': 4, 'residual_rms_nm': [100., 200.], 'shifted_residual_nm': 300.,
    'atmosphere': {
        'r0_m': .165, 'r0_wavelength_m': 500e-9, 'L0_m': 75.,
        'fractions': [.51/.97, .11/.97, .06/.97, .06/.97, .10/.97, .08/.97, .05/.97],
        'heights_m': [0., 500., 1000., 2000., 4000., 8000., 16000.],
        'wind_m_s': [6.7, 13.9, 20.8, 29., 29., 29., 29.],
        'wind_direction_rad': [0., np.pi/3, -np.pi/3, -np.pi, -4*np.pi/3, -np.pi/6, np.pi/8],
        'kc': 10., 'screen_n': 128,
        'model': 'HCIPy frozen-flow von Karman + spatial AO high-pass; no measured WFS/DM controller',
        'r0_reference_note': '500nm assumption: not stated in the 2024 table',
    },
    'budgets': {'cpu_seconds': 57600, 'gpu_seconds': 7200, 'ram_bytes': 16*1024**3, 'data_bytes': 4*1024**3},
    'thresholds': {'image_l1': 1e-3, 'coefficient_refinement_nm': 1., 'noiseless_rmse_nm': 2.},
}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_hash():
    if str(hp.Configuration())!=HCIPY_RUNTIME:raise ValueError('HCIPy runtime configuration changed')
    pinned=json.loads((ROOT/'runtime-versions.json').read_text(encoding='utf-8'))
    if sys.version.split()[0]!=pinned['python'].split()[0]:
        raise ValueError('Python version differs from the frozen runtime')
    for name, expected in pinned['distributions'].items():
        actual=importlib.metadata.version(name)
        if actual!=expected:raise ValueError(f'runtime dependency changed: {name} {actual}, expected {expected}')
    files={p.name: file_hash(p) for p in sorted(ROOT.glob('*.py')) if not p.name.startswith('test_')}
    for name in ('requirements.lock.txt','runtime-versions.json','telescope_case.json','reference-provenance.json'):
        files[name]=file_hash(ROOT/name)
    files['effective_hcipy_configuration']=HCIPY_RUNTIME
    return digest(files)

def save_json(path, data, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f'preserve existing evidence: {path}')
    temporary = path.with_suffix(path.suffix + '.partial')
    if temporary.exists():
        raise FileExistsError(f'unresolved temporary output: {temporary}')
    def scalar(value):
        if isinstance(value,np.generic):
            return value.item()
        raise TypeError(f'unsupported evidence value: {type(value).__name__}')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False, default=scalar), encoding='utf-8')
    os.replace(temporary, path)

def validate_config(c):
    if c['schema_version'] != 1 or c['roi'] != 16:
        raise ValueError('schema 1 and 16x16 detector required')
    for name in ('diameter_m', 'pixel_mas', 'exposure_s', 'diversity_nm'):
        if isinstance(c[name], bool) or not np.isfinite(c[name]) or c[name] <= 0:
            raise ValueError(f'invalid positive {name}')
    for name in ('read_noise_e', 'background_e', 'dark_e', 'ncpa_nm', 'calibration_error_rms_nm'):
        if not np.isfinite(c[name]) or c[name] < 0:
            raise ValueError(f'invalid nonnegative {name}')
    for name in ('pupil_n', 'basis_reference_n', 'pixel_order', 'wavelength_nodes', 'phase_steps', 'boundary_supersampling'):
        if isinstance(c[name], bool) or not isinstance(c[name], int) or c[name] < 2:
            raise ValueError(f'invalid integer {name}')
    if c['phase_steps'] % 2:
        raise ValueError('paired exposure requires even number of time nodes')
    if isinstance(c['generation_workers'],bool) or not isinstance(c['generation_workers'],int) or not 1<=c['generation_workers']<=8:
        raise ValueError('one to eight generation workers required')
    if set(c['parent_counts'])!={'train','validation','calibration','test','shifted'} or any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in c['parent_counts'].values()):
        raise ValueError('positive integer parent counts for all five splits required')
    a = c['atmosphere']
    lengths = [len(a[k]) for k in ('fractions', 'heights_m', 'wind_m_s', 'wind_direction_rad')]
    if len(set(lengths)) != 1 or min(lengths) < 1 or not np.isclose(sum(a['fractions']), 1):
        raise ValueError('invalid atmospheric layer profile')
    for name in ('r0_m', 'r0_wavelength_m', 'L0_m', 'kc'):
        if not np.isfinite(a[name]) or a[name] <= 0:
            raise ValueError(f'invalid atmosphere {name}')
    if any(not np.isfinite(a[k]).all() for k in ('fractions','heights_m','wind_m_s','wind_direction_rad')) or min(a['fractions'])<=0 or min(a['heights_m'])<0 or min(a['wind_m_s'])<0:
        raise ValueError('finite positive layer weights and nonnegative heights/winds required')
    registration=np.asarray(c.get('pupil_registration_diameter',[0.,0.]))
    if registration.shape!=(2,) or not np.isfinite(registration).all() or np.max(abs(registration))>.1:
        raise ValueError('invalid pupil registration')
    if not 0 < c['band_um'][0] < c['band_um'][1] or any(x <= 0 for x in c['flux_e']):
        raise ValueError('positive ordered bandpass and flux required')
    return c

def keck_coverage(n, diameter, supersampling=4, registration=(0.,0.), pupil='keck'):
    """Subcell area quadrature, real 3mm gaps (no arbitrary gap padding)."""
    extent = 1.1 * diameter
    fine = hp.make_pupil_grid(n*supersampling, extent)
    if pupil=='keck':
        generator = hp.make_keck_aperture(normalized=True, gap_padding=1)
    else:
        generators={'tmt':hp.make_tmt_aperture,'elt':hp.make_elt_aperture,'gmt':hp.make_gmt_aperture}
        if pupil not in generators: raise ValueError('supported published pupil required')
        generator=generators[pupil](normalized=True)
    # Full 8192-square quadrature creates multi-GiB temporaries; row blocks preserve exact nodes.
    xs,ys=fine.separated_coords
    coverage=np.empty((n,n))
    for first in range(0,n,64):
        stop=min(first+64,n)
        block=hp.CartesianGrid(hp.SeparatedCoords([xs,ys[first*supersampling:stop*supersampling]]))
        binary=generator(block.shifted(-np.asarray(registration)*diameter).scaled(1/diameter)).shaped
        coverage[first:stop]=binary.reshape(stop-first,supersampling,n,supersampling).mean(axis=(1,3))
    return hp.make_pupil_grid(n, extent), coverage

def zernike_seeds(x, y, diameter):
    x, y = x/(diameter/2), y/(diameter/2)
    r2 = x*x+y*y
    # Unit-disk conventional Zernikes; actual-pupil normalization below.
    return np.stack([2*x, 2*y, np.sqrt(3)*(2*r2-1),
                     np.sqrt(6)*(x*x-y*y), np.sqrt(6)*2*x*y,
                     np.sqrt(5)*(6*r2*r2-6*r2+1)], axis=-1)

class OpticalModel:
    def __init__(self, config=None, pupil_n=None, pixel_order=None, wavelength_nodes=None, engine='hcipy', basis=None):
        self.config = validate_config(json.loads(json.dumps(config or DEFAULT_CONFIG)))
        if engine != 'hcipy':
            raise ValueError('main data generator is HCIPy')
        self.n = pupil_n or self.config['pupil_n']
        self.pixel_order = pixel_order or self.config['pixel_order']
        wn = wavelength_nodes or self.config['wavelength_nodes']
        self.pupil_grid, self.mask = keck_coverage(self.n, self.config['diameter_m'], self.config['boundary_supersampling'],self.config.get('pupil_registration_diameter',(0.,0.)),self.config.get('pupil','keck'))
        self.dx = 1.1*self.config['diameter_m']/self.n
        self.pupil_power_m2 = float(self.mask.sum()*self.dx**2)
        self.weights = self.mask.ravel()*self.dx**2/self.pupil_power_m2
        # Coverage is an integration weight for the continuous binary aperture,
        # not a gray amplitude transmission. MFT supports per-point input weights.
        self.pupil_grid.weights = self.mask.ravel()*self.dx**2
        if basis is None:
            grid, coverage = keck_coverage(self.config['basis_reference_n'], self.config['diameter_m'], 2,pupil=self.config.get('pupil','keck'))
            w = coverage.ravel()/coverage.sum()
            seeds = zernike_seeds(np.asarray(grid.x), np.asarray(grid.y), self.config['diameter_m'])
            mu = w @ seeds
            gram = (seeds-mu).T @ ((seeds-mu)*w[:, None])
            chol = np.linalg.cholesky(gram)
            self.basis = {'mu': mu.tolist(), 'transform': np.linalg.inv(chol.T).tolist(),
                          'order': ['x_tilt','y_tilt',*MODES], 'target_indices': [2,3,4,5],
                          'conventional_seeds': ['2x/R','2y/R','sqrt3(2r2-1)','sqrt6(x2-y2)','sqrt6(2xy)','sqrt5(6r4-6r2+1)'],
                          'metric': 'normalized geometric illuminated area',
                          'coordinates': 'columns+x, rows+y, OPD exp(+i2piW/lambda), Fourier negative',
                          'units': 'nm OPD RMS', 'reference_n': self.config['basis_reference_n']}
        else:
            self.basis = json.loads(json.dumps(basis))
        q = (zernike_seeds(np.asarray(self.pupil_grid.x), np.asarray(self.pupil_grid.y), self.config['diameter_m'])
             - np.array(self.basis['mu'])) @ np.array(self.basis['transform'])
        self.q_all = q
        self.modes_nm_units = q[:, 2:].T.reshape(4, self.n, self.n)
        self.measurement_hash = digest({'basis': self.basis, 'diameter_m': self.config['diameter_m'],
                                      'roi': 16, 'pixel_mas': self.config['pixel_mas'],
                                      'band_um': self.config['band_um'], 'pupil': self.config.get('pupil','keck'),
                                      'geometry_version': 'HCIPy0.7.1 published pupil; Keck gap_padding1'})
        nodes, weights = np.polynomial.legendre.leggauss(self.pixel_order)
        self.pixel_nodes_rad = ((np.arange(16)+.5-8)[:, None]+nodes[None, :]/2).ravel()*self.config['pixel_mas']*MAS_RAD
        self.pixel_weights = weights*self.config['pixel_mas']*MAS_RAD/2
        output_grid = hp.CartesianGrid(hp.SeparatedCoords([self.pixel_nodes_rad, self.pixel_nodes_rad]))
        self.propagator = hp.FraunhoferPropagator(self.pupil_grid, output_grid, focal_length=1)
        g, wg = np.polynomial.legendre.leggauss(wn)
        lo, hi = np.array(self.config['band_um'])*1e-6
        self.wavelengths = (lo+hi)/2 + g*(hi-lo)/2
        self.spectral_weights = wg/2
        self.static_nm = self._ncpa()
        self.residual_gain = None

    def project_residual(self, residual):
        v = np.asarray(residual, float).reshape(-1).copy()
        if v.size != self.n**2 or not np.isfinite(v).all():
            raise ValueError('finite matching residual map required')
        columns = np.column_stack([np.ones_like(v), self.q_all])
        gram = columns.T @ (self.weights[:, None]*columns)
        v -= columns @ np.linalg.solve(gram, columns.T @ (self.weights*v))
        return v.reshape(self.n, self.n)

    def _ncpa(self):
        x = np.asarray(self.pupil_grid.x).reshape(self.n,self.n)/self.config['diameter_m']
        y = np.asarray(self.pupil_grid.y).reshape(self.n,self.n)/self.config['diameter_m']
        v = self.project_residual(np.sin(12*np.pi*x+.3)*np.cos(10*np.pi*y-.7))
        rms = np.sqrt(self.weights @ v.ravel()**2)
        return v*self.config['ncpa_nm']/rms

    def calibration_error(self,seed):
        """Constant unknown high-order error per parent, independent of labelled modes."""
        rng=np.random.default_rng(seed)
        x=np.asarray(self.pupil_grid.x).reshape(self.n,self.n)/self.config['diameter_m']
        y=np.asarray(self.pupil_grid.y).reshape(self.n,self.n)/self.config['diameter_m']
        error=np.zeros_like(x)
        for _ in range(3):
            kx,ky=rng.uniform(6.,14.,2);phase=rng.uniform(0,2*np.pi)
            error+=rng.normal()*np.sin(2*np.pi*(kx*x+ky*y)+phase)
        error=self.project_residual(error)
        rms=np.sqrt(self.weights@error.ravel()**2)
        return error*(rng.uniform(0.,self.config['calibration_error_rms_nm'])/rms)

    def integrate_density(self, density):
        p = self.pixel_order
        return np.einsum('a,b,iajb->ij', self.pixel_weights, self.pixel_weights,
                         np.asarray(density).reshape(16,p,16,p))

    def image(self, coeff_nm, diversity_nm=None, residual_nm=None, centroid_mas=(0.,0.), derivatives=False):
        c = np.asarray(coeff_nm)
        if c.shape != (4,) or np.iscomplexobj(c) or not np.isfinite(c).all():
            raise ValueError('four finite real modal coefficients required')
        d = np.zeros(4) if diversity_nm is None else np.asarray(diversity_nm, float)
        if d.shape != (4,) or not np.isfinite(d).all() or not np.isfinite(centroid_mas).all():
            raise ValueError('invalid diversity or centroid')
        opd = np.einsum('k,kij->ij', c+d, self.modes_nm_units) + self.static_nm
        if residual_nm is not None:
            r = np.asarray(residual_nm)
            if r.shape != opd.shape or np.iscomplexobj(r) or not np.isfinite(r).all():
                raise ValueError('invalid residual OPD')
            opd += r
        opd = opd.ravel()*1e-9 + MAS_RAD*(centroid_mas[0]*np.asarray(self.pupil_grid.x)+centroid_mas[1]*np.asarray(self.pupil_grid.y))
        slopes = list(self.modes_nm_units.reshape(4,-1)*1e-9)
        if derivatives == 'nuisance':
            slopes += [MAS_RAD*np.asarray(self.pupil_grid.x),MAS_RAD*np.asarray(self.pupil_grid.y)]
        elif derivatives == 'centroid':
            slopes = [MAS_RAD*np.asarray(self.pupil_grid.x),MAS_RAD*np.asarray(self.pupil_grid.y)]
        result = np.zeros((1+len(slopes) if derivatives else 1,16,16))
        for lam, sw in zip(self.wavelengths, self.spectral_weights):
            wave = hp.Wavefront(hp.Field(np.exp(2j*np.pi*opd/lam), self.pupil_grid), lam)
            e = self.propagator(wave).electric_field
            result[0] += sw*self.integrate_density(abs(e)**2/self.pupil_power_m2)
            if derivatives:
                for k, slope in enumerate(slopes):
                    dw = hp.Wavefront(hp.Field(np.asarray(wave.electric_field)*(2j*np.pi/lam)*slope, self.pupil_grid),lam)
                    de = self.propagator(dw).electric_field
                    result[k+1] += sw*self.integrate_density(2*np.real(e.conj()*de)/self.pupil_power_m2)
        if not np.isfinite(result).all() or np.any(result[0] < -1e-15):
            raise FloatingPointError('invalid propagated intensity')
        return (result[0],result[1:]) if derivatives else result[0]

    def independent_image(self, coeff_nm, diversity_nm=None, centroid_mas=(0.,0.)):
        """Direct NumPy quadrature; shares only declared pupil/basis, not HCIPy propagation."""
        if __package__:
            from .reference import field
        else:
            from reference import field
        points = np.c_[self.pupil_grid.x,self.pupil_grid.y]
        d = np.zeros(4) if diversity_nm is None else np.asarray(diversity_nm)
        opd = np.einsum('k,kij->ij',np.asarray(coeff_nm)+d,self.modes_nm_units).ravel()*1e-9+self.static_nm.ravel()*1e-9
        opd += MAS_RAD*(points[:,0]*centroid_mas[0]+points[:,1]*centroid_mas[1])
        result = np.zeros((16,16))
        for lam, sw in zip(self.wavelengths,self.spectral_weights):
            e = field(points,self.mask.ravel()*self.dx**2,opd,lam,self.pixel_nodes_rad,self.pixel_nodes_rad)
            result += sw*self.integrate_density(abs(e)**2/(lam**2*self.pupil_power_m2))
        return result

    def probabilities(self, coeff_nm, residual_sequence_nm, centroid_mas=(0.,0.), diversity_scale=1.):
        r = np.asarray(residual_sequence_nm)
        if r.ndim != 3 or r.shape[1:] != (self.n,self.n) or len(r)%2 or len(r)<2:
            raise ValueError('an even residual time sequence is required')
        single = np.zeros((1,16,16)); pair = np.zeros((2,16,16))
        n = len(r)
        for i, residual in enumerate(r):
            single[0] += self.image(coeff_nm,[0,self.config['diversity_nm']*diversity_scale,0,0],residual,centroid_mas)/n
            channel = int(i >= n//2)
            pair[channel] += self.image(coeff_nm,[channel*self.config['diversity_nm']*diversity_scale,0,0,0],residual,centroid_mas)/(n//2)
        return single, pair

    def acquire(self, coeff_nm, residual_sequence_nm, rng, flux_e, condition='nominal', centroid_mas=(0.,0.)):
        scale = 1.1 if condition == 'diversity' else 1.
        single, pair = self.probabilities(coeff_nm,residual_sequence_nm,centroid_mas,scale)
        rn = 10. if condition == 'read_noise' else self.config['read_noise_e']
        a = detector_read(single*flux_e,rng,self.config['background_e']+self.config['dark_e'],rn,condition)
        b = detector_read(pair*flux_e/2,rng,(self.config['background_e']+self.config['dark_e'])/2,rn,condition)
        return {'images_single':a.astype('float32'),'images_pair':b.astype('float32'),
                'labels_nm':np.asarray(coeff_nm,float),'flux_e':float(flux_e)}

def detector_read(source_e, rng, background_e, read_noise_e, condition='nominal'):
    source = np.asarray(source_e)
    if source.ndim < 2 or np.iscomplexobj(source) or source.dtype.kind not in 'fiu' or not np.isfinite(source).all() or np.any(source<0):
        raise ValueError('real finite nonnegative source electrons required')
    if not np.isfinite([background_e,read_noise_e]).all() or min(background_e,read_noise_e)<0:
        raise ValueError('finite nonnegative noise parameters required')
    charge = rng.poisson(source+background_e).astype(float)
    if condition == 'ipc':
        kernel = np.array([[0,.02,0],[.02,.92,.02],[0,.02,0]])
        charge = np.stack([ndimage.convolve(frame,kernel,mode='constant') for frame in charge.reshape(-1,16,16)]).reshape(charge.shape)
    image = charge+rng.normal(0,read_noise_e,size=charge.shape)
    if condition == 'saturation':
        image = np.minimum(image,2000.)
    if condition == 'bad_pixels':
        # Common acquisition guard treats nonfinite pixels as unusable; never drop the row.
        image[...,3,9] = np.nan
    return image

class ResidualSequence:
    """Independent HCIPy layers, declared filtered AO surrogate, ensemble gain."""
    def __init__(self, optical, parent_seed, gain=1., shifted=False):
        self.optical = optical; self.gain = gain; self.shifted = shifted
        a = optical.config['atmosphere']; self.a = a
        self.n = a['screen_n']; self.grid = hp.make_pupil_grid(self.n,1.1*optical.config['diameter_m'])
        cn = hp.Cn_squared_from_fried_parameter(a['r0_m'],a['r0_wavelength_m'])
        self.layers = []
        seeds = np.random.SeedSequence(parent_seed).spawn(len(a['fractions']))
        for f,h,v,angle,seed in zip(a['fractions'],a['heights_m'],a['wind_m_s'],a['wind_direction_rad'],seeds):
            self.layers.append(hp.FiniteAtmosphericLayer(self.grid,cn*f,a['L0_m'],
                velocity=v*np.array([np.cos(angle),np.sin(angle)]),height=h,oversampling=2,seed=seed))
        k = np.fft.fftfreq(self.n)*self.n/1.1
        kx,ky = np.meshgrid(k,k)
        kc = 6. if shifted else a['kc']
        self.filter = (kx*kx+ky*ky)/(kx*kx+ky*ky+kc*kc)

    def at(self,t):
        opd = np.zeros((self.n,self.n))
        for layer in self.layers:
            layer.evolve_until(float(t))
            opd += np.asarray(layer.phase_for(self.a['r0_wavelength_m'])).reshape(self.n,self.n)*self.a['r0_wavelength_m']/(2*np.pi)*1e9
        opd = np.fft.ifft2(np.fft.fft2(opd)*self.filter).real
        if self.n != self.optical.n:
            # Interpolate the same physical screen; this is not a fresh fine-grid realization.
            opd = ndimage.zoom(opd,self.optical.n/self.n,order=3,mode='nearest',prefilter=True,grid_mode=True)
        return self.optical.project_residual(opd)*self.gain

    def acquisition(self,index,steps=None):
        steps = steps or self.optical.config['phase_steps']
        exposure = self.optical.config['exposure_s']
        return np.array([self.at(index*exposure+(i+.5)*exposure/steps) for i in range(steps)])

def calibrate_residual_gain(model, shifted=False):
    rms = []
    for seed in (190731,88213,71921):
        sequence = ResidualSequence(model,seed,shifted=shifted)
        for t in (0.,.017,.053):
            r = sequence.at(t)
            rms.append(float(np.sqrt(model.weights @ r.ravel()**2)))
    norm = float(np.sqrt(np.mean(np.square(rms))))
    if not np.isfinite(norm) or norm<=0:
        raise FloatingPointError('invalid independent residual gain')
    return 1/norm, {'developer_seeds':[190731,88213,71921],'raw_rms_nm':rms,'ensemble_rms_nm':norm,
                    'projection':'piston, physical tilt and all four target modes; controlled benchmark, not measured AO'}

def physical_fit(model, images, design, initial=None, max_nfev=30, two_steps=False):
    """Nominal residual-free approximation; fits nuisance flux/background/centroid, no truth."""
    y = np.asarray(images,float)
    channels = 1 if design == 'single' else 2
    if y.shape != (channels,16,16) or not np.isfinite(y).all():
        return {'status':'NO_ESTIMATE','coeff_nm':None}
    capture=None
    if initial is None and model.n>512 and not two_steps:
        # n-squared propagation dominates multi-start cost; refine its blind coarse solution.
        if not hasattr(model,'_capture_model'):
            model._capture_model=OpticalModel(model.config,pupil_n=512,pixel_order=6,basis=model.basis)
        capture=physical_fit(model._capture_model,y,design,max_nfev=max_nfev)
        initial=np.asarray(capture['coeff_nm'])
    d = ([0,model.config['diversity_nm'],0,0],) if channels==1 else ([0,0,0,0],[model.config['diversity_nm'],0,0,0])
    variance = np.maximum(y,0)+model.config['read_noise_e']**2
    scale = np.sqrt(variance)
    starts = [np.zeros(4)] if initial is None else [np.asarray(initial,float)]
    if initial is None and not two_steps:
        starts += [np.array([v,0.,0.,0.]) for v in (-100.,100.)]
    def templates(a):
        return np.array([model.image(a[:4],diversity_nm=v,centroid_mas=a[4:6]) for v in d])
    def residual(a):
        p = templates(a)
        fitted = []
        for pp,yy,ss in zip(p,y,scale):
            x = np.c_[pp.ravel(),np.ones(256)]/ss.ravel()[:,None]
            flux,bg = np.linalg.lstsq(x,yy.ravel()/ss.ravel(),rcond=None)[0]
            fitted.append((max(flux,0)*pp+bg-yy)/ss)
        return np.asarray(fitted).ravel()
    best = None
    for start in starts:
        solution = least_squares(residual,np.r_[np.clip(start,-LIMITS,LIMITS),0.,0.],
            bounds=(np.r_[-LIMITS,-30.,-30.],np.r_[LIMITS,30.,30.]),
            x_scale=np.r_[LIMITS,20.,20.],max_nfev=3 if two_steps else max_nfev,
            ftol=1e-7,xtol=1e-7,gtol=1e-7,diff_step=1e-3)
        if best is None or np.linalg.norm(solution.fun)<np.linalg.norm(best.fun):
            best = solution
    return {'status':'OK' if best.success or two_steps else 'ITERATION_CAP',
            'coeff_nm':best.x[:4].tolist(),'centroid_mas':best.x[4:6].tolist(),
            'cost':float(best.fun@best.fun),'nfev':int(best.nfev),
            'blind_capture':None if capture is None else {'pupil_n':512,'fixed_starts':3,'status':capture['status'],'nfev':capture['nfev']},
            'approximation':'static calibrated PSF with fitted nuisance; no exposure residual oracle'}

def linearized_fit(model,images,design,initial=None,iterations=5,exact_iterations=False):
    """Calibrated LiFT-style analytical derivative update; not an upstream reproduction."""
    y = np.asarray(images,float)
    if not np.isfinite(y).all():
        return {'status':'NO_ESTIMATE','coeff_nm':None}
    d = ([0,model.config['diversity_nm'],0,0],) if design=='single' else ([0,0,0,0],[model.config['diversity_nm'],0,0,0])
    a = np.r_[np.zeros(4) if initial is None else np.clip(np.asarray(initial,float),-LIMITS,LIMITS),0.,0.]
    for _ in range(iterations):
        columns=[]; differences=[]
        for yy,div in zip(y,d):
            p,j = model.image(a[:4],div,centroid_mas=a[4:],derivatives='nuisance')
            variance = np.maximum(yy,0)+model.config['read_noise_e']**2
            root = np.sqrt(variance.ravel())
            nuis = np.c_[p.ravel(),np.ones(256)]/root[:,None]
            flux,bg = np.linalg.lstsq(nuis,yy.ravel()/root,rcond=None)[0]
            flux = max(flux,0)
            r = (yy-flux*p-bg).ravel()/root
            jac = flux*j.reshape(6,-1).T/root[:,None]
            q = np.linalg.qr(nuis)[0]
            columns.append(jac-q@(q.T@jac)); differences.append(r-q@(q.T@r))
        step = np.linalg.lstsq(np.vstack(columns),np.concatenate(differences),rcond=1e-8)[0]
        bounds=np.r_[LIMITS,30.,30.]
        a = np.clip(a+np.clip(step,-np.r_[np.full(4,50.),5.,5.],np.r_[np.full(4,50.),5.,5.]),-bounds,bounds)
        if not exact_iterations and np.linalg.norm(step)<.01:
            break
    return {'status':'OK','coeff_nm':a[:4].tolist(),'centroid_mas':a[4:].tolist(),
            'approximation':'four-mode LiFT-style nominal Jacobian with fitted centroid/flux/background; unknown residual'}
