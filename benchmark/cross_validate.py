"""Frozen, source-pinned OOPAO validation; never fits a network or calibration.

OOPAO performs both pupil propagation and the evolving atmospheric realization.
Shared pupil/basis, bandpass, pixel quadrature and detector statistics express
matched experimental inputs, not independent implementations of those inputs.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.ndimage import map_coordinates
from scipy import ndimage


OOPAO_REVISION = "e8e9aa60cf99f4ab21a4dae7c29aae9b9ec6ec87"
VENDOR = Path(__file__).resolve().parent / "vendor" / "OOPAO"
_CUDA_DLL_HANDLES = []
_CUDA_BOOTSTRAP = None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _vendor_identity() -> dict:
    """Refuse unreviewed, unpinned, changed or additional Python source files."""
    manifest = _json(VENDOR / "source-pin.json")
    if manifest["revision"] != OOPAO_REVISION:
        raise RuntimeError("OOPAO revision differs from the reviewed pin")
    listed = {name: digest for name, digest in manifest["files"].items()
              if name.startswith("OOPAO/") and name.endswith(".py")}
    actual = {p.relative_to(VENDOR).as_posix(): _sha(p)
              for p in (VENDOR / "OOPAO").rglob("*.py")}
    if listed != actual:
        raise RuntimeError("OOPAO Python source does not match source-pin.json")
    if "OOPAO/Telescope.py" not in listed or "OOPAO/Atmosphere.py" not in listed:
        raise RuntimeError("OOPAO source pin lacks required propagation/atmosphere")
    os.environ["OOPAO_BACKEND"] = "cpu"
    os.environ["OOPAO_PRECISION"] = "64"
    os.environ["OOPAO_GPU_RESIDENT"] = "0"
    loaded = sys.modules.get("OOPAO")
    if loaded is not None and Path(loaded.__file__).resolve().parent != VENDOR / "OOPAO":
        raise RuntimeError("A different OOPAO source is already imported")
    if loaded is None:
        # Upstream __init__ prints a Unicode banner and overwrites a legacy
        # precision file selected by sys.path. It exports no scientific objects.
        # Register a package namespace instead: source science modules are used
        # unchanged, with precision configured explicitly via runtime.py.
        spec = importlib.machinery.ModuleSpec("OOPAO", loader=None, is_package=True)
        spec.submodule_search_locations = [str(VENDOR / "OOPAO")]
        namespace = importlib.util.module_from_spec(spec)
        namespace.__file__ = str(VENDOR / "OOPAO" / "__init__.py")
        sys.modules["OOPAO"] = namespace
    return {"revision": OOPAO_REVISION, "source_pin_sha256": _sha(VENDOR / "source-pin.json"),
            "python_files": len(actual), "backend": "cpu", "precision": 64,
            "package_initialization": "namespace loader bypasses upstream banner and legacy-file overwrite; scientific modules unchanged"}


def _sample_fft_field(field: np.ndarray, angular_spacing: float,
                      xy_rad: np.ndarray) -> np.ndarray:
    """Interpolate OOPAO's complex FFT after removing its array-origin phase.

    The pinned PropagateField's even-size PSF samples lie on half pixels.
    Removing the input-array origin phase before interpolation is essential:
    interpolating raw complex values would attenuate the PSF substantially.
    """
    size = field.shape[0]
    if field.shape != (size, size) or size % 2:
        raise ValueError("OOPAO interpolation requires a full even-sized FFT")
    coordinates = xy_rad / angular_spacing + (size - 1) / 2
    if np.any(coordinates < 3) or np.any(coordinates > size - 4):
        raise ValueError("Detector extends beyond the OOPAO FFT interpolation support")
    # A quintic spline prefilter is an IIR operation: retain a 32-pixel halo,
    # then verify its boundary truncation against full filtering in _selfcheck.
    lower = np.maximum(0, np.floor(coordinates.min(axis=1)).astype(int) - 32)
    upper = np.minimum(size, np.ceil(coordinates.max(axis=1)).astype(int) + 33)
    x0, y0 = lower
    x1, y1 = upper
    patch = field[y0:y1, x0:x1]
    # Transfer only this patch when native OOPAO returns a CuPy Fourier plane.
    patch = patch.get() if hasattr(patch, "get") else np.asarray(patch)
    frequency_x = np.arange(x0, x1) - (size - 1) / 2
    frequency_y = np.arange(y0, y1) - (size - 1) / 2
    phase_x = np.exp(2j * np.pi * frequency_x * ((size - 1) / 2) / size)
    phase_y = np.exp(2j * np.pi * frequency_y * ((size - 1) / 2) / size)
    # Keep the full native FFT's origin and denominator, even after cropping.
    centered = patch * phase_y[:, None] * phase_x[None, :]
    # xy_rad is [x, y]; scipy indexes [row=y, column=x].
    indexes = np.stack((coordinates[1] - y0, coordinates[0] - x0))
    sampled = (map_coordinates(centered.real, indexes, order=5, mode="constant")
               + 1j * map_coordinates(centered.imag, indexes, order=5, mode="constant"))
    return np.abs(sampled) ** 2


def _native_fft_backend(native_size: int) -> tuple[object, dict]:
    """Select an existing verified GPU; never install a backend or change vendor code."""
    record = {"backend": "numpy_cpu", "precision_bits": 64, "native_fft_size": native_size}
    # An 8192-square complex128 array alone is 1 GiB. Several native temporaries
    # and cuFFT workspaces cannot safely share an 8 GiB laptop GPU at that size.
    if native_size > 4096:
        record["reason"] = "CPU refinement: native FFT exceeds bounded 4096-square GPU path"
        return np, record
    try:
        global _CUDA_BOOTSTRAP
        os.environ.setdefault("CUPY_CACHE_DIR", str(Path(__file__).resolve().parent / "runs" / "cupy-cache"))
        if os.name == "nt" and _CUDA_BOOTSTRAP is None:
            packages = (("nvidia-cuda-runtime-cu12", "nvidia/cuda_runtime"),
                        ("nvidia-cuda-nvrtc-cu12", "nvidia/cuda_nvrtc"),
                        ("nvidia-cufft-cu12", "nvidia/cufft"))
            locations = {name: Path(importlib.metadata.distribution(name).locate_file(relative)).resolve()
                         for name, relative in packages}
            libraries = [location / "bin" for location in locations.values()]
            if not all(path.is_dir() for path in libraries):
                raise RuntimeError("Installed NVIDIA wheel DLL directory is missing")
            os.environ["CUDA_PATH"] = str(locations["nvidia-cuda-runtime-cu12"])
            for directory in libraries:
                _CUDA_DLL_HANDLES.append(os.add_dll_directory(str(directory)))
            os.environ["PATH"] = os.pathsep.join(map(str, libraries)) + os.pathsep + os.environ.get("PATH", "")
            _CUDA_BOOTSTRAP = {"cuda_path": os.environ["CUDA_PATH"],
                               "dll_directories": list(map(str, libraries)),
                               "packages": {name: importlib.metadata.version(name) for name, _ in packages}}
        record["cuda_bootstrap"] = _CUDA_BOOTSTRAP
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("No visible CUDA device")
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        estimated_native_bytes = 6 * native_size ** 2 * 16
        record.update({"cupy_version": cp.__version__, "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
                       "gpu_free_bytes_at_selection": int(free_bytes), "gpu_total_bytes": int(total_bytes),
                       "estimated_native_working_bytes": estimated_native_bytes})
        if free_bytes < estimated_native_bytes + 1024 ** 3:
            raise RuntimeError("Insufficient GPU headroom for native complex128 FFT temporaries")
        # Direct evidence that CuPy can execute and return FP64 arithmetic.
        probe = cp.asarray([1., 2.], dtype=cp.float64)
        if not np.array_equal(cp.asnumpy(probe + probe), [2., 4.]):
            raise RuntimeError("CuPy FP64 probe failed")
        record.update({"backend": "cupy_cuda", "fp64_probe": "PASS",
                       "native_dispatch": "PropagateField get_array_module(amplitude)",
                       "host_transfer": "detector Fourier patch plus 32-pixel halo only"})
        return cp, record
    except Exception as error:
        record["reason"] = f"Explicit CPU fallback: {type(error).__name__}: {error}"
        return np, record


def _make_model(config: dict, padding: int = 4, basis: dict | None = None):
    """Use core's acquisition/noise chain with an independently propagated PSF."""
    if __package__:
        from .core import OpticalModel
    else:
        from core import OpticalModel
    # Bootstrap before native imports: OOPAO.tools imports CuPy's dispatcher.
    native_array_backend, native_backend_info = _native_fft_backend(padding * int(config["pupil_n"]))
    from OOPAO.Telescope import Telescope
    from OOPAO.Source import Source

    class OopaoModel(OpticalModel):
        def __init__(self):
            super().__init__(config, engine="hcipy", basis=basis)
            diameter = float(self.n * self.dx)
            # OOPAO casts pupil to bool. Reflectivity carries fractional amplitude.
            with contextlib.redirect_stdout(io.StringIO()):
                self.tel = Telescope(self.n, diameter, pupil=np.asarray(self.mask) > 0)
                self.src = Source("H", magnitude=0, display_properties=False)
                self.src * self.tel
            self.tel.pupilReflectivity = np.asarray(self.mask, dtype=float).copy()
            self.fft_padding = padding
            self.native_array_backend, self.native_backend_info = native_array_backend, native_backend_info
            self.native_amplitude = self.native_array_backend.asarray(self.mask, dtype=self.native_array_backend.float64)

        def image(self, coeff_nm, diversity_nm=None, residual_nm=None,
                  centroid_mas=(0, 0), derivatives=False):
            if derivatives:
                values = self.image(coeff_nm, diversity_nm, residual_nm, centroid_mas)
                steps = np.eye(4) * 0.1
                jac = np.stack([(self.image(np.asarray(coeff_nm) + step, diversity_nm,
                                           residual_nm, centroid_mas)
                                - self.image(np.asarray(coeff_nm) - step, diversity_nm,
                                             residual_nm, centroid_mas)) / 0.2
                               for step in steps])
                return values, jac
            coefficients = np.asarray(coeff_nm, dtype=float)
            diversity = np.zeros(4) if diversity_nm is None else np.asarray(diversity_nm, dtype=float)
            if diversity.shape != (4,) or not np.isfinite(diversity).all():
                raise ValueError("Four finite diversity coefficients required")
            if diversity_nm is not None:
                coefficients = coefficients + diversity
            if coefficients.shape != (4,) or not np.isfinite(coefficients).all():
                raise ValueError("Four finite modal coefficients required")
            opd_nm = np.einsum("i,ijk->jk", coefficients, self.modes_nm_units) + self.static_nm
            if residual_nm is not None:
                residual = np.asarray(residual_nm)
                if residual.shape != opd_nm.shape or np.iscomplexobj(residual) or not np.isfinite(residual).all():
                    raise ValueError("Finite matching real residual map required")
                opd_nm = opd_nm + residual
            result = np.zeros((16, 16), dtype=float)
            xx, yy = np.meshgrid(self.pixel_nodes_rad, self.pixel_nodes_rad)
            xy = np.stack((xx.ravel(), yy.ravel()))
            shift_rad = np.asarray(centroid_mas) * np.pi / (180 * 3600 * 1000)
            if shift_rad.shape != (2,) or not np.isfinite(shift_rad).all():
                raise ValueError("Two finite centroid coordinates required")
            xy = xy.reshape(2, -1) - shift_rad[:, None]
            amplitude = np.asarray(self.mask, dtype=float)
            # mask is geometric cell-area quadrature, not a gray transmission.
            # The optical normalization is the continuous sampled pupil area.
            energy = float(np.sum(amplitude))
            for wavelength, spectral_weight in zip(self.wavelengths, self.spectral_weights):
                phase = self.native_array_backend.asarray(2 * np.pi * opd_nm * 1e-9 / wavelength,
                                                         dtype=self.native_array_backend.float64)
                self.tel.PropagateField(self.native_amplitude, phase,
                                        self.fft_padding,
                                        img_resolution=self.fft_padding * self.n)
                spacing = wavelength / (self.n * self.dx * self.fft_padding)
                density = _sample_fft_field(self.tel.focal_EMF, spacing, xy)
                # Unit-normalized full-plane density, never normalize the detector crop.
                density = density / (energy * spacing ** 2)
                result += float(spectral_weight) * self.integrate_density(density)
            if not np.all(np.isfinite(result)) or np.min(result) < 0:
                raise RuntimeError("Invalid OOPAO propagated PSF")
            return result

    return OopaoModel()


def _matched_atmosphere(model, config: dict):
    """Build native multilayer OOPAO turbulence, without HCIPy screen reuse."""
    from OOPAO.Atmosphere import Atmosphere
    from OOPAO.Telescope import Telescope
    from OOPAO.Source import Source
    a = config["atmosphere"]
    sampling_time = float(config["exposure_s"] / config["phase_steps"])
    reference = float(a["r0_wavelength_m"])
    r0_at_500nm = float(a["r0_m"]) * (500e-9 / reference) ** (6 / 5)
    with contextlib.redirect_stdout(io.StringIO()):
        # Match the declared atmospheric sampling, including finite-grid bandwidth.
        tel = Telescope(a["screen_n"], model.n * model.dx, samplingTime=sampling_time)
        src = Source("H", magnitude=0, display_properties=False)
        src * tel
        atm = Atmosphere(tel, r0=r0_at_500nm, L0=float(a["L0_m"]),
                         windSpeed=a["wind_m_s"], fractionalR0=a["fractions"],
                         windDirection=(90 - np.rad2deg(a["wind_direction_rad"])).tolist(), altitude=a["heights_m"],
                         angular_spectrum_propagation=False, mode=2, param=None)
        atm.initializeAtmosphere(tel,compute_covariance=True)
        tel + atm
    return atm


def _filtered_residual(opd_m: np.ndarray, model, cutoff_cycles_per_m: float) -> np.ndarray:
    """Declared spatial AO approximation, independently applied to OOPAO OPD."""
    screen_n = opd_m.shape[0]
    f = np.fft.fftfreq(screen_n, d=model.n * model.dx / screen_n)
    frequency = np.hypot(f[:, None], f[None, :])
    transfer = frequency ** 2 / (frequency ** 2 + cutoff_cycles_per_m ** 2)
    result = np.fft.ifft2(np.fft.fft2(opd_m) * transfer).real * 1e9
    if screen_n != model.n:
        result = ndimage.zoom(result, model.n / screen_n, order=3,
                              mode="nearest", prefilter=True, grid_mode=True)
    pupil = np.asarray(model.mask)
    # The injected four slowly varying targets are excluded from the disturbance.
    # Project the exact frozen basis along with atmospheric piston and tip/tilt.
    axis = (np.arange(model.n) - (model.n - 1) / 2) * model.dx
    xx, yy = np.meshgrid(axis, axis)
    basis = np.concatenate((np.ones((1, model.n, model.n)), xx[None], yy[None],
                            np.asarray(model.modes_nm_units)))
    flat = basis.reshape(7, -1)
    weights = pupil.ravel()
    gram = (flat * weights) @ flat.T
    coefficients = np.linalg.solve(gram, (flat * weights) @ result.ravel())
    result -= np.einsum("i,ijk->jk", coefficients, basis)
    return result


def _residual_stats(frames: np.ndarray, mask: np.ndarray) -> dict:
    weight = mask
    values = frames[:, mask > 0]
    rms = np.sqrt(np.sum(frames ** 2 * weight, axis=(1, 2)) / weight.sum())
    correlation = None
    if len(frames) > 1:
        correlation = float(np.mean(np.sum(values[1:] * values[:-1], axis=1)
                            / np.maximum(np.linalg.norm(values[1:], axis=1)
                                         * np.linalg.norm(values[:-1], axis=1), 1e-30)))
    power = np.mean(np.abs(np.fft.fft2(frames, axes=(1, 2))) ** 2, axis=0)
    f = np.fft.fftfreq(frames.shape[1])
    r = np.hypot(f[:, None], f[None, :])
    edges = np.linspace(0, np.sqrt(0.5), 17)
    radial = [float(power[(r >= lo) & (r < hi)].mean())
              for lo, hi in zip(edges[:-1], edges[1:])]
    return {"rms_mean_nm": float(rms.mean()), "rms_min_nm": float(rms.min()),
            "rms_max_nm": float(rms.max()), "lag_one_correlation": correlation,
            "radial_psd_power": radial, "radial_psd_edges_cycles_per_pixel": edges.tolist()}


def _developer_statistics(model, atmosphere, config: dict, cutoff: float) -> tuple[float, dict]:
    """Compare simulator nuisance ensembles before opening held-out observations.

    HCIPy is used only to measure the reference distribution in this gate;
    final OOPAO observations retain independently generated native screens.
    These tolerances are declared before measuring, not fitted to test results.
    """
    if __package__:
        from .core import ResidualSequence
    else:
        from core import ResidualSequence
    dt = config["exposure_s"] / config["phase_steps"]
    reference, independent, powers = [], [], []
    native_seeds = list(range(1_500_000_000, 1_500_000_008))
    reference_seeds = list(range(1_400_000_000, 1_400_000_008))
    for native_seed, reference_seed in zip(native_seeds, reference_seeds):
        with contextlib.redirect_stdout(io.StringIO()):
            atmosphere.generateNewPhaseScreen(native_seed)
        frames = []
        for i in range(config["phase_steps"]):
            atmosphere.telescope.samplingTime = dt / 2 if i == 0 else dt
            with contextlib.redirect_stdout(io.StringIO()):
                atmosphere.update()
            frames.append(_filtered_residual(np.asarray(atmosphere.OPD), model, cutoff))
        frames = np.asarray(frames)
        powers.append(float(np.sum(frames ** 2 * model.mask) / (len(frames) * np.sum(model.mask))))
        independent.append(_residual_stats(frames, model.mask))
        reference_frames = ResidualSequence(model, reference_seed).acquisition(0)
        reference.append(_residual_stats(reference_frames, model.mask))
    base_rms = float(np.sqrt(np.mean(powers)))
    if not np.isfinite(base_rms) or base_rms <= 0:
        raise RuntimeError("OOPAO developer calibration produced no residual power")
    spectra = [np.mean([item["radial_psd_power"] for item in engine], axis=0)
               for engine in (reference, independent)]
    normalized = [power / power.sum() for power in spectra]
    spectral_distance = float(np.sum(np.abs(normalized[0] - normalized[1])))
    correlations = [float(np.mean([item["lag_one_correlation"] for item in engine]))
                    for engine in (reference, independent)]
    lag_difference = float(abs(correlations[0] - correlations[1]))
    matched = spectral_distance <= .35 and lag_difference <= .05
    return base_rms, {"status": "PASS" if matched else "PARTIAL",
        "oopao_seeds": native_seeds, "hcipy_reference_seeds": reference_seeds,
        "independent_oopao_raw_rms_nm": base_rms,
        "normalization": "independent global simulator gains; not per frame, parent or test",
        "oopao_sequences": independent, "hcipy_reference_sequences": reference,
        "normalized_radial_psd": {"hcipy": normalized[0].tolist(), "oopao": normalized[1].tolist()},
        "normalized_radial_psd_l1": spectral_distance,
        "mean_lag_one_correlation": {"hcipy": correlations[0], "oopao": correlations[1]},
        "lag_one_absolute_difference": lag_difference,
        "declared_tolerances": {"normalized_radial_psd_l1_max": .35, "lag_one_difference_max": .05},
        "tolerance_scope": "initial practical distribution gate; not proof of identical stochastic processes"}


def validate(run_dir: Path, max_seconds=600, parent_count=100, acquisitions=5) -> dict:
    """Validate frozen model, writing new evidence and independent OOPAO shards.

    Budget expiry preserves completed parents and returns PARTIAL, never PASS.
    Physics disagreement returns FAIL before generating inference observations.
    Root execution owner remains responsible for bounding individual native calls.
    """
    start = time.monotonic()
    run_dir = Path(run_dir).resolve()
    if parent_count < 1 or acquisitions < 1 or max_seconds <= 0:
        raise ValueError("Positive parent count, acquisitions and time budget required")
    frozen_path = run_dir / "models" / "frozen.json"
    if not frozen_path.is_file():
        raise RuntimeError("OOPAO validation requires models/frozen.json before test access")
    frozen = _json(frozen_path)
    config = _json(run_dir / "config.json")
    readiness = _json(run_dir / "readiness.json")
    dataset = _json(run_dir / "dataset.json")
    if __package__:
        from .core import digest, source_hash
    else:
        from core import digest, source_hash
    if frozen.get("config_hash") != digest(config) or frozen.get("source_hash") != source_hash():
        raise RuntimeError("Current configuration/source differs from frozen training identity")
    if readiness.get("status") != "PASS":
        raise RuntimeError("Readiness must pass before OOPAO validation")
    for identity in ("config_hash", "source_hash"):
        if not frozen.get(identity) or frozen[identity] != dataset.get(identity):
            raise RuntimeError(f"Frozen model / training dataset {identity} mismatch")
        if readiness.get(identity) != frozen[identity]:
            raise RuntimeError(f"Readiness / frozen model {identity} mismatch")
    if _sha(run_dir / "dataset.json") != frozen["dataset_manifest_sha256"]:
        raise RuntimeError("Training dataset manifest changed after freeze")
    for name, checksum in frozen["artifacts"].items():
        artifact = (run_dir / "models" / name).resolve()
        if not artifact.is_relative_to(run_dir / "models") or _sha(artifact) != checksum:
            raise RuntimeError(f"Frozen model artifact changed: {name}")
    output = run_dir / "oopao"
    output.mkdir(exist_ok=False)
    evidence = {"status": "RUNNING", "frozen_sha256": _sha(frozen_path),
                "config_hash": frozen["config_hash"], "source_hash": frozen["source_hash"],
                "requested_parents": parent_count, "acquisitions_per_parent": acquisitions,
                "independent_atmosphere": True, "independent_propagation": True,
                "shared_chain": ["pupil/basis", "bandpass", "pixel quadrature", "detector statistics"],
                "limitations": ["Filtered phase residual, not a full telescope AO controller",
                                "Geometric turbulence; scintillation is not enabled",
                                "Validation does not establish bench or on-sky performance"]}
    try:
        evidence["oopao"] = _vendor_identity()
        model = _make_model(config, basis=frozen["basis"])
        if frozen.get("measurement_hash") != model.measurement_hash:
            raise RuntimeError("OOPAO measurement pupil/basis differs from frozen model")
        refined = _make_model(config, padding=8, basis=frozen["basis"])
        evidence["native_fft_backends"] = {"data": model.native_backend_info,
                                           "refinement": refined.native_backend_info,
                                           "atmosphere": "numpy_cpu"}
        evidence["fourier_sampling"] = {"default_padding": 4, "refinement_padding": 8,
                                        "relative_l1_threshold": 1e-3}
        evidence["pupil_quadrature"] = {"pupil_area_m2": model.pupil_power_m2,
                "native_fft_discrete_to_continuous_energy_ratio": float(np.sum(model.mask ** 2) / np.sum(model.mask)),
                "interpretation": "boundary area quadrature, not gray transmission; core continuum gate governs discretization"}
        trials = [np.zeros(4), np.array([120., -80., 60., -70.]),
                  np.array([-120., 80., -60., 70.])]
        cases = [(coefficients, diversity, (0., 0.)) for coefficients in trials
                 for diversity in (np.array([0., 200., 0., 0.]), np.array([200., 0., 0., 0.]))]
        cases.append((np.array([73., 42., -61., 27.]), np.array([0., 200., 0., 0.]), (3., -7.)))
        optical = []
        for coefficients, diversity, centroid in cases:
            actual = model.image(coefficients, diversity_nm=diversity, centroid_mas=centroid)
            fine = refined.image(coefficients, diversity_nm=diversity, centroid_mas=centroid)
            # Explicit base implementation call: HCIPy reference only in this gate.
            expected = super(type(model), model).image(coefficients, diversity_nm=diversity,
                                                      centroid_mas=centroid)
            optical.append({"coefficients_nm": coefficients.tolist(),
                            "diversity_nm": diversity.tolist(), "centroid_mas": list(centroid),
                            "relative_l1": float(np.sum(np.abs(actual - expected)) / expected.sum()),
                            "padding_relative_l1": float(np.sum(np.abs(actual - fine)) / fine.sum()),
                            "captured_fraction": float(actual.sum())})
        evidence["identical_input_optics"] = optical
        if any(c["relative_l1"] >= 1e-3 or c["padding_relative_l1"] >= 1e-3 for c in optical):
            evidence["status"] = "FAIL"
            evidence["failure_class"] = "OPTICAL_DISAGREEMENT"
            return evidence
        del refined
        atmosphere = _matched_atmosphere(model, config)
        a = config["atmosphere"]
        dt = float(config["exposure_s"] / config["phase_steps"])
        samples = int(round(0.010 / dt))
        if samples < 2 or samples % 2 or abs(samples * dt - .010) > 1e-12:
            raise ValueError("OOPAO time samples must divide 10 ms and split equally into 5 ms")
        cutoff = float(a["kc"]) / float(config["diameter_m"])
        evidence["atmosphere_sampling"] = {"native_screen_n": a["screen_n"], "optical_pupil_n": model.n,
                "upsampling": "cubic map interpolation; same finite-grid bandwidth as HCIPy",
                "wind_conversion": "OOPAO angle_deg=90-HCIPy angle_deg for matching x,y velocity"}
        ranges = np.array([150., 100., 100., 100.])
        shards, stats = [], []
        base_rms, matched_statistics = _developer_statistics(model, atmosphere, config, cutoff)
        evidence["matched_developer_statistics"] = matched_statistics
        for parent in range(parent_count):
            if time.monotonic() - start >= max_seconds:
                break
            seed = 1_600_000_000 + parent
            rng = np.random.default_rng(seed)
            with contextlib.redirect_stdout(io.StringIO()):
                atmosphere.generateNewPhaseScreen(seed)
            frames = []
            for frame_index in range(samples * acquisitions):
                atmosphere.telescope.samplingTime = dt / 2 if frame_index == 0 else dt
                with contextlib.redirect_stdout(io.StringIO()):
                    atmosphere.update()
                frames.append(_filtered_residual(np.asarray(atmosphere.OPD), model, cutoff))
            frames = np.asarray(frames)
            # Global gain from independent developer calibration; preserve fluctuations.
            target = 100. if parent % 2 == 0 else 200.
            if not np.all(np.isfinite(frames)):
                raise RuntimeError("OOPAO atmospheric screen has no finite residual power")
            frames *= target / base_rms
            stats.append({"parent_id": f"oopao_{parent:04d}", "seed": seed,
                          "target_sequence_rms_nm": target, **_residual_stats(frames, model.mask)})
            # Keep AO statistics/gain pure. A separately seeded unknown error is
            # constant for this parent and is never an estimator input.
            calibration_seed = seed + 71_003
            static_error = np.asarray(model.calibration_error(calibration_seed))
            if (static_error.shape != frames.shape[1:] or np.iscomplexobj(static_error)
                    or not np.isfinite(static_error).all()):
                raise RuntimeError("Invalid parent calibration-error map")
            static_rms = float(np.sqrt(model.weights @ static_error.ravel() ** 2))
            stats[-1].update({"ao_statistics_exclude_static_calibration_error": True,
                             "calibration_error_seed": calibration_seed,
                             "calibration_error_rms_nm": static_rms,
                             "calibration_error_max_rms_nm": config["calibration_error_rms_nm"]})
            frames += static_error[None, :, :]
            rows, centroids = [], []
            for item in range(acquisitions):
                coefficient = rng.uniform(-ranges, ranges)
                flux = float(config["flux_e"][item % len(config["flux_e"])])
                centroid = rng.uniform(-20., 20., 2)
                centroids.append(centroid)
                rows.append(model.acquire(coefficient, frames[item * samples:(item + 1) * samples],
                                          rng, flux_e=flux, condition="nominal", centroid_mas=centroid))
            name = f"parent_{parent:04d}.npz"
            path = output / name
            with path.open("xb") as stream:
                np.savez_compressed(stream,
                                    images_single=np.stack([row["images_single"] for row in rows]).astype("float32"),
                                    images_pair=np.stack([row["images_pair"] for row in rows]).astype("float32"),
                                    labels_nm=np.stack([row["labels_nm"] for row in rows]).astype("float32"),
                                    flux_e=np.asarray([row["flux_e"] for row in rows], dtype="float32"),
                                    seed=np.asarray(seed, dtype="int64"),
                                    condition=np.asarray(["nominal"] * acquisitions),
                                    acquisition_index=np.arange(acquisitions),
                                    centroid_truth_mas=np.asarray(centroids),
                                    calibration_error_seed=np.asarray(calibration_seed, dtype="int64"),
                                    calibration_error_rms_nm=np.asarray(static_rms),
                                    calibration_identity=np.asarray(model.measurement_hash))
            shards.append({"path": name, "split": "oopao",
                           "parent_id": f"oopao_{parent:04d}", "sha256": _sha(path),
                           "acquisitions": acquisitions})
        manifest = {"schema_version": 1, "shards": shards, "config_hash": frozen["config_hash"],
                    "source_hash": frozen["source_hash"], "oopao_revision": OOPAO_REVISION,
                    "measurement_hash": model.measurement_hash,
                    "basis": model.basis,
                    "frozen_sha256": _sha(frozen_path), "calibration_identity": dataset.get("calibration_identity"),
                    "counts": {"parents": len(shards), "acquisitions": len(shards) * acquisitions}}
        _write(output / "oopao_dataset.json", manifest)
        _write(output / "atmosphere_statistics.json", {"sequences": stats,
               "dt_s": dt, "normalization": "global developer gain, not per parent or frame",
               "ao_filter": "spatial_frequency^2/(spatial_frequency^2+declared_cutoff^2)",
               "projection": "piston, physical tilt, all four slowly varying targets",
               "statistics_scope": "pure atmospheric residual before unknown per-parent static calibration error"})
        evidence["completed_parents"] = len(shards)
        evidence["dataset_sha256"] = _sha(output / "oopao_dataset.json")
        evidence["residual_statistics"] = {"parents": len(stats),
                "mean_rms_nm": float(np.mean([s["rms_mean_nm"] for s in stats])) if stats else None,
                "mean_lag_one_correlation": float(np.mean([s["lag_one_correlation"] for s in stats])) if stats else None,
                "statistics_scope": "pure native AO residual; per-parent static errors excluded; no test-based gain calibration"}
        if __package__:
            from .learning import evaluate
        else:
            from learning import evaluate
        if shards:
            evidence["frozen_inference"] = evaluate(run_dir, manifest_path=output / "oopao_dataset.json")
        distribution_matched = matched_statistics["status"] == "PASS"
        evidence["status"] = "PASS" if len(shards) == parent_count and distribution_matched else "PARTIAL"
        if not distribution_matched:
            evidence["failure_class"] = "ATMOSPHERIC_DISTRIBUTION_SHIFT"
            evidence["interpretation"] = "Independent simulator challenge; declared residual statistics did not match"
        if len(shards) != parent_count:
            evidence["resource_limit"] = "TIME_BUDGET"
            if distribution_matched:
                evidence["failure_class"] = "TIME_BUDGET"
        if _sha(frozen_path) != evidence["frozen_sha256"]:
            raise RuntimeError("Frozen model manifest changed during OOPAO validation")
        return evidence
    except Exception as error:
        evidence["status"] = "BLOCKED"
        evidence["failure_class"] = "UNAVAILABLE_OR_INVALID_DEPENDENCY"
        evidence["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        evidence["elapsed_seconds"] = time.monotonic() - start
        _write(output / "validation.json", evidence)


def _selfcheck() -> None:
    """Small runnable convention check; no simulator/runtime claim."""
    size = 64
    axis = np.arange(size) - (size - 1) / 2
    xx, yy = np.meshgrid(axis, axis)
    centered = np.exp(-(xx ** 2 + yy ** 2) / 100).astype(complex)
    phase = np.exp(2j * np.pi * axis * ((size - 1) / 2) / size)
    raw = centered / phase[:, None] / phase[None, :]
    points = np.array([[-2.5, 0.5, 2.5], [.5, .5, .5]])
    actual = _sample_fft_field(raw, 1., points)
    expected = np.exp(-2 * (points[0] ** 2 + points[1] ** 2) / 100)
    assert np.allclose(actual, expected, rtol=1e-9), (actual, expected)
    # Reproduce only the pinned FFT centering algebra on a tiny synthetic array;
    # compare it to direct pupil quadrature at arbitrary angular coordinates.
    pupil_n = 8
    pupil_axis = np.arange(pupil_n) - (pupil_n - 1) / 2
    px, py = np.meshgrid(pupil_axis, pupil_axis)
    pupil = np.exp(-(px ** 2 + py ** 2) / 20) * np.exp(.1j * px + .07j * py)
    support = np.pad(pupil, (size - pupil_n) // 2)
    indexes = np.arange(size)
    ix, iy = np.meshgrid(indexes, indexes)
    phasor = np.exp(-1j * np.pi * (size + 1) / size * (ix + iy))
    native_convention = np.fft.fft2(support * phasor) / size
    sampled = _sample_fft_field(native_convention, 1., points)
    direct = np.array([abs(np.sum(pupil * np.exp(-2j * np.pi * (x * px + y * py) / size)) / size) ** 2
                       for x, y in points.T])
    assert np.allclose(sampled, direct, rtol=1e-4), (sampled, direct)
    assert np.isclose(np.sum(np.abs(native_convention) ** 2), np.sum(np.abs(pupil) ** 2))
    # Compare the previous full-plane quintic filter with the cropped filter.
    # No propagation engine or CUDA device is used by these numerical controls.
    rng = np.random.default_rng(38192)
    for native_size in (128, 256):
        arbitrary = rng.normal(size=(native_size, native_size)) + 1j * rng.normal(size=(native_size, native_size))
        axis = np.arange(native_size) - (native_size - 1) / 2
        phase = np.exp(2j * np.pi * axis * ((native_size - 1) / 2) / native_size)
        centered = arbitrary * phase[:, None] * phase[None, :]
        queries = rng.uniform(-10, 10, (2, 120))
        coordinates = queries + (native_size - 1) / 2
        indexes = np.stack((coordinates[1], coordinates[0]))
        expected = np.abs(map_coordinates(centered.real, indexes, order=5, mode="constant")
                          + 1j * map_coordinates(centered.imag, indexes, order=5, mode="constant")) ** 2
        actual = _sample_fft_field(arbitrary, 1., queries)
        relative_l1 = float(np.sum(np.abs(actual - expected)) / np.sum(expected))
        assert relative_l1 <= 1e-10, (native_size, relative_l1)


if __name__ == "__main__":
    _selfcheck()
    print("OOPAO adapter interpolation convention self-check passed; simulator not exercised")
