# Frozen OOPAO validation

`cross_validate.validate(run_dir, max_seconds=600, parent_count=100, acquisitions=5)`
is the final validation boundary. It requires a passed HCIPy readiness record,
a checksummed training manifest, and `models/frozen.json` with checksummed model
artifacts. It neither trains a network nor changes interval calibration.

## Reviewed source and runtime

The exact source pin is
`e8e9aa60cf99f4ab21a4dae7c29aae9b9ec6ec87` of
[cheritier/OOPAO](https://github.com/cheritier/OOPAO/tree/e8e9aa60cf99f4ab21a4dae7c29aae9b9ec6ec87).
The execution owner obtains and reviews source before using it. Place the source
under `benchmark/vendor/OOPAO`, with `source-pin.json` containing `revision` and
`files` (relative paths to SHA256 checksums). The adapter verifies the complete
`OOPAO/**/*.py` inventory and refuses altered or additional Python modules.

Pinned native APIs inspected for this adapter are
[Telescope.PropagateField](https://github.com/cheritier/OOPAO/blob/e8e9aa60cf99f4ab21a4dae7c29aae9b9ec6ec87/OOPAO/Telescope.py),
[Atmosphere.initializeAtmosphere, generateNewPhaseScreen and update](https://github.com/cheritier/OOPAO/blob/e8e9aa60cf99f4ab21a4dae7c29aae9b9ec6ec87/OOPAO/Atmosphere.py),
and [Source](https://github.com/cheritier/OOPAO/blob/e8e9aa60cf99f4ab21a4dae7c29aae9b9ec6ec87/OOPAO/Source.py).
Use the benchmark's locked NumPy/SciPy environment, with installed Astropy,
Matplotlib, jsonpickle, joblib and scikit-image. Record all resolved versions in the
execution owner's dependency lock. Upstream `requirements.txt` contains
incompatible duplicate version pins; it is not a usable installation lock.

The adapter selects OOPAO CPU, double precision and nonresident arrays before
import, keeping its atmosphere on CPU. The native `Telescope.PropagateField`
dispatches on the supplied amplitude array: verified existing CuPy float64
amplitude/phase arrays use its unchanged CUDA FFT path. There is no functional
`isGpu` switch in this telescope class.
The adapter checks device availability, memory headroom and a small executed
FP64 arithmetic probe. It records the actual backend and any explicit CPU
fallback reason. Native FFT sizes above 4096 remain on CPU, including the
8192-square refinement path, to protect an 8 GiB GPU from large workspaces.
No GPU dependency is installed automatically; the execution owner freezes the
existing CuPy/CUDA dependency identity. Its kernel cache stays project-local.
On Windows the adapter locates the installed NVIDIA runtime, NVRTC and cuFFT
wheels through distribution metadata before importing CuPy. It sets the process
CUDA header path, prepends their DLL directories and retains native DLL-directory
handles. Resolved paths and package versions are recorded; no machine-wide
environment setting or installation changes. The cache defaults to
`benchmark/runs/cupy-cache`.
The upstream package initializer prints a banner and overwrites a legacy
precision file selected from `sys.path`. The adapter registers a package
namespace and imports the checksummed scientific modules directly, preserving
them unchanged while skipping this initializer. `runtime.py` reads the explicit
precision setting. The bypass is recorded in validation evidence.
The public parameters are the historical Keck/TRICK case described in
`TELESCOPE_AO.md`, with a declared residual approximation rather than measured
or proprietary AO telemetry.

## What is independent

OOPAO's native Fourier propagation computes every validation PSF. Its native
multilayer `Atmosphere` generates and evolves new phase screens using different
parent and developer seeds. No HCIPy propagation or HCIPy residual realization
is used to produce the independent dataset.

The pupil cell-area quadrature, frozen modes, photon bandpass, detector pixel
quadrature and detector statistics are shared experimental inputs. Shared
detector statistics do not establish an independent detector implementation.
The mask is fractional *geometric cell area*, not physical gray transmission:
OOPAO propagates the weighted pupil samples and normalizes to the corresponding
continuous pupil area. Its native pupil setter casts to Boolean; direct
`PropagateField` preserves the supplied weighted amplitude.

OOPAO's full Fourier plane retains its unitary energy scaling. Transfer only
the Fourier patch covering detector coordinates with a 32-pixel halo to CPU.
The complex field removes the original full FFT array-origin phase before
quintic interpolation, angular quadrature and detector pixel integration.
The full FFT's origin and frequency spacing remain unchanged by the patch.
Known complex 128/256-square controls require patch-versus-full relative L1 at
most `1e-10`. The cropped image is never renormalized. Broadband wavelengths
retain their detected-photon weights.
Fourfold versus eightfold padding must agree below relative L1 `1e-3`;
identical pupil/OPD HCIPy versus OOPAO images must also agree below `1e-3` before
independent observations are generated. This tests matched sampled optics;
continuous pupil and temporal convergence remain the main readiness checks.
The identical-input gate includes a signed `(3, -7)` mas centroid. Independent
nominal acquisitions draw unknown centroids uniformly within ±20 mas on each
axis, matching the training range. Centroid truth is saved as generator metadata
and is excluded from estimator inputs.
Each nominal parent also receives a constant independently seeded unknown
calibration error with RMS drawn up to 20 nm. It is added after atmospheric
scaling; pure AO statistics and the developer gain exclude that static map.
Only its seed and actual RMS are retained as offline diagnostics, never model
inputs.

The evolving residual applies the declared spatial AO filter independently to
OOPAO OPD. It excludes piston, tip/tilt and the four injected slow modes using
the frozen pupil-weighted basis. Independent developer screens calibrate one
global gain for nominal 100/200 nm levels; sequence and frame fluctuations
remain. The amplitude filter is `k²/(k²+kc²)`, matching the training condition.
Use the declared atmospheric `screen_n`, with cubic interpolation to the optical
pupil grid, so an enlarged validation screen does not silently increase its
spatial bandwidth. Convert the OOPAO wind angle to preserve the declared
physical x/y velocity. Save radial PSD summaries and temporal lag-one correlations alongside
the dataset. Geometric phase turbulence excludes scintillation. This is neither
a full AO loop nor an on-sky telescope residual validation.

Eight independent 10 ms developer sequences per engine compare normalized
radial PSD shape and lag-one correlation. Declared initial tolerances are PSD
L1 distance at most 0.35 and absolute correlation difference at most 0.05.
These are practical distribution checks, not proof that the stochastic
processes are identical. Failed matching labels the final result
`PARTIAL / ATMOSPHERIC_DISTRIBUTION_SHIFT`; frozen inference may still describe
the independent engine challenge. No held-out observation tunes this gate.

## Evidence and failure behavior

Create new evidence under `run_dir/oopao`; refuse existing output. Save
`validation.json`, `atmosphere_statistics.json`, `oopao_dataset.json` and one
checksummed NPZ shard per completed independent parent. Shard paths are relative
to the external manifest, with the same image/label shapes as the training
manifest. Inference receives count images and frozen calibration only.

`validation.json` separates optical disagreement from frozen estimator metrics.
`FAIL / OPTICAL_DISAGREEMENT` stops before independent dataset generation.
Budget expiry preserves completed parents and reports `PARTIAL / TIME_BUDGET`;
it never claims the requested parent count completed. Missing pins, artifacts
or dependencies remain explicit errors. The execution owner must supervise
individual native calls, because Python deadline checks occur at natural yields.

`python benchmark/cross_validate.py` performs small interpolation convention,
Parseval-energy and full-versus-patch controls. Those checks do not exercise
OOPAO, the atmospheric engine, CUDA or trained
networks. Only saved real-run evidence establishes those behaviors. Model error,
cross-engine optical disagreement and missing validation are separate outcomes;
a simulator pass does not imply estimator improvement or telescope readiness.
