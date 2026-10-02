# Four-mode PSF benchmark implementation

Authorized by the user's 30 September 2026 implementation request, then explicit authorization for necessary parallel subagents. Root remains the sole scientific execution/repair owner. Four bounded children own telescope_case.json/TELESCOPE_AO.md, learning.py/test_learning.py, cross_validate.py/OOPAO_VALIDATION.md, and read-only physics review, respectively. No remote services, proprietary data or overlapping edits. New implementation lives only in benchmark/. Existing design/ and literature/ work is preserved. Static reference geometry is reused read-only with dependency hashes.

Scope: HCIPy 0.7.1 generation, signed slow focus/two astigmatisms/primary spherical aberration, single and paired acquisitions, local learned/classical comparison, then frozen OOPAO cross-engine validation. Latest steering requires a known telescope/AO first case: published Keck/TRICK pupil, atmosphere, AO cadence and residual envelope. Public parameters do not establish measured residual time-series; any idealized correction is labeled a surrogate. TMT/ELT/GMT generation is conditional on this first benchmark passing.

Start: 2026-09-30 20:50 UTC. Implementation estimate 90 minutes, warning 120 minutes, bounded checkpoint 180 minutes. Scientific compute ceilings: 16 CPU-hours, 2 GPU-hours, 16 GiB RAM, 4 GiB retained data. Natural yields and child progress are observed at most 30 seconds apart. A measured 100-acquisition canary selects 20, 10 or 5 acquisitions per independent sequence without dropping difficult conditions. Physics failure blocks dataset generation and training.

Readiness evidence will record source/configuration/dependency identities, runtime versions, pupil/basis conventions, optical and count-moment checks, temporal refinement, recoverability and ambiguity checks, actual timing, bounded output/cancellation/resume behavior, and explicit limitations. One bounded repair follows a failed boundary; a second distinct post-admission failure invalidates readiness. Frozen test data cannot tune model/calibration choices.

The project-local venv inherits the already verified science environment to reuse its installed CUDA PyTorch. Additional simulator dependencies are installed only in that venv. Record its full resolved distribution list, local simulator pin and shared base identity; a clean isolated rebuild must use the recorded CUDA wheel index/version. This is an explicit environment dependency, not a claim of a self-contained venv.

Checkpoint 2026-10-01 07:45 UTC: the implementation estimate was exceeded by
the necessary independent propagation, nuisance-profiled signed-recovery and
joint pupil/pixel/wavelength controls. The verified final-resolution gate took
1,527 s wall and 1,507 s CPU; earlier development gates were preserved.
There was an approximately eight-hour idle/delivery gap after that completed
run, with no dataset generation or model training in that interval. Source
identity is now changing once more for the user's newly requested portable
hardware setup and pinned CUDA backend. Existing gate evidence remains useful
numerical evidence but does not admit this changed source.

One estimate revision: the remaining scientific execution is governed by the
measured 100-acquisition canary rather than the original 90-minute coding
estimate. Hard resource ceilings and scientific scope remain unchanged.
Reserve 20% of the CPU ceiling for independent validation, comparisons and
earlier development checks. If even five acquisitions per parent cannot fit,
stop with a resource-limit report. Root remains the single execution owner.
The new hardware worker owns only hardware.py/test_hardware.py; the final
physics reviewer is read-only and has completed its review. New source must
pass a fresh physics gate before production.
