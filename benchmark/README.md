# Four-mode Keck/TRICK PSF benchmark

Generate noisy broadband PSFs with **HCIPy 0.7.1**, train separate single-image
and paired-image four-output estimators, then test frozen weights using native
**OOPAO** propagation and independent evolving atmosphere.

The first case uses published historical Keck/TRICK parameters and a declared
filtered AO residual surrogate. It is not measured Keck telemetry or a complete
Keck AO controller. See [TELESCOPE_AO.md](TELESCOPE_AO.md) for sources and known
unknowns. Focus, two astigmatisms and primary spherical are signed **nm OPD RMS**
in a frozen basis normalized on the actual pupil. The saved basis records its
transformation from conventional Zernike seeds. Atmospheric target-mode
projection makes this a controlled experiment, not arbitrary on-sky recovery.

## Environment

Run commands from the `infrared-lowfs` project root in PowerShell. The verified
runtime is managed CPython **3.11.15**, uv **0.11.24**, Windows AMD64, and PyTorch
**2.11.0+cu128** with CUDA **12.8**. `requirements.lock.txt` pins the scoped
scientific dependency closure of 51 packages; it does not freeze unrelated
shared packages, GPU drivers or wheel hashes. `runtime-versions.json` records
that same scoped closure and the shared base identity. Readiness records the
versions actually exercised.

The current `.venv` uses `--system-site-packages` and an explicit
`codex-science.pth` pointing to the shared science environment. This is an
external runtime dependency. To recreate that arrangement **on a fresh
checkout**, retaining the already installed shared CUDA PyTorch:

```powershell
uv --cache-dir .\benchmark\.uv-cache venv --python 3.11.15 --system-site-packages .\benchmark\.venv
$benchPython = Join-Path $PWD 'benchmark\.venv\Scripts\python.exe'
$sharedPackages = 'C:\Users\mojta\.venvs\codex-science\Lib\site-packages'
Set-Content -LiteralPath .\benchmark\.venv\Lib\site-packages\codex-science.pth -Value $sharedPackages -Encoding ascii
$localPins = Get-Content .\benchmark\requirements.lock.txt | Where-Object { $_ -match '^[A-Za-z0-9_-]+==' -and $_ -notmatch '^torch==' }
uv --cache-dir .\benchmark\.uv-cache pip install --python $benchPython --no-deps $localPins
& $benchPython -c "import torch; assert torch.__version__ == '2.11.0+cu128'; assert torch.version.cuda == '12.8'; print('CUDA available:', torch.cuda.is_available())"
```

`uv pip freeze` does not include packages reached through this custom `.pth`.
The lock was therefore captured with Python distribution metadata from the
actual interpreter. Only PyTorch needs to remain inherited in the recipe above;
other scoped pins can be installed locally.

For a **self-contained alternative on a fresh checkout**, omit system-site
packages and the `.pth`, and install the CUDA wheel stack locally:

```powershell
uv --cache-dir .\benchmark\.uv-cache venv --python 3.11.15 .\benchmark\.venv
$benchPython = Join-Path $PWD 'benchmark\.venv\Scripts\python.exe'
uv --cache-dir .\benchmark\.uv-cache pip install --python $benchPython --torch-backend cu128 --no-deps -r .\benchmark\requirements.lock.txt
```

These are alternative recipes; do not recreate the environment during an
admitted scientific run. The isolated recipe has not been rebuilt in this
checkout. Its official PyTorch wheel index is
`https://download.pytorch.org/whl/cu128`; no CUDA packages are installed into the
Codex bundled interpreter or Hermes environment.

OOPAO is vendored from revision
`e8e9aa60cf99f4ab21a4dae7c29aae9b9ec6ec87`, with archive SHA256
`4c547280c653f307e418eda51f911a6f58d589441796fb07e1e7e2116b21dd98` and per-source
checksums in `vendor/OOPAO/source-pin.json`. Obtain and review that exact official
source through the execution owner; upstream's contradictory requirements file
is not the runtime lock. The adapter bypasses only the package initializer's
banner and legacy precision-file overwrite; scientific modules remain native
and unchanged. See [OOPAO_VALIDATION.md](OOPAO_VALIDATION.md).

## Run the pipeline

The package entry point is `python -m benchmark`. Before the physics gate it
checks the actual host: available RAM, physical CPU cores and a synchronized
CUDA arithmetic probe. It saves `hardware.json`, selects bounded CPU workers,
and freezes CUDA or CPU training and its batch size in the run configuration.
Native OOPAO independently verifies its installed CuPy backend and records any
CPU fallback. Hardware adaptation never lowers optical resolution, changes
the measurement, removes difficult conditions or relaxes a physics threshold.

For a hardware-only check, use a fresh directory:

```powershell
& $benchPython -m benchmark configure --run .\benchmark\runs\hardware-check
```

Inspect its `hardware.json` and `auto-config.json`. A GPU is optional; CPU
training is supported. Insufficient RAM returns `RESOURCE_LIMIT` before
scientific work. The 100-acquisition canary then chooses dataset size from
measured timing on that host.

Choose a **new run directory** beneath `benchmark/runs`. The optional
`--config path.json` is accepted by `gate`; its frozen `config.json` controls
later stages. Use one execution owner. Stop after any nonzero exit code.

```powershell
$benchPython = Join-Path $PWD 'benchmark\.venv\Scripts\python.exe'
$benchmarkRun = '.\benchmark\runs\keck-001'
& $benchPython -m benchmark gate --run $benchmarkRun --max-seconds 2400
& $benchPython .\benchmark\cli.py canary --run $benchmarkRun --count 100 --max-seconds 1800
& $benchPython .\benchmark\cli.py generate --run $benchmarkRun --max-seconds 57600
& $benchPython .\benchmark\cli.py train --run $benchmarkRun --max-seconds 7200
& $benchPython .\benchmark\cli.py evaluate --run $benchmarkRun
& $benchPython .\benchmark\cli.py compare --run $benchmarkRun --count 10 --max-seconds 1800
& $benchPython .\benchmark\cli.py challenges --run $benchmarkRun --count 20 --max-seconds 1800
& $benchPython .\benchmark\cli.py oopao --run $benchmarkRun --count 100 --max-seconds 1800
```

Run each command separately and inspect its result before continuing. The
canary measures 100 acquisitions and chooses 20, 10 or 5 acquisitions per parent
within the declared budget. If five cannot fit, generation is not admitted.
Initial ceilings are 16 CPU-hours, two GPU-hours, 16 GiB RAM and 4 GiB retained
data. Training bounds its complete training/calibration wall time at 7,200
seconds, a conservative GPU occupancy ceiling rather than active kernel timing.
`--max-seconds` bounds each stage; record cumulative actual use separately.
Checks occur at natural yields, so the execution owner must also supervise long
native calls. Real readout/switching overhead is unknown and is reported as such.

The physics gate checks independent optical agreement, pupil/pixel/temporal
refinement, detector count moments and signed recovery/ambiguities. The declared
numerical limits are relative image L1 below `1e-3`, coefficient refinement below
1 nm and noiseless RMSE below 2 nm per mode. Failed readiness stops generation
and training. Do not relax a failed limit to produce a trained checkpoint.

Training/validation/calibration/test/shifted splits keep every replica and both
acquisition designs inside the same atmospheric parent split. Models consume
count images and independent calibration only. They never receive residual
phase, nuisance truth or generator seeds as features. Networks use two hidden
layers of 32 units; training uses three seeds, validation selection and separate
interval calibration before `models/frozen.json` is written.

Nominal parents also contain one independently seeded unknown high-order
calibration error, constant across their acquisitions, with RMS drawn up to
20 nm. It is added after the pure AO gain calibration. Its seed and actual RMS
are offline metadata only; they are excluded from estimator inputs and pure AO
statistics.

## Evidence and recovery

| Artifact | Purpose |
|---|---|
| `config.json`, `readiness.json`, `canary.json` | Frozen configuration, physics admission and measured generation estimate |
| `data/*.npz`, `generation-state.json`, `dataset.json` | Checksummed parent shards, completed-parent resume state and final manifest |
| `models/*`, `models/frozen.json` | Learned/ridge checkpoints, selection, calibration and exact artifact hashes |
| `evaluation.json`, `classical.json`, `challenges.json` | Held-out errors, paired sequence uncertainty, failure examples and runtime evidence |
| `oopao/validation.json`, `oopao/atmosphere_statistics.json`, `oopao/oopao_dataset.json` | Independent-engine optical checks, residual distribution checks and frozen inference |
| `large_telescopes.json` and per-case pilot manifests | Conditional TMT, ELT and GMT optical pilots; no trained transfer model or real AO controller |

Re-run `generate` after a bounded interruption to resume **completed parents**
only. It verifies source/configuration identity and shard checksums. It rejects
changed identities, modified shards and unindexed outputs. An incomplete parent
can be regenerated only after the execution owner inspects its retained partial
evidence. Finished datasets and models are not overwritten. Source changes
invalidate readiness; begin a fresh run. OOPAO preserves partial evidence and
refuses an existing output directory; it does not silently restart or recalibrate
the trained model.

The independent direct-propagation reference is the byte-identical local
`reference.py` snapshot, not the other writer's live `design/fraunhofer.py`.
`reference-provenance.json` records its SHA256
`217619a30972ab6a2df05b68c1d6994e11e984395343d145968265185fc98a0e`.
Source identity includes that reference, the dependency lock, scoped runtime
snapshot, telescope configuration and reference provenance.

OOPAO checks identical pupil/OPD inputs and fourfold/eightfold sampling before
generating 100 new parents with five acquisitions each. Independent developer
ensembles compare filtered residual PSD shape and temporal correlation. Failed
matching reports `PARTIAL / ATMOSPHERIC_DISTRIBUTION_SHIFT`; time exhaustion
reports partial completion. Neither result is a matched-distribution validation
pass. Keep optical disagreement separate from estimator generalization error.

The 2024 [infrared LOWFS paper](https://arxiv.org/html/2410.12084v1) used a
ResNet18-based architecture. This initial small MLP is not that baseline; do not
claim improvement over the published method without implementing its relevant
architecture and matching its measurement/calibration conditions. After the
Keck pipeline and independent-engine admission pass, the conditional optical
pilot command is:

```powershell
& $benchPython .\benchmark\cli.py telescopes --run $benchmarkRun --count 5 --max-seconds 1800
```

It generates separate TMT, ELT and GMT pupil/basis cases with declared residual
surrogates. It does not validate a real telescope controller or transfer the
trained Keck model. Bench/on-sky validation and trained cross-telescope transfer
remain follow-on work.

Quick local checks exercise small logic without launching a scientific campaign:

```powershell
& $benchPython .\benchmark\cli.py --help
& $benchPython .\benchmark\test_core.py
& $benchPython .\benchmark\test_learning.py
& $benchPython .\benchmark\test_hardware.py
& $benchPython .\benchmark\cross_validate.py
```

The adapter self-check verifies interpolation/energy algebra only. Saved native
run evidence is required to claim actual simulator, training or validation
behavior.
