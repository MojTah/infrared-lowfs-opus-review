# Current PSF benchmark evidence

Generation and model freezing completed in `runs/keck-benchmark-05`: 20
physical acquisitions for each 620 parents and 37,200 flux rows per design.
Each held-out split contains 100 parents, 2,000 physical acquisitions and
6,000 flux rows. Flux replicas share their physical parent; uncertainty uses
parents as the sampling units.
The GPU cutover preserved all2,280 then-saved acquisitions and their hashes.
The unchanged native engine retains its historical physics gate below;
the new controller/backend has separate source and admission identities.
FP64 GPU optics passed signed recovery,10/20-frame integration, guards,
fallback and three sampling profiles (relative L1≤2.21e-16). Twelve complete
TRAIN parents matched native CPU count arrays bitwise. Real interruption and
resumption passed. Four/eight native phase workers with one GPU produced
0.602/0.607 acquisitions/s, so production uses four phase workers.
All six MLPs with two 32-unit hidden layers and both ridge models are frozen with train-only
preprocessing, validation-only selection and calibration-only intervals.
Validation selected seed 11 for single images and seed 37 for pairs.
Frozen training elapsed 202.344 seconds on CUDA. A supervisor exit-timestamp
failure prevented final CPU/peak recording; the preserved training received
a conservative 5,041 CPU-second charge. No refitting occurred. The separate
corrected supervisor completed held-out evaluation and flux reporting with
exit 0, verified child accounting and terminal metadata.

## Frozen held-out results

These RMSE values concern the four target modes, not total AO residual error.
Every numerical held-out evaluation followed model freezing.

| Split/design | MLP wavefront RMSE (nm) | Parent-bootstrap 95% interval (nm) | Ridge RMSE (nm) |
|---|---:|---:|---:|
| Nominal single | 60.38 | 58.80–61.98 | 80.26 |
| Nominal pair | 63.05 | 61.84–64.19 | 76.95 |
| Shifted single | 106.85 | 105.12–108.48 | 109.04 |
| Shifted pair | 106.10 | 104.43–107.62 | 107.66 |

Nominal MLP-minus-ridge standardized MSE differences have paired parent-bootstrap
95% intervals [-0.05389, -0.04914] for single images and [-0.03908, -0.03532]
for pairs. Both shifted intervals cross zero, so shifted performance does
not establish a clear advantage over ridge. The shifted split changes both
residual conditions and diversity. These intervals condition on fixed models.

| Nominal flux (electrons) | Single RMSE (nm) | Pair RMSE (nm) |
|---:|---:|---:|
| 1,000 | 81.14 | 93.15 |
| 10,000 | 47.67 | 45.43 |
| 100,000 | 45.61 | 34.41 |

Aggregate nominal marginal interval coverage is approximately 94–96%.
At 1,000 electrons, coverage falls to 86.7–90.4% for single images and
83.9–86.3% for pairs. Shifted aggregate coverage falls to 66.3–82.4% and
76.1–83.0%, respectively. These are empirical marginal intervals, not joint
or guaranteed 95% coverage. No nonfinite prediction occurred; this does not
imply absence of scientific estimation errors.

Nominal warmed batch-one CUDA medians are 0.341 ms for single images and
0.573 ms for pairs, including preprocessing, transfers and synchronization.
Corresponding CPU ridge medians are 0.0118 and 0.0155 ms. These timings exclude
acquisition, detector readout, switching and controller latency.

Evidence: evaluation.json SHA256
8d0b660a917db68edb6bd88a92281f54b0bdbde2250f6e1ba4637f110faf71ec;
flux-evaluation.json SHA256
b56096e07aad9bdd895852d0c1f521259810bec58ed22024d8a2700e8aeeacf5;
models/frozen.json SHA256
f2d02f1f79145a2014d4b785af840fd8bb7d1a6ad7bdd950b991f0f651f5b54d.
Files remain in the ignored local run directory. Challenge/OOPAO
validation and conditional telescope pilots remain pending.

## Completed classical subset, 2026-10-02

The restarted comparison completed through the unchanged verified supervisor
with exit0: twenty distinct parents, ten per condition, and 200 unique finite
estimates. Each parent contributes one identical 10,000-electron observation
to all five methods and both designs. There were no time/RAM caps.

| Method | Nominal single | Nominal pair | Shifted single | Shifted pair |
|---|---:|---:|---:|---:|
| MLP | 59.09 | 65.93 | 73.93 | 82.99 |
| Ridge | 76.36 | 79.13 | 90.67 | 93.45 |
| LiFT-style | 28.12 | 80.39 | 133.92 | 193.05 |
| Physical multistart | 22.68 | 21.88 | 107.14 | 153.13 |
| MLP plus two updates | 49.61 | 45.73 | 103.34 | 150.21 |

Errors are four-mode nm OPD RMS point estimates. Nominal physical fits can
reduce error at much greater implementation cost; physical refinement fails
to improve robustness on the combined shifted subset. The paired intervals
test standardized MSE with150/100/100/100 nm mode scales, not directly these
unscaled wavefront-RMSE differences. This small subset is development
evidence, not fresh confirmation or a full-population result.

Single/pair nominal median calls are8.7/17.3s LiFT-style,53.2/99.1s multistart,
and3.5/6.9s hybrid. These ten interleaved calls differ from the separately
warmed1,000-call MLP timing protocol. They establish this implementation's
observed cost, not the best possible physical-solver runtime.

The nonlinear method first uses three starts on a512-square coarse pupil,
then refines its best capture on the1024-square pupil. Each fit has a30-function
evaluation cap per start. LiFT-style uses up to five analytical Jacobian
updates and is not an exact upstream reproduction; the hybrid uses exactly
two updates. All receive nominal calibration without residual/nuisance truth.

classical.json SHA256:
b24077a467db8dcac0a67946d589977533555d992a2649a3fdcd40a988554378.
Wall4261.257s; CPU4410.0625s; peak sampled aggregate working set1.100GiB.
See ledger/checkpoints/classical-and-manuscript-2026-10-02.md for provenance,
paired intervals, the manuscript review and challenge admission.

## Historical native physics evidence

Historical native checkpoint2026-10-01. Engine source commit d955961; final-source run
`runs/keck-benchmark-03`. Detailed JSON is retained locally in the ignored run
directory. Configuration, source, runtime, basis and hardware identities are
saved with the evidence. Earlier development results do not admit changed source.

The final physics gate passed:

| Check | Result | Starting numerical gate |
|---|---:|---:|
| Independent direct propagation versus HCIPy |1.38e-16 maximum relative L1|1e-3|
| Joint pupil/pixel/wavelength refinement |8.522e-4 maximum relative L1|1e-3|
|200/300nm residual plus calibration-error refinement |8.246e-4 maximum relative L1|1e-3|
| Blind noiseless recovery, focus |0.08393nm RMSE|2nm|
| Blind noiseless recovery, astigmatism cosine |0.13316nm RMSE|2nm|
| Blind noiseless recovery, astigmatism sine |0.01595nm RMSE|2nm|
| Blind noiseless recovery, spherical |0.05519nm RMSE|2nm|
| Largest blind absolute coefficient error |0.20832nm|not a separate gate|
| Exposure-time refinement |9.74e-5 maximum relative L1|1e-3|

Detector count moments passed, including negative read-noise values and the
pair's two read contributions. The opposite-sign/mixed-mode candidate bank
found no ambiguous candidates at100,000 electrons after nuisance fitting.
This finite search does not prove global uniqueness or faint-source recovery.
Full conventional-basis transformation is saved in readiness.json.

Gate wall time1565.43s; measured CPU1537.23s; Windows peak working set2.93GiB.
The hardware stage verified actual RTX4060 CUDA arithmetic, chose CUDA training
with batch128 and four CPU workers, and retained the16GiB RAM ceiling. Mock
checks verify missing/broken CUDA, CPU fallback and insufficient-RAM rejection.
CPU selection remains consistent through frozen inference and comparisons.

Native OOPAO evidence remains preliminary: the pinned engine's GPU propagation
agreed with HCIPy at1.04936e-6 relative L1 for one identical1024-pupil input.
One independent atmospheric-screen API smoke passed. These are not a complete
frozen-model validation or an estimator-generalization result.

The100-acquisition timing canary passed in1169.73s wall/1149.05s CPU. Its
11.4905CPU s/acquisition selected five acquisitions for all620 parents:
projected35620CPU s (9.9hours), or9065s (2.5hours) at four workers.
Two planned120-second generation segments exercised package-mode stop/resume.
That early checkpoint retained ten parent shards/150rows (437131bytes); all hashes matched,
the original five were unchanged, no partial files or generator commands remain.
The cumulative generation counter is960.05CPU s; valid data can resume.

The current trained checkpoint and held-out metrics are reported above.
Challenge comparisons and full independent OOPAO validation subsequently
completed. Telescope pilots remain conditional numerical feasibility and require
native optical convergence. Preliminary smoke evidence above is historical.

The initial telescope uses published historical Keck parameters with a
synthetic pupil proxy and declared filtered residual surrogate. It does not
reproduce a complete controller, measured telemetry or current KAPA performance.
No comparison claim against the author's2024 ResNet18 method is made.

## Matched architecture development results, 2026-10-02

All twelve CNN/ResNet18 fits and both original/full OOPAO evaluations completed
native exit0. Artifact identity, all-seed reporting, validation-only selection
and full failure denominators are verified.

| Method | Nominal single | Nominal pair | Shift single | Shift pair | OOPAO single | OOPAO pair |
|---|---:|---:|---:|---:|---:|---:|
| MLP |60.38|63.05|106.85|106.10|61.57|63.44|
| Ridge |80.26|76.95|109.04|107.66|81.38|78.22|
| Compact CNN |58.52|61.20|108.17|106.24|58.29|62.27|
| ResNet18 |53.68|57.17|106.01|107.19|53.74|58.19|

Four-mode wavefront nm RMS, equal-third flux weighting. Nominal conditional
paired intervals favor architecture extensions. Shift is mixed: ResNet18 pair
wavefront RMSE worsens1.09nm versus MLP,95% interval0.18–2.13nm; standardized-
MSE includes zero. CNN pair standardized-MSE improves while wavefront RMSE is
inconclusive. These test sets remain development evidence; fresh confirmation
is pending.

Warmed single median/p95/p99 latency:0.423/0.623/0.815ms MLP,1.012/1.304/2.001ms
CNN,3.871/5.238/7.177ms ResNet18. Pair medians0.441/1.021/3.928ms respectively.
Acquisition/controller time excluded. ResNet18 median costs about9times MLP.

Actual paired-shift canary passed192 finite estimates/four optical checks,
maximum relative L1 7.36e-17. This admits full diagnostics, not a20-parent result.
Full diagnostics, untouched confirmation and bounded refinement remain.
Details: ledger/checkpoints/matched-models-and-diagnostic-admission-2026-10-02.md.


## Completed paired diagnostics and pause

Twenty parents completed eight cells with3,840 finite estimates, native exit0,280.83s wall and388.78 charged CPU seconds. MLP amplitude200→300nm increases average cell wavefront RMSE27.68/24.74nm single/pair; cutoff10→6 increases16.36/12.01nm. Exploratory marginal intervals exclude zero for amplitude/spectrum in every method/design. Neural diversity main-effect intervals include zero; interactions remain. Calibration maps were fixed, so their error contribution was not independently isolated.

Research is paused. Draft0.4 integrates recovered LOWFS-01 introduction, architecture comparisons and diagnostics. Confirmation/physical follow-on have not run. Two accepted Astra review11 gaps remain before admission. See ledger/checkpoints/research-pause-and-draft04-2026-10-02.md.
