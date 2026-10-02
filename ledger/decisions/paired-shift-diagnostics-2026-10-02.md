# Paired residual and diversity diagnostics

Mode: Classic Codex, one writer/execution owner, existing authorized Astra review.
This registers development diagnostics before their data exist. Actual execution
requires full OOPAO PASS, all twelve sealed architecture fits and source-matched
architecture evaluations on both original and OOPAO parents. No fitting occurs.

Twenty new diagnostic parents span the full 2 x 2 x 2 factorial:

- Residual ensemble amplitude: 200 or 300 nm.
- Residual-filter cutoff: 10 or 6 cycles per pupil diameter.
- Diversity calibration multiplier: 1.0 or 1.1.

The 200/10/1.0 anchor separates the shift from an otherwise nominal 200 nm tier;
it is not the original equal 100/200 nm mixture. Each parent shares one target
draw, centroid, raw atmospheric seed and unknown high-order calibration map
across all conditions. HCIPy layers use the same seed in each spectrum branch;
their filtered cubes use the existing independently calibrated gain for that
cutoff. Amplitude scales those cubes without per-frame or test-derived RMS
normalization. The same calibration map is added after amplitude scaling.
Detector noise draws are independent across conditions and flux replicas.

Each condition produces the existing single/paired designs and all three flux
replicas. The planned experiment has 160 condition-specific acquisitions,
480 flux rows per design and 3,840 estimator calls for frozen MLP, ridge, compact
CNN and ResNet18. The predictors receive count images and frozen nominal
calibration only. Targets, atmosphere, amplitude, cutoff, diversity shift and
unknown calibration truth are unavailable to inference.

The seed entropy is [20260930, 2910357, parent index]. A separate one-parent
canary uses the first diagnostic parent, which is also included in the twenty
development parents. Its duplicate execution does not add an independent
parent. Untouched confirmation will require another registered seed namespace
after method/guard choices are frozen. No adaptive expansion, retraining or
selection based on these parents is treated as confirmation.

The canary must preserve all eight conditions, both designs and four methods,
with 192 finite estimates. It compares native CPU and the previously admitted
FP64 GPU forward arithmetic at the anchor and combined-shift conditions,
requiring relative image L1 below 1e-3 in both designs. This is an arithmetic
canary, not a new independent optical engine or accuracy threshold. Only a
source/model-matched canary permits the twenty-parent run. Admission also requires
native exit0 without supervision failure, the matching helper hash and reporting
reserve, an intact checksum-verified count shard and a matching dataset/report.

The report includes every planned observation, parent-weighted errors, explicit
conditional available-estimate errors, and the registered twice-label-limit
penalty for absent estimates. A stopped run fills unexecuted rows and remains
PARTIAL. It cannot be used as a complete result or silently restarted over its
prefix. Parent count, all condition/flux rows and failure counts remain visible.
Count arrays and parent seeds are saved; large residual cubes are not retained.

Factorial effects use -1/+1 coding in the stated condition order and equal parent
and flux weights. Main effects average high-minus-low errors over the other
factors. Every effect is twice its orthogonal regression coefficient; this same
normalization applies to interactions. Report standardized penalized MSE and
population wavefront-RMSE effects separately. Parent bootstrap intervals are
unadjusted, marginal and conditional on frozen models. They are exploratory;
do not infer a familywise significant result from searching seven effects.
The whole one-parent canary report has point diagnostics and no inferential
intervals, including nested per-condition and paired-comparison reports.
Observed call timings are interleaved batch-one measurements in this experiment,
not warmed tail-latency guarantees. Keep the separate 1,000-call architecture
latency benchmark's sampling scope when comparing implementation costs.

Frozen engine, basis, measurement, source/protocol, model, runtime and resource
identities remain required. Use the native supervisor after its existing OOPAO
run terminates; no overlapping scientific stages. Keep the 48 CPU-hour, 16 GiB,
4 GiB ceilings and 6,000 CPU-second reporting reserve. New GPU rendering is not
initial architecture fitting. Expensive runs need current capacity and the
bounded real canary; preserve valid state after failure and reapply readiness.
