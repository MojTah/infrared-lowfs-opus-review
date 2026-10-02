# Matched architecture study

Mode: Classic Codex; one writer/execution owner, with the authorized existing
Astra chat providing bounded read-only review. This is a post-baseline study.
Existing test outcomes motivated it; fresh confirmation is required after model
selection. The current frozen benchmark source, images, models and results stay
unchanged. Experiment code lives in ledger/analysis/ and outputs in ignored runs.

Question: with identical detector information and training parents, does retained
spatial structure or capacity improve four-mode error, failures, uncertainty and
measured inference latency? Neither a deeper network nor a hybrid is presumed
to improve performance under residual or calibration mismatch.

Compare the preserved MLP/ridge with two additional controls:

1. Compact residual CNN: a 16-channel 3x3 stride-one stem, two residual blocks,
   one 2x2 average pool, retained 8x8 spatial features and a 32-unit/four-output
   regression head. Fewer than 50,000 parameters for either measurement design.
2. Torchvision ResNet18: no downloaded weights, one/two ordered input channels,
   a 3x3 stride-one stem, omitted initial max-pool and four-output head. No image
   resizing. This is an architecture adaptation, not exact 2024 reproduction.

The published 2024 paper identifies ResNet18 but does not fully specify the
original head, preprocessing or optimizer. Preserve this replication gap;
do not claim superiority over that exact method. Standard ImageNet weights are
an optional future initialization experiment, not a wavefront-trained model.
Sources checked: https://arxiv.org/html/2410.12084v1, section 4; official builder
https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html.
Actual local import: torchvision 0.26.0+cu128; resnet18(weights=None) succeeded.

Use the original 300/60/60 training/tuning/interval-calibration parents, both
designs and all three flux replicas. Reuse train-fitted asinh preprocessing and
signed target scaling. Match baseline Adam at 0.001, standardized four-mode MSE,
up to 100 epochs, ten stale validation epochs and batch size at most 128.
Use fixed seeds 11/23/37; select by parent-weighted validation loss only, report
all seeds, and calibrate intervals on independent calibration parents.
No natural-image augmentation, unseen truth, test-informed correction or
artificial upsampling. Any physically broadened training is a separate ablation.

Complete baseline comparisons, detector challenges and independent OOPAO checks
before scientific model-study training. Small synthetic CPU controls may run
while the baseline executes. GPU, real-data, resource and stop/resume admission
are still required. Freeze exact experiment source/protocol/data/runtime identities
and preserve completed seed checkpoints. Keep the 48 CPU-hour, 16 GiB RAM and
4 GiB retained-data ceiling; reserve at least 6,000 CPU seconds for final reporting
and bounded manuscript verification. Profile actual training before allocating
the remaining initial two-hour GPU-occupancy envelope; no cloud expenditure.

The prepared admission uses separate native profile, interrupt, resume and seal
invocations after full OOPAO PASS. Profile each architecture/design on all 18,000
training and 3,600 validation rows for two epochs, without reading test data.
Canary checkpoints live separately from scientific study checkpoints; restored
CUDA predictions must match exactly. Interrupt after one actual CUDA optimizer
step and resume in another native invocation. The canary and scientific trainer
share the same completed-seed skip, fit and exclusive checkpoint path.

The conservative projection is the four measured two-epoch invocation times,
scaled to 100 epochs and three seeds, with a 1.5 safety factor. It is an estimate,
not a guaranteed bound; native guards enforce the actual wall/CPU/RAM ceilings.
Imports and admission count toward the native deadline. Leave a further 60-second
supervisor cleanup allowance inside the remaining 7,200-second envelope. Reject
admission when the projection does not fit; do not change epochs, seeds or the
matched protocol silently to obtain a favorable result.
Every profile and lifecycle checkpoint/report pair must exist and match its
recorded measured evidence before sealing. Training binds the profile checksum
and validates completed scientific seed pairs before calculating unfinished work.
On restart, only unfinished fits consume the new projection; previous native
invocations still count toward the cumulative GPU envelope.

The prepared evaluator checks all twelve sealed checkpoints and validation-only
selection before held-out access. It reports every seed, parent-paired contrasts
against the frozen MLP, and separate standardized MSE and wavefront-RMSE intervals.
Each parent has equal weight and each flux has one-third weight. This also
corrects the OOPAO 40/40/20 acquisition mixture. Nonfinite predictions retain all
planned rows with the registered twice-label-limit penalty. Conditional errors
are omitted if failures prevent a complete finite aggregate. Warming and 1,000
batch-one latency calls are repeated for selected MLP, ridge and architecture
controls on the same hardware; acquisition and controller latency remain outside
that measurement. Actual admission and scientific evaluation remain unverified.

After the architecture-only comparison, use independently seeded paired
diagnostic parents to separate residual amplitude (200/300 nm), spectrum
(nominal/shifted cutoff) and diversity calibration (1.0/1.1), then combine them.
Diagnose before broadening training. Treat those outcomes as development.
Freeze the selected method and uncertainty/guard choices before independent
confirmation. Compare two physical steps from learned and classical initializers
using the same frozen inference calibration and photon budget. Count failures,
iteration cost and latency in every planned-observation denominator.

Telescope expansion, real control-loop claims and proprietary observations retain
their separate evidence gates. A useful null remains acceptable; a manuscript
requires a supported contribution, not merely more trained architectures.
