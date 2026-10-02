# Expanded PSF campaign admission

The user approved the largest proposed option in the original PSF chat:
20 genuine acquisitions for each of 620 atmospheric parents, 48 cumulative
CPU-hours and up to eight CPU workers. RAM remains 16 GiB, retained data 4 GiB,
and initial training GPU occupancy two hours. Laptop GPU generation may be
used only after numerical and throughput checks. No cloud service is admitted.

T001 root remains the scientific execution and repair owner. The old
`benchmark/runs/keck-benchmark-03` campaign stopped at a safe parent boundary:
219 training parents, 3,285 rows, 13,196.20 generation CPU seconds, unchanged
shard hashes and no partial files. Its original source, configuration,
readiness and timing records remain historical evidence.

The expanded driver is new generator source with its own frozen checksum and
sampling/execution policy. It must not hide behind the legacy engine hash.
The unchanged engine and measurement may inherit byte-identical physics
evidence. New driver, sampling, worker count and any GPU backend require their
own admission. Downstream entrypoints must validate both identities and exact
coverage before training or frozen inference.

Grouped construction: retained five-acquisition prefix plus a disjoint
15-acquisition continuation. Retained prefixes remain
byte-identical. Continuations replay the first five native atmospheric
acquisitions before advancing to index five, retain the same atmospheric and
static-calibration parent seeds, and use explicit independent continuation
label/noise streams. Fresh parents produce a continuous 20-acquisition file
with the identical stream transition at index five, avoiding redundant replay.
Completeness means 620 distinct parents, unchanged split allocation, indices
0 through 19 exactly once, three declared flux replicas per acquisition and
both image designs in each row: 37,200 rows, including 18,000 training rows.
Flux replicas and both designs remain grouped by physical parent.

Admission completed: CPU four/eight-worker comparison, resource enforcement,
source/policy rejection, prefix/continuity/coverage checks and real stop/resume.
The separate FP64 GPU admission passed complex-field/image agreement,
signed recovery, centroid/diversity/calibration,10/20-frame averaging,
finite/nonnegative guards and optical-only fallback. Twelve complete TRAIN
parents matched saved CPU arrays bitwise. Real GPU interruption/resume passed.
Four/eight phase workers achieved0.602/0.607 acquisitions/s at3.52/5.02GiB
aggregate RAM, so production uses four phase workers and one GPU owner.
The original GPU admission JSON's unused `metrics.status` initialization
remains preserved; its top-level `status: PASS` and explicit checks are authoritative.
Training waits for the complete planned manifest. Held-out arrays stay closed
until `models/frozen.json` exists.

The inherited gate, timing canary and old generation total 15,882.48 measured
CPU seconds. Carry these, the prior-development allowance, all new admission,
discarded work, initialization, shutdown and downstream stages into 48 hours.
Do not silently reduce the target if 20 acquisitions cannot fit.

Independent static reviews completed without science execution. They require
new-driver provenance/coverage admission and measured worker admission. Later
classical coarse capture needs external supervision. OOPAO's five-exposure
flux mixture is 40/40/20 rather than equal thirds, so compare flux-specific
errors before interpreting aggregate engine-transfer differences.

Active command: `benchmark/.venv/Scripts/python.exe -B
ledger/analysis/expanded_campaign/gpu_campaign.py --run
benchmark/runs/keck-benchmark-05 --max-seconds 30000`.
The original engine configuration stays byte-identical. `execution-policy.json`
binds controller and GPU sources, admission evidence and every pre-GPU shard.
Initial cutover policy SHA256:
`48932ccf6223397b68367193cefb28b61dbf3060b900b32d7e2e3377ee20f780`.
The conservative projection is34.57 CPUh before the reserved downstream work.

CPU/controller identities are archived in `policy-history/native-cpu-001`.
Ordinary migration failures roll back only the controller and three JSON files.
After a hard interruption, first verify that no owner/process remains; then run
`benchmark/.venv/Scripts/python.exe -B
ledger/analysis/expanded_campaign/recover_gpu_transition.py --history native-cpu-001`
and `campaign.py check` before resuming. Recovery verifies all backup hashes
and preserves attempted files. It does not alter acquisition arrays.

Training command after full publication:
`powershell.exe -NoProfile -ExecutionPolicy Bypass -File
ledger/analysis/T001-run-stage.ps1 -Stage train -Run
benchmark/runs/keck-benchmark-05 -MaxSeconds 7200`.
Use explicit counts10 for compare,20 for challenges,100 for OOPAO and5 for
conditional telescope pilots. The launcher default100 is unsuitable for compare/challenges.

For conditional pilots after OOPAO PASS, use the separate reviewed launcher:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ledger/analysis/T001-run-pilot-stage.ps1 -Run benchmark/runs/keck-benchmark-05 -MaxSeconds 7200 -Count 5
```

Its syntax and inherited process guard equality passed static checks.
Production pilot execution remains unexercised. Recompute the provisional
deadline from remaining cumulative CPU allowance. Preserve the sealed
T001-run-stage.ps1 bytes; its telescope command invokes the native count bug.

Generation and training completed. The sealed supervisor then lost its
launcher exit timestamp during resource finalization. The separate
T001-run-verified-stage.ps1 retains that handle and passed a real CPU canary.
Use it for subsequent stages, with the explicit counts above. It also supports
`-Stage flux`, and `-Stage flux -Manifest <OOPAO-manifest>` for separate reports.
The pilot entrypoint forwards to this corrected supervisor. Frozen models
were preserved; training CPU is explicitly charged by a conservative bound.
See ledger/checkpoints/generation-freeze-supervisor-repair-2026-10-01.md.
