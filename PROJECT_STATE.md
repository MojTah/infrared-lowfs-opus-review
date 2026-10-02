# Public consultation state

Snapshot date: 2026-10-02. Source development version: 0.8.1. Manuscript: draft 0.4. Packaging version impact: NONE; scientific source and results are unchanged.

The research is paused. This repository supplies material for an independent opinion on the AI component and the next defensible comparison. It does not authorize further experiments.

The task estimates signed focus, two astigmatism coefficients and spherical aberration in nm OPD RMS from infrared focal-plane images. Existing single-image and paired-image measurements use a historical Keck/TRICK-inspired proxy. The target is four-mode error, not total AO residual reconstruction.

Completed source evidence includes a 620-parent simulation dataset, 37,200 flux rows per design, frozen MLP/ridge controls, a 20-parent classical comparison, detector challenges, an independent OOPAO check, 12 matched CNN/ResNet18 fits, and 20-parent paired diagnostics. Rows derived from one physical parent are correlated; they are not independent training or uncertainty units.

Nominal single/pair RMSE is 60.38/63.05 nm for MLP, 58.52/61.20 nm for compact CNN, and 53.68/57.17 nm for adapted ResNet18. Shifted single/pair results are 106.85/106.10, 108.17/106.24 and 106.01/107.19 nm respectively. These are development results. ResNet18 costs about nine times the MLP's median warmed inference time in the matched timing comparison; acquisition and controller latency are excluded. See the JSON reports for denominators, intervals, failures and weighting.

The adapted ResNet18 is not an exact reproduction of the author's 2024 system. LiFT-style code is not an exact upstream LiFT reproduction. Simulator agreement under shared assumptions does not establish on-sky validity or superiority over published methods. Fresh confirmation, a complete realistic AO/controller test and bench/sky validation remain outstanding.

The latest public-source proposal prioritizes calibrated single-image sensing with existing TRICK astigmatism, realistic residual statistics and intensity integration before further model tuning. This is a proposed direction for independent assessment, not a required conclusion. Any useful additional inputs must be measurable at deployment; target truth and simulator-only nuisance truth are prohibited.

Before an authorized scientific resume, the source project retains two recorded gaps: complete continuation seed fields and secondary direct paired wavefront-RMSE contrasts. Untouched confirmation has not run. The source cumulative resource ceiling is 172,800 CPU seconds, 16 GiB RAM, 4 GiB retained data and 7,200 initial GPU-fitting seconds. Its latest accounting leaves approximately 30,313.66 CPU seconds outside a 6,000-second reporting reserve. These are planning constraints, not a budget for the external reviewer to spend.

Private observational material is outside this package. Do not infer its calibration, labels, compatibility, reuse rights or availability. Recommend any real-data validation only as a separate author decision with explicit access and provenance requirements.
