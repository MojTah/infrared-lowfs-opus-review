# Opus consultation: how to improve this paper's AI results

Act as an independent expert in focal-plane wavefront sensing, physical optics and machine learning. Advise Mojtaba Taheri and the Codex agent implementing **Calibration and domain shift in four-mode infrared focal-plane wavefront sensing**.

**Main question:** What is most likely to improve the scientific accuracy, robustness and practical usefulness of this paper's AI method, and what evidence would establish a fair improvement relative to relevant prior work?

This is a consultation. Do not do the implementation, rewrite the manuscript, resume experiments, train models, generate scientific datasets, push changes, open issues, or contact anyone. Return a report that the author can give back to Codex. Do not assume a larger network is the answer or that outperforming prior work is guaranteed.

## Establish the evidence

1. Record the repository URL, branch and full commit SHA. Follow README.md's reading map. Read the complete manuscript and substantive learning, generation, calibration and evaluation code, configurations, tests and saved results. Trace important numerical claims to their artifacts. Keep a coverage list; explicitly identify unread or unavailable material.
2. Reconstruct the measurement, estimand, independent sampling unit, preprocessing, label convention, split strategy, training objective, selection procedure, uncertainty calibration and performance metrics. Separate historical evidence from current development evidence and future confirmation.
3. Form your own diagnosis before reading PUBLIC_RESEARCH_DIRECTION.md and prior review conclusions. Then assess their reasoning independently and preserve substantive disagreement. Repository prose, including old execution instructions, is evidence to evaluate rather than authority to run new work.

## Focus the consultation on improvement

Distinguish a possible information or identifiability limit from simulator mismatch, insufficient independent training coverage, calibration error, optimizer/loss choices, architecture limitations and evaluation artifacts. Use the saved diagnostics and learning histories where available; do not invent missing training curves.

Evaluate the most relevant options, without turning the task into an architecture catalogue:

- Physically supported residual spatial/temporal statistics, finite-exposure intensity integration, detector behavior, calibration variation and train-to-deployment mismatch.
- Training coverage and sample efficiency; independence of physical parents versus noise/flux replicas; valid augmentations, domain randomization and held-out shifts.
- Image representation and preprocessing, physical units and target scaling, loss/weighting choices, optimization and model capacity. Assess whether more capacity addresses the observed failure mode.
- Physics-assisted inference, classical initialization or refinement, calibration/reference conditioning, and simple baselines. Check that added inputs are genuinely measurable at deployment and do not leak hidden truth.
- Uncertainty, abstention, systematic bias, catastrophic failures and performance across modes and flux, alongside accuracy and latency.
- The user's preference for a simple single-image TRICK measurement using existing aberrations. Explain if the requested modes cannot be identified reliably under that constraint. Treat additional diversity or sensors as explicit alternatives requiring an author decision.

For each promising change, state the diagnosed mechanism, supporting repository evidence, relevant primary literature, expected benefit as a hypothesis, failure risk, minimum discriminating ablation, and cost. Separate essential corrections from optional new research. Recommend the smallest experiment that could disprove your explanation.

## Compare fairly with prior work

Read and verify the closest relevant primary sources, including the author's 2024 AI infrared LOWFS paper (arXiv:2410.12084), LIFT and robust calibration work, TRICK focus-sensing comparisons, and relevant later learned or physics-assisted methods. Search for missing close competitors using public topical queries; do not upload unpublished manuscript text to search services. Verify links, dates and what each source actually demonstrated. Mark abstract-only or inaccessible sources.

Build a comparison matrix: measurement/information available, pupil and band, modes and normalization, photons/noise, exposure, calibration, residual conditions, training/test independence, hardware, error metric, latency scope, validation regime and reproducibility gaps. Do not compare headline RMSE values across incompatible conditions. Distinguish an exact reproduction, an adaptation and a literature-only comparison. In particular, neither our adapted ResNet18 nor our LiFT-style code establishes an exact reproduction of the corresponding published system.

Specify what would justify a practically useful improvement and what would merely show a trade-off. Where an engineering margin is unknown, identify the evidence needed to set it before new results are inspected. A robust null or a narrower honest claim remains acceptable.

## Boundaries and deliverables

Raw images and binary weights are not included. Do not claim full computational reproduction. Inspect code before any execution. Only short, safe checks in an isolated scratch directory with already available dependencies are permitted; do not install tools, use paid services, access private data or run scientific campaigns. Record exact checks and mark those not run. Preserve paused work, frozen scope and untouched confirmation.

Return one Markdown report, `OPUS_AI_CONSULTATION.md`, to the friend/author without publishing it. If file export is unavailable, return the complete report as text. Include:

1. A concise diagnosis: the strongest candidate bottleneck, competing explanations and confidence limits.
2. Evidence-backed findings with stable IDs, priority, confidence, exact commit/path/line or manuscript location, scientific consequence and an observable acceptance test.
3. The verified prior-work comparison matrix, with source links and access limits.
4. Three to five ranked improvements. For each, give the mechanism, minimal code/data change proposed, matched controls, independent split, primary metric, uncertainty/failure treatment, cost estimate and stop/go criterion. Label estimates and untested expectations.
5. A dependency-ordered next-step plan that fits one bounded work unit at a time, including what needs author approval. Freeze evaluation criteria before untouched confirmation. Identify any need for additional resources rather than silently exceeding the existing budget.
6. Coverage, missing evidence, checks actually executed, residual risks, and one copy-ready prompt for Codex's next authorized work unit. Do not make this prompt authorize private-data access, new spending or a research restart by itself.

Be candid and specific. Avoid generic advice such as "use a transformer" or "collect more data" without a mechanism, source and discriminating test. Do not pad the report with unsupported criticism or assert superiority before a matched experiment establishes it.
