# Infrared LOWFS: independent AI-method consultation

This temporary review package concerns **Calibration and domain shift in four-mode infrared focal-plane wavefront sensing**, working manuscript draft 0.4 by Mojtaba Taheri. It is a snapshot for an independent consultation, not a new experimental result or a submission-ready software release.

**Start with [REVIEW.md](REVIEW.md).** The requested outcome is a specific, evidence-based plan to improve the AI results and compare them fairly with relevant prior methods. The reviewer should advise the implementation agent, not implement the project.

## Reading map

| Material | Location | Purpose |
|---|---|---|
| Review request and deliverables | [REVIEW.md](REVIEW.md) | Exact consultation scope |
| Current scope and limitations | [PROJECT_STATE.md](PROJECT_STATE.md) | Public summary of the paused study |
| Complete manuscript source | [manuscript/infrared-lowfs-validation.tex](manuscript/infrared-lowfs-validation.tex) | Draft 0.4; tables and bibliography are embedded |
| Existing evidence summary | [benchmark/RESULTS.md](benchmark/RESULTS.md) | Dated results, including historical sections |
| Baseline forward model and learning | [benchmark/](benchmark/) | Optics, detector, inference, evaluation and tests |
| CNN/ResNet18 implementation | [ledger/analysis/architecture_models.py](ledger/analysis/architecture_models.py) | Exact architecture and fit code |
| Training/selection and evaluation | [ledger/analysis/architecture_study.py](ledger/analysis/architecture_study.py), [architecture_evaluate.py](ledger/analysis/architecture_evaluate.py) | Data partitions, model freezing, metrics and paired comparisons |
| Diagnostic experiment | [ledger/analysis/shift_diagnostics.py](ledger/analysis/shift_diagnostics.py) | Paired residual-amplitude, spectrum and diversity comparisons |
| Expanded data generation | [ledger/analysis/expanded_campaign/](ledger/analysis/expanded_campaign/) | Current generation and flux-weighting implementation |
| Saved numerical reports | [benchmark/runs/keck-benchmark-05/](benchmark/runs/keck-benchmark-05/) | Selected result JSON, protocols and manifests, including per-seed records |
| Prior-work starting set | [literature/review.md](literature/review.md), [references.csv](literature/references.csv), [design/closest-prior-work.md](design/closest-prior-work.md) | Sources to verify and extend independently |
| Latest public-source direction | [PUBLIC_RESEARCH_DIRECTION.md](PUBLIC_RESEARCH_DIRECTION.md) | Explicitly shortened decision record; read after forming an initial view |
| Copy provenance | [SNAPSHOT_MANIFEST.json](SNAPSHOT_MANIFEST.json) | File hashes, sizes, source revision and exclusions |

## Evidence and execution limits

The selected source working files were copied from scientific repository revision `6f838fa421e85a44c7e2b1ac9878c530f994ebfb` on 2026-10-02. The source checkout also contained uncommitted work. The manifest identifies the actual copied bytes; the review repository's own commit identifies this consultation snapshot. Old admission hashes and dated status paragraphs retain their original meaning.

Raw simulated image arrays, trained binary weights, third-party vendor trees, environments, private postdoc material, and original Git history are excluded. Result files include aggregate and diagnostic numerical evidence, but this package alone cannot reproduce scientific inference or retraining. Missing arrays/weights must be reported as a limitation, not treated as an error in the method. No previously private real-data contents are supplied.

Some retained documents describe historical launch commands and local Windows paths. They are evidence, not instructions to launch those commands. The reviewer must not resume paused research. The baseline lock predates the architecture extension; the architecture protocol additionally records `torchvision 0.26.0+cu128`. No fresh installation or portable runtime validation is claimed.

The author requested temporary public access for approximately four days. The intended end is **2026-10-06 at 07:00 America/Halifax (10:00 UTC)**. Returning the repository to private will not retract copies already made. Public availability is for review and does not represent journal submission or a new open-source license grant.
