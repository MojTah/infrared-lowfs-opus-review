# Public-source research direction (excerpt)

This is an explicitly shortened copy of the 2026-10-02 decision record. The private-data section and identifiers are omitted. Proposed direction remains open to independent review.

# TRICK measurement and machine learning research review

Reviewed 2026-10-02 against frozen scientific checkpoint `6f838fa`. Audience: Mojtaba and the project reviewer. This record answers the exposure, residual spectrum, calibration-input and network questions. It changes the proposed research direction; it does not report a new performance result. The autonomous goal and periodic review remain paused.

**Decision:** make calibrated single-image sensing with TRICK's existing optical aberrations the primary instrument case. Establish a realistic residual and integration model before further network tuning. Keep the MLP as the inexpensive baseline and ResNet18 as the demonstrated nominal-accuracy reference. Public telemetry and physical AO models provide a better basis for the next training distribution than another arbitrary cutoff sweep.

## Current models and the measured bottleneck

All neural models are PyTorch supervised regressors predicting focus, two astigmatism coefficients and spherical aberration in nanometres. The baseline MLP has two 32-unit ReLU hidden layers. The compact CNN uses a 16-channel stem, two residual blocks and a small regression head. ResNet18 starts without pretrained weights and uses a native-resolution one/two-channel input and four-output head.

| Model | Single-image parameters | Nominal single / pair RMSE nm | Changed-condition single / pair RMSE nm | Single median inference ms |
|---|---:|---:|---:|---:|
| MLP | 9,412 | 60.38 / 63.05 | 106.85 / 106.10 | 0.423 |
| Compact residual CNN | 42,372 | 58.52 / 61.20 | 108.17 / 106.24 | 1.012 |
| ResNet18 | 11,169,732 | 53.68 / 57.17 | 106.01 / 107.19 | 3.871 |

These are four-mode wavefront RMSE values with equal-third flux weighting. Timing includes preprocessing and inference but excludes acquisition, readout and control. Source: `benchmark/RESULTS.md`, `benchmark/learning.py`, `ledger/analysis/architecture_models.py`, and the frozen architecture evaluation.

ResNet18 improves nominal accuracy by approximately 9–11%, including a similar advantage on the independent OOPAO evaluation. It costs about nine times the MLP's median inference time. Its advantage largely disappears under changed conditions; paired wavefront RMSE is slightly worse. This supports retaining it as a reference, without treating capacity as the main robustness remedy.

The 20-parent paired diagnostics associate residual RMS 200→300 nm with approximately 25–28 nm additional neural wavefront error, and the spectrum change with approximately 10–17 nm. These are exploratory effects within this surrogate. They do not establish a telescope error budget or irreducible floor. Bright images also retain substantial changed-condition error, so photons alone do not resolve it.

The networks share train-fitted asinh(counts/100), per-pixel standardization, Adam at 0.001, batch size 128, up to 100 epochs and patience 10. Three fixed seeds are selected using independent validation parents. There are 300 training parents; the 18,000 rows per design include correlated acquisitions and noise/flux realizations. Training loss was not retained, so existing validation histories cannot diagnose overfitting or optimizer failure. Longer training or a broad hyperparameter search is not justified by those logs.

The loss scales focus by 150 nm and the other modes by 100 nm. Equal physical focus error therefore receives 4/9 of another mode's squared-error weight. Preserve that historical objective. Before a new experiment, declare whether the primary utility is four-mode physical wavefront error or a focus-control error budget, and retain both kinds of reporting.

## Exposure and detector evidence

TRICK's published on-sky slow-focus test used **1 ms frames accumulated into 1 s images with 1 Hz updates**. Bench results used 1 s frames and five-frame coadds. Simulations considered 1–10 ms frames and 0.5–1 s integration. The modeled 5/35 s correction intervals are separate from detector exposure. Bench comparisons include fitted bias/gain calibration; on-sky focus-estimate scatter is not independently measured reconstruction error. [Salgueiro et al. 2026, sections 5–7](https://arxiv.org/html/2602.15746v1).

The author's previous simulation used 10 ms accumulation over ten 1 kHz AO steps. That supports the historical comparison, but does not make 10 ms the preferred slow-focus integration. [Taheri et al. 2024, sections 2–3](https://arxiv.org/html/2410.12084v1).

Historical TRICK hardware uses an H2RG near-infrared detector, 50 mas pixels and H/Ks bands. Available windows include 2×2, 4×4, 8×8 and 16×16. For a 4×4 window, reported effective read noise is at most 3 electrons at 30–200 Hz, 4 electrons at 600 Hz and 5 electrons at 1 kHz. Raw nondestructive-read times and maximum detector/controller rates are distinct from those effective exposures. These figures do not validate our fixed 5-electron noise for every 16×16 acquisition. [van Dam et al. 2019, section 3](https://ao4elt6.copl.ulaval.ca/proceedings/401-39ys-231.pdf).

Frames can be synthesized by combining/subtracting nondestructive reads; resets introduce gaps. A model that independently adds one read-noise draw to each short exposure needs verification against that read sequence. [Rampy et al. 2015, section 2.1](https://escholarship.org/content/qt6317r29b/qt6317r29b.pdf).

**Proposed measurement:** retain 10 ms as the short-frame benchmark and use a 1 s integrated image as the primary slow-focus case. Use 0.5 s as one documented sensitivity case. This is a design recommendation, not an established optimum. Compare at fixed incident photon rate, with exposure-dependent background and effective read noise; preserve equal-total-photon comparisons when isolating optical designs. Estimate the exposure-weighted target state first, and report lag separately before making controller claims. Average propagated intensities over evolving residuals, not the pupil phase before propagation.

## Existing diversity and useful calibration inputs

TRICK has approximately 200 nm RMS dichroic-induced astigmatism, used to resolve focus sign. Thus the user's preference for zero **added** diversity is compatible with this instrument. The published evidence supports slow focus and does not establish joint recovery of all four targets. [Salgueiro et al. 2026, sections 2–4](https://arxiv.org/html/2602.15746v1).

The current single-image simulator includes that nominal astigmatism. The paired simulator instead uses zero/focus diversity and omits the instrumental astigmatism (`benchmark/core.py:288`). Its results describe an alternative optical configuration. They cannot be presented as two exposures from unchanged TRICK hardware. Any future TRICK pair must retain the instrument phase in both exposures, with an intentional probe added separately.

Under our additive phase model, instrumental and target astigmatism in the same basis mode enter through their sum. An unrecognized instrument change from 200 to 220 nm can appear as 20 nm target astigmatism. This is an identifiability result from the model, not a measured TRICK drift. An independently anchored calibration is needed to assign an absolute target coefficient. Repeated science frames or extra network outputs cannot remove a shared unknown calibration offset.

The pupil and ideal PSF are useful suggestions, with different roles. A constant pupil/reference supplies no varying per-observation information in a fixed-calibration dataset, though an architecture may use it as an inductive bias. For transfer between calibration states, the smallest useful comparison is image-only versus image plus an independently measured **calibrated reference PSF and metadata**. The actual reference includes instrumental aberrations, pixel sampling and bandpass. A diffraction-limited ideal PSF alone omits those effects; an intensity reference also does not uniquely specify complex pupil phase.

Hold out entire calibration states, propagate reference uncertainty and supply the estimated calibration available at deployment. Never supply hidden simulator residual or calibration truth. Check signed response and cross-talk using the actual pupil, sampling, noise and integration. A network cannot repair a genuinely ambiguous measurement. Instrument names differ; the detector geometry, known phase, bandpass, read sequence and loop role must be specified for each camera.

## Meaning of the residual spatial cutoff

The present residual filter is a **smooth surrogate**, not a measured telescope boundary:

`H(k) = k² / (k² + kc²)` for phase amplitude; the corresponding power is multiplied by `H(k)²`.

Here k counts spatial cycles across the pupil. Small k means broad wavefront patterns; large k means finer ripples. At k=kc, amplitude is halved and power quartered. For a 10.949 m aperture, ten cycles correspond roughly to a 1.1 m pattern scale and six to 1.8 m. This has no connection to the star's optical wavelength spectrum or an image cropping threshold.

Reducing kc from 10 to 6 admits more broad residual structure before the ensemble is rescaled to its declared RMS. There is no sharp removal of every frequency below kc. Our diagnostics show sensitivity to this changed distribution; they do not show that either value matches a particular telescope.

A realistic residual needs components from fitting, servo lag, WFS noise, aliasing and field/laser geometry, plus temporal correlations and sometimes vibrations. A Keck II analytical example predicts 141.2 nm including fitting under its specified controller and turbulence parameters; that is not current TRICK performance. Its component spectra differ in slope and direction, which a radial knob cannot reproduce. [Correia et al. 2017](https://arxiv.org/pdf/1709.05090), [author OOMAO implementation](https://github.com/cmcorreia/LAM-Public).

## Public datasets and code to use first

| Resource | Verified availability and useful content | Limit for this project |
|---|---|---|
| [Keck Observatory Archive NIRC2 TRS](https://vmkoaweb.ipac.caltech.edu/UserGuide/trs_data.html) | Documented IDL `.sav` telemetry products for closed-loop observations from 18 August 2019. Listed fields include timestamps, residual wavefront/RMS, DM and tip-tilt commands, centroid offsets, subaperture intensity and configuration information. | No individual file fetched. Units, reconstruction basis, influence functions, calibration and completeness remain unverified. Keck II/NIRC2 is a transferable reference, not Keck I/TRICK truth. |
| [AOT public release paper](https://arxiv.org/html/2312.08300v1), [proof-of-concept dataset DOI](https://doi.org/10.5281/zenodo.8192741) | Authors report public GALACSI/CIAO/NAOMI/ERIS telemetry via ESO programs 60.A-9278(B/C/D/E), plus PAPYRUS and an ESO subset on Zenodo. | File list, sizes, data license, actual cadence and calibrated fields were not verified: live Zenodo access failed. These are other instruments and provide statistical/method checks. The bibliography gives the dataset DOI; a paper footnote instead points to the software DOI. |
| [aotpy](https://github.com/STAR-PORT/aotpy) | Public BSD-3-Clause reader/translator source for AOT FITS. | A standard can represent measurements, commands and timestamps; representation does not prove those fields exist in each released file. No installation performed. |
| [P3](https://github.com/astro-tiptop/P3), [TIPTOP](https://github.com/astro-tiptop/TIPTOP), [component coverage](https://astro-tiptop-services.github.io/astro-tiptop-services/docs/general/error_breakdown/) | Public BSD-3-Clause/MIT sources and physically parameterized 2D residual PSD components; P3 includes NIRC2 configuration examples. | Analytical PSD/PSF predictions do not automatically provide correlated finite-exposure realizations or complete engineering/static errors. Static inspection only. |
| [OOPAO](https://github.com/cheritier/OOPAO), [closed-loop documentation](https://cheritier.github.io/OOPAO/closed_loop/run_cl.html) | Public GPL-3.0 source and documented evolving atmosphere, DM/WFS correction and residual saving. Existing project-pinned source is a reusable starting point. | Verify needed capabilities in our pinned revision. A realistic configured controller has not been admitted by the existing independent propagation comparison. |

The archive discovery qualifies the earlier request-only description: **some Keck telemetry is archived**, while the official AO page also describes full-rate `.npz` telemetry provided to interested observers. No anonymous TRICK image/focus-stage dataset was identified in this search. [Keck AO](https://www2.keck.hawaii.edu/inst/ao/).

Measured temporal evidence also matters: a Keck predictive-control study reports a residual feature near 60 Hz and distinguishes reconstructed controlled-mode residuals from complete optical truth. [van Kooten et al. 2022](https://arxiv.org/html/2205.14164v1). A later daytime KAPA rejection measurement is consistent with 0.7 ms latency at 1.5 kHz and gain 0.5; that illustrated noise response is not an atmospheric spectrum or TRICK exposure requirement. [KAPA commissioning 2026](https://arxiv.org/html/2608.07769v1).

## Finite next experiment and stopping rules

1. **Verify one released telemetry example and its calibration.** Establish timestamp units, wavefront basis, controller state and noise contribution. Extract RMS distribution, spatial covariance and temporal PSD. Stop the telescope-specific claim if calibration is missing; retain it only as a transferable prior. Raw DM commands are not residual optical truth.
2. **Match one bounded closed-loop residual model.** Include unsensed fitting and relevant field effects. Project nuisance onto the benchmark's pupil-weighted complement of piston, tilt and the four target modes, and report removed variance. A total telescope RMS cannot be substituted for our target-removed high-order RMS. Check the spatial and temporal statistics before constructing new training data.
3. **Fix the calibrated single-image measurement.** Keep existing astigmatism, H-band sampling and an instrument-appropriate window/read sequence. Verify opposite-sign response, calibration-offset limits and intensity coaddition at the selected integrations. Retain all four target modes and report focus separately.
4. **Make one controlled training comparison.** First compare the same MLP trained with the historical versus physically supported residual/calibration distribution, matching independent-parent count, labels, seeds and optimizer budget. If variable calibration is material, add one calibrated-reference/metadata comparison with whole states held out. Retain ResNet18 as a capacity reference under matched data; do not launch a broad architecture search. Log training and validation loss by mode and flux.
5. **Freeze criteria before fresh evaluation and then stop.** Require paired improvement on new parents, a declared acceptable nominal regression, stable signed response, coverage by mode/flux/calibration and a measured end-to-end deadline. Set numerical engineering margins from the intended TRICK error/control budget before inspecting new results. Any unsuccessful comparison yields a bounded negative result and a documented limitation, not another automatic tuning cycle.

These stages require a separately scoped scientific launch and readiness/resource admission. The autonomous goal remains paused. Existing confirmation seed-field and secondary paired-RMSE review gaps also remain. No new dataset, model fit, simulation, inference or controller run occurred in this review. It does not establish superiority over the author's previous paper or other methods.

