# Infrared focal-plane LOWFS: evidence and collision review

Cutoff: 2026-09-30, America/Halifax. Initial public-source review. RESULT: PARTIAL. This is sufficient to frame a narrow benchmark; it does not certify novelty, a complete systematic review, operational readiness, or a telescope error budget.

## Finding and decision

**INFERENCE:** A new generic neural Zernike regressor, a physics-consistency loss, or a fast forward pass alone would have weak differentiation. The credible question is whether calibrated, pupil-aware refinement improves signed error, failure rate and latency together under held-out infrared sampling, residual-aberration and calibration shifts. That remains a hypothesis.

**Recommendation:** conditionally proceed to one Keck/TRICK-like *slow signed-focus* simulation benchmark. Freeze photon budget, exposure, diversity, pupil, detector sampling and independent test conditions before training. Extend to fast TTF, ELT-class pupils and differential piston only after the corresponding information and control questions pass. This recommendation does not authorize training, proprietary data acquisition or collaborator contact.

The catalogue contains 38 distinct primary-verified scholarly identities, plus one unresolved close collision. Verification depth varies explicitly: a verified title/DOI is not a verified performance result. Official facility documents are separate in requirements.csv. No negative search establishes absence of competing work.

## Closest collisions

| Work | SOURCE STATEMENT and validation | Consequence for the proposed work |
|---|---|---|
| [Taheri et al. 2024, R06](https://arxiv.org/html/2410.12084v1) | ResNet18 estimates low-order modes from infrared focal-plane images; H-band simulation and K-band Keck I **internal calibration-source** measurements. Cross-band focus comparison removes the mean offset. | The author's own prior work is mandatory. Atmospheric on-sky validation and absolute focus calibration cannot be attributed to these measurements. |
| [Salgueiro et al. 2026, R07](https://arxiv.org/html/2602.15746v1) | GS, LIFT and Gaussian focus estimators are compared for TRICK H/K; GS is selected for residual robustness and tested in a February 2025 on-sky focus loop. Routine use remains prospective in the manuscript. | A fair challenge must reproduce GS as used, and correctly sampled LIFT. Speed alone is insufficient. Public operational history belongs to the coordinator's Keck review. |
| [Kuznetsov et al. 2024, R08](https://arxiv.org/html/2406.08529v1) | MUSE/IRLOS uses an NCPA-calibration stage followed by LIFT TTF. DIP fits a differentiable PSF using autograd. Simulation and on-sky tests support slow/truth focus sensing. | “Physics plus calibration plus LIFT” already exists. Compare calibration burden, nuisance robustness and independent-condition generalization. |
| [Orban de Xivry et al. 2021, R12](https://doi.org/10.1093/mnras/stab1634) | CNN modal and phase reconstruction are studied against fundamental limits using diverse images in simulation. | Generic CNN reconstruction and photon-limit comparisons are established. Match optical information before comparing errors. |
| [Singh et al. 2026, R20](https://arxiv.org/html/2609.00737v1) | Learned H-band LLOWFS controls differential petal piston in a preliminary SCExAO 1 kHz on-sky loop. TT is not controlled; instability remains under investigation. | Fast learned infrared sensing and petal control are close prior work, although their rejected-light optics differ. |
| [Landman et al. 2025, R25](https://arxiv.org/html/2503.16690v1) | Neural unmodulated-pyramid reconstruction runs on sky at 2 kHz. RTX4090 FP32 transfers plus inference: median 240 us, p95 243 us, max 247 us. Faster FP16 was not used on sky. | Named-hardware inference timing is an available standard. These numbers are not total acquisition-to-actuation latency or a photon-equivalent plain-PSF benchmark. |
| [Lin et al. 2025, R30](https://arxiv.org/html/2505.00765v1) | A spectrally dispersed 19-port lantern uses linear SVD reconstruction on sky: 1.7 kHz and reported 3.77-frame latency. Calibration, spectral overlap and recapture range matter. | Real-time sensing does not require a neural estimator. Port/spectral hardware information must be costed separately. |
| [Wu et al. 2020, R31](https://pmc.ncbi.nlm.nih.gov/articles/PMC7506609/) and [Gao et al. 2026, R32](https://doi.org/10.1364/OE.606688) | PD-CNN reports about 0.5 ms inference; (PD)2 uses self-supervised physical image consistency. Generic simulation/bench validation. | Sub-millisecond inference and physics-driven learning are not new by themselves; infrared atmospheric robustness is not established by these generic benches. |

An August 2026 [MORFEO control paper, R35](https://arxiv.org/html/2608.13728v1) explicitly includes a notch-filtered defocused LIFT petalometer. Its dedicated Plantet et al. proceeding, U01, was found through the primary reference list and conference program, but the full record could not be retrieved. It is an unresolved novelty collision, not evidence of an empty niche.

## Method families and fair information accounting

| Family | Measurement and prerequisite | Appropriate comparison |
|---|---|---|
| Gaussian/ellipticity | Parametric spot shape; diversity and nuisance calibration determine signed focus interpretation. | Cheap focus reference, including failure under residual-induced shape changes. |
| GS / phase retrieval | Iteration between physical constraints; initialization, sampling, support and phase ambiguities matter. | Same band-integrated pixel data and calibration; freeze stopping rule. |
| LIFT | Linearized intensity Jacobian and weighted likelihood with known diversity; iterative relinearization expands local range. | Correct pixel-integrated model, diversity and NCPA calibration; fit nuisance terms consistently. |
| Fast & Furious | Sequential focal images and known control diversity; weak-phase/pupil assumptions, with FF-GS extensions. | Sequence-aware baseline, accounting for extra frames and DM excitation. |
| Learned focal-image estimator | Training distribution and physical diversity carry prior information. | Independent sequences/conditions, matched calibration data, uncertainty and out-of-distribution failure. |
| Coronagraphic LOWFS | Focal mask and reflective Lyot stop encode otherwise rejected light. | Instrument alternative with throughput, optical changes, modes and detector costs declared. |
| Zernike / pyramid WFS | A focal phase mask or prism converts phase into **pupil-image** intensity. | Hardware alternative; not an algorithm receiving an unchanged focal-plane PSF. |
| Photonic lantern | Mode mixing and possibly spectral dispersion generate output-port intensities. | Hardware alternative with finite port count, coupling and degeneracy constraints. |

[Meimon et al. 2010, R01](https://doi.org/10.1364/OL.35.003036) provides LIFT's original identity; its full text was not obtained. [Plantet et al. 2011, R02](https://ao4elt2.lesia.obspm.fr/sites/ao4elt2/IMG/pdf/113meimon.pdf) supplies the inspected equations and monochromatic/broadband laboratory evidence. R03-R05 verify later LIFT and Keck lineage only; their inaccessible full texts cannot substantiate additional numerical claims.

[Fast & Furious, R09](https://arxiv.org/abs/1406.1006), its [SCExAO on-sky work, R10](https://arxiv.org/abs/2005.12097) and [Keck work, R11](https://arxiv.org/abs/2107.07601) prevent omission of a serious non-neural focal-image family. Their abstracts were read; exact filter/cadence details need full-text extraction before implementation.

[Quesnel et al., R13](https://arxiv.org/abs/2210.00632) uses vortex phase diversity in learned reconstruction. [LLOWFS R18-R19](https://arxiv.org/abs/1506.06298), [Zernike-mask R21](https://arxiv.org/abs/1305.5143), [ZELDA on-sky tests R22](https://arxiv.org/abs/1806.06158), [Keck IR pyramid R23](https://doi.org/10.1117/1.JATIS.6.3.039003), and [lantern R26-R29](https://arxiv.org/abs/2208.10563) establish multiple physical encoding alternatives. R22 is the **2018 SPIE preprint**, not the separate 2019 A&A ZELDA study.

**INFERENCE from measurement models:** global piston has no intensity signature. Single symmetric focused images can retain sign/phase degeneracies; a network prior cannot manufacture missing information. Declare the pupil, OPD convention, basis normalization and diversity. Differential segment/petal piston needs its own basis, capture-range study and optical encoding. Do not merge it silently into low-order Zernike regression.

## Telescope and instrument cases

requirements.csv is the authoritative extracted matrix. UNKNOWN means unavailable or unverified in this pass, not zero or unconstrained. Published design envelopes, historical architecture, simulations and measured operation are distinct statuses.

- **TMT:** current [official NFIRAOS/instrument overview](https://www.tmt.org/page/instruments-adaptive-optics) assigns NIR TT/TTF roles to IRIS OIWFS and truth TT/flexure to ODGW. NFIRAOS's up-to-800 Hz high-order rate is not an OIWFS allocation. [Dunn et al. 2014, R34](https://arxiv.org/pdf/1407.2999) gives a historical 1.16-2.31 um optical design and interchangeable TT imaging/2x2 SH. Current detector, sensor pixels, magnitude thresholds and total latency remain unverified. MODHIS OIWFS design is still being finalized in the public overview.
- **ELT:** [MORFEO R35](https://arxiv.org/html/2608.13728v1) separates J/H fast LO from R/I reference sensors; Tables 1-3 provide sampling and magnitude-dependent timing baselines. These are not measured control latency. [HARMONI R37](https://arxiv.org/pdf/2607.20330) replaces original LTAO with MORFEO MCAO following rescope: three FREDA J/H TTFS arms and ALICE visible truth SH sensors. MICADO NGS timings must not be transplanted to HARMONI. [METIS R38](https://doi.org/10.1007/s10686-024-09968-2) combines K/H pyramid sensing with science-focal-plane QACITS/ALWFS auxiliary loops. Its Table 2 Strehl/contrast values are not a LOWFS-only nm allocation. Section 5.2.1 specifies a 909 us RTC computation ceiling; this excludes full acquisition-to-actuation latency.
- **GMT:** [public LTAO control architecture](https://gmto.github.io/gmt_docs/SWC_architecture/tcs/wavefront_control/ltao_obsmode.html) describes approximately 10 Hz OIWFS focus, asynchronous TT, and separate segment-phasing/edge-sensor cadences. It cites older architecture and incomplete full-loop modeling. The current large architecture PDF exceeded the retrieval tool limit; current instrument detector/band/latency requirements remain UNKNOWN. A snippet is insufficient to assert a current 1/2 kHz NGWS requirement.

Official [TMT optics](https://www.tmt.org/page/optics), [ELT facts](https://elt.eso.org/about/facts/) and [GMT primary mirrors](https://giantmagellan.org/telescope-primary-mirrors/) define three distinct segmented aperture classes. They do not alone supply runnable instrument pupils. Exact segment gaps, missing segments, obscurations, spiders, pupil rotation and registration need source/version-controlled prescriptions. An illustrative mask must be labelled illustrative.

**INFERENCE:** “ELT ready” cannot follow from one ideal annular or segmented mask. Likewise, sky coverage needs distributions of star colors/fluxes, patrol geometry, AO residuals and detector/throughput models; no sky-coverage claim is supported by the present review.

## Surviving differentiators and decisive first test

The following are **HYPOTHESES**, not originality findings:

1. Physically sampled signed-focus reconstruction with calibrated abstention under unseen residual spectra and NCPA/diversity shifts, outperforming the best calibrated classical estimator at the same information budget.
2. Lightweight learned initialization or bounded refinement whose accuracy/failure/latency trade-off survives independent physical conditions, rather than merely accelerating an unmatched optimizer.
3. Transfer across declared pupil/sampling conditions with explicit calibration cost and degradation, demonstrated before extrapolation to telescope-class cases.

For the first benchmark, retain Gaussian/ellipticity, empirical GS from R07, a physically sampled GS variant, calibrated LIFT including R08's calibration concept, a simple non-neural regression, R06 reproduction and an appropriate modern learned comparator. The empirical GS sampling/gain reported in R07 must be documented rather than silently corrected; its physically correct counterpart is a separate comparison.

Test known signed focus on held-out independent sequences. Vary photon flux, broadband integration, sub-Nyquist sampling, background/read noise, pupil displacement, diversity miscalibration, NCPA and AO-residual spectra. Fit/calibrate on separate conditions. Report bias, modal RMSE, sign errors, catastrophic failures, abstention coverage and calibration effort. A random image split alone does not establish independence.

Measure exposure/readout, preprocessing, transfers, reconstruction, communication and actuation separately; report p50/p95/p99 and deadlines on named hardware. Add closed-loop tests only after static sign/scale and independent forward-model checks pass. Exact numerical success margins belong to the selected use case and must be frozen before decisive data, not selected after observing a favorable result.

**GO:** a narrow reproduction and benchmark design after missing critical methods are retrieved and physical observability is settled. **NO-GO:** manuscript novelty claims, broad TMT/ELT/GMT readiness, fast-loop superiority, or new training before the coordinator integrates this evidence with design/ and keck/.

A useful null result could show that calibrated LIFT or GS dominates under realistic shifts, or that calibration—not learning—is the limiting factor. That outcome would answer the research question without an arbitrary “AI wins” criterion.

## Material limits

Public text and small PDFs only; no data downloads, code execution for science, training, installs, proprietary measurements, external contacts, email or Drive. Literature and requirements extraction are not runtime validation.

ADS web access failed; the search is not ADS-complete. Several publisher/conference full texts and the dedicated MORFEO LIFT proceeding remain unread. Coverage includes close classical, learned, physics-guided and hardware-encoded families across Keck, VLT, Subaru and Magellan, plus TMT/ELT/GMT designs. It is a bounded collision review, not an exhaustive worldwide census. See search-log.md for exact queries, retrieval limits and verification instructions.
