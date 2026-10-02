# Closest prior work and collision assessment

Date/cutoff: 2026-09-30. Evidence: small public text/PDF review, not a systematic-review completeness claim. Verdict: **NEAR COLLISION; novelty unresolved.** Generic AI LOWFS, learned initialization, calibrated physical refinement and single-shot hybrid sensing cannot be claimed as new. The defensible candidate is a narrowly measured robustness/error/failure/total-latency improvement under realistic infrared sampling and independently held-out mismatch.

## Mandatory full-text anchors

| ID | Verified source and reading level | What it establishes; consequence for this project |
|---|---|---|
| T24 | Taheri et al., *AI-Powered Low-Order Focal Plane Wavefront Sensing in Infrared* (2024), [full text](https://arxiv.org/html/2410.12084v1). Methods/results Sections 2-5 read. | ResNet18 focus estimation from OOMAO simulations with Keck-like residuals, diversity and sampling; real K-band results use a calibration source. Several displayed comparisons remove mean focus. This is the author's mandatory prior AI comparator, not evidence of atmospheric on-sky validation or uncorrected absolute bias. The text has inconsistent stated dataset totals and ramp-step counts; exact reproduction needs the original manifest/preprocessing/code. Do not infer confirmed leakage from an incompletely described image split. |
| S26 | Salgueiro et al., *Slow focus sensor for the Keck I laser guide star adaptive optics system using focal plane wavefront sensing* (accepted 2026), [full text](https://arxiv.org/html/2602.15746v1). Main text and Appendices A-B read. | GS was selected for 2025 sky tests for residual robustness; all three methods met the slow computational budget. Sampling/gain distinctions are addressed in the pipeline. This instrument-specific result is not a general rejection of LIFT or an established need for neural latency. Current operational availability belongs to the lead's review. |
| K24 | Kuznetsov et al., *Striving towards robust phase diversity on-sky: Implementing LIFT for VLT/MUSE-NFM*, A&A 687 A221 (2024), DOI 10.1051/0004-6361/202449860. [Full text](https://arxiv.org/html/2406.08529v1), [journal PDF](https://www.aanda.org/articles/aa/pdf/2024/07/aa49860-24.pdf). Main text and appendices read. | Offline NCPA calibration with differentiable optics precedes online low-order estimation. Short-exposure fitting and temporal averaging are distinguished. Improved on-sky calibration tests use narrow H-band plus DSM diversity; full broadband lens calibration and slow closed-loop use remain future steps in that account. Some low-flux tests add noise to bright data. Calibration/model fidelity is therefore a substantive competing explanation for improvements, not a detail that AI may bypass. |

Reading level describes this review, not independently reproduced results. No original models or proprietary data were executed. A claimed exact reproduction requires source/configuration identity and matching units, calibration and metrics; otherwise use **method-level reimplementation**.

## Architecture and recent-method collisions

| ID | Primary source | Evidence examined and collision |
|---|---|---|
| P18 | Paine and Fienup, *Smart starting guesses from machine learning for phase retrieval*, SPIE 10698, 106985W (2018), DOI 10.1117/12.2307858. [Author-hosted PDF](https://sites.rochester.edu/fienup/wp-content/uploads/2019/08/SWP_StartML-PR_SPIE106985W-2018.pdf). Related journal article: *Machine learning for improved image-based wavefront sensing*, Optics Letters 43, 1235-1238, DOI 10.1364/OL.43.001235; [author publication list](https://sites.rochester.edu/fienup/james-r-fienup/publications/). | Conference PDF/abstract and author metadata examined; the linked conference PDF is not the journal full text. Neural coefficient estimates initialize nonlinear phase retrieval. A learned initializer followed by physical optimization is an established architecture. The telescope/sampling/latency conditions still need a new matched test. |
| X21 | Orban de Xivry et al., *Focal Plane Wavefront Sensing using Machine Learning: Performance of Convolutional Neural Networks compared to Fundamental Limits* (2021). [Primary manuscript](https://arxiv.org/html/2106.04456v1), [MNRAS article](https://academic.oup.com/mnras/article/505/4/5702/6295324). | Abstract and simulation/measurement methods inspected. CNNs use diversity images with idealized circular pupils and photon noise. Useful for matched-information/noise bounds; a two-image result is not a fair baseline against a single exposure unless both receive the same information and photon budget. |
| B26 | Moayed Baharlou et al., *An end-to-end hybrid deep-learning approach for single-shot wavefront sensing and correction*, Nature Communications 17, 6340 (2026), DOI 10.1038/s41467-026-72364-1. [Primary full text](https://www.nature.com/articles/s41467-026-72364-1). | Main methods/results and availability statements read. A learned optical encoder and ResNet34-based APN decoder jointly sense/correct wavefronts in visible-light experiments. Generic hybrid/single-shot claims collide. Its optical mask changes information; use a separate matched-mask track. Code/data are restricted during an ongoing patent process in the availability statement. A fixed-diversity ResNet34 adaptation is an independent architecture control, not exact reproduction or an infrared telescope validation. |
| C25 | Kou et al., *Single-Shot Wavefront Sensing in Focal Plane Imaging Using Transformer Networks*, Optics 6(1), 11 (2025), DOI 10.3390/opt6010011. [Publisher](https://www.mdpi.com/2673-3269/6/1/11). | Publisher-indexed method passages examined; direct full-page/PDF retrieval failed. NAM-CoAtNet uses phase modulation and compares other learned architectures. Do not credit sign recovery to an unmodulated intensity image when optical encoding supplies information. Full protocol and code availability must be verified before choosing this as a decisive comparator. |
| V25 | Liu et al., *Physics-informed deep learning for accurate and efficient wavefront sensing in adaptive optics*, Optics Communications 596, 132458 (2025), DOI 10.1016/j.optcom.2025.132458. [Publisher](https://www.sciencedirect.com/science/article/pii/S0030401825009861). | Publisher abstract/method excerpts only. PI-VMamba combines a visual state-space model and physical wavefront/PSF priors. Physics-informed learning is already a close class. Full implementation, data and comparable infrared performance were not established here. |
| L25 | Yuxuan Liu et al., *Training networks without wavefront label for pixel-based wavefront sensing*, Frontiers in Physics 13, 1537756 (2025), DOI 10.3389/fphy.2025.1537756. [Publisher](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2025.1537756/full). | Primary indexed text only; direct full-text retrieval did not complete. PSF-based training without wavefront labels is adjacent to residual/physical-consistency losses. Detailed reproduction and applicability remain unresolved; no exact performance claim is imported. |
| U20/24 | Naimipour, Khobahi and Soltanalian, *Unfolded Algorithms for Deep Phase Retrieval*, [2020 preprint](https://arxiv.org/html/2012.11102v1). Expanded journal version: Naimipour, Khobahi, Soltanalian, Safavi and Shaw, Algorithms 17(12), 587 (2024), DOI 10.3390/a17120587, [publisher](https://www.mdpi.com/1999-4893/17/12/587). | Preprint abstract/model and several method sections inspected; journal publisher-indexed method/conclusion passages checked, direct page retrieval failed. Learned iterations/steps and measurement design already exist. Keep the two author/version records distinct. A generic unrolled phase-retrieval label is not a new contribution. |
| U25 | Shi, Gao and Zhang, *UPrime: Unrolled Phase Retrieval Iterative Method with provable convergence*, Signal Processing 226, 109640 (2025), DOI 10.1016/j.sigpro.2024.109640. [Publisher abstract](https://www.sciencedirect.com/science/article/pii/S0165168424002603). | Publisher-indexed abstract/highlights only. A recent model-driven/unrolled comparator class; neither its coded-diffraction information nor its stated performance can be assumed transferable to this focal-plane measurement. |
| R26 | Taşkın et al., *Exploring reinforcement learning to enhance focal-plane wavefront control for vortex coronagraphs*, [2026 preprint](https://arxiv.org/html/2609.01199v1). | Abstract and early methods inspected. Simulated ELT/METIS-related control uses temporal image/action history. It is an adjacent NCPA/controller collision, not the same single-frame NGS focus measurement. Keep sensing and closed-loop control claims separate. Refereed-publication status not established in this review. |
| O19 | Ovadia et al., *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift*, NeurIPS (2019), [primary preprint](https://arxiv.org/abs/1906.02530). | Abstract-level general uncertainty evidence. Calibration under one distribution does not establish reliable confidence under a shift. This motivates empirical optical shift tests, not a claim that a particular uncertainty technique is certified here. |

## Collision verdict and testable opening

| Proposed claim | Assessment | Required alternative evidence |
|---|---|---|
| First AI infrared LOWFS | Already collides with T24 | Do not claim |
| First neural initializer plus physics | Already collides with P18 and related phase retrieval | Do not claim |
| First calibrated differentiable focal-plane solver | Already collides with K24 | Do not claim |
| First hybrid/single-shot learned sensing | Already collides with B26 and encoded measurements | Do not claim |
| First unrolled/physics-informed estimator | Already collides with U20/24, U25 and V25 | Do not claim |
| Better robustness/error/failure/total latency for a specified infrared regime | Open hypothesis; S26 and K24 are strong alternatives | Shared calibration/information, untouched physical shifts, complete pipeline timing and meaningful margins |
| Transfer to TMT/ELT/GMT | Unproven | Public instrument-specific models, whole-pupil holdouts and separately reported recalibration |
| Reliable uncertainty/fallback | Unproven | Coverage, accepted fraction, shift failures and fallback timing; no universal OOD guarantee |

The first pilot should earn its complexity. A null result showing that correct sampling/calibration removes the apparent learning advantage is useful. A positive result must identify which failure mechanism the initializer resolves and whether that benefit survives independent calibration and full timing.

## Search record and remaining evidence

Search date: 2026-09-30. Public web indexing, arXiv full texts, author-hosted material and publisher pages were used. Representative literal queries:

```
focal plane wavefront sensing deep learning physics informed unrolled 2024 2025 2026 single image
single shot wavefront sensing neural uncertainty calibration domain shift 2025 2026
focal plane wavefront sensing neural phase retrieval differentiable optical model recurrent 2024 2025
"wavefront" "unrolled" "focal" learning
"focal-plane" "uncertainty" "neural"
"wavefront sensing" "generalization" neural network 2025
phase retrieval "unrolled" deep learning optics 2024 2025
site:ui.adsabs.harvard.edu "wavefront" "deep learning" "2025" "focal"
```

The ADS-index query did not yield useful records; an authenticated/comprehensive ADS search was not performed. Publisher access/indexing limitations mean this is a collision screen, not an exhaustive worldwide survey. The literature chat owns the broader scholarly inventory and instrument specifications; this document does not claim its target source count is complete.

Before a decisive comparison, resolve T24 reproduction assets/rights; verify a current learned method's complete protocol and usable implementation; obtain source-backed pupil/detector/diversity data; and freeze an independent calibration and test plan. Unknown code availability is not evidence that no code exists. The B26 restriction is explicit; no contact or download permission is inferred for private materials.

Deliverable status: design and bounded protocol complete; novelty, exact AI reproduction, actual compute quotas, bench rights and instrument readiness unresolved. Hugging Face access reported by Mojtaba is recorded in the pipeline/protocol as a later option, without a remote run or upload.
