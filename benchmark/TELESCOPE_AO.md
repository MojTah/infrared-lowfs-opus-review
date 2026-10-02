# First public telescope and AO case

Source review: 30 September 2026. The executable benchmark must start with **historical Keck I LGS AO / TRICK**. TMT, ELT and GMT are conditional transfer cases. Machine-readable provenance and unknowns are in `telescope_case.json`.

## What is reproducible now

Use HCIPy 0.7.1's published Keck aperture rather than a generic 36-hexagon mask. Its code includes segmentation, obstruction and spiders, and its documentation cites agreement with Keck internal simulations. The cited validation publications concern Keck II; call it a published Keck pupil model, not an as-built Keck I/TRICK calibration. [HCIPy aperture documentation](https://docs.hcipy.org/0.7.0/api/hcipy.aperture.make_keck_aperture.html)

The function signature is `make_keck_aperture(normalized=False, with_spiders=True, with_segment_gaps=True, gap_padding=10, segment_transmissions=1, return_segments=False)`. Freeze `normalized=True`, physical `gap_padding=1`, and evaluate at pupil coordinates divided by 10.949 m. This rescales the library's native 10.95 m geometry while preserving its ratios. Integrate pupil edges and check sampling convergence: physical 3 mm gaps and 26 mm spiders can disappear on a coarse point-sampled grid. The tenfold gap padding is a visualization/small-grid convenience, not physical gap evidence.

Pinned source was read directly from the public `hcipy-0.7.1-py3-none-any.whl` in memory, without installation. SHA-256: `33ca4a70fcf62f53b95875ee9e41ed199d17eb2d39076b2b9a34192be3f72f60`. Functions reviewed: Keck, TMT, ELT and GMT aperture generators, plus `ModalAdaptiveOpticsLayer`. [Package release](https://pypi.org/project/hcipy/0.7.1/)

The TRICK geometry is anchored to 50 mas/pixel, 10.949 m pupil diameter, H/K wavelengths 1.65/2.18 micrometres, and approximately 200 nm RMS natural Noll-Z6 astigmatism. The same paper's bench residual test used about 200–300 nm RMS from DM-command replay; it does **not** publish that command sequence or a reproducible residual PSD. The sodium-height PSD with alpha 35 m²/Hz and exponent −1.9 describes slow focus, not high-order AO residuals. [Salgueiro 2026, §§4–6.2](https://arxiv.org/html/2602.15746v1)

## Source-backed atmosphere and controller

The strongest public recipe in the required prior work is the OOMAO Keck simulation configuration below. [Taheri 2024, §2/Table 1 and §3.1](https://arxiv.org/html/2410.12084v1)

| Quantity | Published value |
|---|---|
| Fried parameter / outer scale | 0.165 m / 75 m |
| Heights, km | 0, 0.5, 1, 2, 4, 8, 16 |
| Raw layer strengths | 0.51, 0.11, 0.06, 0.06, 0.10, 0.08, 0.05 |
| Wind speeds, m/s | 6.7, 13.9, 20.8, 29, 29, 29, 29 |
| Wind directions, radians | 0, π/3, −π/3, −π, −4π/3, −π/6, π/8 |
| HOWFS / DM | 20×20 SH / 21×21 actuator grid |
| Controller | 1 kHz, gain 0.25, one-step lag |
| TRICK acquisition | 10 ms, accumulated over ten AO steps |
| Noisy study conditions | Photon noise and 5-electron read RMS |

The layer strengths sum to **0.97**, not one. Preserve the published vector and explicitly renormalize for a simulator requiring normalized strengths. The reference wavelength of r0 is absent from the table. If 500 nm is needed, label it a conventional modeling assumption. DM influence, WFS detector/wavelength, reconstructor regularization, LGS cone and off-axis geometry are also insufficiently specified for exact replication.

Current Keck I KAPA is different: the official overview specifies a 349-actuator Xinetics DM, 20×20 SH at up to 2 kHz, and four LGS on a 15.2-arcsec-radius square. The historical 1 kHz recipe must retain its date and system label. [Current Keck AO overview](https://www2.keck.hawaii.edu/inst/ao/)

## Residual fidelity and acceptance

An RMS-normalized, spatially filtered frozen-flow screen is an **AO residual surrogate**. Using a real pupil and published atmosphere does not make it measured Keck residuals or a full Keck AO simulation. HCIPy's own modal layer is explicitly an infinite-SNR approximation. [Modal-layer API](https://docs.hcipy.org/0.7.0/api/hcipy.atmosphere.ModalAdaptiveOpticsLayer.html)

For a stronger public temporal model, historical Keck characterization provides camera integration, zero-order hold, delay and controller transfer functions. With `s=i2πf`, its turbulence rejection is `R(f)=1/(1+H(f))`, where `H=H_stare H_delay H_DM H_ZOH`; stare/hold is `(1-exp(-sT))/(sT)` and delay is `exp(-s tau)`. The same source supplies a measured 349-actuator DM influence approximation: two Gaussians with weights 2/−1 and widths 0.54/0.85 subaperture. These calibrations remain historical. [van Dam 2004, §§2C and 3C1, equations 3 and 13–20](https://www2.keck.hawaii.edu/optics/aodocs/ApplOptVol43_29oct2004.pdf)

The public Keck LGS characterization uses the same residual/noise transfer framework and illustrates a 300 Hz LGS case with separate bandwidth and measurement-noise terms. It is useful for validating model shape, not for substituting a 2006 error budget for current TRICK conditions. [van Dam 2006, §2.2/Fig. 1](https://www2.keck.hawaii.edu/optics/aodocs/MvDetal2006PASP118_310.pdf)

HCIPy's official Shack–Hartmann tutorial provides a measured-slope reconstruction and DM-integrator loop; reuse it when the stronger simulation is needed. Do not silently call the modal/spatial-filter route its equivalent. The development documentation was inspected; the versioned 0.7.0 tutorial exceeded the web tool's 4 MiB response limit. Reconcile any reused tutorial calls with the pinned 0.7.1 source. [Official SH tutorial](https://docs.hcipy.org/dev/tutorials/ShackHartmannWFS/ShackHartmannWFS.html)

Required residual evidence before any system-performance claim:

- Save actual spatial PSD, temporal PSD/autocorrelation, residual RMS distribution and exposure convergence by independent parent sequence.
- Record layer/reference-wavelength assumptions, correction/filter/operator, loop timing, gain/delay, amplitude normalization and low-order projection.
- Treat 100/200 nm training and 300 nm challenge as the study's chosen perturbation grid. A published 200–300 nm bench range does not establish the 100 nm case as a median system value.
- Scale residual amplitude once per sequence if prescribed; framewise RMS renormalization changes physical temporal statistics and must be separately disclosed.
- Preserve the four injected coefficients as the slow target. Report whether residual projection removes those modes: otherwise atmospheric low-order residuals contribute an irreducible confusion term.
- Do not give estimators the simulated test residual, exact nuisance values or per-test phase-derived PSFs. Freeze inference residual statistics on development/calibration sequences.

No measured residual time series was accessed. Keck's public page says AO telemetry may be supplied to observers; that is not a public download or permission to obtain private data.

## Conditional TMT, ELT and GMT transfer

Only admit these after the Keck optical/count/temporal/recovery gates and the frozen-model independent-engine test. A different diameter needs new pupil/basis normalization and angular sampling; it is not a resized Keck PSF. Keep matched dimensionless sampling separate from physical instrument pixels. Derive sodium-focus coefficients on the actual pupil; at fixed normalized geometry, the same sodium-height disturbance grows as diameter squared.

| Candidate | Public facts to freeze | Still required |
|---|---|---|
| TMT NFIRAOS / IRIS OIWFS | 30 m, 492 hexagonal segments; two DMs 63×63/76×76 at 0/11.8 km; six LGS, five on a 35-arcsec ring plus central; up to 800 Hz HO loop; three IRIS OIWFS | Actual OIWFS pixel scale, bandpass, detector/calibration and residual statistics; 800 Hz is not an OIWFS allocation |
| ELT MORFEO / MICADO | 39 m, 798 segments; six 68×68 LGS SH at 500 Hz; J/H LO full-aperture 7 mas or 2×2 SH 15 mas; LO 100–1000 Hz; visible 8×8 references 10–100 Hz | A specific LO sensor/diversity and residual model; full-aperture LO primarily senses TT, focus uses 2×2 SH; dedicated LIFT channel targets petals |
| GMT historical LTAO OIWFS | Seven 8.4 m mirrors, 25.4 m extent; historical architecture combines about 10 Hz OIWFS focus with high-pass tomographic focus | Current OIWFS band/pixels/detector, controller/residuals and allocations; source contains 2013 references and incomplete TBA tables |

Sources: [TMT optics](https://www.tmt.org/page/optics), [NFIRAOS/IRIS official overview](https://www.tmt.org/page/instruments-adaptive-optics), [ESO ELT facts](https://elt.eso.org/about/facts/), [MORFEO 2026, §2/Table 1](https://arxiv.org/html/2608.13728v1), [GMT primary mirrors](https://giantmagellan.org/telescope-primary-mirrors/), [GMT historical LTAO architecture, §5.5.3](https://gmto.github.io/gmt_docs/SWC_architecture/tcs/wavefront_control/ltao_obsmode.html).

HCIPy has named generators for all three pupils. Their source geometry epochs must remain explicit: native diameters are TMT 30 m, ELT 39.14634 m and GMT 25.448 m. ELT's generator cites an older construction proposal; the nominal modern 39 m label is not a reason to hide that difference. Freeze physical or normalized rescaling deliberately and rerun convergence/independent propagation. Segment/petal piston remains outside the four global-mode estimator.

HARMONI and METIS are distinct ELT systems. MICADO's MORFEO NGS specifications cannot be copied into them. Current HARMONI source review and METIS reference records already live in `literature/requirements.csv`; this first-stage implementation does not select them or assert their sensor/residual allocations.

## Verification and limits

This deliverable is a source/configuration freeze. JSON parsing, source-wheel checksum and source/API inspection are static checks. No telescope-control, optics, atmosphere, learned model or performance experiment was run by this researcher. No code or environment package was installed. Exact 2024 reproduction and current observatory readiness remain unverified.

## Installed aperture and conditional-transfer follow-up

Installed HCIPy **0.7.1** was inspected using `benchmark/.venv/Scripts/python.exe`. Constructor-only checks returned **492 TMT, 798 ELT and 7 GMT segments**. No atmospheric or focal-plane simulation was launched. These are credible published geometry proxies; facility-wide pupil names do not freeze instrument stops, orientation or AO residuals.

Minimum calls are `make_tmt_aperture(normalized=True, with_spiders=True, segment_transmissions=1, return_segments=True)`, the equivalent `make_elt_aperture` call, and `make_gmt_aperture(normalized=True, with_spiders=True, return_segments=True)`. GMT has no `segment_transmissions` keyword. None accepts diameter, obstruction, gap or spider width overrides. Keep native geometry or document uniform rescaling through the grid; use a versioned instrument pupil file when its geometry differs.

| Installed geometry | Embedded model and required distinction |
|---|---|
| TMT | 30 m; 1.44 m corner-to-corner segments; 2.5 mm gaps; 3.636 m circular pupil shadow plus missing central segments; six 0.22 m supports. The 3.1 m mechanical secondary in official optics is not automatically the projected pupil shadow. Source supports/shadow derive from Jensen-Clem 2021 Figure 5. |
| ELT | 39.14634 m; 1.45 m segments; 4 mm gaps; central missing-segment mask parameter 9.4136 m; six 0.4 m supports. This older construction-proposal model differs from METIS's 0.54 m supports and TIPTOP MORFEO's 38.5 m effective pupil. |
| GMT | 25.448 m; central projected segment 8.293 m with 3.495 m central hole; six outer 8.35 m segments projected at 13.522° as ellipses; central truss and diverging supports. Seven identical circles omit source geometry. |

For H-band at 1.65 µm, calculations of `lambda/(2D)` give Nyquist pixel scales **5.67 mas TMT, 4.36 mas ELT and 6.70 mas GMT**. Retaining 50 mas means 4.41, 5.73 and 3.73 `lambda/D` per pixel, respectively. At the 1.50 µm band edge, Nyquist scales tighten to 5.16, 3.97 and 6.09 mas. Keep 50 mas only as an explicitly severe undersampling experiment; freeze physical instrument sampling separately. These are calculated sampling limits, not claimed detector specifications.

Additional public residual resources were reviewed without installation or simulation:

- **NFIRAOS:** public MAOS site-profile documentation, example configurations and telescope wind/vibration PSD inputs at revision `16aed2e10885c83e27d087738fdfc90f6a027ccd`. The reviewed example uses **600 Hz**, so it cannot silently inherit today's up-to-800-Hz design label. Recursive includes, instrument epochs and implementation errors need their own freeze. [Pinned NFIRAOS configuration](https://github.com/lianqiw/maos/blob/16aed2e10885c83e27d087738fdfc90f6a027ccd/config/maos/examples/nfiraos_lgs.conf), [pinned documentation](https://github.com/lianqiw/maos/blob/16aed2e10885c83e27d087738fdfc90f6a027ccd/docs/43_nfiraos.md).
- **MORFEO:** TIPTOP revision `3d8c20888b40d8fa8750caec2a46535b164b2697` includes a 35-layer median atmosphere, public ELT pupil/static-WFE files and windshake PSD. Its configuration uses 38.5 m, 500 Hz HO with gain 0.25 and three-step delay, and 250 Hz LO. This is a stronger system-specific analytical PSD comparator. It does not directly supply correlated 10 ms phase histories. Its reference pixel scale is 125 mas, differing from the 2026 control-paper value 165 mas; keep epochs explicit. [Pinned MORFEO configuration](https://github.com/astro-tiptop/TIPTOP/blob/3d8c20888b40d8fa8750caec2a46535b164b2697/tiptop/perfTest/MORFEO.ini), [error coverage and limitations](https://astro-tiptop-services.github.io/astro-tiptop-services/docs/general/error_breakdown/).
- **HARMONI:** the 2026 rescope uses MORFEO MCAO but retains its own three-arm NGS system: ALICE visible 8×8 truth sensors and FREDA J/H TTFS sensors with ADC/focus stages. No physical TTFS pixel scale or residual phase dataset was frozen here. Science 6/25 mas spaxels are not those sensor pixels. [Dohlen 2026, §4.2](https://arxiv.org/pdf/2607.20330).
- **METIS:** the 2024 methods describe COMPASS 5.0 with custom METIS/reconstruction/CCS modules and 1 kHz simulations, using 0.54 m pupil spiders. Selected outputs are offered upon request, not as an identified public residual download. Stock COMPASS alone does not establish reproduction. K-band pyramid sensing and MIR auxiliary NCPA loops require a separate case. [Feldt 2024, §§2.6 and 6.1, Data Availability](https://link.springer.com/article/10.1007/s10686-024-09968-2).
- **GMT:** no current public, complete system-specific residual configuration or time-series dataset was established in this bounded review. Retain unknown fields; the historical OIWFS focus architecture is not sufficient to instantiate present detector/controller performance.

Keck success permits a geometry-transfer control using built-in pupils; it does not authorize an instrument-performance claim. A later NFIRAOS/MORFEO/GMT residual case needs source identity, model assumptions, physical units, source/model epoch, temporal realization and the same independent numerical gates before training or evaluation.
