"""Conditional PSF pilot on HCIPy's published TMT, ELT and GMT pupil proxies.

This extends a successful Keck benchmark's numerical pipeline. It does not
implement NFIRAOS, MORFEO, METIS or a GMT AO controller, or transfer a network.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import psutil

try:
    from .core import (LIMITS, MAS_RAD, MODES, OpticalModel, ResidualSequence,
                       calibrate_residual_gain, detector_read, digest,
                       file_hash, save_json, source_hash)
except ImportError:
    from core import (LIMITS, MAS_RAD, MODES, OpticalModel, ResidualSequence,
                      calibrate_residual_gain, detector_read, digest,
                      file_hash, save_json, source_hash)


CASES = (("tmt", 30.0), ("elt", 39.14634), ("gmt", 25.448))
TRUTH = np.array([73., 42., -61., 27.])
DIVERSITIES = (np.zeros(4), np.array([0., 200., 0., 0.]),
               np.array([200., 0., 0., 0.]))
THRESHOLD = 1e-3


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _basis_hash(basis):
    value = json.dumps(basis, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(value.encode()).hexdigest()


def _admission(run_dir):
    """Require the original successful, frozen and independently checked case."""
    run_dir = Path(run_dir).resolve()
    config = _read(run_dir / "config.json")
    readiness = _read(run_dir / "readiness.json")
    frozen_path = run_dir / "models" / "frozen.json"
    frozen = _read(frozen_path)
    dataset_path = run_dir / "dataset.json"
    dataset = _read(dataset_path)
    validation = _read(run_dir / "oopao" / "validation.json")
    if config.get("pupil", "keck") != "keck":
        raise ValueError("Original benchmark must use the known Keck pupil")
    if readiness.get("status") != "PASS" or validation.get("status") != "PASS":
        raise ValueError("Keck readiness and independent OOPAO validation must both PASS")
    identities = {"config_hash": digest(config), "source_hash": source_hash()}
    for key, expected in identities.items():
        for name, record in (("readiness", readiness), ("frozen", frozen),
                             ("dataset", dataset), ("OOPAO", validation)):
            if record.get(key) != expected:
                raise ValueError(f"{name} {key} differs from current original benchmark")
    if validation.get("frozen_sha256") != file_hash(frozen_path):
        raise ValueError("OOPAO did not validate the current frozen manifest")
    if frozen.get("dataset_manifest_sha256") != file_hash(dataset_path):
        raise ValueError("Original training dataset manifest changed after freeze")
    if frozen.get("modal_order") != MODES or frozen.get("coefficient_unit") != "nm OPD RMS":
        raise ValueError("Frozen coefficient convention differs from the benchmark")
    if _basis_hash(frozen["basis"]) != frozen.get("basis_sha256"):
        raise ValueError("Frozen actual-pupil basis changed")
    if frozen["basis"] != dataset.get("basis") or frozen["basis"] != readiness.get("basis"):
        raise ValueError("Original dataset/readiness/frozen basis mismatch")
    measurement = frozen.get("measurement_hash")
    if not measurement or any(r.get("measurement_hash") != measurement for r in (dataset, readiness)):
        raise ValueError("Original measurement identity mismatch")
    model_dir = run_dir / "models"
    artifacts = frozen.get("artifacts", {})
    for design in ("single", "pair"):
        selected = frozen["models"][design]["selected_checkpoint"]
        if selected not in artifacts:
            raise ValueError(f"No sealed {design} deployment checkpoint")
    for name, checksum in artifacts.items():
        path = (model_dir / name).resolve()
        if not path.is_relative_to(model_dir) or file_hash(path) != checksum:
            raise ValueError(f"Frozen model artifact changed: {name}")
    return config, frozen, {"frozen_sha256": file_hash(frozen_path), **identities,
                            "oopao_validation_sha256": file_hash(run_dir / "oopao" / "validation.json")}


def _guard(start, seconds, ram_bytes, pupil_n=None, supersampling=4):
    if time.monotonic() - start >= seconds:
        raise TimeoutError("Large-telescope elapsed-time ceiling reached")
    rss = psutil.Process().memory_info().rss
    # Conservative admission estimate for aperture evaluation, modes and MFT.
    # It is a guard before allocation; actual RSS is checked at every yield.
    estimate = 0 if pupil_n is None else pupil_n ** 2 * (128 * supersampling ** 2 + 512)
    if rss + estimate >= ram_bytes:
        raise MemoryError("Large-telescope pupil would exceed the declared RAM ceiling")


def _relative_l1(first, second):
    denominator = float(np.sum(second))
    if denominator <= 0 or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise FloatingPointError("Nonfinite or empty physical image")
    return float(np.sum(np.abs(first - second)) / denominator)


def _converged_model(config, start, seconds, ram_bytes):
    """Retain one actual-pupil basis across independent/refined calculations."""
    evidence = {"status": "RUNNING", "relative_l1_threshold": THRESHOLD,
                "mixed_coefficients_nm": TRUTH.tolist(),
                "diversities_nm": [d.tolist() for d in DIVERSITIES],
                "refinements": [], "independent_propagation": {}}
    _guard(start, seconds, ram_bytes, 1024, config["boundary_supersampling"])
    model = OpticalModel(config, pupil_n=512)
    basis = model.basis
    evidence["basis_sha256"] = _basis_hash(basis)
    reference = OpticalModel(config, pupil_n=128, basis=basis)
    independent = []
    for diversity in DIVERSITIES:
        _guard(start, seconds, ram_bytes)
        independent.append(_relative_l1(reference.image(TRUTH, diversity),
                                         reference.independent_image(TRUTH, diversity)))
        _guard(start, seconds, ram_bytes)
    evidence["independent_propagation"] = {
        "pupil_n": 128, "relative_l1": independent,
        "scope": "Same declared pupil/basis/OPD; direct NumPy integration independent of HCIPy propagation",
        "status": "PASS" if max(independent) < THRESHOLD else "FAIL"}
    del reference
    gc.collect()
    if max(independent) >= THRESHOLD:
        evidence["status"] = "FAIL"
        return None, basis, evidence
    for refined_n in (1024, 2048):
        _guard(start, seconds, ram_bytes, refined_n, config["boundary_supersampling"])
        begin = time.monotonic()
        refined = OpticalModel(config, pupil_n=refined_n, basis=basis)
        disagreements = []
        for diversity in DIVERSITIES:
            _guard(start, seconds, ram_bytes)
            disagreements.append(_relative_l1(model.image(TRUTH, diversity),
                                               refined.image(TRUTH, diversity)))
            _guard(start, seconds, ram_bytes)
        elapsed = time.monotonic() - begin
        evidence["refinements"].append({"pupil_n": model.n, "refined_pupil_n": refined_n,
                                         "relative_l1": disagreements, "elapsed_s": elapsed})
        if max(disagreements) < THRESHOLD:
            evidence.update(status="PASS", admitted_pupil_n=model.n,
                            refined_pupil_n=refined_n)
            del refined
            gc.collect()
            return model, basis, evidence
        del model
        model = refined
        gc.collect()
        if refined_n == 1024 and seconds - (time.monotonic() - start) < 4 * elapsed:
            evidence.update(status="LIMIT", reason="Measured refinement cost projects beyond remaining budget")
            return None, basis, evidence
    evidence.update(status="LIMIT", reason="Image convergence failed through 1024 to 2048; tolerance retained")
    return None, basis, evidence


def _save_acquisition(path, values):
    """Exclusive file creation prevents replacing any completed parent."""
    with path.open("xb") as stream:
        np.savez_compressed(stream, **values)
    return file_hash(path)


def simulate(run_dir, max_seconds=1800, count=5):
    """Return an immutable-summary payload; CLI owns large_telescopes.json.

    Each parent contributes one labelled acquisition and both acquisition
    designs. Networks are neither fitted nor evaluated on these new pupils.
    A refused admission creates no directories or simulated observations.
    """
    if (isinstance(count, bool) or not isinstance(count, int) or count < 1
            or isinstance(max_seconds, bool) or not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("Positive finite time ceiling and integer parent count required")
    start = time.monotonic()
    run_dir = Path(run_dir).resolve()
    if (run_dir / "large_telescopes.json").exists():
        raise FileExistsError("Preserve existing large-telescope summary")
    try:
        original_config, frozen, identity = _admission(run_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"status": "NOT_ADMITTED", "reason": str(error), "cases": [],
                "required": "Keck readiness PASS, intact frozen models, matching OOPAO validation PASS"}
    output = run_dir / "large_telescopes"
    output.mkdir(exist_ok=False)
    seconds = min(float(max_seconds), float(original_config["budgets"]["cpu_seconds"]))
    ram_bytes = min(int(original_config["budgets"]["ram_bytes"]), 16 * 1024 ** 3)
    data_bytes = min(int(original_config["budgets"]["data_bytes"]), 4 * 1024 ** 3)
    retained_bytes = sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file())
    report = {"schema_version": 1, "status": "RUNNING", "original_case": original_config["case"],
              "original_identity": identity, "source_hash": source_hash(),
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "requested_parents_per_case": count, "acquisitions_per_parent": 1,
              "max_seconds": seconds, "ram_bytes_ceiling": ram_bytes,
              "retained_data_bytes_ceiling": data_bytes, "cases": [],
              "residual_model": "Reused Keck-like filtered frozen-flow AO challenge; ensemble 100 nm OPD RMS",
              "residual_reference": "Original seven-layer profile and spatial filter; independent new parent seeds",
              "limitations": ["Published HCIPy 0.7.1 geometric pupil proxies, not as-built pupil metrology",
                              "No instrument-specific NFIRAOS, MORFEO, METIS or GMT AO residuals",
                              "No real AO controller, scintillation or telescope performance prediction",
                              "New actual-pupil basis for each telescope; no Keck network transfer claim",
                              "No ML trained for these cases; labelled simulation pilot only",
                              "Convergence checks three specified mixed-mode/diversity inputs, not global parameter space",
                              "Time/RAM guards apply at native-call boundaries; execution owner bounds the process"]}
    for case_index, (name, diameter) in enumerate(CASES):
        case_output = output / name
        case_output.mkdir(exist_ok=False)
        record = {"pupil": name, "status": "RUNNING", "diameter_m": diameter, "shards": []}
        report["cases"].append(record)
        try:
            _guard(start, seconds, ram_bytes)
            config = json.loads(json.dumps(original_config))
            config.update(case=f"published_{name.upper()}_pupil_Keck_like_residual_challenge",
                          pupil=name, diameter_m=diameter, pupil_n=512, basis_reference_n=1024,
                          pixel_mas=config["band_um"][0] * 1e-6 / (2 * diameter * MAS_RAD),
                          phase_steps=10, residual_rms_nm=[100.], flux_e=[10000.],
                          parent_counts={"large_telescope_pilot": count})
            config["atmosphere"]["model"] = report["residual_model"]
            record.update(pixel_mas=config["pixel_mas"],
                          pixel_rule="lambda_min/(2*published_HCIPy_diameter), radians converted to mas")
            save_json(case_output / "config.requested.json", config)
            model, basis, convergence = _converged_model(config, start, seconds, ram_bytes)
            if model is not None:
                config["pupil_n"] = model.n
                model.config["pupil_n"] = model.n
            save_json(case_output / "config.json", config)
            save_json(case_output / "basis.json", basis)
            save_json(case_output / "optical_convergence.json", convergence)
            record.update(config_sha256=file_hash(case_output / "config.json"),
                          basis_sha256=file_hash(case_output / "basis.json"),
                          convergence_sha256=file_hash(case_output / "optical_convergence.json"),
                          optical_status=convergence["status"])
            if model is None:
                record.update(status=convergence["status"], reason=convergence.get("reason", "Independent optical disagreement"))
                save_json(case_output / "manifest.json", {
                    **record, "mode_order": MODES, "label_units": "nm OPD RMS",
                    "original_identity": identity, "ml_trained": False,
                    "real_ao_controller": False, "limitations": report["limitations"]})
                gc.collect()
                continue
            record["admitted_pupil_n"] = model.n
            record["measurement_hash"] = model.measurement_hash
            _guard(start, seconds, ram_bytes)
            gain, calibration = calibrate_residual_gain(model)
            save_json(case_output / "residual_calibration.json", {
                "gain_per_nm": gain, "target_ensemble_rms_nm": 100., **calibration,
                "scope": "Independent developer seeds; one ensemble gain, no per-parent/frame normalization"})
            record["residual_calibration_sha256"] = file_hash(case_output / "residual_calibration.json")
            for parent in range(count):
                _guard(start, seconds, ram_bytes)
                parent_seed = 730000000 + case_index * 100000 + parent
                child_seeds = np.random.SeedSequence(parent_seed).spawn(3)
                atmospheric_seed, coefficient_seed, detector_seed = [
                    int(seed.generate_state(1)[0]) for seed in child_seeds]
                truth = np.random.default_rng(coefficient_seed).uniform(-LIMITS, LIMITS)
                atmosphere = ResidualSequence(model, atmospheric_seed, gain * 100.)
                residual = atmosphere.acquisition(0, steps=10)
                single, pair = model.probabilities(truth, residual)
                rng = np.random.default_rng(detector_seed)
                background = config["background_e"] + config["dark_e"]
                images = {"images_single": detector_read(single * 10000., rng, background, config["read_noise_e"]).astype("float32"),
                          "images_pair": detector_read(pair * 5000., rng, background / 2, config["read_noise_e"]).astype("float32"),
                          "labels_nm": truth[None, :], "flux_e": np.array([10000.])}
                # Match existing shard convention: leading dimension is acquisition.
                images["images_single"] = images["images_single"][None, :]
                images["images_pair"] = images["images_pair"][None, :]
                if not all(np.isfinite(v).all() for v in images.values()):
                    raise FloatingPointError("Nonfinite pilot acquisition")
                fractions = {"single": single.sum(axis=(-2, -1)).tolist(),
                             "pair_per_exposure": pair.sum(axis=(-2, -1)).tolist()}
                if any(x < 0 or x > 1 + THRESHOLD for v in fractions.values() for x in v):
                    raise FloatingPointError("Invalid cropped source probability; never renormalized")
                _guard(start, seconds, ram_bytes)
                if retained_bytes + 65536 > data_bytes:
                    raise MemoryError("Retained dataset ceiling reached")
                filename = f"parent_{parent:04d}.npz"
                checksum = _save_acquisition(case_output / filename, images)
                retained_bytes += (case_output / filename).stat().st_size
                record["shards"].append({"path": str((case_output / filename).relative_to(run_dir)),
                    "sha256": checksum, "parent_id": f"{name}_pilot_{parent:04d}",
                    "parent_seed": parent_seed, "seed_spawn_order": ["atmosphere", "coefficients", "detector"],
                    "atmospheric_seed": atmospheric_seed, "coefficient_seed": coefficient_seed,
                    "detector_seed": detector_seed,
                    "acquisitions": 1, "phase_nodes": 10, "expected_source_e_total": 10000.,
                    "source_fraction_in_crop": fractions,
                    "calibration_identity": record["measurement_hash"],
                    "residual_rms_nm_by_node": [float(np.sqrt(model.weights @ r.ravel() ** 2)) for r in residual]})
                del atmosphere, residual
                gc.collect()
            record["status"] = "PASS"
            del model
            gc.collect()
        except (TimeoutError, MemoryError) as error:
            record.update(status="LIMIT", reason=str(error))
        except (FloatingPointError, ValueError) as error:
            record.update(status="FAIL", reason=str(error))
        finally:
            model = None
            gc.collect()
        save_json(case_output / "manifest.json", {
            **record, "mode_order": MODES, "label_units": "nm OPD RMS",
            "original_identity": identity, "inference_inputs": "Images and independent per-case calibration only",
            "truth_access": "Offline labels; residual RMS is diagnostic evidence, never estimator input",
            "source_photons_outside_crop": "Preserved as lost probability; source flux is not ROI-normalized",
            "pair_source_e": [5000., 5000.], "single_exposure_s": .01, "pair_exposure_s": [.005, .005],
            "switching_readout_overhead_s": None, "no_real_instrument_latency_claim": True,
            "ml_trained": False, "real_ao_controller": False, "limitations": report["limitations"]})
    report.update(status="PASS" if all(c["status"] == "PASS" for c in report["cases"]) else "PARTIAL",
                  elapsed_s=time.monotonic() - start,
                  retained_data_bytes=sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file()))
    if file_hash(run_dir / "models" / "frozen.json") != identity["frozen_sha256"] or source_hash() != identity["source_hash"]:
        report.update(status="FAIL", reason="Original frozen identity changed during pilot")
    return report
