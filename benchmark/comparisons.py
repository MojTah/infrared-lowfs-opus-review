"""Bounded, frozen-model comparisons and independently generated challenges.

Every method sees identical detector observations and nominal calibration.
Failure rates retain every planned observation; conditional errors are labelled.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch

try:
    from . import core, learning
except ImportError:
    import core
    import learning

METHODS = ("mlp", "ridge", "lift_style", "physical_multistart", "mlp_two_steps")
CHALLENGES = ("nominal", "read_noise", "ipc", "saturation", "diversity", "bad_pixels",
              "pixel_scale", "pupil_registration", "calibration_error")


def _admit(run_dir):
    run_dir = Path(run_dir).resolve()
    config = json.loads((run_dir / "config.json").read_text())
    readiness = json.loads((run_dir / "readiness.json").read_text())
    frozen = json.loads((run_dir / "models" / "frozen.json").read_text())
    manifest = learning._manifest(run_dir / "dataset.json")
    current = core.source_hash()
    if readiness.get("status") != "PASS":
        raise ValueError("Physics readiness must be PASS")
    for identity, expected in (("source_hash", current), ("config_hash", core.digest(config))):
        if any(record.get(identity) != expected for record in (readiness, frozen, manifest)):
            raise ValueError(f"Comparison admission {identity} mismatch")
    if core.file_hash(run_dir / "dataset.json") != frozen["dataset_manifest_sha256"]:
        raise ValueError("Frozen dataset manifest mismatch")
    if (manifest["measurement_hash"] != frozen["measurement_hash"]
            or readiness["measurement_hash"] != frozen["measurement_hash"]
            or learning._basis_hash(manifest["basis"]) != frozen["basis_sha256"]
            or learning._basis_hash(frozen["basis"]) != frozen["basis_sha256"]):
        raise ValueError("Frozen measurement/basis mismatch")
    if frozen["modal_order"] != core.MODES or frozen["coefficient_unit"] != "nm OPD RMS":
        raise ValueError("Frozen coefficient order/units mismatch")
    for filename, digest in frozen["artifacts"].items():
        path = (run_dir / "models" / filename).resolve()
        if not path.is_relative_to(run_dir / "models") or core.file_hash(path) != digest:
            raise ValueError(f"Frozen artifact mismatch: {filename}")
    return run_dir, config, readiness, frozen, manifest


def _guard(start, seconds, ram_bytes):
    if time.perf_counter() - start >= seconds:
        raise TimeoutError("Comparison/challenge time ceiling")
    if psutil.Process().memory_info().rss > ram_bytes:
        raise MemoryError("Comparison/challenge RAM ceiling")


def _protect(model, guard):
    original = model.image
    def image(*args, **kwargs):
        guard()
        result = original(*args, **kwargs)
        guard()
        return result
    model.image = image
    return model


def _predictors(run_dir, frozen, device):
    predictors = {}
    for design in ("single", "pair"):
        specification = frozen["models"][design]
        model, preprocessing = learning._load_checkpoint(
            run_dir / "models" / specification["selected_checkpoint"], device)
        with np.load(run_dir / "models" / specification["ridge"]["checkpoint"], allow_pickle=False) as saved:
            ridge_preprocessing = {"mean": saved["feature_mean"].copy(), "std": saved["feature_std"].copy(),
                                   "asinh_scale_e": float(saved["asinh_scale_e"])}
            weights, intercept = saved["weights"].copy(), saved["intercept"].copy()
        predictors[design] = {
            "mlp": lambda image, m=model, p=preprocessing: learning._predict_network(m, image[None], p, device)[0],
            "ridge": lambda image, p=ridge_preprocessing, w=weights, b=intercept:
                ((learning._features(image[None], p) @ w + b) * learning.LABEL_SCALE)[0]}
    return predictors


def _invoke(predict, image, design, device, guard):
    expected = (1 if design == "single" else 2, 16, 16)
    if np.asarray(image).shape != expected or not np.isfinite(image).all():
        return {"status": "NO_ESTIMATE", "coeff_nm": None, "latency_ms": 0.0,
                "reason": "invalid/nonfinite detector observation"}
    start = time.perf_counter()
    try:
        guard()
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = predict(image)
        if device.type == "cuda":
            torch.cuda.synchronize()
        guard()
        if not isinstance(result, dict):
            result = {"status": "OK", "coeff_nm": np.asarray(result).tolist()}
        result = dict(result)
        if result.get("coeff_nm") is not None:
            coefficients = np.asarray(result["coeff_nm"], dtype=float)
            if coefficients.shape != (4,) or not np.isfinite(coefficients).all():
                raise FloatingPointError("Nonfinite or invalid coefficient estimate")
            result["coeff_nm"] = coefficients.tolist()
    except (TimeoutError, MemoryError) as exc:
        result = {"status": "TIME_CAP" if isinstance(exc, TimeoutError) else "RAM_CAP",
                  "coeff_nm": None, "reason": str(exc)}
    except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as exc:
        result = {"status": "NO_ESTIMATE", "coeff_nm": None,
                  "reason": f"{type(exc).__name__}: {exc}"}
    result["latency_ms"] = (time.perf_counter() - start) * 1000
    return result


def _row(parent, split, design, truth, method, estimate, acquisition=0):
    record = {"parent_id": parent, "split": split, "design": design,
              "acquisition_index": acquisition, "truth_nm": np.asarray(truth).tolist(),
              "method": method, **estimate}
    coefficients = estimate.get("coeff_nm")
    if coefficients is not None:
        error = np.asarray(coefficients) - truth
        record.update(error_nm=error.tolist(), reconstructed_error_nm=float(np.linalg.norm(error)))
    else:
        record.update(error_nm=None, reconstructed_error_nm=None)
    return record


def _summary(rows):
    parents = sorted({r["parent_id"] for r in rows})
    grouped, available_bias, available_mse = [], [], []
    for parent in parents:
        group = [r for r in rows if r["parent_id"] == parent]
        errors = [np.asarray(r["error_nm"]) for r in group if r["error_nm"] is not None]
        if errors:
            available_bias.append(np.mean(errors, axis=0))
            available_mse.append(np.mean(np.square(errors), axis=0))
        penalized = [np.square(r["error_nm"] if r["error_nm"] is not None else 2 * core.LIMITS) for r in group]
        grouped.append(np.r_[np.mean(penalized, axis=0),
                             np.mean([r["coeff_nm"] is None for r in group]),
                             np.mean([r["status"] != "OK" for r in group]),
                             np.mean([r["status"] in ("ITERATION_CAP", "TIME_CAP", "RAM_CAP") for r in group])])
    grouped = np.asarray(grouped)
    average = grouped.mean(axis=0)
    latencies = [r["latency_ms"] for r in rows if r["coeff_nm"] is not None]
    return {"n_planned_observations": len(rows), "n_parents": len(parents),
            "n_estimates": sum(r["coeff_nm"] is not None for r in rows),
            "parents_with_estimates": len(available_bias),
            "coefficient_bias_nm_available": np.mean(available_bias, axis=0).tolist() if available_bias else None,
            "coefficient_rmse_nm_available": np.sqrt(np.mean(available_mse, axis=0)).tolist() if available_mse else None,
            "reconstructed_wavefront_rmse_nm_available": float(np.sqrt(np.mean(available_mse, axis=0).sum())) if available_mse else None,
            "penalized_coefficient_rmse_nm_all": np.sqrt(average[:4]).tolist(),
            "penalized_wavefront_rmse_nm_all": float(np.sqrt(average[:4].sum())),
            "no_estimate_fraction_all": float(average[4]), "non_OK_fraction_all": float(average[5]),
            "cap_fraction_all": float(average[6]),
            "ci95_parent_bootstrap_penalized_rmse_nm": learning._bootstrap_interval(grouped[:, :4], lambda m: np.sqrt(m)),
            "latency_ms_actual_batch1_available": {"p50": float(np.percentile(latencies, 50)),
                                         "p95": float(np.percentile(latencies, 95)),
                                         "p99": float(np.percentile(latencies, 99))} if latencies else None,
            "metric_weighting": "equal parent; equal acquisition within each parent",
            "failure_policy": "all planned observations retained; absent estimate penalty=2*label_limit per mode; available errors explicitly conditional"}


def _report(rows):
    groups, comparisons = {}, {}
    for split in sorted({r["split"] for r in rows}):
        for design in ("single", "pair"):
            subset = [r for r in rows if r["split"] == split and r["design"] == design]
            if not subset:
                continue
            key = f"{split}/{design}"
            groups[key] = {method: _summary([r for r in subset if r["method"] == method])
                           for method in sorted({r["method"] for r in subset})}
            baseline = {(r["parent_id"], r["acquisition_index"]): r for r in subset if r["method"] == "mlp"}
            comparisons[key] = {}
            for method in sorted({r["method"] for r in subset} - {"mlp"}):
                delta = {}
                for row in [r for r in subset if r["method"] == method]:
                    base = baseline[(row["parent_id"], row["acquisition_index"])]
                    score = lambda r: np.mean(np.square(np.asarray(r["error_nm"] if r["error_nm"] is not None else 2 * core.LIMITS) / core.LIMITS))
                    delta.setdefault(row["parent_id"], []).append(score(row) - score(base))
                values = np.asarray([np.mean(v) for v in delta.values()])[:, None]
                comparisons[key][method] = {"minus_mlp_standardized_penalized_mse": float(values.mean()),
                                           "ci95_paired_parent_bootstrap": learning._bootstrap_interval(values, lambda m: m)}
    return {"metrics": groups, "paired_comparisons": comparisons, "raw": rows}


def _subset(run_dir, manifest, parent_count):
    selected = []
    for split in ("test", "shifted"):
        seen = set()
        for shard in sorted((s for s in manifest["shards"] if s["split"] == split), key=lambda s: (s["parent_id"], s["path"])):
            if shard["parent_id"] in seen:
                continue
            path = run_dir / shard["path"]
            if core.file_hash(path) != shard["sha256"]:
                raise ValueError("Comparison shard checksum mismatch")
            with np.load(path, allow_pickle=False) as saved:
                ids = np.flatnonzero(np.isclose(saved["flux_e"], 10000.0, rtol=0, atol=1e-6))
                if not len(ids):
                    continue
                index = int(ids[0])
                selected.append({"parent_id": shard["parent_id"], "split": split, "row_index": index,
                                 "path": shard["path"], "sha256": shard["sha256"], "flux_e": 10000.0,
                                 "truth": saved["labels_nm"][index].astype(float),
                                 "single": saved["images_single"][index].copy(), "pair": saved["images_pair"][index].copy()})
            seen.add(shard["parent_id"])
            if len(seen) >= parent_count:
                break
        if not seen:
            raise ValueError(f"No 1e4-electron acquisition in {split}")
    return selected


def compare(run_dir, max_seconds=1800, parent_count=10):
    """Compare five estimators on the same first 1e4-electron held-out observations."""
    if not 1 <= parent_count <= 10 or max_seconds <= 0:
        raise ValueError("Require 1..10 parents per split and positive time budget")
    start = time.perf_counter()
    run_dir, config, readiness, frozen, manifest = _admit(run_dir)
    output = run_dir / "classical.json"
    if output.exists():
        raise FileExistsError("Preserve existing classical comparison")
    guard = lambda: _guard(start, max_seconds, config["budgets"]["ram_bytes"])
    guard()
    selected = _subset(run_dir, manifest, parent_count)
    model = _protect(core.OpticalModel(config, basis=frozen["basis"]), guard)
    if model.measurement_hash != frozen["measurement_hash"]:
        raise ValueError("Classical calibration measurement mismatch")
    torch.set_num_threads(1)
    device = learning.execution_device(frozen['device'])
    predictors = _predictors(run_dir, frozen, device)
    rows = []
    for item in selected:
        for design in ("single", "pair"):
            inference = predictors[design]
            functions = {**inference,
                         "lift_style": lambda image: core.linearized_fit(model, image, design, iterations=5),
                         "physical_multistart": lambda image: core.physical_fit(model, image, design, max_nfev=30),
                         "mlp_two_steps": lambda image: core.linearized_fit(model, image, design,
                             initial=inference["mlp"](image), iterations=2, exact_iterations=True)}
            for method in METHODS:
                hardware = device if method in ("mlp", "mlp_two_steps") else torch.device("cpu")
                estimate = _invoke(functions[method], item[design], design, hardware, guard)
                rows.append(_row(item["parent_id"], item["split"], design, item["truth"], method, estimate, item["row_index"]))
        print("classical", item["parent_id"], "elapsed_s", round(time.perf_counter() - start, 2), flush=True)
    result = {"schema_version": 1, "status": "PARTIAL" if any(r["status"] in ("TIME_CAP", "RAM_CAP") for r in rows) else "COMPLETE",
              "source_hash": frozen["source_hash"], "config_hash": frozen["config_hash"],
              "measurement_hash": frozen["measurement_hash"], "frozen_sha256": core.file_hash(run_dir / "models" / "frozen.json"),
              "readiness_sha256": core.file_hash(run_dir / "readiness.json"),
              "subset": [{k: v for k, v in item.items() if k not in ("truth", "single", "pair")} for item in selected],
              "calibration": "nominal static PSF only; independent centroid/flux/background fitting; no residual/nuisance truth",
              "lift_identity": "calibrated LiFT-style analytical Jacobian; not exact upstream LIFT",
              "hybrid_updates": 2, "physical_initializations_nm": [[0,0,0,0],[-100,0,0,0],[100,0,0,0]],
              "max_seconds": max_seconds, "elapsed_s": time.perf_counter() - start,
              "latency_scope": "actual batch-one calls including preprocessing/transfers/synchronization; loading excluded",
              **_report(rows)}
    core.save_json(output, result)
    return result


def challenges(run_dir, max_seconds=600, parent_count=10):
    """Fresh independent parents; one acquisition each, all detector/optical challenges.

    parent_count is total fresh parents (maximum 20), not parents per split.
    Nonfinite bad pixels return NO_ESTIMATE directly through the common guard.
    """
    if not 1 <= parent_count <= 20 or max_seconds <= 0:
        raise ValueError("Require 1..20 fresh parents and positive time budget")
    start = time.perf_counter()
    run_dir, config, readiness, frozen, _ = _admit(run_dir)
    output = run_dir / "challenges.json"
    if output.exists():
        raise FileExistsError("Preserve existing challenge evaluation")
    guard = lambda: _guard(start, max_seconds, config["budgets"]["ram_bytes"])
    guard()
    nominal = _protect(core.OpticalModel(config, basis=frozen["basis"]), guard)
    guard()
    if nominal.measurement_hash != frozen["measurement_hash"]:
        raise ValueError("Nominal challenge measurement mismatch")
    models = {"nominal": nominal}
    for name, patch in (("pixel_scale", {"pixel_mas": config["pixel_mas"] * 1.02}),
                        ("pupil_registration", {"pupil_registration_diameter": [.01, 0.]}),
                        ("calibration_error", {})):
        changed = json.loads(json.dumps(config))
        changed.update(patch)
        models[name] = _protect(core.OpticalModel(changed, basis=frozen["basis"]), guard)
        guard()
    calibration = models["calibration_error"]
    x, y = np.asarray(calibration.pupil_grid.x), np.asarray(calibration.pupil_grid.y)
    pattern = calibration.project_residual((np.sin(16 * np.pi * x / config["diameter_m"]) *
                                           np.cos(14 * np.pi * y / config["diameter_m"])).reshape(calibration.n, calibration.n))
    rms = np.sqrt(calibration.weights @ pattern.ravel() ** 2)
    if not np.isfinite(rms) or rms <= 0:
        raise FloatingPointError("Projected calibration challenge has zero/nonfinite power")
    calibration.static_nm += pattern * (20.0 / rms)
    device = learning.execution_device(frozen['device'])
    torch.set_num_threads(1)
    predictors = _predictors(run_dir, frozen, device)
    rows, seeds = [], []
    for parent in range(parent_count):
        parent_id = f"fresh-challenge-{parent:04d}"
        seed = int(np.random.SeedSequence([config["seed"], 923781, parent]).generate_state(1)[0])
        seeds.append({"parent_id": parent_id, "seed": seed,
                      "phase_seed": seed + 1, "calibration_error_seed": seed + 2})
        rng = np.random.default_rng(seed)
        truth = rng.uniform(-core.LIMITS, core.LIMITS)
        centroid = rng.uniform(-20, 20, 2)
        base_single, base_pair = None, None
        try:
            guard()
            gain = readiness["residual_gain"]["nominal_per_nm"] * config["residual_rms_nm"][parent % 2]
            residual = core.ResidualSequence(nominal, seed + 1, gain).acquisition(0)
            # One unknown static high-order map per parent, matching nominal generation.
            residual += nominal.calibration_error(seed + 2)[None]
            guard()
        except (TimeoutError, MemoryError) as exc:
            residual = None
            generation_failure = "TIME_CAP" if isinstance(exc, TimeoutError) else "RAM_CAP"
        for condition in CHALLENGES:
            try:
                if residual is None:
                    raise TimeoutError("Fresh phase generation was capped")
                guard()
                model = models.get(condition, nominal)
                # Detector variants share the same physical exposure; noise draws remain independent.
                if condition == "nominal":
                    base_single, base_pair = model.probabilities(truth, residual, centroid)
                if condition in ("pixel_scale", "pupil_registration", "calibration_error", "diversity"):
                    single, pair = model.probabilities(truth, residual, centroid, 1.1 if condition == "diversity" else 1.0)
                else:
                    if base_single is None:
                        raise TimeoutError("Nominal exposure unavailable after generation cap")
                    single, pair = base_single, base_pair
                rn = 10.0 if condition == "read_noise" else config["read_noise_e"]
                detector_condition = condition if condition in ("ipc", "saturation", "bad_pixels") else "nominal"
                images = {"single": core.detector_read(single * 10000, rng, config["background_e"] + config["dark_e"], rn, detector_condition),
                          "pair": core.detector_read(pair * 5000, rng, (config["background_e"] + config["dark_e"]) / 2, rn, detector_condition)}
                generation_failure = None
            except (TimeoutError, MemoryError) as exc:
                images = None
                generation_failure = "TIME_CAP" if isinstance(exc, TimeoutError) else "RAM_CAP"
            for design in ("single", "pair"):
                for method in ("mlp", "ridge"):
                    if images is None:
                        estimate = {"status": generation_failure, "coeff_nm": None, "latency_ms": 0.0,
                                    "reason": "challenge generation resource cap"}
                    else:
                        estimate = _invoke(predictors[design][method], images[design], design,
                                           device if method == "mlp" else torch.device("cpu"), guard)
                    rows.append(_row(parent_id, condition, design, truth, method, estimate))
            print("challenge", parent_id, condition, "elapsed_s", round(time.perf_counter() - start, 2), flush=True)
    result = {"schema_version": 1, "status": "PARTIAL" if any(r["status"] in ("TIME_CAP", "RAM_CAP") for r in rows) else "COMPLETE",
              "source_hash": frozen["source_hash"], "config_hash": frozen["config_hash"],
              "measurement_hash": frozen["measurement_hash"], "frozen_sha256": core.file_hash(run_dir / "models" / "frozen.json"),
              "parent_count": parent_count, "acquisitions_per_parent": 1, "seeds": seeds,
              "limitations": ["outside-crop IPC influx is not modeled"],
              "challenge_settings": {"source_e": 10000, "read_noise_e": 10, "ipc_neighbor_fraction": .02,
                                     "saturation_e": 2000, "diversity_multiplier": 1.1,
                                     "bad_pixel": [3,9], "pixel_scale_multiplier": 1.02,
                                     "pupil_registration_diameter": [.01,0.], "projected_calibration_error_rms_nm": 20,
                                     "nominal_calibration_error_rms_nm_range": [0, config["calibration_error_rms_nm"]]},
              "inference_calibration": "unchanged frozen nominal metadata; changed-model nuisance truth never supplied",
              "max_seconds": max_seconds, "elapsed_s": time.perf_counter() - start, **_report(rows)}
    core.save_json(output, result)
    return result
