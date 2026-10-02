"""Paired post-baseline amplitude/spectrum/diversity diagnostics; never refits.

One parent shares targets, centroid, unknown high-order calibration and raw
atmospheric seed across eight controlled conditions. Detector draws differ by
condition. GPU/native admission follows full OOPAO and frozen architectures.
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import sys
import time

NATIVE_STARTED = time.monotonic()
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
# Establish the frozen one-thread native settings before numerical imports.
from benchmark import core, cli, comparisons, learning as learn
import numpy as np
import torch

import architecture_evaluate as evaluation
import architecture_study as study

sys.path.insert(0, str(Path(__file__).with_name("expanded_campaign")))
import blocks
import gpu_backend
import gpu_mft

CONDITIONS = tuple(product((200., 300.), (10., 6.), (1., 1.1)))
METHODS = ("mlp", "ridge", "compact_cnn", "resnet18")
EFFECTS = ("amplitude", "spectrum", "diversity", "amplitude:spectrum",
           "amplitude:diversity", "spectrum:diversity", "amplitude:spectrum:diversity")


def condition_name(values):
    amplitude, cutoff, diversity = values
    return f"a{amplitude:g}_k{cutoff:g}_d{diversity:g}"


def parent_spec(config, index):
    seeds = np.random.SeedSequence([config["seed"], 2910357, index]).generate_state(4)
    rng = np.random.default_rng(int(seeds[0]))
    return {"parent_id": f"shift-diagnostic-{index:04d}", "truth_nm": rng.uniform(-core.LIMITS, core.LIMITS),
            "centroid_mas": rng.uniform(-20, 20, 2), "phase_seed": int(seeds[1]),
            "calibration_seed": int(seeds[2]), "noise_root_seed": int(seeds[3])}


def identity(run, count):
    if count not in (1, 20):
        raise ValueError("One-parent admission canary or the registered twenty-parent study required")
    run, manifest, baseline, protocol = study.identity(run)
    output, architecture = evaluation.frozen_study(run, protocol)
    for filename, expected_manifest in (("evaluation.json", run / "dataset.json"),
                                       ("oopao-evaluation.json", run / "oopao/oopao_dataset.json")):
        checked = study.read(output / filename)
        if (checked["status"] != "COMPLETE" or checked["identity"] != protocol
                or checked["architecture_frozen_sha256"] != learn._sha(output / "frozen.json")
                or checked["baseline_frozen_sha256"] != learn._sha(run / "models/frozen.json")
                or checked["evaluation_manifest_sha256"] != learn._sha(expected_manifest)
                or checked["reporter_sha256"] != learn._sha(evaluation.__file__)):
            raise ValueError("Complete matching architecture comparison on original and OOPAO parents required")
    record, config = cli.admitted(run)
    sources = (Path(__file__), Path(gpu_backend.__file__), Path(gpu_mft.__file__),
               PROJECT / "ledger/decisions/paired-shift-diagnostics-2026-10-02.md")
    return run, baseline, architecture, record, config, {
        "schema_version": 1, "sources": {p.resolve().relative_to(PROJECT).as_posix(): learn._sha(p) for p in sources},
        "baseline_frozen_sha256": learn._sha(run / "models/frozen.json"),
        "architecture_frozen_sha256": learn._sha(output / "frozen.json"),
        "architecture_reporter_sha256": learn._sha(evaluation.__file__),
        "measurement_hash": baseline["measurement_hash"], "basis_sha256": baseline["basis_sha256"],
        "engine_source_hash": baseline["source_hash"], "config_hash": baseline["config_hash"],
        "parent_count": count, "condition_order": [list(c) for c in CONDITIONS],
        "flux_e": config["flux_e"], "modal_order": learn.MODES,
        "seed_entropy": [config["seed"], 2910357, "parent index"],
        "scope": "post-baseline development; no untouched confirmation or telescope-performance claim",
        "inference_calibration": "frozen nominal metadata; no nuisance/residual truth supplied"}


def predictors(run, baseline, architecture, device):
    result = comparisons._predictors(run, baseline, device)
    for design in study.DESIGNS:
        for method in study.models.ARCHITECTURES:
            spec = architecture["models"][method][design]
            model, preprocess = study.load_checkpoint(run / "architecture-study-01" / spec["selected_checkpoint"], device)
            result[design][method] = lambda image, m=model, p=preprocess: learn._predict_network(m, image[None], p, device)[0]
    return result


def validate_arrays(arrays, declared):
    blocks._validate_arrays(arrays, 8, np.asarray(declared, dtype=float))
    labels = arrays["labels_nm"].reshape(8, 3, 4)
    if not np.array_equal(labels, np.broadcast_to(labels[:1, :1], labels.shape)):
        raise ValueError("All eight conditions must share each parent's target")


def factorial_effects(values):
    """Columns are standardized penalized MSE and penalized wavefront squared error."""
    values = np.asarray(values, dtype=float)
    if (values.ndim != 3 or values.shape[1:] != (8, 2) or not len(values)
            or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError("Finite parent x eight conditions x two scores required")
    signs = np.asarray(list(product((-1, 1), repeat=3)))
    codes = np.column_stack([signs[:, 0], signs[:, 1], signs[:, 2],
                            signs[:, 0] * signs[:, 1], signs[:, 0] * signs[:, 2],
                            signs[:, 1] * signs[:, 2], np.prod(signs, axis=1)]) / 4
    def transform(mean):
        return np.r_[mean[:, 0] @ codes, np.sqrt(mean[:, 1]) @ codes]
    mean = values.mean(axis=0)
    point = transform(mean)
    interval = learn._bootstrap_interval(values, transform) if len(values) > 1 else None
    return {"effect_order": list(EFFECTS), "standardized_penalized_mse_effect": point[:7].tolist(),
            "penalized_wavefront_rmse_effect_nm": point[7:].tolist(),
            "ci95_parent_bootstrap": interval, "ci_order": ["standardized_MSE effects[7]", "wavefront_RMSE effects[7]"],
            "n_parents": len(values), "definition": "twice orthogonal regression coefficients with -1/+1 factors; main effects are average high minus low; higher effects use the same normalization",
            "scope": "exploratory unadjusted marginal intervals conditional on fixed models; no familywise guarantee; canary has no intervals"}


def factor_report(rows):
    names = [condition_name(c) for c in CONDITIONS]
    report = {}
    for design in study.DESIGNS:
        report[design] = {}
        for method in METHODS:
            selected = [r for r in rows if r["design"] == design and r["method"] == method]
            parents = sorted({r["parent_id"] for r in selected})
            values = []
            for parent in parents:
                cell_scores = []
                for name in names:
                    cell = [r for r in selected if r["parent_id"] == parent and r["split"] == name]
                    if len(cell) != 3 or {r["acquisition_index"] for r in cell} != {0, 1, 2}:
                        raise ValueError("Three declared flux rows per parent/condition/method/design required")
                    errors = np.asarray([r["error_nm"] if r["error_nm"] is not None else 2 * core.LIMITS for r in cell])
                    cell_scores.append([np.square(errors / core.LIMITS).mean(), np.square(errors).sum(axis=1).mean()])
                values.append(cell_scores)
            report[design][method] = factorial_effects(values)
    return report


def fill_unexecuted(rows, config, count, reason):
    seen = {(r["parent_id"], r["split"], r["design"], r["method"], r["acquisition_index"]) for r in rows}
    for index in range(count):
        spec = parent_spec(config, index)
        for condition in CONDITIONS:
            for design in study.DESIGNS:
                for method in METHODS:
                    for flux_index in range(3):
                        key = (spec["parent_id"], condition_name(condition), design, method, flux_index)
                        if key not in seen:
                            rows.append(comparisons._row(spec["parent_id"], key[1], design, spec["truth_nm"], method,
                                {"status": "NOT_EXECUTED", "coeff_nm": None, "latency_ms": 0., "reason": reason}, flux_index))
    expected = count * 8 * 2 * 4 * 3
    if len(rows) != expected or len({(r["parent_id"], r["split"], r["design"], r["method"], r["acquisition_index"]) for r in rows}) != expected:
        raise ValueError("Planned diagnostic rows duplicated or missing")


def point_only(value):
    """The whole one-parent report has point diagnostics, without sampling intervals."""
    if isinstance(value, dict):
        return {k: point_only(v) for k, v in value.items() if not k.startswith("ci95") and k != "ci_order"}
    if isinstance(value, list):
        return [point_only(v) for v in value]
    return value


def require_canary(output, expected):
    canary = study.read(output / "report.json")
    resource = study.read(output.parent / "t001-shift-diagnostics-canary-resources.json")
    dataset = study.read(output / "dataset.json")
    if (canary["status"] != "PASS_ENGINE_COUNTERFACTUAL_CANARY" or canary["identity"] != expected
            or canary["completed_parent_shards"] != 1 or canary["planned_estimates"] != 192
            or canary["finite_estimates"] != 192 or len(canary["raw"]) != 192
            or resource["exit_code"] != 0 or resource["supervision_failure"] is not None
            or resource["cpu_reserved_for_reporting_seconds"] != study.RESERVE_CPU_SECONDS
            or resource["helper_source_sha256"] != learn._sha(__file__)
            or dataset["status"] != canary["status"] or dataset["diagnostic_identity"] != expected
            or dataset["shards"] != canary["shards"] or len(canary["shards"]) != 1):
        raise ValueError("Complete matching terminal native canary, dataset and report required")
    if any(r["status"] != "OK" or np.asarray(r["coeff_nm"]).shape != (4,)
           or not np.isfinite(r["coeff_nm"]).all() for r in canary["raw"]):
        raise ValueError("Every canary prediction must be finite and OK")
    fill_unexecuted(canary["raw"], {"seed": expected["seed_entropy"][0]}, 1, "missing canary row")
    optics = canary["optical_checks"]
    expected_optics = {(condition_name(CONDITIONS[i]), design) for i in (0, 7) for design in study.DESIGNS}
    if (len(optics) != 4 or {(r["condition"], r["design"]) for r in optics} != expected_optics
            or any(not r["pass"] or not np.isfinite(r["relative_L1"]) or not 0 <= r["relative_L1"] < 1e-3 for r in optics)):
        raise ValueError("All four declared native/GPU image checks required")
    root = output.resolve()
    for shard in canary["shards"]:
        path = (root / shard["path"]).resolve()
        if not path.is_relative_to(root) or learn._sha(path) != shard["sha256"]:
            raise ValueError("Canary shard missing, changed or outside its output root")
        with np.load(path, allow_pickle=False) as arrays:
            validate_arrays({k: arrays[k] for k in arrays.files}, expected["flux_e"])


def run_study(run, count, max_seconds, cpu_before_launch):
    run, baseline, architecture, record, config, protocol = identity(run, count)
    output = run / ("shift-diagnostics-canary-01" if count == 1 else "shift-diagnostics-01")
    if output.exists():
        raise FileExistsError("Preserve previous diagnostic outputs; inspect rather than restart blindly")
    if count == 20:
        require_canary(run / "shift-diagnostics-canary-01", dict(protocol, parent_count=1))
    resource_guard, storage_guard = study.limits(run, max_seconds, cpu_before_launch, output / "stop-request.json")
    def guard():
        if time.monotonic() - NATIVE_STARTED >= max_seconds:
            raise TimeoutError("Diagnostic deadline includes native imports/admission")
        resource_guard()
    guard()
    output.mkdir()
    learn._json(output / "protocol.json", protocol)
    rows, shards, optical_checks, failure = [], [], [], None
    with study.owner(output):
        try:
            torch.set_num_threads(1)
            device = learn.execution_device(baseline["device"])
            selected = predictors(run, baseline, architecture, device)
            cp = gpu_mft.cuda_backend(output / "cupy-cache")
            model = core.OpticalModel(config, basis=baseline["basis"])
            guard()
            gpu_backend.attach_device_phase(model, cp, 64)
            reference = core.OpticalModel(config, basis=baseline["basis"]) if count == 1 else None
            for index in range(count):
                guard()
                spec = parent_spec(config, index)
                calibration = model.calibration_error(spec["calibration_seed"])
                phases = {cutoff: core.ResidualSequence(model, spec["phase_seed"],
                    record["residual_gain"]["nominal_per_nm" if cutoff == 10 else "shifted_per_nm"],
                    shifted=cutoff == 6).acquisition(0) for cutoff in (10., 6.)}
                guard()
                stored = {key: [] for key in ("images_single", "images_pair", "labels_nm", "flux_e")}
                for condition_index, condition in enumerate(CONDITIONS):
                    guard()
                    amplitude, cutoff, diversity = condition
                    residual = phases[cutoff] * amplitude + calibration[None]
                    probabilities = model.probabilities(spec["truth_nm"], residual, spec["centroid_mas"], diversity)
                    if count == 1 and condition_index in (0, 7):
                        native = reference.probabilities(spec["truth_nm"], residual, spec["centroid_mas"], diversity)
                        for design, actual, expected in zip(study.DESIGNS, probabilities, native):
                            relative = float(np.abs(actual - expected).sum() / expected.sum())
                            optical_checks.append({"condition": condition_name(condition), "design": design,
                                                   "relative_L1": relative, "pass": relative < 1e-3})
                            if not optical_checks[-1]["pass"]:
                                raise ValueError("Counterfactual GPU/native optical canary failed")
                    noise = np.random.default_rng(np.random.SeedSequence([spec["noise_root_seed"], condition_index]))
                    for flux_index, flux in enumerate(config["flux_e"]):
                        images = {"single": core.detector_read(probabilities[0] * flux, noise,
                            config["background_e"] + config["dark_e"], config["read_noise_e"]),
                            "pair": core.detector_read(probabilities[1] * flux / 2, noise,
                            (config["background_e"] + config["dark_e"]) / 2, config["read_noise_e"])}
                        stored["images_single"].append(images["single"].astype("float32"))
                        stored["images_pair"].append(images["pair"].astype("float32"))
                        stored["labels_nm"].append(spec["truth_nm"])
                        stored["flux_e"].append(flux)
                        for design in study.DESIGNS:
                            for method in METHODS:
                                estimate = comparisons._invoke(selected[design][method], images[design], design,
                                    torch.device("cpu") if method == "ridge" else device, guard)
                                rows.append(comparisons._row(spec["parent_id"], condition_name(condition), design,
                                                             spec["truth_nm"], method, estimate, flux_index))
                arrays = {key: np.asarray(value, dtype="float32" if key.startswith("images_") else "float64")
                          for key, value in stored.items()}
                validate_arrays(arrays, config["flux_e"])
                storage_guard(sum(a.nbytes for a in arrays.values()) + 1024**2)
                shard = output / f"parent_{index:04d}.npz"
                with shard.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                shards.append({"path": shard.name, "sha256": learn._sha(shard), "parent_id": spec["parent_id"],
                               "split": "shifted", "rows": 24, "acquisition_start": 0, "acquisition_stop": 8,
                               "phase_seed": spec["phase_seed"], "calibration_seed": spec["calibration_seed"],
                               "noise_root_seed": spec["noise_root_seed"], "condition_order": protocol["condition_order"]})
                del phases, residual
                print("paired diagnostic parent complete", index + 1, "of", count, flush=True)
            guard()
            if identity(run, count)[5] != protocol:
                raise ValueError("Diagnostic source/model/runtime identity changed during execution")
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            fill_unexecuted(rows, config, count, failure or "missing execution row")
            finite = sum(r["coeff_nm"] is not None for r in rows)
            status = "COMPLETE" if failure is None and len(shards) == count else "PARTIAL"
            if count == 1 and status == "COMPLETE" and all(c["pass"] for c in optical_checks) and len(optical_checks) == 4 and finite == len(rows):
                status = "PASS_ENGINE_COUNTERFACTUAL_CANARY"
            result = {"status": status, "identity": protocol, "failure": failure,
                      "completed_parent_shards": len(shards), "planned_estimates": len(rows), "finite_estimates": finite,
                      "optical_checks": optical_checks, "shards": shards,
                      "physical_condition_acquisitions_planned": count * 8, "flux_rows_per_design_planned": count * 24,
                      "elapsed_seconds": time.monotonic() - NATIVE_STARTED,
                      "factorial_effects": factor_report(rows), **comparisons._report(rows)}
            if count == 1:
                result = point_only(result)
                result["interval_scope"] = "one-parent arithmetic/interface canary; all sampling-interval fields suppressed"
            core.save_json(output / "report.json", result)
            core.save_json(output / "dataset.json", {"schema_version": 1, "status": status,
                "source_hash": baseline["source_hash"], "config_hash": baseline["config_hash"],
                "measurement_hash": baseline["measurement_hash"], "basis": baseline["basis"],
                "acquisitions_per_parent": 8, "shards": shards, "diagnostic_identity": protocol})
    if failure:
        raise RuntimeError(failure)
    return {"status": status, "planned_estimates": len(rows), "output": str(output)}


def self_check():
    signs = np.asarray(list(product((-1, 1), repeat=3)))
    linear = 10 + 2 * signs[:, 0] + .5 * signs[:, 1] + .25 * signs[:, 2] + signs[:, 0] * signs[:, 1]
    values = np.tile(np.column_stack([linear, linear**2])[None], (4, 1, 1))
    report = factorial_effects(values)
    np.testing.assert_allclose(report["standardized_penalized_mse_effect"], [4, 1, .5, 2, 0, 0, 0])
    np.testing.assert_allclose(report["penalized_wavefront_rmse_effect_nm"], [4, 1, .5, 2, 0, 0, 0])
    assert factorial_effects(values[:1])["ci95_parent_bootstrap"] is None
    for damaged in (np.full_like(values, np.nan), np.full_like(values, -1)):
        try: factorial_effects(damaged)
        except ValueError: pass
        else: raise AssertionError("Invalid squared-error scores admitted")
    config = {"seed": 20260930}
    assert parent_spec(config, 0)["phase_seed"] == parent_spec(config, 0)["phase_seed"]
    assert parent_spec(config, 0)["phase_seed"] != parent_spec(config, 1)["phase_seed"]
    rows = []
    fill_unexecuted(rows, config, 1, "synthetic CPU failure fixture")
    assert len(rows) == 192 and all(r["coeff_nm"] is None for r in rows)
    assert comparisons._report(rows)["metrics"][condition_name(CONDITIONS[0]) + "/single"]["mlp"]["no_estimate_fraction_all"] == 1.
    assert all(v["n_parents"] == 1 for methods in factor_report(rows).values() for v in methods.values())
    assembled = {"factorial_effects": factor_report(rows), **comparisons._report(rows)}
    assert "ci95_parent_bootstrap_penalized_rmse_nm" in next(iter(next(iter(assembled["metrics"].values())).values()))
    point = point_only(assembled)
    def no_intervals(value):
        if isinstance(value, dict):
            assert not any(k.startswith("ci95") or k == "ci_order" for k in value)
            for nested in value.values(): no_intervals(nested)
        elif isinstance(value, list):
            for nested in value: no_intervals(nested)
    no_intervals(point)
    assert len(point["raw"]) == 192 and point["metrics"] and point["paired_comparisons"]
    fill_unexecuted(rows, config, 1, "must not duplicate rows")
    assert len(rows) == 192
    rows.append(rows[0])
    try: fill_unexecuted(rows, config, 1, "duplicate row fixture")
    except ValueError: pass
    else: raise AssertionError("Duplicate planned row admitted")
    declared = [1000., 10000., 100000.]
    arrays = {"images_single": np.zeros((24, 1, 16, 16), np.float32),
              "images_pair": np.zeros((24, 2, 16, 16), np.float32),
              "labels_nm": np.ones((24, 4), np.float64), "flux_e": np.tile(declared, 8)}
    validate_arrays(arrays, declared)
    arrays["labels_nm"][3:6] = 2
    try: validate_arrays(arrays, declared)
    except ValueError: pass
    else: raise AssertionError("Different targets across paired conditions admitted")
    # Actual artifact-admission function, synthetic CPU fixture only; no native/GPU PASS claim.
    import tempfile
    import json
    with tempfile.TemporaryDirectory(dir=PROJECT / "benchmark/runs") as temporary:
        parent = Path(temporary)
        output = parent / "canary-fixture"
        output.mkdir()
        arrays["labels_nm"][:] = 1
        shard_path = output / "parent_0000.npz"
        with shard_path.open("xb") as stream: np.savez_compressed(stream, **arrays)
        expected = {"fixture": "synthetic artifact gate only", "flux_e": declared, "seed_entropy": [20260930]}
        raw = []
        fill_unexecuted(raw, {"seed": 20260930}, 1, "fixture")
        for row in raw:
            row.update(status="OK", coeff_nm=[0., 0., 0., 0.])
        shards = [{"path": shard_path.name, "sha256": learn._sha(shard_path)}]
        optics = [{"condition": condition_name(CONDITIONS[i]), "design": d, "pass": True, "relative_L1": 0.}
                  for i in (0, 7) for d in study.DESIGNS]
        canary = {"status": "PASS_ENGINE_COUNTERFACTUAL_CANARY", "identity": expected,
            "completed_parent_shards": 1, "planned_estimates": 192, "finite_estimates": 192,
            "raw": raw, "shards": shards, "optical_checks": optics}
        dataset = {"status": canary["status"], "diagnostic_identity": expected, "shards": shards}
        resource = {"exit_code": 0, "supervision_failure": None, "cpu_reserved_for_reporting_seconds": 6000,
                    "helper_source_sha256": learn._sha(__file__)}
        learn._json(output / "report.json", canary)
        # A positive report alone cannot admit a failed/incomplete native invocation.
        try: require_canary(output, expected)
        except FileNotFoundError: pass
        else: raise AssertionError("Positive report without terminal resource/dataset admitted")
        learn._json(parent / "t001-shift-diagnostics-canary-resources.json", resource)
        learn._json(output / "dataset.json", dataset)
        require_canary(output, expected)
        resource["exit_code"] = 1
        (parent / "t001-shift-diagnostics-canary-resources.json").write_text(json.dumps(resource))
        try: require_canary(output, expected)
        except ValueError: pass
        else: raise AssertionError("Failed native exit admitted despite positive report")
        resource["exit_code"] = 0
        (parent / "t001-shift-diagnostics-canary-resources.json").write_text(json.dumps(resource))
        canary["shards"] = [dict(shards[0], path="missing.npz")]
        dataset["shards"] = canary["shards"]
        (output / "report.json").write_text(json.dumps(canary))
        (output / "dataset.json").write_text(json.dumps(dataset))
        try: require_canary(output, expected)
        except FileNotFoundError: pass
        else: raise AssertionError("Missing canary counts admitted")
    print({"status": "PASS_CPU_FACTORIAL_PAIRING_AND_PLANNED_FAILURE_DENOMINATORS", "scope": "synthetic CPU only; no optical/GPU admission"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=3600)
    parser.add_argument("--cpu-before-launch-upper", type=float)
    args = parser.parse_args()
    if args.run is None:
        self_check()
    else:
        print(run_study(args.run, args.count, args.max_seconds, args.cpu_before_launch_upper), flush=True)
