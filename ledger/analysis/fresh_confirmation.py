"""Frozen-choice untouched confirmation; reuse admitted optics, estimators and guards."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

NATIVE_STARTED = time.monotonic()
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from benchmark import core, cli, comparisons, learning as learn
import numpy as np
import torch

import shift_diagnostics as diagnostics
import architecture_evaluate as evaluation
import architecture_study as study

METHODS = diagnostics.METHODS
CONDITIONS = ("nominal", "shifted")
FAMILIES = ("mlp", "compact_cnn", "resnet18")
PROTOCOL = PROJECT / "ledger/decisions/fresh-confirmation-2026-10-02.md"


def parent_spec(config, index, canary=False):
    namespace = 4190356 if canary else 4190357
    seeds = np.random.SeedSequence([config["seed"], namespace, index]).generate_state(4)
    rng = np.random.default_rng(int(seeds[0]))
    return {"parent_id": f"confirmation{'-canary' if canary else ''}-{index:04d}",
            "truth_nm": rng.uniform(-core.LIMITS, core.LIMITS), "centroid_mas": rng.uniform(-20, 20, 2),
            "target_seed": int(seeds[0]), "phase_seed": int(seeds[1]),
            "calibration_seed": int(seeds[2]), "noise_root_seed": int(seeds[3])}


def validate_arrays(arrays, declared):
    diagnostics.blocks._validate_arrays(arrays, 2, np.asarray(declared, dtype=float))
    labels = arrays["labels_nm"].reshape(2, 3, 4)
    if not np.array_equal(labels, np.broadcast_to(labels[:1, :1], labels.shape)):
        raise ValueError("Both counterfactual conditions must share each parent's targets")


def choose(candidates):
    if {r["method"] for r in candidates} != set(FAMILIES) or len(candidates) != 3:
        raise ValueError("Every frozen learned family required before selection")
    if any(not np.isfinite(r["best_validation_standardized_mse"])
           or r["best_validation_standardized_mse"] < 0 or r["parameter_count"] <= 0 for r in candidates):
        raise ValueError("Finite validation loss and positive actual parameter counts required")
    return min(candidates, key=lambda r: (r["best_validation_standardized_mse"],
                                         r["parameter_count"], FAMILIES.index(r["method"])))


def snapshot(run):
    run, baseline, architecture, record, config, diagnostic_identity = diagnostics.identity(run, 20)
    output = run / "shift-diagnostics-01"
    report, dataset = study.read(output / "report.json"), study.read(output / "dataset.json")
    resource = study.read(run / "t001-shift-diagnostics-resources.json")
    if (report["status"] != "COMPLETE" or report["failure"] is not None
            or report["identity"] != diagnostic_identity or report["completed_parent_shards"] != 20
            or report["planned_estimates"] != 3840 or len(report["raw"]) != 3840
            or dataset["status"] != "COMPLETE" or dataset["diagnostic_identity"] != diagnostic_identity
            or dataset["shards"] != report["shards"] or len(report["shards"]) != 20
            or resource["exit_code"] != 0 or resource["supervision_failure"] is not None
            or resource["helper_source_sha256"] != learn._sha(diagnostics.__file__)
            or resource["cpu_reserved_for_reporting_seconds"] != 6000):
        raise ValueError("Complete matching terminal development diagnostics required before confirmation")
    diagnostics.fill_unexecuted(list(report["raw"]), config, 20, "missing prerequisite row")
    for shard in report["shards"]:
        path = (output / shard["path"]).resolve()
        if not path.is_relative_to(output) or learn._sha(path) != shard["sha256"]:
            raise ValueError("Diagnostic count artifact changed")
        with np.load(path, allow_pickle=False) as saved:
            diagnostics.validate_arrays({k: saved[k] for k in saved.files}, config["flux_e"])
    files = (Path(__file__), PROTOCOL, Path(diagnostics.__file__), Path(evaluation.__file__),
             Path(diagnostics.gpu_backend.__file__), Path(diagnostics.gpu_mft.__file__))
    evidence = ("shift-diagnostics-01/report.json", "shift-diagnostics-01/dataset.json",
                "t001-shift-diagnostics-resources.json", "architecture-study-01/evaluation.json",
                "architecture-study-01/oopao-evaluation.json")
    identity = {"schema_version": 1, "sources": {p.resolve().relative_to(PROJECT).as_posix(): learn._sha(p) for p in files},
        "baseline_frozen_sha256": learn._sha(run / "models/frozen.json"),
        "architecture_frozen_sha256": learn._sha(run / "architecture-study-01/frozen.json"),
        "development_evidence_sha256": {name: learn._sha(run / name) for name in evidence},
        "diagnostic_identity": diagnostic_identity, "config_hash": baseline["config_hash"],
        "measurement_hash": baseline["measurement_hash"], "basis_sha256": baseline["basis_sha256"],
        "flux_e": config["flux_e"], "config_seed": config["seed"], "conditions": list(CONDITIONS), "methods": list(METHODS),
        "confirmation_parents": 100, "primary_contrasts": 4, "bootstrap_draws": 9999}
    return run, baseline, architecture, record, config, identity


def check_seeds(run, config):
    existing = set()
    manifest = study.read(run / "dataset.json")
    for shard in manifest["shards"]:
        existing.update(shard[k] for k in ("phase_seed", "calibration_seed", "label_seed", "noise_seed"))
    for row in study.read(run / "challenges.json")["seeds"]:
        existing.update(row[k] for k in ("seed", "phase_seed", "calibration_error_seed"))
    for row in study.read(run / "oopao/atmosphere_statistics.json")["sequences"]:
        existing.update((row["seed"], row["calibration_error_seed"]))
    for index in range(20):
        spec = diagnostics.parent_spec(config, index)
        existing.update(spec[k] for k in ("phase_seed", "calibration_seed", "noise_root_seed"))
        existing.add(int(np.random.SeedSequence([config["seed"], 2910357, index]).generate_state(4)[0]))
    used = set()
    old_ids = {s["parent_id"] for s in manifest["shards"]}
    old_ids.update(s["parent_id"] for s in study.read(run / "oopao/oopao_dataset.json")["shards"])
    old_ids.update(r["parent_id"] for r in study.read(run / "challenges.json")["raw"])
    old_ids.update(diagnostics.parent_spec(config, i)["parent_id"] for i in range(20))
    for canary, count in ((True, 1), (False, 100)):
        for index in range(count):
            spec = parent_spec(config, index, canary)
            seeds = [spec[k] for k in ("target_seed", "phase_seed", "calibration_seed", "noise_root_seed")]
            if spec["parent_id"] in old_ids or len(set(seeds)) != 4 or set(seeds) & (existing | used):
                raise ValueError("Confirmation/canary identity or numerical seed collision")
            used.update(seeds)
    return {"status": "PASS", "existing_unique_seed_values": len(existing), "new_unique_seed_values": len(used)}


def select_choices(baseline, architecture):
    choices = {}
    for design in study.DESIGNS:
        base = baseline["models"][design]
        selected = next(r for r in base["seeds"] if r["seed"] == base["selected_seed"])
        parameters = sum(p.numel() for p in learn._network(256 if design == "single" else 512).parameters())
        candidates = [{**selected, "method": "mlp", "parameter_count": parameters}]
        for family in study.models.ARCHITECTURES:
            spec = architecture["models"][family][design]
            candidates.append({**next(r for r in spec["seeds"] if r["seed"] == spec["selected_seed"]), "method": family})
        winner = choose(candidates)
        choices[design] = {k: winner[k] for k in ("method", "seed", "checkpoint", "parameter_count",
                                                 "best_validation_standardized_mse", "calibration_half_width_nm")}
    return choices


def freeze(run, max_seconds, cpu_before_launch):
    run, baseline, architecture, _, config, identity = snapshot(run)
    guard, _ = study.limits(run, max_seconds, cpu_before_launch, run / "confirmation-stop-request.json")
    guard()
    if any((run / name).exists() for name in ("confirmation-01", "confirmation-canary-01", "confirmation-physical-01")):
        raise FileExistsError("Cannot choose a method after confirmation data exist")
    choices = select_choices(baseline, architecture)
    result = {"status": "FROZEN_BEFORE_CONFIRMATION", "identity": identity, "choices": choices,
              "seed_disjointness": check_seeds(run, config), "inference_guard": "unchanged finite image/estimate guards",
              "selection_rule": "minimum frozen parent-weighted validation MSE; tie fewer parameters then fixed family order"}
    guard()
    learn._json(run / "fresh-confirmation-freeze.json", result)
    return {"status": result["status"], "choices": {d: c["method"] for d, c in choices.items()}}


def frozen(run):
    run, baseline, architecture, record, config, identity = snapshot(run)
    decision = study.read(run / "fresh-confirmation-freeze.json")
    if (decision["status"] != "FROZEN_BEFORE_CONFIRMATION" or decision["identity"] != identity
            or decision["choices"] != select_choices(baseline, architecture)):
        raise ValueError("Freeze choices/calibration/source identities before untouched confirmation")
    return run, baseline, architecture, record, config, decision


def row_key(row):
    return row["parent_id"], row["split"], row["design"], row["method"], row["acquisition_index"]


def fill_unexecuted(rows, config, count, reason):
    if count not in (1, 100):
        raise ValueError("One independent canary or one hundred confirmation parents required")
    seen = {row_key(r) for r in rows}
    for index in range(count):
        spec = parent_spec(config, index, count == 1)
        for condition in CONDITIONS:
            for design in study.DESIGNS:
                for method in METHODS:
                    for flux_index, flux in enumerate(config["flux_e"]):
                        key = (spec["parent_id"], condition, design, method, flux_index)
                        if key not in seen:
                            row = comparisons._row(key[0], condition, design, spec["truth_nm"], method,
                                {"status": "NOT_EXECUTED", "coeff_nm": None, "latency_ms": 0., "reason": reason}, flux_index)
                            rows.append({**row, "flux_e": flux})
    if len(rows) != count * 48 or len({row_key(r) for r in rows}) != count * 48:
        raise ValueError("Every planned confirmation row required exactly once")
    for row in rows:
        index = int(row["parent_id"].rsplit("-", 1)[1])
        spec = parent_spec(config, index, count == 1)
        if (not np.array_equal(row["truth_nm"], spec["truth_nm"])
                or row["flux_e"] != config["flux_e"][row["acquisition_index"]]):
            raise ValueError("Confirmation target or declared flux identity changed")
        if row["coeff_nm"] is None:
            if row["error_nm"] is not None:
                raise ValueError("Absent predictions must retain their declared failure penalty")
        else:
            prediction = np.asarray(row["coeff_nm"])
            if (prediction.shape != (4,) or not np.isfinite(prediction).all()
                    or not np.array_equal(row["error_nm"], prediction - spec["truth_nm"])):
                raise ValueError("Invalid prediction or error identity")


def primary_report(rows, choices, intervals=True):
    lookup = {row_key(r): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate inference observations cannot enter primary contrasts")
    parents = sorted({r["parent_id"] for r in rows})
    columns = [(condition, design) for condition in CONDITIONS for design in study.DESIGNS]
    def score(row):
        error = np.asarray(row["error_nm"] if row["error_nm"] is not None else 2 * core.LIMITS)
        if error.shape != (4,) or not np.isfinite(error).all():
            raise ValueError("Finite four-mode errors or declared failure penalty required")
        return np.square(error / core.LIMITS).mean()
    values = np.asarray([[np.mean([
        score(lookup[(parent, condition, design, choices[design]["method"], f)])
        - score(lookup[(parent, condition, design, "mlp", f)]) for f in range(3)])
        for condition, design in columns] for parent in parents])
    if values.shape != (len(parents), 4) or not len(parents) or not np.isfinite(values).all():
        raise ValueError("Every paired parent/four primary contrasts required")
    result = {"order": [f"{condition}/{design}" for condition, design in columns],
              "selected_family_minus_mlp_standardized_penalized_mse": values.mean(axis=0).tolist(),
              "selected_families": {d: choices[d]["method"] for d in study.DESIGNS},
              "n_paired_parents": len(parents), "fixed_primary_contrasts": 4,
              "weighting": "equal parents; equal three fluxes; paired across conditions and designs",
              "scope": "approximate bootstrap conditional on fixed models/calibration; no exact finite-sample guarantee"}
    if intervals and len(parents) > 1:
        rng = np.random.default_rng(7341982)
        samples = values[rng.integers(0, len(parents), (9999, len(parents)))].mean(axis=1)
        result.update(ci98_75_bonferroni_parent_percentile=np.quantile(samples, [.00625, .99375], axis=0).tolist(),
                      bootstrap_draws=9999, nominal_error_allocation=.05 / 4)
    return result


def calibration_report(rows, baseline, architecture, config):
    report = {}
    for condition in CONDITIONS:
        for design in study.DESIGNS:
            for method in METHODS:
                cell = sorted((r for r in rows if (r["split"], r["design"], r["method"]) ==
                               (condition, design, method)), key=lambda r: (r["parent_id"], r["acquisition_index"]))
                spec = baseline["models"][design] if method in ("mlp", "ridge") else architecture["models"][method][design]
                selected = spec["ridge"] if method == "ridge" else next(s for s in spec["seeds"] if s["seed"] == spec["selected_seed"])
                prediction = np.asarray([r["coeff_nm"] if r["coeff_nm"] is not None else [np.nan] * 4 for r in cell])
                data = {"labels": np.asarray([r["truth_nm"] for r in cell]), "parents": np.asarray([r["parent_id"] for r in cell])}
                measured, _ = evaluation.metrics(prediction, data, np.asarray([r["flux_e"] for r in cell]),
                                                 np.asarray(config["flux_e"]), selected["calibration_half_width_nm"])
                report[f"{condition}/{design}/{method}"] = measured
    return report


def run_confirmation(run, count, max_seconds, cpu_before_launch):
    if count not in (1, 100):
        raise ValueError("Require one independent canary or one hundred untouched parents")
    run, baseline, architecture, record, config, decision = frozen(run)
    identity = {**decision["identity"], "choice_freeze_sha256": learn._sha(run / "fresh-confirmation-freeze.json"),
                "parent_count": count, "seed_namespace": 4190356 if count == 1 else 4190357}
    output = run / ("confirmation-canary-01" if count == 1 else "confirmation-01")
    if output.exists():
        raise FileExistsError("Preserve earlier confirmation artifacts; no blind overwrite/restart")
    if count == 100:
        require_output(run / "confirmation-canary-01", dict(identity, parent_count=1, seed_namespace=4190356),
                       1, "t001-confirmation-canary-resources.json")
    resource_guard, storage_guard = study.limits(run, max_seconds, cpu_before_launch, run / "confirmation-stop-request.json")
    def guard():
        if time.monotonic() - NATIVE_STARTED >= max_seconds:
            raise TimeoutError("Confirmation deadline includes all imports/admission")
        resource_guard()
    guard()
    output.mkdir()
    learn._json(output / "protocol.json", identity)
    rows, shards, optics, failure = [], [], [], None
    with study.owner(output):
        try:
            torch.set_num_threads(1)
            device = learn.execution_device(baseline["device"])
            selected = diagnostics.predictors(run, baseline, architecture, device)
            cp = diagnostics.gpu_mft.cuda_backend(output / "cupy-cache")
            model = core.OpticalModel(config, basis=baseline["basis"])
            guard()
            diagnostics.gpu_backend.attach_device_phase(model, cp, 64)
            reference = core.OpticalModel(config, basis=baseline["basis"]) if count == 1 else None
            for index in range(count):
                guard()
                spec = parent_spec(config, index, count == 1)
                calibration = model.calibration_error(spec["calibration_seed"])
                phases = {condition: core.ResidualSequence(model, spec["phase_seed"],
                    record["residual_gain"]["nominal_per_nm" if condition == "nominal" else "shifted_per_nm"],
                    shifted=condition == "shifted").acquisition(0) for condition in CONDITIONS}
                stored = {k: [] for k in ("images_single", "images_pair", "labels_nm", "flux_e")}
                for condition_index, condition in enumerate(CONDITIONS):
                    guard()
                    amplitude = (200. if count == 1 else 100. if index % 2 == 0 else 200.) if condition == "nominal" else 300.
                    diversity = 1. if condition == "nominal" else 1.1
                    residual = phases[condition] * amplitude + calibration[None]
                    probabilities = model.probabilities(spec["truth_nm"], residual, spec["centroid_mas"], diversity)
                    if reference is not None:
                        native = reference.probabilities(spec["truth_nm"], residual, spec["centroid_mas"], diversity)
                        for design, actual, expected in zip(study.DESIGNS, probabilities, native):
                            relative = float(np.abs(actual - expected).sum() / expected.sum())
                            optics.append({"condition": condition, "design": design, "relative_L1": relative,
                                           "pass": np.isfinite(relative) and 0 <= relative < 1e-3})
                            if not optics[-1]["pass"]:
                                raise ValueError("Confirmation CPU/GPU propagation disagreement")
                    noise = np.random.default_rng(np.random.SeedSequence([spec["noise_root_seed"], condition_index]))
                    for flux_index, flux in enumerate(config["flux_e"]):
                        images = {"single": core.detector_read(probabilities[0] * flux, noise,
                            config["background_e"] + config["dark_e"], config["read_noise_e"]),
                            "pair": core.detector_read(probabilities[1] * flux / 2, noise,
                            (config["background_e"] + config["dark_e"]) / 2, config["read_noise_e"])}
                        images = {d: image.astype("float32") for d, image in images.items()}
                        for design in study.DESIGNS:
                            stored[f"images_{design}"].append(images[design])
                            for method in METHODS:
                                estimate = comparisons._invoke(selected[design][method], images[design], design,
                                    torch.device("cpu") if method == "ridge" else device, guard)
                                row = comparisons._row(spec["parent_id"], condition, design, spec["truth_nm"], method, estimate, flux_index)
                                rows.append({**row, "flux_e": flux})
                        stored["labels_nm"].append(spec["truth_nm"])
                        stored["flux_e"].append(flux)
                arrays = {k: np.asarray(v, dtype="float32" if k.startswith("images_") else "float64") for k, v in stored.items()}
                validate_arrays(arrays, config["flux_e"])
                storage_guard(sum(a.nbytes for a in arrays.values()) + 1024**2)
                path = output / f"parent_{index:04d}.npz"
                with path.open("xb") as stream: np.savez_compressed(stream, **arrays)
                shards.append({"path": path.name, "sha256": learn._sha(path), "parent_id": spec["parent_id"], "rows": 6,
                    "target_seed": spec["target_seed"], "phase_seed": spec["phase_seed"], "calibration_seed": spec["calibration_seed"],
                    "noise_root_seed": spec["noise_root_seed"], "condition_order": list(CONDITIONS),
                    "nominal_amplitude_nm": 200. if count == 1 else 100. if index % 2 == 0 else 200.})
                del phases, residual
                print("untouched confirmation parent complete", index + 1, "of", count, flush=True)
            guard()
            if frozen(run)[5] != decision:
                raise ValueError("Frozen choices/calibration/source changed during confirmation")
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            fill_unexecuted(rows, config, count, failure or "missing execution row")
            finite = sum(r["coeff_nm"] is not None for r in rows)
            status = "COMPLETE" if failure is None and len(shards) == count else "PARTIAL"
            if count == 1 and status == "COMPLETE" and finite == 48 and len(optics) == 4 and all(x["pass"] for x in optics):
                status = "PASS_UNTOUCHED_CONFIRMATION_CANARY"
            report = {"status": status, "failure": failure, "identity": identity, "choices": decision["choices"],
                "completed_parent_shards": len(shards), "planned_estimates": count * 48, "finite_estimates": finite,
                "shards": shards, "optical_checks": optics, "physical_condition_acquisitions_planned": count * 2,
                "flux_rows_per_design_planned": count * 6, "primary_contrasts": primary_report(rows, decision["choices"], count > 1),
                "calibration_and_flux": calibration_report(rows, baseline, architecture, config), **comparisons._report(rows)}
            if count == 1:
                report = diagnostics.point_only(report)
            core.save_json(output / "report.json", report)
            core.save_json(output / "dataset.json", {"status": status, "confirmation_identity": identity,
                "source_hash": baseline["source_hash"], "config_hash": baseline["config_hash"],
                "measurement_hash": baseline["measurement_hash"], "basis": baseline["basis"], "shards": shards})
    if failure:
        raise RuntimeError(failure)
    return {"status": status, "output": str(output), "planned_estimates": count * 48}


def require_output(output, identity, count, resource_name):
    report, dataset = study.read(output / "report.json"), study.read(output / "dataset.json")
    native = study.read(output.parent / resource_name)
    expected_status = "PASS_UNTOUCHED_CONFIRMATION_CANARY" if count == 1 else "COMPLETE"
    if (report["status"] != expected_status or report["failure"] is not None or report["identity"] != identity
            or report["completed_parent_shards"] != count or report["planned_estimates"] != count * 48
            or len(report["raw"]) != count * 48 or len(report["shards"]) != count
            or native["exit_code"] != 0 or native["supervision_failure"] is not None
            or native["helper_source_sha256"] != learn._sha(__file__) or native["cpu_reserved_for_reporting_seconds"] != 6000
            or dataset["status"] != expected_status or dataset["confirmation_identity"] != identity
            or dataset["shards"] != report["shards"]):
        raise ValueError("Matching terminal native confirmation report/dataset/artifacts required")
    fill_unexecuted(list(report["raw"]), {"seed": identity["config_seed"], "flux_e": identity["flux_e"]}, count, "missing admission row")
    if count == 1:
        optics = report["optical_checks"]
        if (report["finite_estimates"] != 48 or any(r["status"] != "OK" for r in report["raw"])
                or {(r["condition"], r["design"]) for r in optics} != {(c, d) for c in CONDITIONS for d in study.DESIGNS}
                or len(optics) != 4 or any(not r["pass"] or not np.isfinite(r["relative_L1"])
                                         or not 0 <= r["relative_L1"] < 1e-3 for r in optics)):
            raise ValueError("Complete finite canary and all four optical checks required")
    root = output.resolve()
    expected_parents = {parent_spec({"seed": identity["config_seed"]}, i, count == 1)["parent_id"] for i in range(count)}
    if {s["parent_id"] for s in report["shards"]} != expected_parents or len({s["path"] for s in report["shards"]}) != count:
        raise ValueError("Every planned independent count shard required exactly once")
    for shard in report["shards"]:
        path = (root / shard["path"]).resolve()
        if not path.is_relative_to(root) or learn._sha(path) != shard["sha256"]:
            raise ValueError("Confirmation shard changed or outside its root")
        with np.load(path, allow_pickle=False) as saved:
            validate_arrays({k: saved[k] for k in saved.files}, identity["flux_e"])
            spec = parent_spec({"seed": identity["config_seed"]}, int(shard["parent_id"].rsplit("-", 1)[1]), count == 1)
            if not np.array_equal(saved["labels_nm"], np.broadcast_to(spec["truth_nm"], (6, 4))):
                raise ValueError("Count shard targets differ from the registered parent")
    return report


def physical_contrasts(rows, choices):
    result = {}
    for condition in CONDITIONS:
        for design in study.DESIGNS:
            parents = sorted({r["parent_id"] for r in rows if r["split"] == condition and r["design"] == design})
            lookup = {(r["parent_id"], r["method"]): r for r in rows if r["split"] == condition and r["design"] == design}
            def grouped(method):
                values = []
                for parent in parents:
                    row = lookup[(parent, method)]
                    error = np.asarray(row["error_nm"] if row["error_nm"] is not None else 2 * core.LIMITS)
                    values.append(np.r_[error**2, np.square(error / core.LIMITS).mean(), row["coeff_nm"] is None])
                return np.asarray(values)
            for method, initializer in (("mlp_two_steps", "mlp"), ("selected_two_steps", choices[design]["method"]),
                                        ("ridge_two_steps", "ridge")):
                contrast = evaluation.paired_scores(grouped(method), grouped(initializer))
                contrast.update(direction="two-step estimate minus matching frozen initializer; negative favors update",
                                weighting="equal paired parents; fixed10000-electron observations",
                                scope="secondary exploratory95% intervals conditional on fixed models; no multiplicity guarantee")
                result[f"{condition}/{design}/{method}"] = contrast
    return result


def run_physical(run, count, max_seconds, cpu_before_launch):
    if count not in (1, 20):
        raise ValueError("One physical canary or the fixed twenty-parent subset required")
    run, baseline, architecture, _, config, decision = frozen(run)
    identity = {**decision["identity"], "choice_freeze_sha256": learn._sha(run / "fresh-confirmation-freeze.json"),
                "parent_count": 1 if count == 1 else 100, "seed_namespace": 4190356 if count == 1 else 4190357}
    input_name = "confirmation-canary-01" if count == 1 else "confirmation-01"
    resource_name = "t001-confirmation-canary-resources.json" if count == 1 else "t001-confirmation-resources.json"
    confirmed = require_output(run / input_name, identity, 1 if count == 1 else 100, resource_name)
    if count == 20:
        canary = study.read(run / "confirmation-physical-canary-01/report.json")
        native = study.read(run / "t001-confirmation-physical-canary-resources.json")
        if (canary["status"] != "PASS_PHYSICAL_CONFIRMATION_CANARY" or canary["failure"] is not None
                or canary["identity"] != dict(identity, parent_count=1, seed_namespace=4190356)
                or canary["parent_count"] != 1 or canary["planned_observations"] != 28
                or len(canary["raw"]) != 28 or any(r["status"] != "OK" or r["coeff_nm"] is None
                    or np.asarray(r["coeff_nm"]).shape != (4,) or not np.isfinite(r["coeff_nm"]).all() for r in canary["raw"])
                or native["exit_code"] != 0 or native["supervision_failure"] is not None
                or native["helper_source_sha256"] != learn._sha(__file__) or native["cpu_reserved_for_reporting_seconds"] != 6000):
            raise ValueError("Complete matching actual physical canary required")
    output = run / ("confirmation-physical-canary-01" if count == 1 else "confirmation-physical-01")
    if output.exists():
        raise FileExistsError("Preserve earlier physical confirmation evidence")
    resource_guard, _ = study.limits(run, max_seconds, cpu_before_launch, run / "confirmation-stop-request.json")
    def guard():
        if time.monotonic() - NATIVE_STARTED >= max_seconds:
            raise TimeoutError("Physical confirmation deadline includes imports/admission")
        resource_guard()
    guard()
    output.mkdir()
    choices = decision["choices"]
    planned = {parent_spec(config, i, count == 1)["parent_id"] for i in range(count)}
    rows = [{**r, "latency_provenance": "cached completed direct confirmation inference"}
            for r in confirmed["raw"] if r["parent_id"] in planned and r["acquisition_index"] == 1]
    if len(rows) != count * 16:
        raise ValueError("All four direct methods on the fixed physical subset required")
    direct = {row_key(r): r for r in rows}
    failure = None
    with study.owner(output):
        try:
            torch.set_num_threads(1)
            device = learn.execution_device(baseline["device"])
            predictors = diagnostics.predictors(run, baseline, architecture, device)
            model = comparisons._protect(core.OpticalModel(config, basis=baseline["basis"]), guard)
            if model.measurement_hash != baseline["measurement_hash"]:
                raise ValueError("Nominal physical calibration changed")
            for shard in confirmed["shards"]:
                if shard["parent_id"] not in planned:
                    continue
                path = run / input_name / shard["path"]
                if learn._sha(path) != shard["sha256"]:
                    raise ValueError("Physical input count artifact changed")
                with np.load(path, allow_pickle=False) as saved:
                    for condition_index, condition in enumerate(CONDITIONS):
                        for design in study.DESIGNS:
                            guard()
                            image = saved[f"images_{design}"][condition_index * 3 + 1]
                            truth = saved["labels_nm"][condition_index * 3 + 1]
                            estimates = {}
                            for method, initializer in (("mlp_two_steps", "mlp"), ("selected_two_steps", choices[design]["method"]),
                                                        ("ridge_two_steps", "ridge")):
                                if method == "selected_two_steps" and initializer == "mlp":
                                    estimate = {**estimates["mlp_two_steps"], "alias_of": "mlp_two_steps",
                                                "latency_provenance": "same actual call; no repeated calculation"}
                                else:
                                    predict = predictors[design][initializer]
                                    function = lambda image, p=predict, d=design: core.linearized_fit(model, image, d,
                                        initial=p(image), iterations=2, exact_iterations=True)
                                    estimate = comparisons._invoke(function, image, design,
                                        torch.device("cpu") if initializer == "ridge" else device, guard)
                                    estimate["latency_provenance"] = "actual initializer plus exactly two nominal analytical updates"
                                estimates[method] = estimate
                                row = comparisons._row(shard["parent_id"], condition, design, truth, method, estimate, 1)
                                rows.append({**row, "flux_e": 10000.})
                print("physical confirmation parent complete", shard["parent_id"], flush=True)
            guard()
            if frozen(run)[5] != decision:
                raise ValueError("Frozen choices/source changed during physical confirmation")
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            seen = {row_key(r) for r in rows}
            for parent in sorted(planned):
                for condition in CONDITIONS:
                    for design in study.DESIGNS:
                        base = direct[(parent, condition, design, "mlp", 1)]
                        for method in ("mlp_two_steps", "selected_two_steps", "ridge_two_steps"):
                            if (parent, condition, design, method, 1) not in seen:
                                row = comparisons._row(parent, condition, design, base["truth_nm"], method,
                                    {"status": "NOT_EXECUTED", "coeff_nm": None, "latency_ms": 0., "reason": failure or "missing call"}, 1)
                                rows.append({**row, "flux_e": 10000.})
            if len(rows) != count * 28 or len({row_key(r) for r in rows}) != count * 28:
                raise ValueError("Every physical planned row must remain in the denominator")
            status = "COMPLETE" if failure is None else "PARTIAL"
            if count == 1 and status == "COMPLETE" and all(r["status"] == "OK" and r["coeff_nm"] is not None for r in rows):
                status = "PASS_PHYSICAL_CONFIRMATION_CANARY"
            result = {"status": status, "failure": failure, "identity": identity,
                "confirmation_report_sha256": learn._sha(run / input_name / "report.json"),
                "parent_count": count, "planned_observations": count * 28, "physical_update_columns": count * 12,
                "physical_calls_executed": sum(r["method"].endswith("two_steps") and "alias_of" not in r
                                                and r["status"] != "NOT_EXECUTED" for r in rows),
                "choices": choices, "calibration": "nominal static PSF only; fitted centroid/flux/background; no residual or unknown map",
                "latency_scope": "direct calls cached from confirmation; physical calls include initializer and exactly two updates; aliases explicit",
                "paired_against_own_initializer": physical_contrasts(rows, choices), **comparisons._report(rows)}
            if count == 1:
                result = diagnostics.point_only(result)
            core.save_json(output / "report.json", result)
    if failure:
        raise RuntimeError(failure)
    return {"status": status, "output": str(output), "planned_observations": count * 28}


def self_check():
    import tempfile
    from unittest.mock import patch
    config = {"seed": 42, "flux_e": [1000., 10000., 100000.]}
    candidates = [{"method": name, "best_validation_standardized_mse": .5, "parameter_count": count}
                  for name, count in zip(FAMILIES, (9000, 40000, 11000000))]
    assert choose(candidates)["method"] == "mlp"
    candidates[2]["best_validation_standardized_mse"] = .4
    assert choose(candidates)["method"] == "resnet18"
    candidates[2]["best_validation_standardized_mse"] = float("nan")
    try: choose(candidates)
    except ValueError: pass
    else: raise AssertionError("Nonfinite validation selection admitted")
    rows = []
    fill_unexecuted(rows, config, 100, "synthetic fixture")
    assert len(rows) == 4800 and all(r["status"] == "NOT_EXECUTED" for r in rows)
    fill_unexecuted(rows, config, 100, "idempotent")
    assert len(rows) == 4800
    factors = {"mlp": 1., "ridge": np.sqrt(2.), "compact_cnn": .5, "resnet18": np.sqrt(.5)}
    for row in rows:
        prediction = np.asarray(row["truth_nm"]) + core.LIMITS * factors[row["method"]]
        row.update(coeff_nm=prediction.tolist(), error_nm=(prediction - row["truth_nm"]).tolist(), status="OK")
    choices = {d: {"method": "compact_cnn"} for d in study.DESIGNS}
    primary = primary_report(rows, choices)
    np.testing.assert_allclose(primary["selected_family_minus_mlp_standardized_penalized_mse"], [-.75] * 4)
    np.testing.assert_allclose(primary["ci98_75_bonferroni_parent_percentile"], [[-.75] * 4] * 2)
    missing = next(r for r in rows if r["parent_id"] == "confirmation-0000" and r["split"] == "nominal"
                   and r["design"] == "single" and r["method"] == "compact_cnn" and r["acquisition_index"] == 0)
    missing.update(coeff_nm=None, error_nm=None, status="NO_ESTIMATE")
    np.testing.assert_allclose(primary_report(rows, choices, False)["selected_family_minus_mlp_standardized_penalized_mse"],
                               [-.7375, -.75, -.75, -.75])
    assert parent_spec(config, 0)["phase_seed"] != parent_spec(config, 0, True)["phase_seed"]
    canary_rows = []
    fill_unexecuted(canary_rows, config, 1, "synthetic artifact fixture")
    for row in canary_rows:
        row.update(coeff_nm=[0.] * 4, error_nm=(-np.asarray(row["truth_nm"])).tolist(), status="OK")
    assembled = diagnostics.point_only({"primary": primary_report(canary_rows, choices, False),
                                        **comparisons._report(canary_rows)})
    def assert_no_intervals(value):
        if isinstance(value, dict):
            assert not any(k.startswith("ci95") or k.startswith("ci98") or k == "ci_order" for k in value)
            for v in value.values(): assert_no_intervals(v)
        elif isinstance(value, list):
            for v in value: assert_no_intervals(v)
    assert_no_intervals(assembled)
    # Synthetic artifact gate; these mocked records are not native or optical evidence.
    with tempfile.TemporaryDirectory(dir=PROJECT / "benchmark/runs") as temporary:
        root = Path(temporary)
        out = root / "confirmation-canary-01"
        out.mkdir()
        truth = parent_spec(config, 0, True)["truth_nm"]
        arrays = {"images_single": np.zeros((6, 1, 16, 16), dtype="float32"),
                  "images_pair": np.zeros((6, 2, 16, 16), dtype="float32"),
                  "labels_nm": np.broadcast_to(truth, (6, 4)).copy(), "flux_e": np.tile(config["flux_e"], 2)}
        validate_arrays(arrays, config["flux_e"])
        shard = out / "parent_0000.npz"
        with shard.open("xb") as stream: np.savez_compressed(stream, **arrays)
        identity = {"config_seed": 42, "flux_e": config["flux_e"]}
        shards = [{"path": shard.name, "sha256": learn._sha(shard), "parent_id": "confirmation-canary-0000"}]
        report = {"status": "PASS_UNTOUCHED_CONFIRMATION_CANARY", "failure": None, "identity": identity,
                  "completed_parent_shards": 1, "planned_estimates": 48, "finite_estimates": 48,
                  "raw": canary_rows, "shards": shards,
                  "optical_checks": [{"condition": c, "design": d, "relative_L1": 0., "pass": True}
                                     for c in CONDITIONS for d in study.DESIGNS]}
        learn._json(out / "report.json", report)
        learn._json(out / "dataset.json", {"status": report["status"], "confirmation_identity": identity, "shards": shards})
        native = {"exit_code": 0, "supervision_failure": None, "helper_source_sha256": learn._sha(__file__),
                  "cpu_reserved_for_reporting_seconds": 6000}
        native_path = root / "native-fixture.json"
        learn._json(native_path, native)
        require_output(out, identity, 1, native_path.name)
        import json
        native["exit_code"] = 1
        native_path.write_text(json.dumps(native))
        try: require_output(out, identity, 1, native_path.name)
        except ValueError: pass
        else: raise AssertionError("Positive canary report admitted after failed native exit")
        learn._json(root / "fresh-confirmation-freeze.json", {"fixture": "no scientific choice"})
        decision = {"identity": identity, "choices": choices}
        module = sys.modules[__name__]
        with patch.object(module, "frozen", return_value=(root, {"device": "cpu", "basis": None}, {}, {}, config, decision)), \
             patch.object(module, "require_output", return_value=report), \
             patch.object(study, "limits", return_value=(lambda: None, lambda size: None)), \
             patch.object(diagnostics, "predictors", return_value={}), \
             patch.object(core, "OpticalModel", side_effect=RuntimeError("deliberate CPU fixture; no optics")):
            try: run_physical(root, 1, 600, 0)
            except RuntimeError as exc: assert "deliberate CPU fixture" in str(exc)
            else: raise AssertionError("Physical fixture failed to exercise partial-finalization branch")
        stopped = study.read(root / "confirmation-physical-canary-01/report.json")
        assert stopped["status"] == "PARTIAL" and stopped["planned_observations"] == len(stopped["raw"]) == 28
        assert sum(r["status"] == "NOT_EXECUTED" for r in stopped["raw"]) == 12
        assert_no_intervals(stopped)
    print({"status": "PASS_CPU_CONFIRMATION_SELECTION_PAIRING_AND_FAILURE_DENOMINATORS",
           "scope": "synthetic CPU/artifact fixtures only; no confirmation images, GPU or optical admission"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--action", choices=("freeze", "generate", "physical"), default="generate")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--max-seconds", type=float, default=3600)
    parser.add_argument("--cpu-before-launch-upper", type=float)
    args = parser.parse_args()
    if args.run is None:
        self_check()
    elif args.action == "freeze":
        print(freeze(args.run, args.max_seconds, args.cpu_before_launch_upper), flush=True)
    elif args.action == "physical":
        print(run_physical(args.run, args.count, args.max_seconds, args.cpu_before_launch_upper), flush=True)
    else:
        print(run_confirmation(args.run, args.count, args.max_seconds, args.cpu_before_launch_upper), flush=True)
