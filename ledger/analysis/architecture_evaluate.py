"""Evaluate sealed architecture controls with parent-paired, equal-flux comparisons.

The original held-out data are development evidence for this extension. No fitting
or threshold selection occurs here. A fresh confirmation manifest comes only after
the development choices are frozen.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

import architecture_study as study
from benchmark import learning as learn

sys.path.insert(0, str(Path(__file__).with_name("expanded_campaign")))
import flux_report


def grouped_scores(prediction, truth, parents, flux, declared):
    """Every planned row counts; missing estimates incur the registered 2*limit penalty."""
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    parents, flux, declared = np.asarray(parents), np.asarray(flux), np.asarray(declared)
    if (prediction.shape != truth.shape or truth.shape != (len(parents), 4)
            or not len(parents) or flux.shape != (len(parents),) or declared.shape != (3,)
            or len(set(declared)) != 3 or not np.isfinite(truth).all()
            or not np.isfinite(declared).all() or np.any(declared <= 0)
            or not np.isin(flux, declared).all()):
        raise ValueError("Invalid planned prediction/truth/parent/flux alignment")
    finite = np.isfinite(prediction).all(axis=1)
    error = np.where(finite[:, None], prediction - truth, 2 * learn.LABEL_SCALE)
    rows = []
    for parent in np.unique(parents):
        groups = []
        for value in declared:
            mask = (parents == parent) & (flux == value)
            if not mask.any():
                raise ValueError("Every parent must cover every declared flux")
            groups.append(np.r_[np.square(error[mask]).mean(axis=0),
                                np.square(error[mask] / learn.LABEL_SCALE).mean(),
                                np.mean(~finite[mask])])
        rows.append(np.mean(groups, axis=0))
    return np.asarray(rows)


def paired_scores(candidate, baseline):
    if candidate.shape != baseline.shape or candidate.ndim != 2 or candidate.shape[1] != 6:
        raise ValueError("Paired parent scores disagree")
    delta = candidate[:, 4:5] - baseline[:, 4:5]
    joint = np.column_stack([candidate[:, :4].sum(axis=1), baseline[:, :4].sum(axis=1)])
    transform = lambda m: np.sqrt(m[0]) - np.sqrt(m[1])
    return {"standardized_penalized_mse_difference": float(delta.mean()),
            "ci95_standardized_penalized_mse_difference": learn._bootstrap_interval(delta, lambda m: m),
            "penalized_wavefront_rmse_difference_nm": float(transform(joint.mean(axis=0))),
            "ci95_penalized_wavefront_rmse_difference_nm": learn._bootstrap_interval(joint, transform),
            "n_paired_parents": len(joint), "direction": "candidate minus frozen MLP; negative favors candidate",
            "weighting": "equal parents; equal one-third flux weights within each parent",
            "scope": "conditional on fixed selected models; no training-seed or calibration-sample uncertainty"}


def metrics(prediction, data, flux, declared, width):
    scores = grouped_scores(prediction, data["labels"], data["parents"], flux, declared)
    finite = np.isfinite(prediction).all(axis=1)
    result = {"planned_rows": len(prediction), "finite_estimates": int(finite.sum()),
              "n_parents": len(scores), "equal_third_no_estimate_fraction": float(scores[:, 5].mean()),
              "equal_third_penalized_wavefront_rmse_nm": float(np.sqrt(scores[:, :4].mean(axis=0).sum())),
              "failure_policy": "all planned rows retained; nonfinite estimate penalty=2*label_limit per mode",
              "empirical_mixture": None, "flux_standardized": None}
    if finite.all():
        result["empirical_mixture"] = learn._metrics(prediction, data["labels"], data["parents"], width)
        result["flux_standardized"] = flux_report.flux_metrics(prediction, data["labels"], data["parents"],
                                                               flux, declared, width)
    else:
        result["conditional_metric_note"] = "Conditional error/coverage omitted because predictions include failures"
    return result, scores


def frozen_study(run, protocol):
    output = run / "architecture-study-01"
    frozen = study.read(output / "frozen.json")
    if frozen["status"] != "COMPLETE" or frozen["identity"] != protocol:
        raise ValueError("Complete source/data/runtime-matched architecture freeze required before held-out access")
    reports = []
    for architecture in study.models.ARCHITECTURES:
        for design in study.DESIGNS:
            reports.extend(frozen["models"][architecture][design]["seeds"])
    if study.select(reports) != frozen["models"]:
        raise ValueError("Validation-only selection or complete seed set changed")
    if (set(frozen["artifacts"]) != {r["checkpoint"] for r in reports}
            or set(frozen["artifacts"]) != {p.name for p in output.glob("*.pt")}
            or learn._sha(output / "admission.json") != frozen["admission_sha256"]):
        raise ValueError("Architecture artifact/admission inventory changed")
    for report in reports:
        key = Path(report["checkpoint"]).stem
        sealed = study.completed(output, key, protocol)
        width = np.asarray(report["calibration_half_width_nm"])
        if (sealed != report or sealed["checkpoint_sha256"] != frozen["artifacts"][report["checkpoint"]]
                or width.shape != (4,) or not np.isfinite(width).all() or np.any(width < 0)):
            raise ValueError("Architecture checkpoint or calibration width changed")
    return output, frozen


def evaluate(run, manifest_path, max_seconds, cpu_before_launch):
    run, _, baseline, protocol = study.identity(run)
    output, frozen = frozen_study(run, protocol)
    guard, _ = study.limits(run, max_seconds, cpu_before_launch, output / "stop-request.json")
    run, checked_baseline, path, manifest, semantics, declared, baseline_sha = flux_report._guard(run, manifest_path)
    assert baseline == checked_baseline
    torch.set_num_threads(1)
    device = learn.execution_device(baseline["device"])
    frozen_sha = learn._sha(output / "frozen.json")
    result = {"status": "COMPLETE", "scope": "post-baseline architecture development evaluation",
              "architecture_frozen_sha256": frozen_sha, "baseline_frozen_sha256": baseline_sha,
              "evaluation_manifest_sha256": learn._sha(path), "reporter_sha256": learn._sha(__file__),
              "flux_reporter_sha256": learn._sha(flux_report.__file__), "identity": protocol,
              "declared_flux_e": declared.tolist(), "splits": {}}
    for split, (counts, kinds) in semantics.items():
        guard()
        data = learn._load_split(path, manifest, split)
        flux = flux_report._load_flux(path, manifest, split, data, declared, kinds)
        designs = {}
        for design in study.DESIGNS:
            guard()
            methods, scores = {}, {}
            spec = baseline["models"][design]
            selected = next(r for r in spec["seeds"] if r["seed"] == spec["selected_seed"])
            model, preprocess = learn._load_checkpoint(run / "models" / selected["checkpoint"], device)
            predict = lambda images: learn._predict_network(model, images, preprocess, device)
            prediction = predict(data[design])
            methods["mlp"], scores["mlp"] = metrics(prediction, data, flux, declared,
                                                       selected["calibration_half_width_nm"])
            if split == next(iter(semantics)):
                methods["mlp"]["latency"] = learn._timing(predict, data[design][:1], device, 1000)
            del predict, model
            with np.load(run / "models" / spec["ridge"]["checkpoint"], allow_pickle=False) as ridge:
                preprocess = {"mean": ridge["feature_mean"], "std": ridge["feature_std"],
                              "asinh_scale_e": float(ridge["asinh_scale_e"])}
                weights, intercept = ridge["weights"], ridge["intercept"]
            predict = lambda images: (learn._features(images, preprocess) @ weights + intercept) * learn.LABEL_SCALE
            prediction = predict(data[design])
            methods["ridge"], scores["ridge"] = metrics(prediction, data, flux, declared,
                                                           spec["ridge"]["calibration_half_width_nm"])
            if split == next(iter(semantics)):
                methods["ridge"]["latency"] = learn._timing(predict, data[design][:1], torch.device("cpu"), 1000)
            del predict, weights, intercept
            for architecture in study.models.ARCHITECTURES:
                spec = frozen["models"][architecture][design]
                seed_metrics = {}
                for seed in spec["seeds"]:
                    guard()
                    model, preprocess = study.load_checkpoint(output / seed["checkpoint"], device)
                    predict = lambda images: learn._predict_network(model, images, preprocess, device)
                    prediction = predict(data[design])
                    measured, grouped = metrics(prediction, data, flux, declared, seed["calibration_half_width_nm"])
                    seed_metrics[str(seed["seed"])] = measured
                    if seed["seed"] == spec["selected_seed"]:
                        methods[architecture], scores[architecture] = measured, grouped
                        if split == next(iter(semantics)):
                            measured["latency"] = learn._timing(predict, data[design][:1], device, 1000)
                    del predict, model
                    guard()
                methods[architecture] = {**methods[architecture], "selected_seed": spec["selected_seed"],
                                         "all_seeds": seed_metrics}
            designs[design] = {"methods": methods, "paired_against_mlp": {
                name: paired_scores(value, scores["mlp"]) for name, value in scores.items() if name != "mlp"}}
        result["splits"][split] = {"counts": counts, "designs": designs}
    guard()
    if (learn._sha(output / "frozen.json") != frozen_sha
            or learn._sha(path) != result["evaluation_manifest_sha256"] or study.identity(run)[3] != protocol):
        raise ValueError("Source/data/runtime/freeze changed during evaluation")
    return result


def self_check():
    parents = np.repeat(["a", "b"], 5)
    flux = np.tile([1000., 10000., 100000., 1000., 10000.], 2)
    truth = np.zeros((10, 4))
    prediction = np.ones((10, 4))
    prediction[flux == 100000] = 3
    scores = grouped_scores(prediction, truth, parents, flux, [1000., 10000., 100000.])
    np.testing.assert_allclose(scores[:, :4], 11 / 3)
    assert paired_scores(scores, scores)["ci95_penalized_wavefront_rmse_difference_nm"] == [0., 0.]
    prediction[0, 0] = np.nan
    failed = grouped_scores(prediction, truth, parents, flux, [1000., 10000., 100000.])
    assert np.isclose(failed[0, 5], 1 / 6) and np.isfinite(failed).all()
    for broken in (flux[:-1], np.full(10, 1000.)):
        try: grouped_scores(prediction, truth, parents, broken, [1000., 10000., 100000.])
        except ValueError: pass
        else: raise AssertionError("Misaligned/missing flux rows admitted")
    print({"status": "PASS_CPU_EQUAL_FLUX_PAIRED_FAILURE_DENOMINATORS", "scope": "synthetic CPU only"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--cpu-before-launch-upper", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.run is None:
        self_check()
    else:
        run = args.run.resolve()
        output = args.output or run / "architecture-study-01/evaluation.json"
        if not output.resolve().is_relative_to(run) or output.exists():
            raise ValueError("Require a new report path inside this run")
        learn._json(output, evaluate(run, args.manifest, args.max_seconds, args.cpu_before_launch_upper))
