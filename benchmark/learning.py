"""Four-mode supervised estimators; test data are read only after model freeze.

The public entrypoints are ``train(run_dir)`` and ``evaluate(run_dir)``.
Generator phase and nuisance truth are deliberately absent from model inputs.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

MODES = ["focus", "astig_cos", "astig_sin", "spherical"]
LABEL_SCALE = np.array([150.0, 100.0, 100.0, 100.0])
SPLITS = {"train", "validation", "calibration", "test", "shifted", "oopao"}
ASINH_SCALE_E = 100.0
COEFFICIENT_UNIT = "nm OPD RMS"


def execution_device(requested):
    """Respect verified CPU fallback throughout training and frozen inference."""
    if requested not in ('cpu','cuda'):raise ValueError('Unsupported configured training device')
    if requested=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('Configured CUDA device is no longer available; freeze a new hardware configuration')
    return torch.device(requested)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    # A failed run preserves its partial artifacts; an existing result is never replaced.
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _basis_hash(basis):
    canonical = json.dumps(basis, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Expected dataset schema_version 1")
    for key in ("config_hash", "source_hash", "measurement_hash"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ValueError(f"Dataset lacks {key}")
    basis = data.get("basis")
    if not isinstance(basis, dict) or not basis:
        raise ValueError("Dataset lacks explicit basis identity")
    indices, order = basis.get("target_indices", []), basis.get("order", [])
    if (indices != [2, 3, 4, 5] or len(order) != 6
            or [order[index] for index in indices] != MODES):
        raise ValueError("Dataset basis target order must match MODES")
    if basis.get("units") != COEFFICIENT_UNIT or not basis.get("coordinates") or not basis.get("metric"):
        raise ValueError("Dataset lacks declared nm OPD RMS units or basis convention")
    transform, mu = np.asarray(basis.get("transform", [])), np.asarray(basis.get("mu", []))
    if (transform.shape != (6, 6) or mu.shape != (6,)
            or not np.isfinite(transform).all() or not np.isfinite(mu).all()):
        raise ValueError("Dataset basis transformation must be finite and six-dimensional")
    _basis_hash(basis)
    parents, paths = {}, set()
    if not data.get("shards"):
        raise ValueError("Dataset has no shards")
    for shard in data["shards"]:
        split, parent = shard["split"], shard["parent_id"]
        if split not in SPLITS or not isinstance(parent, str) or not parent:
            raise ValueError("Invalid split or parent_id")
        if parent in parents and parents[parent] != split:
            raise ValueError(f"Parent {parent!r} leaks across dataset splits")
        parents[parent] = split
        resolved = (path.parent / shard["path"]).resolve()
        if not resolved.is_relative_to(path.parent) or resolved in paths:
            raise ValueError("Shard path escapes dataset root or is duplicated")
        paths.add(resolved)
        digest = shard.get("sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Shard requires a lowercase SHA256 checksum")
    return data


def _load_split(manifest_path, manifest, split):
    rows = [s for s in manifest["shards"] if s["split"] == split]
    if not rows:
        raise ValueError(f"Dataset lacks required split {split}")
    parts = {key: [] for key in ("single", "pair", "labels", "parents")}
    for shard in rows:
        path = (Path(manifest_path).resolve().parent / shard["path"]).resolve()
        if _sha(path) != shard["sha256"]:
            raise ValueError(f"Shard checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as arrays:
            labels = np.asarray(arrays["labels_nm"], dtype=np.float64)
            n = len(labels)
            single = np.asarray(arrays["images_single"], dtype=np.float32)
            pair = np.asarray(arrays["images_pair"], dtype=np.float32)
            flux = np.asarray(arrays["flux_e"], dtype=np.float64)
            if (n == 0 or labels.shape != (n, 4) or single.shape != (n, 1, 16, 16)
                    or pair.shape != (n, 2, 16, 16) or flux.shape != (n,)):
                raise ValueError(f"Unexpected shard array dimensions: {path}")
            if not all(np.isfinite(a).all() for a in (labels, single, pair, flux)):
                raise ValueError(f"Nonfinite shard values: {path}")
            if np.any(flux <= 0):
                raise ValueError("Expected source electron counts must be positive")
            parts["single"].append(single)
            parts["pair"].append(pair)
            parts["labels"].append(labels)
            parts["parents"].append(np.repeat(shard["parent_id"], n))
    return {key: np.concatenate(value) for key, value in parts.items()}


def _parent_weights(parents):
    _, inverse, counts = np.unique(parents, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights / weights.sum()


def _fit_preprocess(images, parents):
    transformed = np.arcsinh(images.reshape(len(images), -1) / ASINH_SCALE_E)
    weight = _parent_weights(parents)[:, None]
    mean = np.sum(transformed * weight, axis=0)
    std = np.sqrt(np.sum((transformed - mean) ** 2 * weight, axis=0))
    return {"asinh_scale_e": ASINH_SCALE_E, "mean": mean.astype(np.float32),
            "std": np.maximum(std, 1e-4).astype(np.float32)}


def _features(images, preprocessing):
    raw = np.asarray(images, dtype=np.float32).reshape(len(images), -1)
    return ((np.arcsinh(raw / preprocessing["asinh_scale_e"])
             - preprocessing["mean"]) / preprocessing["std"]).astype(np.float32)


def _network(input_size):
    return nn.Sequential(nn.Linear(input_size, 32), nn.ReLU(), nn.Linear(32, 32),
                         nn.ReLU(), nn.Linear(32, 4))


def _predict_network(model, images, preprocessing, device, batch_size=512):
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            x = torch.from_numpy(_features(images[start:start + batch_size], preprocessing)).to(device)
            output.append(model(x).cpu().numpy().astype(np.float64) * LABEL_SCALE)
    return np.concatenate(output)


def _weighted_mse(prediction, truth, parents):
    return float(np.sum(np.mean(((prediction - truth) / LABEL_SCALE) ** 2, axis=1)
                        * _parent_weights(parents)))


def _train_seed(x, y, validation_x, validation_y, parents, seed, device, deadline, batch_size=128):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = _network(x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tx = torch.from_numpy(x).to(device)
    ty = torch.from_numpy((y / LABEL_SCALE).astype(np.float32)).to(device)
    vx = torch.from_numpy(validation_x).to(device)
    vy = torch.from_numpy((validation_y / LABEL_SCALE).astype(np.float32)).to(device)
    vw = torch.from_numpy(_parent_weights(parents).astype(np.float32)).to(device)
    best, best_state, stale, history = math.inf, None, 0, []
    for epoch in range(100):
        if time.monotonic() >= deadline:
            raise TimeoutError("Training compute ceiling reached; models are not frozen")
        model.train()
        for ids in np.array_split(rng.permutation(len(x)), math.ceil(len(x) / batch_size)):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(tx[ids]) - ty[ids]) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_loss = float(torch.sum(torch.mean((model(vx) - vy) ** 2, dim=1) * vw).item())
        history.append(validation_loss)
        if validation_loss < best:
            best = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 10:
            break
    model.load_state_dict(best_state)
    return model, {"seed": seed, "best_validation_standardized_mse": best,
                   "epochs": len(history), "validation_history": history}


def _fit_ridge(x, labels, validation_x, validation_labels, validation_parents):
    x, targets = x.astype(np.float64), labels / LABEL_SCALE
    xm, ym = x.mean(axis=0), targets.mean(axis=0)
    centered, yc = x - xm, targets - ym
    eigenvalues, vectors = np.linalg.eigh(centered.T @ centered)
    rhs = vectors.T @ centered.T @ yc
    trials, best = [], None
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        weights = vectors @ (rhs / (np.maximum(eigenvalues, 0)[:, None] + alpha))
        intercept = ym - xm @ weights
        prediction = (validation_x @ weights + intercept) * LABEL_SCALE
        score = _weighted_mse(prediction, validation_labels, validation_parents)
        trials.append({"alpha": alpha, "validation_standardized_mse": score})
        if best is None or score < best[0]:
            best = score, weights, intercept, alpha
    return {"weights": best[1], "intercept": best[2], "alpha": best[3], "trials": trials}


def _intervals(prediction, truth, parents):
    residuals = np.abs(prediction - truth)
    weights = _parent_weights(parents)
    result = []
    for mode in range(4):
        order = np.argsort(residuals[:, mode], kind="stable")
        index = min(np.searchsorted(np.cumsum(weights[order]), 0.95), len(order) - 1)
        result.append(float(residuals[order[index], mode]))
    return result


def _load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = _network(checkpoint["input_size"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    preprocessing = {"asinh_scale_e": checkpoint["asinh_scale_e"],
                     "mean": checkpoint["feature_mean"].numpy(),
                     "std": checkpoint["feature_std"].numpy()}
    return model, preprocessing


def train(run_dir: Path, max_gpu_seconds=7200, seeds=(11, 23, 37)) -> dict:
    """Fit on train/validation, calibrate separately, and freeze before test reads.

    Existing ``models/`` causes an error. A failed run leaves unsealed artifacts
    for inspection and cannot be mistaken for a complete training checkpoint.
    """
    if max_gpu_seconds <= 0 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Require positive training budget and distinct training seeds")
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "dataset.json"
    manifest = _manifest(manifest_path)
    manifest_digest = _sha(manifest_path)
    for split in ("train", "validation", "calibration"):
        if not any(s["split"] == split for s in manifest["shards"]):
            raise ValueError(f"Dataset lacks {split}")
    output = run_dir / "models"
    output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    config_path=run_dir / 'config.json'
    compute=json.loads(config_path.read_text()).get('compute',{}) if config_path.exists() else {}
    requested=compute.get('training_device','cuda' if torch.cuda.is_available() else 'cpu')
    batch_size=compute.get('training_batch_size',128)
    if not isinstance(batch_size,int) or not 1<=batch_size<=128:raise ValueError('Training batch size must be 1..128')
    device = execution_device(requested)
    start = time.monotonic()
    deadline = start + max_gpu_seconds
    training = _load_split(manifest_path, manifest, "train")
    validation = _load_split(manifest_path, manifest, "validation")
    calibration = _load_split(manifest_path, manifest, "calibration")
    frozen = {"schema_version": 1, "modal_order": MODES, "coefficient_unit": COEFFICIENT_UNIT,
              "basis": manifest["basis"], "basis_sha256": _basis_hash(manifest["basis"]),
              "measurement_hash": manifest["measurement_hash"],
              "basis_convention": manifest["basis"]["coordinates"], "label_scale_nm": LABEL_SCALE.tolist(),
              "dataset_manifest_sha256": manifest_digest, "config_hash": manifest["config_hash"],
              "source_hash": manifest["source_hash"], "device": str(device),
              "torch_version": str(torch.__version__), "models": {}, "artifacts": {},
              "compute_budget_seconds": max_gpu_seconds, "training_batch_size": batch_size,
              "budget_measurement": "complete training/calibration wall-clock; conservative GPU occupancy ceiling",
              "training_parents": sorted(set(training["parents"])),
              "validation_parents": sorted(set(validation["parents"])),
              "calibration_parents": sorted(set(calibration["parents"])),
              "interval_method": "empirical 95 percent parent-weighted absolute error; no formal coverage guarantee"}
    for design in ("single", "pair"):
        preprocessing = _fit_preprocess(training[design], training["parents"])
        train_x = _features(training[design], preprocessing)
        validation_x = _features(validation[design], preprocessing)
        seed_reports = []
        for seed in seeds:
            model, report = _train_seed(train_x, training["labels"], validation_x,
                                        validation["labels"], validation["parents"], int(seed), device, deadline,batch_size=batch_size)
            filename = f"{design}_seed_{seed}.pt"
            torch.save({"state_dict": model.state_dict(), "input_size": train_x.shape[1],
                        "asinh_scale_e": ASINH_SCALE_E,
                        "feature_mean": torch.from_numpy(preprocessing["mean"]),
                        "feature_std": torch.from_numpy(preprocessing["std"]),
                        "label_scale_nm": torch.from_numpy(LABEL_SCALE.copy()),
                        "seed": int(seed), "modal_order": MODES}, output / filename)
            prediction = _predict_network(model, calibration[design], preprocessing, device)
            report.update({"checkpoint": filename,
                           "calibration_half_width_nm": _intervals(prediction, calibration["labels"], calibration["parents"])})
            seed_reports.append(report)
        selected = min(seed_reports, key=lambda r: (r["best_validation_standardized_mse"], r["seed"]))
        ridge = _fit_ridge(train_x, training["labels"], validation_x, validation["labels"], validation["parents"])
        ridge_path = output / f"{design}_ridge.npz"
        np.savez(ridge_path, weights=ridge["weights"], intercept=ridge["intercept"],
                 feature_mean=preprocessing["mean"], feature_std=preprocessing["std"],
                 asinh_scale_e=np.array(ASINH_SCALE_E))
        prediction = (_features(calibration[design], preprocessing) @ ridge["weights"] + ridge["intercept"]) * LABEL_SCALE
        frozen["models"][design] = {"selected_seed": selected["seed"],
                                    "selected_checkpoint": selected["checkpoint"], "seeds": seed_reports,
                                    "ridge": {"checkpoint": ridge_path.name, "alpha": ridge["alpha"],
                                              "trials": ridge["trials"],
                                              "calibration_half_width_nm": _intervals(prediction, calibration["labels"], calibration["parents"])}}
    if time.monotonic() >= deadline:
        raise TimeoutError("Training/calibration compute ceiling reached before freeze")
    if _sha(manifest_path) != manifest_digest:
        raise ValueError("Dataset manifest changed during training; models are not frozen")
    frozen["elapsed_seconds"] = time.monotonic() - start
    for path in sorted(output.iterdir()):
        frozen["artifacts"][path.name] = _sha(path)
    _json(output / "frozen.json", frozen)
    return frozen


def _group_statistics(prediction, truth, parents, half_width):
    ids = np.unique(parents)
    error = prediction - truth
    finite = np.isfinite(prediction).all(axis=1)
    stats = []
    for parent in ids:
        mask = parents == parent
        good = mask & finite
        if not np.any(good):
            raise FloatingPointError(f"No finite predictions for parent {parent}")
        selected = error[good]
        covered = (np.abs(error[mask]) <= half_width) & np.isfinite(error[mask])
        stats.append(np.r_[selected.mean(axis=0), (selected ** 2).mean(axis=0),
                           np.mean(np.sum(selected ** 2, axis=1)), covered.mean(axis=0)])
    return np.asarray(stats), finite


def _bootstrap_interval(values, statistic, draws=1000):
    rng = np.random.default_rng(918273)
    samples = np.array([statistic(values[rng.integers(0, len(values), len(values))].mean(axis=0))
                        for _ in range(draws)])
    return np.quantile(samples, [0.025, 0.975], axis=0).tolist()


def _metrics(prediction, truth, parents, half_width):
    half_width = np.asarray(half_width)
    grouped, finite = _group_statistics(prediction, truth, parents, half_width)
    average = grouped.mean(axis=0)
    weight = _parent_weights(parents[finite])
    design = np.column_stack([np.ones(finite.sum()), truth[finite]])
    weighted_design = design * np.sqrt(weight[:, None])
    fit = np.linalg.lstsq(weighted_design, prediction[finite] * np.sqrt(weight[:, None]), rcond=None)[0]
    sign_error = []
    for mode in range(4):
        eligible = finite & (np.abs(truth[:, mode]) >= 20)
        if eligible.any():
            weights = _parent_weights(parents[eligible])
            sign_error.append(float(np.sum((np.sign(prediction[eligible, mode]) != np.sign(truth[eligible, mode])) * weights)))
        else:
            sign_error.append(None)
    return {"n_acquisitions": len(truth), "n_parents": len(grouped),
            "coefficient_bias_nm": average[:4].tolist(),
            "coefficient_rmse_nm": np.sqrt(average[4:8]).tolist(),
            "reconstructed_wavefront_rmse_nm": float(np.sqrt(average[8])),
            "crosstalk_predicted_by_true_slopes": fit[1:].T.tolist(),
            "crosstalk_intercept_nm": fit[0].tolist(), "sign_error_fraction_truth_ge_20nm": sign_error,
            "interval_coverage": average[9:13].tolist(), "interval_width_nm": (2 * half_width).tolist(),
            "failure_fraction": float(np.sum(~finite * _parent_weights(parents))),
            "abstention_fraction": 0.0, "abstention_rule": "none",
            "ci95_parent_bootstrap": _bootstrap_interval(grouped, lambda m: np.r_[m[:4], np.sqrt(m[4:9]), m[9:13]]),
            "ci_order": ["bias[4]", "coefficient_rmse[4]", "wavefront_rmse", "coverage[4]"],
            "metric_weighting": "equal parent weight; within-parent acquisition mean"}


def _failure_examples(prediction, truth, parents, limit=10):
    error = np.sqrt(np.sum((prediction - truth) ** 2, axis=1))
    return [{"acquisition_index": int(index), "parent_id": str(parents[index]),
             "truth_nm": truth[index].tolist(), "prediction_nm": prediction[index].tolist(),
             "reconstructed_error_nm": float(error[index])}
            for index in np.argsort(error)[-limit:][::-1]]


def _timing(predict, image, device, calls):
    if calls < 1:
        raise ValueError("timing_calls must be positive")
    for _ in range(10):
        predict(image)
    durations = []
    for _ in range(calls):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        predict(image)
        if device.type == "cuda":
            torch.cuda.synchronize()
        durations.append((time.perf_counter() - start) * 1000)
    return {"calls": calls, "batch_size": 1, "device": str(device),
            "includes": "signed-count preprocessing, device transfer, inference, output transfer, synchronization",
            "p50_ms": float(np.percentile(durations, 50)), "p95_ms": float(np.percentile(durations, 95)),
            "p99_ms": float(np.percentile(durations, 99))}


def evaluate(run_dir: Path, manifest_path: Path | None = None, timing_calls=1000) -> dict:
    """Evaluate frozen models on test/shifted/OOPAO shards without fitting.

    External manifest shard paths are relative to that manifest's directory.
    Reports are returned to the execution owner, which controls persistence.
    """
    run_dir = Path(run_dir).resolve()
    model_dir = run_dir / "models"
    frozen = json.loads((model_dir / "frozen.json").read_text(encoding="utf-8"))
    if frozen["modal_order"] != MODES or frozen["coefficient_unit"] != COEFFICIENT_UNIT:
        raise ValueError("Frozen coefficient order or units mismatch")
    if _basis_hash(frozen["basis"]) != frozen["basis_sha256"]:
        raise ValueError("Frozen basis identity mismatch")
    for filename, digest in frozen["artifacts"].items():
        artifact = (model_dir / filename).resolve()
        if not artifact.is_relative_to(model_dir) or _sha(artifact) != digest:
            raise ValueError(f"Frozen artifact mismatch: {filename}")
    manifest_path = Path(manifest_path or run_dir / "dataset.json").resolve()
    if manifest_path == run_dir / "dataset.json" and _sha(manifest_path) != frozen["dataset_manifest_sha256"]:
        raise ValueError("Training dataset manifest changed after freeze")
    manifest = _manifest(manifest_path)
    if manifest["measurement_hash"] != frozen["measurement_hash"]:
        raise ValueError("Evaluation measurement_hash mismatch")
    if _basis_hash(manifest["basis"]) != frozen["basis_sha256"]:
        raise ValueError("Evaluation basis identity mismatch")
    heldout = [s for s in ("test", "shifted", "oopao") if any(row["split"] == s for row in manifest["shards"])]
    if not heldout:
        raise ValueError("Evaluation manifest has no test, shifted or OOPAO split")
    known_parents = set(frozen["training_parents"] + frozen["validation_parents"] + frozen["calibration_parents"])
    if any(row["parent_id"] in known_parents for row in manifest["shards"] if row["split"] in heldout):
        raise ValueError("Evaluation parent overlaps a fitting/calibration parent")
    torch.set_num_threads(1)
    device = execution_device(frozen['device'])
    result = {"schema_version": 1, "modal_order": MODES, "frozen_manifest_sha256": _sha(model_dir / "frozen.json"),
              "evaluation_manifest_sha256": _sha(manifest_path), "config_hash": manifest["config_hash"],
              "source_hash": manifest["source_hash"], "measurement_hash": manifest["measurement_hash"],
              "basis_sha256": frozen["basis_sha256"], "coefficient_unit": COEFFICIENT_UNIT, "splits": {}}
    for split in heldout:
        dataset = _load_split(manifest_path, manifest, split)
        split_result = {}
        for design in ("single", "pair"):
            specification = frozen["models"][design]
            seed_results, selected_prediction, selected_predict = {}, None, None
            for report in specification["seeds"]:
                model, preprocessing = _load_checkpoint(model_dir / report["checkpoint"], device)
                prediction = _predict_network(model, dataset[design], preprocessing, device)
                seed_results[str(report["seed"])] = _metrics(prediction, dataset["labels"], dataset["parents"],
                                                            report["calibration_half_width_nm"])
                if report["seed"] == specification["selected_seed"]:
                    selected_prediction = prediction
                    selected_predict = lambda images, m=model, p=preprocessing: _predict_network(m, images, p, device)
            with np.load(model_dir / specification["ridge"]["checkpoint"], allow_pickle=False) as ridge:
                preprocessing = {"mean": ridge["feature_mean"], "std": ridge["feature_std"],
                                 "asinh_scale_e": float(ridge["asinh_scale_e"])}
                weights, intercept = ridge["weights"], ridge["intercept"]
            ridge_predict = lambda images: (_features(images, preprocessing) @ weights + intercept) * LABEL_SCALE
            ridge_prediction = ridge_predict(dataset[design])
            selected_error = np.mean(((selected_prediction - dataset["labels"]) / LABEL_SCALE) ** 2, axis=1)
            ridge_error = np.mean(((ridge_prediction - dataset["labels"]) / LABEL_SCALE) ** 2, axis=1)
            differences = np.array([(selected_error - ridge_error)[dataset["parents"] == p].mean()
                                    for p in np.unique(dataset["parents"])])[:, None]
            split_result[design] = {"selected_seed": specification["selected_seed"],
                                    "learned": seed_results[str(specification["selected_seed"])], "seed_results": seed_results,
                                    "ridge": _metrics(ridge_prediction, dataset["labels"], dataset["parents"],
                                                      specification["ridge"]["calibration_half_width_nm"]),
                                    "paired_learned_minus_ridge_standardized_mse": float(differences.mean()),
                                    "paired_difference_ci95": _bootstrap_interval(differences, lambda m: m)[0:2],
                                    "largest_error_examples": _failure_examples(selected_prediction, dataset["labels"], dataset["parents"]),
                                    "learned_latency": _timing(selected_predict, dataset[design][:1], device, timing_calls),
                                    "ridge_latency": _timing(ridge_predict, dataset[design][:1], torch.device("cpu"), timing_calls)}
        result["splits"][split] = split_result
    return result
