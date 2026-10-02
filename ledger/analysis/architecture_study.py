"""Matched native-pixel models; preserve completed seeds and freeze before testing.

Use the verified stage supervisor for real work. The CPU self-check exercises
serialization, interrupted-seed rejection and source/selection guards only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

NATIVE_STARTED = time.monotonic()

import numpy as np
import psutil
import torch

import architecture_models as models
from benchmark import comparisons, learning as learn

PROJECT = models.PROJECT
SEEDS = (11, 23, 37)
DESIGNS = ("single", "pair")
RESERVE_CPU_SECONDS = 6000


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def source_identity():
    paths = (Path(__file__), Path(models.__file__),
             PROJECT / "ledger/decisions/architecture-study-2026-10-02.md")
    return {p.resolve().relative_to(PROJECT).as_posix(): learn._sha(p) for p in paths}


def identity(run, require_gates=True):
    run, config, readiness, baseline, dataset = comparisons._admit(run)
    if require_gates:
        classical, challenges = read(run / "classical.json"), read(run / "challenges.json")
        oopao = read(run / "oopao/validation.json")
        if (classical["status"] != "COMPLETE" or len(classical["raw"]) != 200
                or challenges["status"] != "COMPLETE" or len(challenges["raw"]) != 720
                or oopao["status"] != "PASS" or oopao["completed_parents"] != 100
                or oopao["acquisitions_per_parent"] != 5):
            raise RuntimeError("Complete baseline comparisons/challenges and full OOPAO PASS required")
        baseline_sha = learn._sha(run / "models/frozen.json")
        if any(record["frozen_sha256"] != baseline_sha for record in (classical, challenges, oopao)):
            raise ValueError("Baseline validation refers to a different frozen model")
    return run, dataset, baseline, {
        "schema_version": 1, "sources": source_identity(),
        "baseline_frozen_sha256": learn._sha(run / "models/frozen.json"),
        "dataset_manifest_sha256": learn._sha(run / "dataset.json"),
        "engine_source_hash": baseline["source_hash"], "measurement_hash": baseline["measurement_hash"],
        "basis_sha256": baseline["basis_sha256"], "modal_order": learn.MODES,
        "coefficient_unit": learn.COEFFICIENT_UNIT, "label_scale_nm": learn.LABEL_SCALE.tolist(),
        "architectures": list(models.ARCHITECTURES), "designs": list(DESIGNS), "seeds": list(SEEDS),
        "epochs_max": 100, "stale_epochs": 10, "batch_size": 128,
        "optimizer": "Adam", "learning_rate": .001, "loss": "standardized four-mode MSE",
        "torch_version": str(torch.__version__), "torchvision_version": importlib.metadata.version("torchvision"),
        "cuda_build": torch.version.cuda, "training_device": baseline["device"],
        "selection": "minimum parent-weighted validation MSE, then lower seed; separately per architecture/design",
        "interval_method": baseline["interval_method"],
        "scope": "post-baseline development; old held-out parents are not fresh confirmation"}


@contextmanager
def owner(output):
    lease = output / "active-owner.json"
    learn._json(lease, {"pid": os.getpid(), "started_unix": time.time()})
    try:
        yield
    finally:
        lease.unlink()


def completed(output, key, expected_identity):
    path, report_path = output / f"{key}.pt", output / f"{key}.json"
    if not path.exists() and not report_path.exists():
        return None
    if not path.is_file() or not report_path.is_file():
        raise RuntimeError(f"Unsealed interrupted seed; preserve and inspect {key}")
    report = read(report_path)
    if (report["identity"] != expected_identity or report["checkpoint"] != path.name
            or learn._sha(path) != report["checkpoint_sha256"]):
        raise ValueError(f"Completed seed identity changed: {key}")
    return report


def save_completed(output, key, model, preprocessing, report, expected_identity, storage_guard=lambda size: None):
    # Exclusive writes never replace a valid seed. An interrupted pair stays unsealed.
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError("Nonfinite model state cannot be checkpointed")
    needed = sum(v.numel() * v.element_size() for v in model.state_dict().values())
    needed += preprocessing["mean"].nbytes + preprocessing["std"].nbytes + 1024 ** 2
    storage_guard(needed)
    with (output / f"{key}.pt").open("xb") as stream:
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "architecture": report["architecture"], "input_size": len(preprocessing["mean"]),
                    "asinh_scale_e": preprocessing["asinh_scale_e"],
                    "feature_mean": torch.from_numpy(preprocessing["mean"]),
                    "feature_std": torch.from_numpy(preprocessing["std"])}, stream)
    result = {**report, "checkpoint": f"{key}.pt", "identity": expected_identity,
              "checkpoint_sha256": learn._sha(output / f"{key}.pt")}
    learn._json(output / f"{key}.json", result)
    return result


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = models.build(checkpoint["architecture"], checkpoint["input_size"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    preprocess = {"asinh_scale_e": checkpoint["asinh_scale_e"],
                  "mean": checkpoint["feature_mean"].numpy(), "std": checkpoint["feature_std"].numpy()}
    return model, preprocess


def fit_or_resume(output, key, protocol, fit, preprocessing, storage_guard):
    report = completed(output, key, protocol)
    if report is not None:
        return report, True
    model, report = fit()
    return save_completed(output, key, model, preprocessing, report, protocol, storage_guard), False


def checked_intervals(prediction, truth, parents):
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    if (prediction.shape != truth.shape or prediction.shape != (len(parents), 4)
            or len(parents) == 0 or not np.isfinite(prediction).all() or not np.isfinite(truth).all()):
        raise ValueError("Calibration requires matching finite four-mode predictions for every row")
    widths = learn._intervals(prediction, truth, parents)
    if np.asarray(widths).shape != (4,) or not np.isfinite(widths).all() or min(widths) < 0:
        raise ValueError("Invalid calibration interval widths")
    return widths


def cpu_guard(cpu_before_launch, native_cpu, ceiling):
    if (not all(math.isfinite(v) and v >= 0 for v in (cpu_before_launch, native_cpu, ceiling))
            or cpu_before_launch + native_cpu + RESERVE_CPU_SECONDS >= ceiling):
        raise TimeoutError("Cumulative CPU ceiling including final-report reserve reached")


def training_seconds_used(run, baseline):
    resources = [read(p) for p in run.glob("t001-*-resources.json")]
    return baseline["elapsed_seconds"] + sum(r["elapsed_s"] for r in resources
                                             if r.get("stage", "").startswith("architecture"))


def limits(run, max_seconds, cpu_before_launch, stop_path):
    """Same admission-inclusive CPU, memory, wall and storage limits for every path."""
    sys.path.insert(0, str(Path(__file__).with_name("expanded_campaign")))
    import campaign
    _, policy, _, _, generation, _ = campaign.check(run, require_dataset=True)
    prior_cpu = campaign.spent_cpu(run, policy, generation)
    if (not math.isfinite(max_seconds) or max_seconds <= 0 or cpu_before_launch is None
            or not math.isfinite(cpu_before_launch) or cpu_before_launch < prior_cpu):
        raise ValueError("Require finite limits and supervisor's completed plus preflight/controller CPU bound")
    start, process = NATIVE_STARTED, psutil.Process()
    def guard():
        if time.monotonic() - start >= max_seconds or stop_path.exists():
            raise TimeoutError("Architecture deadline/stop request; preserve completed seeds")
        cpu_guard(cpu_before_launch, sum(process.cpu_times()[:2]), policy["cpu_ceiling_seconds"])
        if process.memory_info().rss > policy["ram_ceiling_bytes"]:
            raise MemoryError("Architecture RAM ceiling reached")
    def storage_guard(needed):
        retained = sum(p.stat().st_size for p in run.rglob("*") if p.is_file())
        if retained + needed > policy["data_ceiling_bytes"]:
            raise RuntimeError("Insufficient retained-data headroom; preserve completed seeds")
    guard()
    return guard, storage_guard


def projection_seconds(profiles, completed_keys=()):
    expected = {(a, d) for a in models.ARCHITECTURES for d in DESIGNS}
    if {(p["architecture"], p["design"]) for p in profiles} != expected or len(profiles) != 4:
        raise ValueError("Profile every architecture/design exactly once")
    if any(p["epochs"] != 2 or not math.isfinite(p["invocation_seconds"])
           or p["invocation_seconds"] <= 0 for p in profiles):
        raise ValueError("Require two actual timed full-data epochs per model/design")
    expected_keys = {f"{a}_{d}_{s}" for a, d in expected for s in SEEDS}
    if not set(completed_keys).issubset(expected_keys):
        raise ValueError("Completed seed keys must belong to the matched study")
    # ponytail: conservative two-epoch projection; the native guard, not this estimate, enforces the ceiling.
    return math.ceil(1.5 * sum(
        sum(f"{p['architecture']}_{p['design']}_{seed}" not in completed_keys for seed in SEEDS)
        * 100 * p["invocation_seconds"] / 2 for p in profiles))


def verify_canary_checkpoints(output, profiles, resumed, protocol):
    for record in profiles:
        sealed = completed(output, record["key"], protocol)
        if sealed is None or any(record.get(k) != value for k, value in sealed.items()):
            raise ValueError("Every profile checkpoint/report pair must exist and match the measured evidence")
    lifecycle = completed(output, "lifecycle_23", protocol)
    if lifecycle is None or lifecycle["checkpoint_sha256"] != resumed["completed_lifecycle_checkpoint_sha256"]:
        raise ValueError("Completed lifecycle checkpoint/report pair must exist and match resume evidence")


def admit(run, action, max_seconds, cpu_before_launch):
    """Three separate native canaries, then seal admission; never fit scientific seeds."""
    if action not in ("profile", "interrupt", "resume", "seal"):
        raise ValueError("Require a declared admission action")
    run, manifest, baseline, protocol = identity(run)
    output = run / "architecture-admission-01"
    output.mkdir(exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if read(protocol_path) != protocol:
            raise ValueError("Admission source/data/runtime changed; preserve old evidence")
    else:
        learn._json(protocol_path, protocol)
    if training_seconds_used(run, baseline) + max_seconds + 60 > 7200:
        raise ValueError("Admission invocation exceeds remaining initial-training GPU allowance")
    guard, storage_guard = limits(run, max_seconds, cpu_before_launch, output / "stop-request.json")
    torch.set_num_threads(1)
    device = learn.execution_device(baseline["device"])
    if device.type != "cuda":
        raise RuntimeError("Architecture admission requires actual CUDA execution")
    with owner(output):
        if action == "seal":
            profiles = read(output / "profile.json")
            interrupted, resumed = read(output / "interrupt.json"), read(output / "resume.json")
            projection = projection_seconds(profiles["models"])
            resources = {a: read(run / f"t001-architecture-{a}-resources.json")
                         for a in ("profile", "interrupt", "resume")}
            if (any(r["supervision_failure"] or r["cpu_reserved_for_reporting_seconds"] != RESERVE_CPU_SECONDS
                    or r["helper_source_sha256"] != learn._sha(__file__) for r in resources.values())
                    or [resources[a]["exit_code"] for a in ("profile", "interrupt", "resume")] != [0, 75, 0]
                    or any(read(output / f"{a}.json")["identity"] != protocol
                           for a in ("profile", "interrupt", "resume"))
                    or not interrupted["cuda_optimizer_step_completed"]
                    or not resumed["completed_seed_skipped"]
                    or interrupted["native_pid"] == resumed["native_pid"]
                    or any(r["roundtrip"] != "BITWISE_PASS" or r["train_rows"] != 18000
                           or r["validation_rows"] != 3600 for r in profiles["models"])
                    or interrupted["preserved_checkpoint_sha256"] != resumed["preserved_checkpoint_sha256"]):
                raise ValueError("Native success/interruption/resume or matching source/reserve evidence missing")
            verify_canary_checkpoints(output, profiles["models"], resumed, protocol)
            remaining = 7200 - training_seconds_used(run, baseline) - max_seconds - 60
            if projection > remaining:
                raise RuntimeError("Measured conservative full-study projection exceeds remaining GPU allowance")
            guard()
            study = run / "architecture-study-01"
            study.mkdir(exist_ok=True)
            if (study / "protocol.json").exists():
                if read(study / "protocol.json") != protocol:
                    raise ValueError("Study identity changed")
            else:
                learn._json(study / "protocol.json", protocol)
            learn._json(study / "admission.json", {
                "status": "PASS", "identity": protocol,
                "checks": ["actual_cuda_training_all_four_models", "checkpoint_roundtrip",
                           "real_interrupt_and_completed_seed_resume", "training_projection_within_remaining_gpu",
                           "aggregate_supervisor_cpu_reserve"],
                "interruption_scope": "guarded stop after one actual CUDA optimizer step; separate native resume invocation",
                "projected_training_seconds_upper": projection,
                "projection_scope": "two full-data epochs, all four models; 100 epochs x three seeds x 1.5 safety factor; hard native ceilings remain authoritative",
                "gpu_seconds_remaining_after_seal_bound": remaining,
                "profile_sha256": learn._sha(output / "profile.json"),
                "evidence_sha256": {p.name: learn._sha(p) for p in output.glob("*.json")
                                    if p.name != "active-owner.json"},
                "native_resource_sha256": {a: learn._sha(run / f"t001-architecture-{a}-resources.json")
                                           for a in resources}})
            return {"status": "PASS_ARCHITECTURE_ADMISSION", "projected_training_seconds_upper": projection}
        data = {s: learn._load_split(run / "dataset.json", manifest, s) for s in ("train", "validation")}
        guard()
        if action == "profile":
            reports = []
            for design in DESIGNS:
                preprocess = learn._fit_preprocess(data["train"][design], data["train"]["parents"])
                x, vx = (learn._features(data[s][design], preprocess) for s in ("train", "validation"))
                for architecture in models.ARCHITECTURES:
                    key = f"{architecture}_{design}_11"
                    if completed(output, key, protocol) is not None:
                        raise FileExistsError("Profile cannot overwrite or silently reuse an earlier invocation")
                    started = time.monotonic()
                    model, report = models.fit(architecture, x, data["train"]["labels"], vx,
                        data["validation"]["labels"], data["validation"]["parents"], 11, device, guard, epochs=2)
                    report.update(design=design, scope="two-epoch admission canary, not a scientific fit")
                    before = learn._predict_network(model, data["validation"][design][:8], preprocess, device)
                    report = save_completed(output, key, model, preprocess, report, protocol, storage_guard)
                    restored, restored_preprocess = load_checkpoint(output / f"{key}.pt", device)
                    np.testing.assert_array_equal(before, learn._predict_network(restored,
                        data["validation"][design][:8], restored_preprocess, device))
                    guard()
                    reports.append({**report, "key": key, "invocation_seconds": time.monotonic() - started,
                                    "train_rows": len(x), "validation_rows": len(vx), "roundtrip": "BITWISE_PASS"})
                    del model, restored
                    torch.cuda.empty_cache()
                    print("admission profile complete", key, flush=True)
            learn._json(output / "profile.json", {"identity": protocol, "models": reports,
                                                 "projected_training_seconds_upper": projection_seconds(reports)})
            return {"status": "PASS_CUDA_PROFILE_ROUNDTRIP"}
        first = completed(output, "compact_cnn_single_11", protocol)
        if first is None:
            raise ValueError("A sealed first canary checkpoint is required before interruption")
        if action == "resume" and not (output / "interrupt.json").is_file():
            raise ValueError("Lifecycle resume requires the earlier native interruption")
        if action == "interrupt" and (output / "interrupt.json").exists():
            raise FileExistsError("Preserve earlier native interruption evidence")
        preprocess = learn._fit_preprocess(data["train"]["single"], data["train"]["parents"])
        x, vx = (learn._features(data[s]["single"], preprocess) for s in ("train", "validation"))
        calls = 0
        def interrupt_guard():
            nonlocal calls
            guard()
            calls += 1
            # fit guard 1 is admission, 2 precedes first batch, 3 follows its CUDA optimizer step.
            if action == "interrupt" and calls == 3:
                raise InterruptedError("Bounded lifecycle interruption after actual optimizer step")
        def no_refit():
            raise AssertionError("A sealed canary seed must be skipped by the shared training path")
        preserved, skipped = fit_or_resume(output, "compact_cnn_single_11", protocol, no_refit,
                                           preprocess, storage_guard)
        assert skipped and preserved == first
        if completed(output, "lifecycle_23", protocol) is not None:
            raise FileExistsError("Preserve completed lifecycle canary")
        def lifecycle_fit():
            model, report = models.fit("compact_cnn", x[:256], data["train"]["labels"][:256], vx[:128],
                data["validation"]["labels"][:128], data["validation"]["parents"][:128], 23,
                device, interrupt_guard, epochs=2)
            report.update(design="single", scope="bounded lifecycle canary, not a scientific fit")
            return model, report
        try:
            lifecycle, _ = fit_or_resume(output, "lifecycle_23", protocol, lifecycle_fit, preprocess, storage_guard)
        except InterruptedError:
            torch.cuda.synchronize()
            assert completed(output, "compact_cnn_single_11", protocol) == first
            assert not (output / "lifecycle_23.pt").exists() and not (output / "lifecycle_23.json").exists()
            learn._json(output / "interrupt.json", {"identity": protocol, "cuda_optimizer_step_completed": True,
                "preserved_checkpoint_sha256": first["checkpoint_sha256"], "guard_calls": calls,
                "native_pid": os.getpid()})
            return {"status": "EXPECTED_NATIVE_INTERRUPTION", "exit_code": 75}
        if action != "resume" or not (output / "interrupt.json").is_file():
            raise ValueError("Lifecycle resume requires the earlier native interruption")
        assert completed(output, "compact_cnn_single_11", protocol) == first
        learn._json(output / "resume.json", {"identity": protocol, "completed_seed_skipped": skipped,
            "preserved_checkpoint_sha256": first["checkpoint_sha256"], "native_pid": os.getpid(),
            "completed_lifecycle_checkpoint_sha256": lifecycle["checkpoint_sha256"]})
        return {"status": "PASS_NATIVE_COMPLETED_SEED_RESUME"}


def select(reports):
    result = {}
    for architecture in models.ARCHITECTURES:
        result[architecture] = {}
        for design in DESIGNS:
            rows = [r for r in reports if r["architecture"] == architecture and r["design"] == design]
            if len(rows) != len(SEEDS) or sorted(r["seed"] for r in rows) != list(SEEDS):
                raise ValueError("Every declared architecture/design requires all three distinct seeds")
            if any(not math.isfinite(r["best_validation_standardized_mse"]) for r in rows):
                raise ValueError("Nonfinite model selection loss")
            winner = min(rows, key=lambda r: (r["best_validation_standardized_mse"], r["seed"]))
            result[architecture][design] = {"selected_seed": winner["seed"],
                                            "selected_checkpoint": winner["checkpoint"], "seeds": rows}
    return result


def train(run, max_seconds, cpu_before_launch):
    run, manifest, baseline, protocol = identity(run)
    output = run / "architecture-study-01"
    output.mkdir(exist_ok=True)
    if (output / "frozen.json").exists():
        raise FileExistsError("Preserve frozen architecture study")
    if (output / "protocol.json").exists():
        if read(output / "protocol.json") != protocol:
            raise ValueError("Study protocol/source/runtime changed; completed seeds cannot be resumed")
    else:
        learn._json(output / "protocol.json", protocol)
    admission = read(output / "admission.json")
    required_checks = {"actual_cuda_training_all_four_models", "checkpoint_roundtrip",
                       "real_interrupt_and_completed_seed_resume", "training_projection_within_remaining_gpu",
                       "aggregate_supervisor_cpu_reserve"}
    profile_path = run / "architecture-admission-01/profile.json"
    if learn._sha(profile_path) != admission["profile_sha256"]:
        raise ValueError("Measured admission profile changed before training/resume")
    profiles = read(profile_path)
    if profiles["identity"] != protocol:
        raise ValueError("Measured admission profile identity changed")
    validated_keys = {f"{a}_{d}_{seed}" for a in models.ARCHITECTURES for d in DESIGNS for seed in SEEDS
                      if completed(output, f"{a}_{d}_{seed}", protocol) is not None}
    original_projection = admission["projected_training_seconds_upper"]
    projection = projection_seconds(profiles["models"], validated_keys)
    if (admission["status"] != "PASS" or admission["identity"] != protocol
            or not required_checks.issubset(admission["checks"])
            or original_projection != projection_seconds(profiles["models"])
            or not math.isfinite(projection) or not 0 <= projection <= max_seconds):
        raise RuntimeError("Actual matching GPU, checkpoint/lifecycle and resource admission required")
    policy = read(run / "execution-policy.json")
    training_used = training_seconds_used(run, baseline)
    if not math.isfinite(max_seconds) or max_seconds <= 0 or training_used + max_seconds + 60 > 7200:
        raise ValueError("Requested study exceeds remaining cumulative initial-training GPU allowance")
    start = NATIVE_STARTED
    guard, storage_guard = limits(run, max_seconds, cpu_before_launch, output / "stop-request.json")
    with owner(output):
        torch.set_num_threads(1)
        device = learn.execution_device(baseline["device"])
        data = {split: learn._load_split(run / "dataset.json", manifest, split)
                for split in ("train", "validation", "calibration")}
        reports = []
        for design in DESIGNS:
            preprocess = learn._fit_preprocess(data["train"][design], data["train"]["parents"])
            x = learn._features(data["train"][design], preprocess)
            vx = learn._features(data["validation"][design], preprocess)
            for architecture in models.ARCHITECTURES:
                for seed in SEEDS:
                    guard()
                    key = f"{architecture}_{design}_{seed}"
                    def fit_seed():
                        model, report = models.fit(architecture, x, data["train"]["labels"], vx,
                            data["validation"]["labels"], data["validation"]["parents"], seed, device, guard)
                        guard()
                        prediction = learn._predict_network(model, data["calibration"][design], preprocess, device)
                        guard()
                        report.update(design=design, calibration_half_width_nm=checked_intervals(
                            prediction, data["calibration"]["labels"], data["calibration"]["parents"]))
                        return model, report
                    report, _ = fit_or_resume(output, key, protocol, fit_seed, preprocess, storage_guard)
                    reports.append(report)
                    print("architecture seed complete", key, "epochs", report["epochs"], flush=True)
        guard()
        if identity(run)[3] != protocol:
            raise ValueError("Source/data/runtime changed during study; do not freeze")
        if sum(p.stat().st_size for p in run.rglob("*") if p.is_file()) > policy["data_ceiling_bytes"]:
            raise RuntimeError("Retained-data ceiling reached; preserve outputs without sealing")
        expected_checkpoints = {r["checkpoint"] for r in reports}
        if {p.name for p in output.glob("*.pt")} != expected_checkpoints:
            raise ValueError("Unexpected unsealed checkpoint in study directory")
        frozen = {"status": "COMPLETE", "identity": protocol, "models": select(reports),
                  "training_parents": baseline["training_parents"],
                  "validation_parents": baseline["validation_parents"],
                  "calibration_parents": baseline["calibration_parents"],
                  "artifacts": {p.name: learn._sha(p) for p in output.glob("*.pt")},
                  "admission_sha256": learn._sha(output / "admission.json"),
                  "this_invocation_elapsed_seconds": time.monotonic() - start}
        learn._json(output / "frozen.json", frozen)
        return {"status": "COMPLETE", "models": len(reports), "output": str(output)}


def self_check():
    """No real-data or GPU admission. Exercise preservation and validation selection."""
    torch.set_num_threads(1)
    try: admit(Path("missing"), "unknown", 1, None)
    except ValueError: pass
    else: raise AssertionError("Unknown admission action accepted")
    profiles = [{"architecture": a, "design": d, "epochs": 2, "invocation_seconds": 2.}
                for a in models.ARCHITECTURES for d in DESIGNS]
    assert projection_seconds(profiles) == 1800
    for invalid in (profiles[:-1], [dict(p, invocation_seconds=float("nan")) for p in profiles]):
        try: projection_seconds(invalid)
        except ValueError: pass
        else: raise AssertionError("Incomplete/nonfinite CUDA profile admitted")
    keys = {f"{a}_{d}_{s}" for a in models.ARCHITECTURES for d in DESIGNS for s in SEEDS}
    assert projection_seconds(profiles, keys) == 0
    assert projection_seconds(profiles, keys - {"resnet18_pair_37"}) == 150
    with tempfile.TemporaryDirectory(dir=PROJECT / "benchmark/runs") as directory:
        output = Path(directory)
        assert output.resolve().is_relative_to((PROJECT / "benchmark/runs").resolve())
        model = models.build("compact_cnn", 256).eval()
        preprocess = {"asinh_scale_e": 100., "mean": np.zeros(256, np.float32),
                      "std": np.ones(256, np.float32)}
        protocol = {"source": "self-check"}
        saved = save_completed(output, "seed", model, preprocess,
            {"architecture": "compact_cnn", "seed": 11}, protocol)
        assert completed(output, "seed", protocol) == saved
        def no_refit():
            raise AssertionError("Completed seed was refitted")
        resumed, skipped = fit_or_resume(output, "seed", protocol, no_refit, preprocess, lambda size: None)
        assert skipped and resumed == saved
        profile_record = {**saved, "key": "seed"}
        resume_record = {"completed_lifecycle_checkpoint_sha256": "missing"}
        try: verify_canary_checkpoints(output, [profile_record], resume_record, protocol)
        except ValueError: pass
        else: raise AssertionError("Missing lifecycle pair admitted")
        lifecycle = save_completed(output, "lifecycle_23", model, preprocess,
            {"architecture": "compact_cnn", "seed": 23}, protocol)
        resume_record["completed_lifecycle_checkpoint_sha256"] = lifecycle["checkpoint_sha256"]
        verify_canary_checkpoints(output, [profile_record], resume_record, protocol)
        empty = output / "missing-canary-pairs"
        empty.mkdir()
        try: verify_canary_checkpoints(empty, [profile_record], resume_record, protocol)
        except ValueError: pass
        else: raise AssertionError("Absent profile checkpoint pair admitted")
        try: verify_canary_checkpoints(output, [dict(profile_record, checkpoint_sha256="changed")], resume_record, protocol)
        except ValueError: pass
        else: raise AssertionError("Changed profile checkpoint evidence admitted")
        def interrupted_fit():
            raise InterruptedError("synthetic interruption before sealing")
        try: fit_or_resume(output, "interrupted", protocol, interrupted_fit, preprocess, lambda size: None)
        except InterruptedError: pass
        else: raise AssertionError("Interrupted seed was sealed")
        assert completed(output, "interrupted", protocol) is None
        restored, restored_preprocess = load_checkpoint(output / "seed.pt", torch.device("cpu"))
        images = np.ones((2, 1, 16, 16), np.float32)
        np.testing.assert_array_equal(learn._predict_network(model, images, preprocess, torch.device("cpu")),
                                      learn._predict_network(restored, images, restored_preprocess, torch.device("cpu")))
        for key, expected in (("seed", {"source": "changed"}), ("orphan", protocol)):
            if key == "orphan": (output / "orphan.pt").write_bytes(b"interrupted")
            try: completed(output, key, expected)
            except (ValueError, RuntimeError): pass
            else: raise AssertionError("Changed source or unsealed seed admitted")
        try: save_completed(output, "seed", model, preprocess, {"architecture": "compact_cnn"}, protocol)
        except FileExistsError: pass
        else: raise AssertionError("Valid seed overwritten")
        reports = [{"architecture": a, "design": d, "seed": s,
                    "best_validation_standardized_mse": .5 if s in (11, 23) else 1.,
                    "checkpoint": f"{a}_{d}_{s}.pt"}
                   for a in models.ARCHITECTURES for d in DESIGNS for s in SEEDS]
        assert all(v["selected_seed"] == 11 for ds in select(reports).values() for v in ds.values())
        try: select(reports[:-1])
        except ValueError: pass
        else: raise AssertionError("Incomplete seed set frozen")
        with owner(output):
            try:
                with owner(output): pass
            except FileExistsError: pass
            else: raise AssertionError("Two owners admitted")
        assert not (output / "active-owner.json").exists()
        good = np.zeros((100, 4))
        assert checked_intervals(good, good, np.arange(100)) == [0., 0., 0., 0.]
        for damaged in (np.full((100, 3), 0.), good.copy()):
            if damaged.shape[1] == 4: damaged[0, 0] = np.nan
            try: checked_intervals(damaged, good, np.arange(100))
            except ValueError: pass
            else: raise AssertionError("Malformed/nonfinite calibration predictions sealed")
        cpu_guard(100_000, 1000, 172_800)
        try: cpu_guard(166_000, 900, 172_800)
        except TimeoutError: pass
        else: raise AssertionError("Reporting reserve lost at CPU boundary")
        def insufficient(size):
            assert size > 0
            raise RuntimeError("self-check storage ceiling")
        try: save_completed(output, "no-space", model, preprocess, {"architecture": "compact_cnn"}, protocol, insufficient)
        except RuntimeError: pass
        else: raise AssertionError("Storage guard ignored")
        assert not (output / "no-space.pt").exists()
        with torch.no_grad(): next(model.parameters()).flatten()[0] = float("nan")
        try: save_completed(output, "bad-state", model, preprocess, {"architecture": "compact_cnn"}, protocol)
        except ValueError: pass
        else: raise AssertionError("Nonfinite state checkpointed")
        assert not (output / "bad-state.pt").exists()
        # Exercise the actual train admission branch; stop before owner/data/CUDA work.
        from unittest.mock import patch
        budget_run = output / "resume-budget"
        budget_output = budget_run / "architecture-study-01"
        profile_output = budget_run / "architecture-admission-01"
        budget_output.mkdir(parents=True)
        profile_output.mkdir()
        budget_profiles = [dict(p, invocation_seconds=4.) for p in profiles]
        learn._json(profile_output / "profile.json", {"identity": protocol, "models": budget_profiles})
        checks = ["actual_cuda_training_all_four_models", "checkpoint_roundtrip",
                  "real_interrupt_and_completed_seed_resume", "training_projection_within_remaining_gpu",
                  "aggregate_supervisor_cpu_reserve"]
        learn._json(budget_output / "admission.json", {"status": "PASS", "identity": protocol,
            "checks": checks, "projected_training_seconds_upper": 3600,
            "profile_sha256": learn._sha(profile_output / "profile.json")})
        learn._json(budget_run / "execution-policy.json", {})
        module = sys.modules[__name__]
        class AdmissionReached(Exception): pass
        pending = {"resnet18_pair_23", "resnet18_pair_37"}
        with patch.object(module, "identity", return_value=(budget_run, None, {}, protocol)), \
             patch.object(module, "training_seconds_used", return_value=4000), \
             patch.object(module, "completed", side_effect=lambda out, key, ident: None if key in pending else {"synthetic": True}), \
             patch.object(module, "limits", side_effect=AdmissionReached) as budget_guard:
            try: train(budget_run, 600, 100000)
            except AdmissionReached: pass
            else: raise AssertionError("Valid 600-second partial restart did not reach the resource guard")
            assert budget_guard.call_count == 1
        with patch.object(module, "identity", return_value=(budget_run, None, {}, protocol)), \
             patch.object(module, "completed", return_value=None):
            try: train(budget_run, 600, 100000)
            except RuntimeError: pass
            else: raise AssertionError("Unfinished full 3600-second study admitted with 600 seconds")
    print({"status": "PASS_CPU_CHECKPOINT_SELECTION_AND_OWNER", "scope": "synthetic CPU checks only"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--max-seconds", type=float, default=6000)
    parser.add_argument("--cpu-before-launch-upper", type=float)
    parser.add_argument("--action", choices=("train", "profile", "interrupt", "resume", "seal"), default="train")
    args = parser.parse_args()
    if args.run is None:
        self_check()
    else:
        result = (train(args.run, args.max_seconds, args.cpu_before_launch_upper) if args.action == "train"
                  else admit(args.run, args.action, args.max_seconds, args.cpu_before_launch_upper))
        print(result, flush=True)
        sys.exit(result.get("exit_code", 0))
