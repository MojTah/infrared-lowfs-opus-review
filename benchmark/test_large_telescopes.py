"""Development-only checks of conditional admission and retained pilot metadata.

Mock optics are never numerical evidence for any telescope.
"""
import json
from contextlib import contextmanager
from pathlib import Path
import time
import unittest
from unittest.mock import patch
import uuid

import numpy as np

try:
    from . import large_telescopes as pilot
    from .core import DEFAULT_CONFIG
except ImportError:
    import large_telescopes as pilot
    from core import DEFAULT_CONFIG


@contextmanager
def fixture_directory():
    # Native Windows sandbox denies child access after tempfile's mode 0700.
    # Ordinary inherited-ACL directories work; retain these tiny mock fixtures.
    path = Path(__file__).parent / "runs" / ("development_large_test_" + uuid.uuid4().hex)
    path.mkdir(parents=True)
    yield str(path)


class PilotChecks(unittest.TestCase):
    def fixture(self, root):
        config = json.loads(json.dumps(DEFAULT_CONFIG))
        basis = {"mu": [0.] * 6, "transform": np.eye(6).tolist(),
                 "units": "nm OPD RMS", "development_only": True}
        identity = {"config_hash": pilot.digest(config), "source_hash": pilot.source_hash(),
                    "measurement_hash": "development-only", "basis": basis}
        (root / "models").mkdir()
        frozen = {**identity, "modal_order": pilot.MODES, "coefficient_unit": "nm OPD RMS",
                  "basis_sha256": pilot._basis_hash(basis), "models": {}, "artifacts": {}}
        for design in ("single", "pair"):
            filename = design + ".pt"
            (root / "models" / filename).write_bytes(b"development-only mocked checkpoint")
            frozen["models"][design] = {"selected_checkpoint": filename}
            frozen["artifacts"][filename] = pilot.file_hash(root / "models" / filename)
        (root / "dataset.json").write_text(json.dumps(identity))
        frozen["dataset_manifest_sha256"] = pilot.file_hash(root / "dataset.json")
        (root / "config.json").write_text(json.dumps(config))
        (root / "readiness.json").write_text(json.dumps({**identity, "status": "PASS"}))
        (root / "models" / "frozen.json").write_text(json.dumps(frozen))
        (root / "oopao").mkdir()
        (root / "oopao" / "validation.json").write_text(json.dumps({
            **identity, "status": "PASS", "frozen_sha256": pilot.file_hash(root / "models" / "frozen.json")}))
        return basis

    def test_admission_never_creates_science_without_pass(self):
        with fixture_directory() as temp:
            root = Path(temp)
            with patch.object(pilot, "OpticalModel") as optical:
                self.assertEqual(pilot.simulate(root)["status"], "NOT_ADMITTED")
                self.assertFalse((root / "large_telescopes").exists())
                optical.assert_not_called()
            self.fixture(root)
            self.assertEqual(pilot._admission(root)[0]["pupil"], "keck")
            validation = pilot._read(root / "oopao" / "validation.json")
            validation["status"] = "PARTIAL"
            (root / "oopao" / "validation.json").write_text(json.dumps(validation))
            with patch.object(pilot, "OpticalModel") as optical:
                self.assertEqual(pilot.simulate(root)["status"], "NOT_ADMITTED")
                optical.assert_not_called()

    def test_frozen_artifact_tampering_is_refused(self):
        with fixture_directory() as temp:
            root = Path(temp)
            self.fixture(root)
            (root / "models" / "single.pt").write_bytes(b"changed")
            self.assertEqual(pilot.simulate(root)["status"], "NOT_ADMITTED")
            self.assertFalse((root / "large_telescopes").exists())

    def test_refinement_escalates_without_relaxing_tolerance(self):
        class FakeOptical:
            def __init__(self, config, pupil_n, basis=None):
                self.n = pupil_n
                self.basis = basis or {"development_only": True}
            def image(self, truth, diversity):
                return np.ones((16, 16)) * (1.02 if self.n == 512 else 1.)
            def independent_image(self, truth, diversity):
                return self.image(truth, diversity)
        with patch.object(pilot, "OpticalModel", FakeOptical), patch.object(pilot, "_guard"):
            model, basis, evidence = pilot._converged_model(DEFAULT_CONFIG, time.monotonic(), 10000, 2**34)
        self.assertEqual(evidence["status"], "PASS")
        self.assertEqual(model.n, 1024)
        self.assertEqual(evidence["relative_l1_threshold"], 1e-3)
        self.assertEqual([r["refined_pupil_n"] for r in evidence["refinements"]], [1024, 2048])

    def test_mock_pilot_preserves_budget_seeds_crop_and_signed_images(self):
        class FakeModel:
            n = 2
            config = {}
            measurement_hash = "new-development-only-basis"
            weights = np.full(4, .25)
            def probabilities(self, truth, residual):
                return np.ones((1, 16, 16)) * .001, np.ones((2, 16, 16)) * .002
        class FakeResidual:
            def __init__(self, model, seed, gain):
                self.seed = seed
            def acquisition(self, index, steps):
                self.assert_steps = steps
                return np.zeros((steps, 2, 2))
        with fixture_directory() as temp:
            root = Path(temp)
            basis = self.fixture(root)
            convergence = {"status": "PASS", "relative_l1_threshold": 1e-3}
            with patch.object(pilot, "_converged_model", return_value=(FakeModel(), basis, convergence)), \
                    patch.object(pilot, "ResidualSequence", FakeResidual), \
                    patch.object(pilot, "calibrate_residual_gain", return_value=(.1, {"developer_seeds": [1, 2, 3]})):
                result = pilot.simulate(root, max_seconds=60, count=2)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(len(result["cases"]), 3)
            for record in result["cases"]:
                self.assertEqual(len(record["shards"]), 2)
                self.assertTrue((root / "large_telescopes" / record["pupil"] / "manifest.json").is_file())
                self.assertEqual(record["pixel_mas"], 1.5e-6 / (2 * record["diameter_m"] * pilot.MAS_RAD))
                shard = record["shards"][0]
                self.assertEqual(shard["phase_nodes"], 10)
                self.assertEqual(shard["expected_source_e_total"], 10000.)
                self.assertIsInstance(shard["atmospheric_seed"], int)
                np.testing.assert_allclose(shard["source_fraction_in_crop"]["single"], [.256])
                np.testing.assert_allclose(shard["source_fraction_in_crop"]["pair_per_exposure"], [.512, .512])
                with np.load(root / shard["path"], allow_pickle=False) as data:
                    self.assertEqual(data["images_single"].shape, (1, 1, 16, 16))
                    self.assertEqual(data["images_pair"].shape, (1, 2, 16, 16))
                    self.assertEqual(data["labels_nm"].shape, (1, 4))
                    self.assertEqual(data["images_single"].dtype, np.float32)
                self.assertEqual(pilot.file_hash(root / shard["path"]), shard["sha256"])
            with self.assertRaises(FileExistsError):
                pilot.simulate(root, count=2)

    def test_time_and_memory_guards(self):
        with self.assertRaises(TimeoutError):
            pilot._guard(time.monotonic() - 2, 1, 2**34)
        with self.assertRaises(MemoryError):
            pilot._guard(time.monotonic(), 1, 1)


if __name__ == "__main__":
    unittest.main()
