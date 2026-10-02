"""Small development checks; these data are not scientific benchmark evidence."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

try:
    from . import learning
except ImportError:
    import learning


class LearningChecks(unittest.TestCase):
    def test_verified_cpu_choice_survives_cuda_availability(self):
        with patch.object(torch.cuda,'is_available',return_value=True) as available:
            self.assertEqual(learning.execution_device('cpu').type,'cpu')
            available.assert_not_called()
        with patch.object(torch.cuda,'is_available',return_value=False):
            with self.assertRaises(RuntimeError):learning.execution_device('cuda')

    def fixture(self, root):
        rows = []
        for index, split in enumerate(("train", "validation", "calibration", "test")):
            filename = f"{split}.npz"
            rng = np.random.default_rng(index)
            np.savez(root / filename, images_single=rng.normal(size=(4, 1, 16, 16)).astype(np.float32),
                     images_pair=rng.normal(size=(4, 2, 16, 16)).astype(np.float32),
                     labels_nm=rng.normal(size=(4, 4)), flux_e=np.ones(4) * 1000)
            digest = hashlib.sha256((root / filename).read_bytes()).hexdigest()
            rows.append({"path": filename, "split": split, "parent_id": split, "sha256": digest})
        path = root / "dataset.json"
        basis = {"order": ["x_tilt", "y_tilt", *learning.MODES], "target_indices": [2, 3, 4, 5],
                 "mu": [0.0] * 6, "transform": np.eye(6).tolist(), "units": "nm OPD RMS",
                 "coordinates": "development-only OPD exp(+i2piW/lambda), Fourier negative",
                 "metric": "development-only normalized illuminated area"}
        path.write_text(json.dumps({"schema_version": 1, "config_hash": "config", "source_hash": "source",
                                    "measurement_hash": "development-measurement", "basis": basis, "shards": rows}))
        return path

    def test_split_leakage_and_checksums(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path = self.fixture(Path(temp))
            manifest = learning._manifest(path)
            dataset = learning._load_split(path, manifest, "train")
            self.assertEqual(dataset["labels"].shape, (4, 4))
            # Test shards may be absent at training time: metadata inspection must not open them.
            (Path(temp) / "test.npz").unlink()
            learning._manifest(path)
            learning._load_split(path, manifest, "train")
            manifest["shards"][1]["parent_id"] = "train"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "leaks"):
                learning._manifest(path)

    def test_signed_features_and_parent_weighting(self):
        images = np.array([-1000, 0, 1000], dtype=np.float32).reshape(3, 1, 1, 1)
        parents = np.array(["a", "a", "b"])
        preprocessing = learning._fit_preprocess(images, parents)
        features = learning._features(images, preprocessing)
        self.assertTrue(np.all(np.diff(features[:, 0]) > 0))
        np.testing.assert_allclose(learning._parent_weights(parents), [.25, .25, .5])
        np.testing.assert_allclose((features[:, 0] * [.25, .25, .5]).sum(), 0, atol=1e-7)
        # Metrics average parent means, rather than allowing a prolific parent to dominate.
        truth = np.zeros((3, 4))
        prediction = np.array([[1] * 4, [1] * 4, [3] * 4])
        metrics = learning._metrics(prediction, truth, parents, [10] * 4)
        np.testing.assert_allclose(metrics["coefficient_bias_nm"], [2] * 4)
        np.testing.assert_allclose(metrics["coefficient_rmse_nm"], [np.sqrt(5)] * 4)

    def test_checkpoint_weights_only_reproducibility(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            torch.manual_seed(5)
            images = np.random.default_rng(5).normal(size=(4, 1, 16, 16)).astype(np.float32)
            preprocessing = learning._fit_preprocess(images, np.array(["a"] * 4))
            model = learning._network(256)
            path = Path(temp) / "weights.pt"
            torch.save({"input_size": 256, "state_dict": model.state_dict(), "asinh_scale_e": 100.0,
                        "feature_mean": torch.from_numpy(preprocessing["mean"]),
                        "feature_std": torch.from_numpy(preprocessing["std"])}, path)
            loaded, restored = learning._load_checkpoint(path, torch.device("cpu"))
            before = learning._predict_network(model, images, preprocessing, torch.device("cpu"))
            after = learning._predict_network(loaded, images, restored, torch.device("cpu"))
            np.testing.assert_array_equal(before, after)

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path = self.fixture(Path(temp))
            manifest = json.loads(path.read_text())
            manifest["shards"][0]["path"] = "../outside.npz"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "escapes"):
                learning._manifest(path)

    def test_basis_metadata_validation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path = self.fixture(Path(temp))
            original = path.read_text()
            for mutation, message in (
                (lambda m: m.pop("measurement_hash"), "measurement_hash"),
                (lambda m: m["basis"].update(units="phase radians"), "units"),
                (lambda m: m["basis"].update(order=["x_tilt", "y_tilt", "spherical", "astig_cos", "astig_sin", "focus"]), "order"),
            ):
                manifest = json.loads(original)
                mutation(manifest)
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, message):
                    learning._manifest(path)

    def test_tiny_development_train_freeze_evaluate(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            self.fixture(root)
            with patch.object(torch.cuda, "is_available", return_value=False):
                frozen = learning.train(root, max_gpu_seconds=120, seeds=(11,))
            with patch.object(torch.cuda, 'is_available',return_value=True):
                result = learning.evaluate(root, timing_calls=2)
            self.assertEqual(result['splits']['test']['pair']['learned_latency']['device'],'cpu')
            self.assertEqual(frozen["modal_order"], learning.MODES)
            self.assertEqual(result["splits"]["test"]["pair"]["learned"]["n_acquisitions"], 4)
            self.assertTrue((root / "models" / "frozen.json").exists())
            # External engines may change nuisance/configuration, but must preserve the measurement and basis.
            external = json.loads((root / "dataset.json").read_text())
            external["config_hash"] = "shifted-config"
            external["source_hash"] = "other-engine-source"
            external_path = root / "external.json"
            external_path.write_text(json.dumps(external))
            with patch.object(torch.cuda, "is_available", return_value=False):
                learning.evaluate(root, manifest_path=external_path, timing_calls=1)
            external["measurement_hash"] = "different-measurement"
            external_path.write_text(json.dumps(external))
            with self.assertRaisesRegex(ValueError, "measurement_hash mismatch"):
                learning.evaluate(root, manifest_path=external_path, timing_calls=1)
            external["measurement_hash"] = frozen["measurement_hash"]
            external["basis"]["transform"][2][2] = 2.0
            external_path.write_text(json.dumps(external))
            with self.assertRaisesRegex(ValueError, "basis identity mismatch"):
                learning.evaluate(root, manifest_path=external_path, timing_calls=1)
            # Repeating training cannot overwrite a frozen checkpoint.
            with self.assertRaises(FileExistsError):
                learning.train(root, seeds=(11,))
            # Modified checkpoint bytes are rejected before loading/inference.
            checkpoint = root / "models" / frozen["models"]["single"]["selected_checkpoint"]
            with checkpoint.open("ab") as stream:
                stream.write(b"modified")
            with self.assertRaisesRegex(ValueError, "artifact mismatch"):
                learning.evaluate(root, timing_calls=1)


if __name__ == "__main__":
    unittest.main()
