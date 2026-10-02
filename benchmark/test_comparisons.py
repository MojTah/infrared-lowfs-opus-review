"""Development-only guards and accounting checks; no optical campaign."""
import unittest
from unittest.mock import Mock

import numpy as np
import torch

try:
    from . import comparisons
except ImportError:
    import comparisons


class ComparisonChecks(unittest.TestCase):
    def test_nonfinite_observation_never_calls_estimator(self):
        image = np.zeros((1,16,16))
        image[0,3,9] = np.nan
        predictor, guard = Mock(), Mock()
        result = comparisons._invoke(predictor, image, "single", torch.device("cpu"), guard)
        self.assertEqual(result["status"], "NO_ESTIMATE")
        predictor.assert_not_called()
        guard.assert_not_called()

    def test_time_cap_retained(self):
        predictor = Mock(return_value=np.zeros(4))
        result = comparisons._invoke(predictor, np.zeros((2,16,16)), "pair", torch.device("cpu"),
                                     Mock(side_effect=TimeoutError("development cap")))
        self.assertEqual(result["status"], "TIME_CAP")
        self.assertIsNone(result["coeff_nm"])
        predictor.assert_not_called()

    def test_failure_denominator_and_paired_uncertainty(self):
        rows = []
        for parent in ("a", "b"):
            rows.append(comparisons._row(parent,"test","single",np.zeros(4),"mlp",
                                         {"status":"OK","coeff_nm":[1]*4,"latency_ms":1.0}))
            rows.append(comparisons._row(parent,"test","single",np.zeros(4),"ridge",
                                         {"status":"NO_ESTIMATE","coeff_nm":None,"latency_ms":.5}))
        report = comparisons._report(rows)
        summary = report["metrics"]["test/single"]["ridge"]
        self.assertEqual(summary["n_planned_observations"],2)
        self.assertEqual(summary["no_estimate_fraction_all"],1.0)
        self.assertIsNone(summary["coefficient_rmse_nm_available"])
        np.testing.assert_allclose(summary["penalized_coefficient_rmse_nm_all"],2*comparisons.core.LIMITS)
        self.assertGreater(report["paired_comparisons"]["test/single"]["ridge"]["minus_mlp_standardized_penalized_mse"],0)

    def test_invalid_prediction_rejected(self):
        result = comparisons._invoke(lambda image: np.array([1,np.nan,3,4]),np.zeros((1,16,16)),
                                     "single",torch.device("cpu"),lambda:None)
        self.assertEqual(result["status"],"NO_ESTIMATE")


if __name__ == "__main__":
    unittest.main()
