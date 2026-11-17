import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jevstudy.calibration import (apply_scalar_logistic, apply_temperature,
                                  fit_scalar_logistic, fit_temperature)
from jevstudy.simulate import generate_records


class CalibrationTests(unittest.TestCase):
    def test_scalar_fit_requires_calibration_rows(self):
        rows = [{"split": "calibration", "signals": {"s": x}, "correct": y}
                for x, y in [(0.1, 0), (0.2, 0), (0.8, 1), (0.9, 1)]]
        params = fit_scalar_logistic(rows, "s")
        self.assertGreaterEqual(params["slope"], 0)
        self.assertLess(apply_scalar_logistic(.1, params), apply_scalar_logistic(.9, params))
        with self.assertRaisesRegex(ValueError, "calibration rows only"):
            fit_scalar_logistic([{"split": "test", "signals": {"s": .5}, "correct": 1}], "s")

    def test_temperature_is_only_fitted_on_calibration_rows(self):
        rows = [r for r in generate_records(seed=99, items_per_domain=10)
                if r["model_id"] == "jev_sim" and r["domain"] == "science_demo" and r["split"] == "calibration"]
        temperature = fit_temperature(rows)
        self.assertGreater(temperature, 0)
        probs = apply_temperature(rows[0]["option_probabilities"], temperature)
        self.assertAlmostEqual(sum(probs.values()), 1.0)
        bad = copy.deepcopy(rows[0])
        bad["split"] = "test"
        with self.assertRaisesRegex(ValueError, "calibration rows only"):
            fit_temperature([bad])


if __name__ == "__main__":
    unittest.main()

