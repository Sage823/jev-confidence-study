import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jevstudy.metrics import (accuracy, auroc, aurc, brier_multiclass, ece,
                              nll, order_agreement, risk_at_coverage)


class MetricTests(unittest.TestCase):
    def test_auroc_ties_and_spearman(self):
        from jevstudy.metrics import spearman
        self.assertAlmostEqual(auroc([0.5, 0.5, 0.9, 0.1], [1, 0, 1, 0]), 0.875)
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 2, 1]), -1.0)
    def test_hand_computable_metrics(self):
        self.assertEqual(accuracy([1, 0, 1]), 2 / 3)
        self.assertAlmostEqual(auroc([0.9, 0.2, 0.8, 0.1], [1, 0, 1, 0]), 1.0)
        self.assertAlmostEqual(auroc([0.5, 0.5], [1, 0]), 0.5)
        self.assertAlmostEqual(brier_multiclass([{"A": .8, "B": .2}], ["A"]), .08)
        self.assertAlmostEqual(nll([{"A": .5, "B": .5}], ["A"]), 0.69314718056, places=8)
        self.assertAlmostEqual(ece([.8, .2], [1, 0], bins=2), .2)
        self.assertEqual(order_agreement(["A", "B"], ["A", "C"]), .5)

    def test_risk_curve_and_fixed_coverage(self):
        scores = [.9, .8, .1]
        correct = [1, 0, 0]
        self.assertAlmostEqual(risk_at_coverage(scores, correct, 1 / 3), 0)
        self.assertGreater(aurc(scores, correct), 0)

    def test_equal_frequency_ece(self):
        value = ece([.1, .2, .8, .9], [0, 0, 1, 1], bins=2, strategy="equal_frequency")
        self.assertAlmostEqual(value, .15)


if __name__ == "__main__":
    unittest.main()

