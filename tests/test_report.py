import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jevstudy.report import analyze_records
from jevstudy.simulate import generate_records


class ReportTests(unittest.TestCase):
    def test_report_marks_simulation_and_pairs_order_conditions(self):
        report = analyze_records(generate_records(seed=11, items_per_domain=8))
        self.assertEqual(report["source_kind"], "simulation")
        self.assertIn("not empirical evidence", report["warning"])
        self.assertTrue(report["order_stability"])
        self.assertTrue(report["domain_macro_averages"])
        self.assertEqual(report["domain_macro_averages"][0]["n_domains"], 3)
        self.assertEqual(report["order_stability"][0]["n_paired_test_items"], 4)
        first_group = report["groups"][0]
        self.assertIn("held_out_calibration", first_group["signals"]["native_probability"])
        self.assertEqual(first_group["sample_audit"]["invalid_samples"], 0)


if __name__ == "__main__":
    unittest.main()

