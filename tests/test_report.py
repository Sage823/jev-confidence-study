import sys
import unittest
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jevstudy.report import analyze_records
from jevstudy.schema import ValidationError
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

    def test_cross_domain_fits_exclude_target_and_missing_signal_is_counted(self):
        records = generate_records(seed=15, items_per_domain=8)
        test_row = next(row for row in records if row["split"] == "test"
                        and row["condition_id"] == "canonical_order"
                        and row["model_id"] == "jev_sim")
        del test_row["self_report_probability"]
        del test_row["signals"]["self_report"]
        report = analyze_records(records)
        group = next(row for row in report["groups"] if row["model_id"] == "jev_sim"
                     and row["domain"] == test_row["domain"]
                     and row["condition_id"] == "canonical_order")
        self.assertEqual(group["signals"]["self_report"]["n_missing"], 1)
        cross = group["signals"]["native_probability"]["calibration"]["cross_domain"]
        self.assertNotIn(test_row["domain"], cross["fit_domains"])
        self.assertEqual(cross["n_calibration"], 8)
        self.assertIn("paired_model_accuracy", report)

    def test_test_labels_do_not_change_fit_parameters(self):
        records = generate_records(seed=16, items_per_domain=8)
        first = analyze_records(records)
        changed = deepcopy(records)
        for row in changed:
            if row["split"] == "test":
                row["gold_key"] = next(key for key in row["options"] if key != row["gold_key"])
                row["correct"] = int(row["answer_key"] == row["gold_key"])
        second = analyze_records(changed)
        def parameters(report):
            return next(row["signals"]["native_probability"]["calibration"]["joint"]["parameters"]
                        for row in report["groups"] if row["condition_id"] == "canonical_order"
                        and "parameters" in row["signals"]["native_probability"]["calibration"]["joint"])
        self.assertEqual(parameters(first), parameters(second))

    def test_unmatched_model_items_are_rejected_by_default(self):
        records = generate_records(seed=18, items_per_domain=8)
        incomplete = [row for row in records if not (
            row["model_id"] == "qwen_sim" and row["split"] == "test"
            and row["condition_id"] == "canonical_order" and row["item_id"].endswith("0004"))]
        with self.assertRaises(ValidationError):
            analyze_records(incomplete)
        report = analyze_records(incomplete, allow_incomplete=True)
        self.assertEqual(report["alignment"], "incomplete")


if __name__ == "__main__":
    unittest.main()

