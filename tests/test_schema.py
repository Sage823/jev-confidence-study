import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jevstudy.schema import ValidationError, validate_records
from jevstudy.simulate import generate_records


class SchemaTests(unittest.TestCase):
    def test_generated_records_validate(self):
        validate_records(generate_records(seed=9, items_per_domain=8))

    def test_rejects_probability_sum_error(self):
        row = copy.deepcopy(generate_records(seed=9, items_per_domain=8)[0])
        row["option_probabilities"]["A"] += 0.1
        with self.assertRaisesRegex(ValidationError, "sum to 1"):
            validate_records([row])

    def test_rejects_item_split_leakage(self):
        rows = generate_records(seed=9, items_per_domain=8)
        duplicate = copy.deepcopy(rows[0])
        duplicate["model_id"] = "new_model"
        duplicate["split"] = "test"
        with self.assertRaisesRegex(ValidationError, "both calibration and test"):
            validate_records([rows[0], duplicate])

    def test_rejects_duplicate_records(self):
        row = generate_records(seed=9, items_per_domain=8)[0]
        with self.assertRaisesRegex(ValidationError, "duplicate"):
            validate_records([row, copy.deepcopy(row)])


if __name__ == "__main__":
    unittest.main()

