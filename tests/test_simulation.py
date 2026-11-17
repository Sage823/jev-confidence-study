import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jevstudy.simulate import generate_records


class SimulationTests(unittest.TestCase):
    def test_seed_repeats_logical_records(self):
        first = generate_records(seed=123, items_per_domain=8)
        second = generate_records(seed=123, items_per_domain=8)
        for rows in (first, second):
            for row in rows:
                row.pop("created_at_utc")
        self.assertEqual(first, second)
        self.assertNotEqual(first, generate_records(seed=124, items_per_domain=8))

    def test_simulation_is_labeled_and_has_both_splits(self):
        rows = generate_records(items_per_domain=8)
        self.assertEqual({row["source_kind"] for row in rows}, {"simulation"})
        self.assertEqual({row["split"] for row in rows}, {"calibration", "test"})


if __name__ == "__main__":
    unittest.main()

