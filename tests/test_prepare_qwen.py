import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jevstudy.adapters.qwen_local import run_one_qwen
from jevstudy.prepare import freeze_dataset, normalize_row
from jevstudy.schema import validate_records


class FakeQwen:
    model_id = "Qwen/Qwen3-8B"
    revision = "fixture-commit"

    def __init__(self):
        self.calls = []

    def score_labels(self, prompt, keys):
        self.calls.append(("score", prompt))
        return {key: (3.0 if key == "A" else 0.0) for key in keys}, 30

    def generate_text(self, prompt, *, thinking, sample, seed, max_new_tokens):
        self.calls.append(("generate", prompt, thinking, sample, seed))
        if sample:
            return "Reason. FINAL: A", 35, 5
        return "0.72", 34, 2


class PreparationAndQwenTests(unittest.TestCase):
    def test_five_source_shapes(self):
        rows = [
            ("sciq", {"question": "Q?", "distractor1": "a", "distractor2": "b",
                      "distractor3": "c", "correct_answer": "d"}),
            ("hellaswag", {"ctx": "A story", "endings": ["a", "b", "c", "d"], "label": "2"}),
            ("medmcqa", {"question": "Q?", "opa": "a", "opb": "b", "opc": "c",
                         "opd": "d", "cop": 1, "choice_type": "single"}),
            ("strategyqa", {"question": "Q?", "answer": True}),
            ("mmlu", {"question": "Q?", "choices": ["a", "b", "c", "d"], "answer": 2}),
        ]
        for dataset, row in rows:
            with self.subTest(dataset=dataset):
                item = normalize_row(dataset, row, "test", 0, 4901)
                self.assertIn(item["gold_key"], item["options"])
                self.assertEqual(len(item["options"]), 2 if dataset == "strategyqa" else 4)

    def test_freeze_keeps_order_pair_and_split_disjoint(self):
        def sciq(index):
            return {"question": f"Question {index}?", "distractor1": f"a{index}",
                    "distractor2": f"b{index}", "distractor3": f"c{index}",
                    "correct_answer": f"d{index}"}
        rows = [sciq(index) for index in range(8)]
        frozen = freeze_dataset("sciq", rows, rows, calibration_source_split="train",
                                test_source_split="train", revision="fixture-hash",
                                n_calibration=2, n_test=3, order_items=2, seed=5)
        self.assertEqual(len(frozen), 7)
        canonical = {item["item_id"]: item for item in frozen if item["condition_id"] == "canonical_order"}
        reversed_rows = [item for item in frozen if item["condition_id"] == "reversed_order"]
        self.assertEqual(len(reversed_rows), 2)
        for item in reversed_rows:
            self.assertEqual(item["option_order"], list(reversed(canonical[item["item_id"]]["option_order"])))
            self.assertEqual(item["gold_key"], canonical[item["item_id"]]["gold_key"])

    def test_qwen_keeps_base_answer_and_separates_samples(self):
        item = normalize_row("strategyqa", {"question": "Can fish fly?", "answer": False}, "test", 0, 1)
        item["split"] = "test"
        fake = FakeQwen()
        record = run_one_qwen(item, fake, "fixture-run", sample_count=5, seed=7)
        validate_records([record])
        self.assertEqual(record["answer_key"], "A")
        self.assertEqual(record["majority_vote_answers"]["5"], "A")
        self.assertEqual(record["signals"]["sample_consistency_5"], 1.0)
        self.assertEqual(record["input_tokens"], 30 + 34 + 5 * 35)
        self.assertEqual(record["output_tokens"], 2 + 5 * 5)
        self.assertIn("fixed as A", fake.calls[1][1])
        self.assertNotIn("Answer with exactly one option letter", fake.calls[1][1])


if __name__ == "__main__":
    unittest.main()
