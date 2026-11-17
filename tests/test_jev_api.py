import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jevstudy.adapters.jev_api import (build_choice_payload, call_jev,
                                        prediction_from_response)


def example_item():
    return {
        "item_id": "sciq-test-1",
        "domain": "science",
        "split": "test",
        "question": "Which planet is known as the Red Planet?",
        "options": {"A": "Venus", "B": "Mars", "C": "Jupiter", "D": "Mercury"},
        "gold_key": "B",
        "option_order": ["A", "B", "C", "D"],
        "source": {"dataset": "example-fixture"},
    }


class JevApiTests(unittest.TestCase):
    def test_request_payload_matches_system_one_choice_shape(self):
        payload = build_choice_payload(example_item(), "jev-1.13.0")
        self.assertEqual(payload["model"], "jev-1.13.0")
        self.assertEqual(payload["state"], example_item()["question"])
        question = payload["questions"]["answer"]
        self.assertEqual(question["type"], "choice")
        self.assertEqual(question["criteria"]["B"], "Mars")
        self.assertNotIn("gold_key", json_dumps(payload))

    def test_request_respects_permuted_option_order(self):
        item = example_item()
        item["option_order"] = ["D", "C", "B", "A"]
        payload = build_choice_payload(item, "jev-1.13.0")
        self.assertEqual(list(payload["questions"]["answer"]["criteria"]), ["D", "C", "B", "A"])

    def test_response_is_normalized_and_raw_values_are_retained(self):
        item = example_item()
        payload = build_choice_payload(item, "jev-1.13.0")
        response = {
            "model": "jev-1.13.0",
            "answers": {"answer": {"type": "choice", "choice": "B", "confidence": 0.6,
                                    "probabilities": {"A": 0.2, "B": 0.6, "C": 0.1, "D": 0.1}}},
            "usage": {"input_tokens": 12, "output_tokens": 3},
        }
        row = prediction_from_response(item, payload, response, 10.5, "run-test")
        self.assertEqual(row["source_kind"], "model_output")
        self.assertEqual(row["answer_key"], "B")
        self.assertAlmostEqual(sum(row["option_probabilities"].values()), 1.0)
        self.assertEqual(row["raw_option_probabilities"], response["answers"]["answer"]["probabilities"])
        self.assertEqual(row["input_tokens"], 12)
        self.assertEqual(row["signals"]["native_probability"], 0.6)

    def test_missing_api_key_fails_before_network(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "no request was sent"):
                call_jev({"model": "jev-1.13.0"})


def json_dumps(value):
    import json
    return json.dumps(value)


if __name__ == "__main__":
    unittest.main()
