"""Convert model outputs into scores where larger means more likely correct."""

from __future__ import annotations

import math
from typing import Any


def selected_probability(row: dict[str, Any]) -> float:
    return float(row["option_probabilities"][row["answer_key"]])


def normalized_entropy_confidence(row: dict[str, Any]) -> float:
    probs = row["option_probabilities"].values()
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    maximum = math.log(len(row["option_probabilities"]))
    return 1.0 - entropy / maximum if maximum else 1.0


def self_report(row: dict[str, Any]) -> float:
    return float(row["self_report_probability"])


def sample_consistency(row: dict[str, Any]) -> float:
    samples = row.get("sampled_answer_keys", [])
    valid = [key for key in samples if key in row["options"]]
    if not valid:
        return float("nan")
    return sum(key == row["answer_key"] for key in valid) / len(valid)


def correctness(row: dict[str, Any]) -> int:
    return int(row["answer_key"] == row["gold_key"])


def extract_signals(row: dict[str, Any]) -> dict[str, float]:
    signals = {
        "native_probability": selected_probability(row),
        "entropy_confidence": normalized_entropy_confidence(row),
    }
    if "self_report_probability" in row:
        signals["self_report"] = self_report(row)
    samples = row.get("sampled_answer_keys", [])
    if samples and any(key in row["options"] for key in samples):
        signals["sample_consistency"] = sample_consistency(row)
    for budget in (5, 10, 20):
        if len(samples) >= budget and all(key in row["options"] for key in samples[:budget]):
            signals[f"sample_consistency_{budget}"] = (
                sum(key == row["answer_key"] for key in samples[:budget]) / budget)
    return signals

