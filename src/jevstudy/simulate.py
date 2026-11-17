"""Generate deterministic fake prediction records for pipeline checks."""

from __future__ import annotations

import hashlib
import random
from datetime import datetime, timezone
from typing import Any

from .signals import extract_signals


def generate_records(seed: int = 4901, items_per_domain: int = 24) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    domains = ("science_demo", "commonsense_demo", "medicine_demo")
    models = (("jev_sim", 0.72, 5.5), ("qwen_sim", 0.68, 4.1))
    run_id = f"simulation-{seed}"
    for domain_index, domain in enumerate(domains):
        for item_index in range(items_per_domain):
            split = "calibration" if item_index < items_per_domain // 2 else "test"
            item_id = f"{domain}-item-{item_index:04d}"
            gold_key = rng.choice(["A", "B", "C", "D"])
            difficulty = rng.random()
            options = {key: f"Fake {domain} option {key} for item {item_index}" for key in "ABCD"}
            for model_id, base_acc, sharpness in models:
                for condition_id, option_order in (("canonical_order", list("ABCD")),
                                                   ("reversed_order", list("DCBA"))):
                    target_accuracy = max(0.15, min(0.95, base_acc - 0.35 * (difficulty - 0.5)))
                    is_correct = rng.random() < target_accuracy
                    answer_key = gold_key if is_correct else rng.choice([key for key in "ABCD" if key != gold_key])
                    confidence = min(0.96, max(0.28, 0.46 + sharpness * (0.12 + rng.random() * 0.07)))
                    remaining = (1.0 - confidence) / 3
                    probs = {key: remaining for key in "ABCD"}
                    probs[answer_key] = confidence
                    report = min(0.99, max(0.01, confidence + rng.gauss(0, 0.12)))
                    samples = [answer_key if rng.random() < confidence else rng.choice([k for k in "ABCD" if k != answer_key])
                               for _ in range(10)]
                    record = {
                    "run_id": run_id,
                    "source_kind": "simulation",
                    "source": {"generator": "jevstudy.simulate", "version": "1"},
                    "protocol_hash": hashlib.sha256(f"{seed}:{items_per_domain}".encode()).hexdigest(),
                    "item_id": item_id,
                    "domain": domain,
                    "split": split,
                    "question": f"Synthetic placeholder question {item_index} in {domain}.",
                    "options": options,
                    "gold_key": gold_key,
                    "model_id": model_id,
                    "condition_id": condition_id,
                    "answer_key": answer_key,
                    "option_order": option_order,
                    "option_probabilities": probs,
                    "self_report_probability": report,
                    "sampled_answer_keys": samples,
                    "model_revision": "synthetic-v1",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "latency_ms": round(rng.uniform(2, 80), 2),
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "simulated_cost": round(rng.uniform(0, 0.001), 6),
                    }
                    record["signals"] = extract_signals(record)
                    record["correct"] = int(answer_key == gold_key)
                    records.append(record)
    return records

