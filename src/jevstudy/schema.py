"""JSONL reading and validation for normalized prediction records."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


class ValidationError(ValueError):
    """Raised when an input record violates the project schema."""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValidationError(f"{path}:{line_number}: each JSONL line must be an object")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def validate_records(records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValidationError("no prediction records found")
    seen: set[tuple[str, str, str, str]] = set()
    split_by_item_domain: dict[tuple[str, str], str] = {}
    for index, row in enumerate(records, start=1):
        required = ("run_id", "source_kind", "item_id", "domain", "split", "model_id",
                    "condition_id", "answer_key", "gold_key", "options", "option_order",
                    "option_probabilities")
        missing = [field for field in required if field not in row]
        if missing:
            raise ValidationError(f"record {index}: missing fields {missing}")
        if row["source_kind"] not in {"simulation", "benchmark", "model_output"}:
            raise ValidationError(f"record {index}: invalid source_kind")
        if row["split"] not in {"calibration", "test"}:
            raise ValidationError(f"record {index}: split must be calibration or test")
        if not isinstance(row["options"], dict) or len(row["options"]) < 2:
            raise ValidationError(f"record {index}: options must contain at least two key/text pairs")
        keys = list(row["options"])
        if set(row["option_order"]) != set(keys) or len(row["option_order"]) != len(keys):
            raise ValidationError(f"record {index}: option_order must contain each option key once")
        if row["answer_key"] not in row["options"] or row["gold_key"] not in row["options"]:
            raise ValidationError(f"record {index}: answer_key and gold_key must be valid option keys")
        probs = row["option_probabilities"]
        if not isinstance(probs, dict) or set(probs) != set(keys):
            raise ValidationError(f"record {index}: option_probabilities must align with option keys")
        vals = list(probs.values())
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in vals):
            raise ValidationError(f"record {index}: probabilities must be finite nonnegative numbers")
        total = sum(vals)
        if not math.isclose(total, 1.0, rel_tol=1e-7, abs_tol=1e-7):
            raise ValidationError(f"record {index}: normalized probabilities must sum to 1 (got {total})")
        if "self_report_probability" in row:
            p = row["self_report_probability"]
            if not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
                raise ValidationError(f"record {index}: self_report_probability must be in [0, 1]")
        key = (row["run_id"], row["item_id"], row["model_id"], row["condition_id"])
        if key in seen:
            raise ValidationError(f"record {index}: duplicate run/item/model/condition {key}")
        seen.add(key)
        identity = (row["run_id"], row["item_id"])
        previous = split_by_item_domain.setdefault(identity, row["split"])
        if previous != row["split"]:
            raise ValidationError(f"record {index}: item appears in both calibration and test splits")

