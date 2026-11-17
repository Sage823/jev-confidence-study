"""One append-only JSONL runner for Jev and local Qwen conditions."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from . import __version__
from .prepare import validate_items
from .schema import read_jsonl, validate_records


def run_items(items: list[dict[str, Any]], output_dir: str | Path, *,
              model_name: str, configuration: dict[str, Any],
              infer: Callable[[dict[str, Any], str], dict[str, Any]],
              input_hash: str | None = None) -> dict[str, Any]:
    validate_items(items)
    root = Path("runs").resolve()
    target = Path(output_dir).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError("model outputs must be written to a new directory inside runs/")
    target.mkdir(parents=True, exist_ok=False)
    run_id = f"run-{uuid4().hex[:12]}"
    successes = failures = 0
    predictions_path = target / "predictions.jsonl"
    errors_path = target / "errors.jsonl"
    for item in items:
        try:
            record = infer(item, run_id)
            if configuration.get("gpu_usd_per_hour") is not None:
                record["cost_usd"] = record["latency_ms"] * configuration["gpu_usd_per_hour"] / 3_600_000
                record["cost_basis"] = "estimated_gpu_time_excludes_model_load"
            elif configuration.get("input_usd_per_million") is not None:
                if record.get("input_tokens") is not None and record.get("output_tokens") is not None:
                    record["cost_usd"] = (
                        record["input_tokens"] * configuration["input_usd_per_million"]
                        + record["output_tokens"] * configuration["output_usd_per_million"]) / 1_000_000
                    record["cost_basis"] = "estimated_from_user_supplied_token_rates"
            validate_records([record])
            if any(record[field] != item[field] for field in
                   ("item_id", "domain", "split", "gold_key", "options", "condition_id", "option_order")):
                raise ValueError("model record does not match the frozen input item")
            with predictions_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            successes += 1
        except Exception as exc:
            error = {"run_id": run_id, "item_id": item["item_id"], "domain": item["domain"],
                     "condition_id": item["condition_id"], "model": model_name,
                     "error_type": type(exc).__name__, "error": str(exc)}
            with errors_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(error, ensure_ascii=False, sort_keys=True) + "\n")
            failures += 1
    if successes:
        validate_records(read_jsonl(predictions_path))
    metadata = {
        "run_id": run_id, "source_kind": "model_output", "model": model_name,
        "configuration": configuration, "input_sha256": input_hash,
        "requested_count": len(items), "success_count": successes, "failure_count": failures,
        "package_version": __version__, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "predictions_sha256": _file_digest(predictions_path),
        "errors_sha256": _file_digest(errors_path),
    }
    (target / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def _file_digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
