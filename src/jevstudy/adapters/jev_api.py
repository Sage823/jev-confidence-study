"""Minimal Jev HTTP adapter using only Python's standard library.

No request is made by importing this module. A call happens only when the
explicit ``run-jev`` CLI command invokes ``call_jev``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

from ..signals import extract_signals

API_URL = "https://api.typesafe.ai/v1/systemone"


def build_choice_payload(item: dict[str, Any], model: str) -> dict[str, Any]:
    """Convert a normalized multiple-choice item to a System One request."""
    required = ("item_id", "domain", "split", "question", "options", "gold_key")
    missing = [field for field in required if field not in item]
    if missing:
        raise ValueError(f"item is missing fields: {missing}")
    if item["split"] not in {"calibration", "test"}:
        raise ValueError("item split must be calibration or test")
    options = item["options"]
    if not isinstance(options, dict) or len(options) < 2:
        raise ValueError("options must map at least two keys to option text")
    if item["gold_key"] not in options:
        raise ValueError("gold_key must be one of the option keys")
    option_order = item.get("option_order", list(options))
    if set(option_order) != set(options) or len(option_order) != len(options):
        raise ValueError("option_order must contain each option key exactly once")
    criteria = {str(key): str(options[key]) for key in option_order}
    return {
        "model": model,
        "state": str(item["question"]),
        "questions": {
            "answer": {
                "type": "choice",
                "instructions": "Choose the option that correctly answers the question in the state.",
                "criteria": criteria,
            }
        },
    }


def call_jev(payload: dict[str, Any], api_key: str | None = None,
             timeout_seconds: float = 60.0) -> dict[str, Any]:
    """Make one authenticated POST. Retries are intentionally disabled."""
    key = api_key or os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError("TYPESAFE_API_KEY is not set; no request was sent")
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"Jev API returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Jev API: {exc.reason}") from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Jev API response was not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Jev API response must be a JSON object")
    return parsed


def prediction_from_response(item: dict[str, Any], payload: dict[str, Any],
                             response: dict[str, Any], latency_ms: float,
                             run_id: str) -> dict[str, Any]:
    """Normalize one Choice response while retaining its raw distribution."""
    try:
        answer = response["answers"]["answer"]
        answer_key = str(answer["choice"])
        raw_probs = answer["probabilities"]
    except (KeyError, TypeError) as exc:
        raise ValueError("response is missing answers.answer.choice/probabilities") from exc
    options = item["options"]
    if answer_key not in options:
        raise ValueError(f"Jev returned unknown choice key: {answer_key}")
    if not isinstance(raw_probs, dict) or set(raw_probs) != set(options):
        raise ValueError("Jev probability keys do not match the requested option keys")
    raw = {str(key): float(value) for key, value in raw_probs.items()}
    if any(not math.isfinite(value) or value < 0 for value in raw.values()):
        raise ValueError("Jev probabilities must be finite and nonnegative")
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("Jev returned a probability distribution with zero total")
    normalized = {key: value / total for key, value in raw.items()}
    usage = response.get("usage") or {}
    request_fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    record = {
        "run_id": run_id,
        "source_kind": "model_output",
        "source": item.get("source", {}),
        "protocol_hash": request_fingerprint,
        "item_id": str(item["item_id"]),
        "domain": str(item["domain"]),
        "split": str(item["split"]),
        "question": str(item["question"]),
        "options": options,
        "gold_key": str(item["gold_key"]),
        "model_id": f"jev:{response.get('model', payload['model'])}",
        "condition_id": str(item.get("condition_id", "canonical_order")),
        "answer_key": answer_key,
        "option_order": item.get("option_order", list(options)),
        "raw_option_probabilities": raw,
        "option_probabilities": normalized,
        "model_revision": response.get("model", payload["model"]),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "latency_ms": round(latency_ms, 3),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "raw_response": response,
    }
    record["signals"] = extract_signals(record)
    record["correct"] = int(answer_key == record["gold_key"])
    return record


def run_one(item: dict[str, Any], model: str, run_id: str,
            api_key: str | None = None) -> dict[str, Any]:
    payload = build_choice_payload(item, model)
    start = time.perf_counter()
    response = call_jev(payload, api_key=api_key)
    latency_ms = (time.perf_counter() - start) * 1000
    return prediction_from_response(item, payload, response, latency_ms, run_id)
