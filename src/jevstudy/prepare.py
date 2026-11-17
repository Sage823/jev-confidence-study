"""Freeze licensed, locally supplied benchmark rows into one shared protocol."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .schema import ValidationError, read_jsonl, write_jsonl

DATASET_URLS = {
    "sciq": "https://huggingface.co/datasets/allenai/sciq",
    "hellaswag": "https://huggingface.co/datasets/Rowan/hellaswag",
    "medmcqa": "https://huggingface.co/datasets/openlifescienceai/medmcqa",
    "strategyqa": "https://huggingface.co/datasets/voidful/StrategyQA",
    "mmlu": "https://huggingface.co/datasets/cais/mmlu",
}


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_row(dataset: str, row: dict[str, Any], source_split: str,
                  index: int, seed: int) -> dict[str, Any] | None:
    """Map one source row to the shared question/option/gold schema.

    Returns None only for MedMCQA rows explicitly marked as multi-answer.
    """
    name = dataset.lower()
    if name not in DATASET_URLS:
        raise ValueError(f"unknown dataset: {dataset}")
    if name == "sciq":
        question = str(row["question"])
        choices = [str(row[f"distractor{i}"]) for i in (1, 2, 3)] + [str(row["correct_answer"])]
        answer_index = 3
        source_id = row.get("id")
    elif name == "hellaswag":
        question = f"Complete the following context: {row['ctx']}"
        choices = [str(value) for value in row["endings"]]
        answer_index = int(row["label"])
        source_id = row.get("ind")
    elif name == "medmcqa":
        if str(row.get("choice_type", "single")).lower() != "single":
            return None
        question = str(row["question"])
        choices = [str(row[key]) for key in ("opa", "opb", "opc", "opd")]
        answer_index = int(row["cop"])
        source_id = row.get("id")
    elif name == "strategyqa":
        question = str(row["question"])
        choices = ["Yes", "No"]
        answer = row["answer"]
        if isinstance(answer, bool):
            answer_index = 0 if answer else 1
        elif str(answer).strip().lower() in {"yes", "true", "1"}:
            answer_index = 0
        elif str(answer).strip().lower() in {"no", "false", "0"}:
            answer_index = 1
        else:
            raise ValueError(f"unrecognized StrategyQA answer: {answer!r}")
        source_id = row.get("qid") or row.get("id")
    else:  # mmlu
        question = str(row["question"])
        choices = [str(value) for value in row["choices"]]
        answer = row["answer"]
        answer_index = "ABCD".index(answer.upper()) if isinstance(answer, str) and answer.upper() in "ABCD" else int(answer)
        source_id = row.get("id")
    if len(choices) not in (2, 4) or not 0 <= answer_index < len(choices):
        raise ValueError(f"{name} row {index}: invalid options or gold answer")
    if not question.strip() or any(not choice.strip() for choice in choices):
        raise ValueError(f"{name} row {index}: blank question or option")
    source_id = str(source_id) if source_id is not None else _digest({"q": question, "choices": choices})[:20]
    item_id = f"{name}:{source_split}:{source_id}"
    shuffled = [(choice, pos == answer_index) for pos, choice in enumerate(choices)]
    random.Random(f"{seed}:{item_id}").shuffle(shuffled)
    keys = "ABCD"[:len(shuffled)]
    options = {key: pair[0] for key, pair in zip(keys, shuffled)}
    gold_key = next(key for key, pair in zip(keys, shuffled) if pair[1])
    return {
        "item_id": item_id,
        "domain": name,
        "split": "",  # Set only after the frozen calibration/test sample is chosen.
        "condition_id": "canonical_order",
        "question": question,
        "options": options,
        "option_order": list(keys),
        "gold_key": gold_key,
        "semantic_hash": _digest({"q": question.strip().casefold(),
                                  "choices": sorted(choice.strip().casefold() for choice in choices)}),
        "source": {"dataset": name, "url": DATASET_URLS[name],
                   "source_split": source_split, "source_id": source_id},
    }


def freeze_dataset(dataset: str, calibration_rows: list[dict[str, Any]],
                   test_rows: list[dict[str, Any]], *, calibration_source_split: str,
                   test_source_split: str, revision: str, n_calibration: int,
                   n_test: int, seed: int, order_items: int = 0,
                   source_url: str | None = None) -> list[dict[str, Any]]:
    if not revision or revision.lower() in {"main", "latest"}:
        raise ValueError("pin a source revision rather than using main/latest")
    if min(n_calibration, n_test) < 1 or not 0 <= order_items <= n_test:
        raise ValueError("sample counts must be positive; order_items must be between 0 and n_test")
    prepared: list[list[dict[str, Any]]] = []
    for rows, source_split in ((calibration_rows, calibration_source_split), (test_rows, test_source_split)):
        normalized = []
        for index, row in enumerate(rows):
            item = normalize_row(dataset, row, source_split, index, seed)
            if item is not None:
                item["source"]["revision"] = revision
                if source_url:
                    item["source"]["url"] = source_url
                normalized.append(item)
        prepared.append(normalized)
    semantic_seen: set[str] = set()
    selected: list[list[dict[str, Any]]] = []
    for rows, count, split in zip(prepared, (n_calibration, n_test), ("calibration", "test")):
        candidates = sorted(rows, key=lambda item: item["item_id"])
        random.Random(f"{seed}:{dataset}:{split}").shuffle(candidates)
        chosen = []
        for item in candidates:
            if item["semantic_hash"] in semantic_seen:
                continue
            semantic_seen.add(item["semantic_hash"])
            item["split"] = split
            chosen.append(item)
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValidationError(f"{dataset}: requested {count} {split} items, found {len(chosen)} unique labeled items")
        selected.append(sorted(chosen, key=lambda item: item["item_id"]))
    frozen = selected[0] + selected[1]
    order_sample = random.Random(f"{seed}:{dataset}:order").sample(selected[1], order_items)
    for item in sorted(order_sample, key=lambda row: row["item_id"]):
        permuted = {**item, "condition_id": "reversed_order",
                    "option_order": list(reversed(item["option_order"]))}
        frozen.append(permuted)
    validate_items(frozen)
    return frozen


def validate_items(items: list[dict[str, Any]]) -> None:
    if not items:
        raise ValidationError("no frozen items")
    keys: set[tuple[str, str]] = set()
    item_metadata: dict[str, tuple[str, str, str, str]] = {}
    for index, item in enumerate(items, 1):
        required = {"item_id", "domain", "split", "condition_id", "question", "options",
                    "option_order", "gold_key", "source"}
        if missing := required - set(item):
            raise ValidationError(f"item {index}: missing {sorted(missing)}")
        if item["split"] not in {"calibration", "test"}:
            raise ValidationError(f"item {index}: invalid split")
        if item["gold_key"] not in item["options"]:
            raise ValidationError(f"item {index}: gold key missing from options")
        if len(item["option_order"]) != len(item["options"]) or set(item["option_order"]) != set(item["options"]):
            raise ValidationError(f"item {index}: option_order does not match options")
        key = (item["item_id"], item["condition_id"])
        if key in keys:
            raise ValidationError(f"item {index}: duplicate item/condition {key}")
        keys.add(key)
        stable = (item["domain"], item["split"], item["gold_key"], _digest(item["options"]))
        previous = item_metadata.setdefault(item["item_id"], stable)
        if previous != stable:
            raise ValidationError(f"item {index}: item identity changed across conditions")


def write_frozen(path: str | Path, items: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=False)
    write_jsonl(target / "items.jsonl", items)
    manifest = {**manifest, "item_count": len(items),
                "items_sha256": hashlib.sha256((target / "items.jsonl").read_bytes()).hexdigest()}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def combine_frozen(paths: list[str | Path]) -> list[dict[str, Any]]:
    combined = [item for path in paths for item in read_jsonl(path)]
    validate_items(combined)
    semantic_owner: dict[str, tuple[str, str]] = {}
    for item in combined:
        semantic_hash = item.get("semantic_hash")
        if semantic_hash:
            previous = semantic_owner.setdefault(semantic_hash, (item["item_id"], item["split"]))
            if previous != (item["item_id"], item["split"]):
                raise ValidationError("semantically identical item appears more than once across frozen datasets")
    return sorted(combined, key=lambda item: (item["domain"], item["item_id"], item["condition_id"]))
