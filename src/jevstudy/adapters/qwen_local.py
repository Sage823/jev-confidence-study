"""Qwen3 local inference: fixed answer, confidence, then optional samples.

Transformers and Torch are imported only when a real local run is requested.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Protocol

from ..signals import extract_signals


def choice_prompt(item: dict[str, Any], instruction: str = "Answer with exactly one option letter and no explanation.") -> str:
    lines = [str(item["question"]), "Options:"]
    lines.extend(f"{key}. {item['options'][key]}" for key in item["option_order"])
    lines.append(instruction)
    return "\n".join(lines)


def parse_probability(text: str) -> float | None:
    stripped = text.strip()
    try:
        value = float(stripped)
    except ValueError:
        return None
    return value if math.isfinite(value) and 0 <= value <= 1 else None


def parse_final_choice(text: str, keys: list[str]) -> str | None:
    matches = re.findall(r"\bFINAL\s*:\s*([A-Z])\b", text.upper())
    return matches[-1] if matches and matches[-1] in keys else None


def majority_vote(samples: list[str | None], keys: list[str]) -> str | None:
    if not samples or any(sample not in keys for sample in samples):
        return None
    counts = Counter(samples)
    return min(keys, key=lambda key: (-counts[key], keys.index(key)))


class QwenEngine(Protocol):
    model_id: str
    revision: str

    def score_labels(self, prompt: str, keys: list[str]) -> tuple[dict[str, float], int]: ...
    def generate_text(self, prompt: str, *, thinking: bool, sample: bool,
                      seed: int, max_new_tokens: int) -> tuple[str, int, int]: ...


class TransformersQwenEngine:
    """Load one pinned Qwen3 checkpoint using Transformers on the local machine."""

    def __init__(self, model_id: str, revision: str):
        if model_id not in {"Qwen/Qwen3-8B", "Qwen/Qwen3-32B"}:
            raise ValueError("model_id must be Qwen/Qwen3-8B or Qwen/Qwen3-32B")
        if not revision or revision.lower() in {"main", "latest"}:
            raise ValueError("pin a model commit or release revision")
        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install the optional Qwen dependencies: python -m pip install -e '.[qwen]'") from exc
        self.torch = torch
        self.model_id = model_id
        self.revision = revision
        self.torch_version = torch.__version__
        self.transformers_version = transformers.__version__
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto")
        self.model.eval()

    def _inputs(self, prompt: str, thinking: bool) -> Any:
        messages = [{"role": "user", "content": prompt}]
        chat = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        return self.tokenizer(chat, return_tensors="pt").to(self.model.device)

    def score_labels(self, prompt: str, keys: list[str]) -> tuple[dict[str, float], int]:
        token_ids = {}
        for key in keys:
            ids = self.tokenizer.encode(key, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"option code {key!r} is not one token for this Qwen tokenizer")
            token_ids[key] = ids[0]
        inputs = self._inputs(prompt, thinking=False)
        with self.torch.inference_mode():
            next_logits = self.model(**inputs).logits[0, -1]
        return {key: float(next_logits[token_id]) for key, token_id in token_ids.items()}, int(inputs.input_ids.shape[-1])

    def generate_text(self, prompt: str, *, thinking: bool, sample: bool,
                      seed: int, max_new_tokens: int) -> tuple[str, int, int]:
        inputs = self._inputs(prompt, thinking=thinking)
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)
        kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": sample,
                                  "pad_token_id": self.tokenizer.eos_token_id}
        if sample:
            kwargs.update({"temperature": 0.6, "top_p": 0.95, "top_k": 20})
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        input_count = int(inputs.input_ids.shape[-1])
        generated = output[0][input_count:]
        return (self.tokenizer.decode(generated, skip_special_tokens=True),
                input_count, int(generated.shape[-1]))


def run_one_qwen(item: dict[str, Any], engine: QwenEngine, run_id: str, *,
                 sample_count: int = 20, self_report: bool = True,
                 seed: int = 4901, max_sample_tokens: int = 512) -> dict[str, Any]:
    if sample_count not in {0, 5, 10, 20}:
        raise ValueError("sample_count must be 0, 5, 10, or 20")
    keys = list(item["option_order"])
    prompt = choice_prompt(item)
    start = time.perf_counter()
    logits, base_input_tokens = engine.score_labels(prompt, keys)
    if set(logits) != set(keys) or any(not math.isfinite(value) for value in logits.values()):
        raise ValueError("Qwen option logits must contain one finite value per option key")
    peak = max(logits.values())
    weights = {key: math.exp(value - peak) for key, value in logits.items()}
    total = sum(weights.values())
    probs = {key: value / total for key, value in weights.items()}
    answer_key = max(keys, key=lambda key: probs[key])  # Fixed before any extra call.
    cost_by_signal: dict[str, dict[str, int]] = {
        "native_probability": {"input_tokens": base_input_tokens, "output_tokens": 0}}
    record = {
        "run_id": run_id, "source_kind": "model_output", "source": item.get("source", {}),
        "item_id": item["item_id"], "domain": item["domain"], "split": item["split"],
        "question": item["question"], "options": item["options"], "gold_key": item["gold_key"],
        "model_id": engine.model_id, "model_revision": engine.revision,
        "condition_id": item.get("condition_id", "canonical_order"),
        "option_order": keys, "answer_key": answer_key, "option_probabilities": probs,
        "raw_option_logits": logits, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "signal_token_usage": cost_by_signal,
    }
    protocol = {"model": engine.model_id, "revision": engine.revision, "prompt": prompt,
                "samples": sample_count, "self_report": self_report,
                "max_sample_tokens": max_sample_tokens, "seed": seed}
    record["protocol_hash"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if self_report:
        report_prompt = choice_prompt(
            item, f"A separate answer is fixed as {answer_key}. Estimate the probability that this "
                  "answer is correct. Output only a decimal between 0 and 1.")
        raw, used_input, used_output = engine.generate_text(
            report_prompt, thinking=False, sample=False, seed=seed, max_new_tokens=24)
        record["self_report_raw"] = raw
        parsed = parse_probability(raw)
        if parsed is not None:
            record["self_report_probability"] = parsed
        else:
            record["self_report_error"] = "invalid probability output"
        cost_by_signal["self_report"] = {"input_tokens": used_input, "output_tokens": used_output}
    samples: list[str | None] = []
    trace_usage: list[dict[str, int]] = []
    raw_traces: list[str] = []
    if sample_count:
        sample_prompt = choice_prompt(item, "Reason briefly and finish with FINAL: <option letter>.")
        for index in range(sample_count):
            raw, used_input, used_output = engine.generate_text(
                sample_prompt, thinking=True, sample=True, seed=seed + index,
                max_new_tokens=max_sample_tokens)
            raw_traces.append(raw)
            samples.append(parse_final_choice(raw, keys))
            trace_usage.append({"input_tokens": used_input, "output_tokens": used_output})
        record["sampled_answer_keys"] = samples
        record["sampled_raw_traces"] = raw_traces
        record["sample_token_usage"] = trace_usage
        record["sample_seeds"] = [seed + index for index in range(sample_count)]
        record["majority_vote_answers"] = {
            str(budget): majority_vote(samples[:budget], keys)
            for budget in (5, 10, 20) if sample_count >= budget}
        for budget in (5, 10, 20):
            if sample_count >= budget:
                cost_by_signal[f"sample_consistency_{budget}"] = {
                    "input_tokens": sum(usage["input_tokens"] for usage in trace_usage[:budget]),
                    "output_tokens": sum(usage["output_tokens"] for usage in trace_usage[:budget])}
    record["input_tokens"] = (base_input_tokens
                              + cost_by_signal.get("self_report", {}).get("input_tokens", 0)
                              + sum(usage["input_tokens"] for usage in trace_usage))
    record["output_tokens"] = (cost_by_signal.get("self_report", {}).get("output_tokens", 0)
                               + sum(usage["output_tokens"] for usage in trace_usage))
    record["latency_ms"] = round((time.perf_counter() - start) * 1000, 3)
    record["signals"] = extract_signals(record)
    record["correct"] = int(answer_key == item["gold_key"])
    return record
