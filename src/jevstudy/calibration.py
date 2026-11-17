"""Calibration transforms trained only from explicitly supplied calibration rows."""

from __future__ import annotations

import math
from typing import Any


def fit_scalar_logistic(rows: list[dict[str, Any]], signal_name: str,
                        iterations: int = 4000, learning_rate: float = 0.08,
                        l2: float = 1e-3) -> dict[str, float]:
    if not rows or any(row["split"] != "calibration" for row in rows):
        raise ValueError("fit_scalar_logistic accepts calibration rows only")
    xs = [float(row["signals"][signal_name]) for row in rows]
    ys = [int(row["correct"]) for row in rows]
    if len(set(ys)) < 2:
        raise ValueError("calibration rows must contain both correct and incorrect answers")
    # Center/scale for stable optimization. Project slope to nonnegative to preserve ranking.
    mean_x = sum(xs) / len(xs)
    scale = math.sqrt(sum((x - mean_x) ** 2 for x in xs) / len(xs)) or 1.0
    zs = [(x - mean_x) / scale for x in xs]
    intercept, slope = 0.0, 0.0
    for _ in range(iterations):
        grad_b = grad_w = 0.0
        for z, y in zip(zs, ys):
            p = _sigmoid(intercept + slope * z)
            grad_b += p - y
            grad_w += (p - y) * z
        grad_b /= len(rows)
        grad_w = grad_w / len(rows) + l2 * slope
        intercept -= learning_rate * grad_b
        slope = max(0.0, slope - learning_rate * grad_w)
    return {"intercept": intercept, "slope": slope, "mean": mean_x, "scale": scale}


def apply_scalar_logistic(value: float, params: dict[str, float]) -> float:
    z = (value - params["mean"]) / params["scale"]
    return _sigmoid(params["intercept"] + params["slope"] * z)


def fit_temperature(rows: list[dict[str, Any]], min_t: float = 0.25,
                    max_t: float = 4.0, steps: int = 300) -> float:
    if not rows or any(row["split"] != "calibration" for row in rows):
        raise ValueError("fit_temperature accepts calibration rows only")
    gold = [row["gold_key"] for row in rows]
    distributions = [row["option_probabilities"] for row in rows]
    # Uniform log-temperature grid is deterministic and sufficient for the scaffold.
    best_t, best_loss = 1.0, math.inf
    lo, hi = math.log(min_t), math.log(max_t)
    for step in range(steps + 1):
        temperature = math.exp(lo + (hi - lo) * step / steps)
        loss = 0.0
        for probs, target in zip(distributions, gold):
            logits = {key: math.log(max(1e-15, p)) / temperature for key, p in probs.items()}
            peak = max(logits.values())
            denom = sum(math.exp(value - peak) for value in logits.values())
            loss -= math.log(max(1e-15, math.exp(logits[target] - peak) / denom))
        loss /= len(rows)
        if loss < best_loss:
            best_t, best_loss = temperature, loss
    return best_t


def apply_temperature(probabilities: dict[str, float], temperature: float) -> dict[str, float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = {key: math.log(max(1e-15, value)) / temperature for key, value in probabilities.items()}
    peak = max(logits.values())
    exps = {key: math.exp(value - peak) for key, value in logits.items()}
    total = sum(exps.values())
    return {key: value / total for key, value in exps.items()}


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)

