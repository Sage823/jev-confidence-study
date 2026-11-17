"""Small dependency-free metrics with explicit conventions."""

from __future__ import annotations

import math
from collections.abc import Sequence


def accuracy(correct: Sequence[int]) -> float:
    if not correct:
        return math.nan
    return sum(correct) / len(correct)


def brier_multiclass(probabilities: Sequence[dict[str, float]], gold_keys: Sequence[str]) -> float:
    if len(probabilities) != len(gold_keys) or not probabilities:
        raise ValueError("probabilities and gold_keys must have equal nonzero length")
    values = []
    for probs, gold in zip(probabilities, gold_keys):
        if gold not in probs:
            raise ValueError("gold key missing from probability mapping")
        values.append(sum((p - float(key == gold)) ** 2 for key, p in probs.items()))
    return sum(values) / len(values)


def nll(probabilities: Sequence[dict[str, float]], gold_keys: Sequence[str], eps: float = 1e-15) -> float:
    if len(probabilities) != len(gold_keys) or not probabilities:
        raise ValueError("probabilities and gold_keys must have equal nonzero length")
    return -sum(math.log(max(eps, probs[gold])) for probs, gold in zip(probabilities, gold_keys)) / len(gold_keys)


def ece(confidence: Sequence[float], correct: Sequence[int], bins: int = 10,
        strategy: str = "equal_width") -> float:
    if len(confidence) != len(correct) or not confidence:
        raise ValueError("confidence and correct must have equal nonzero length")
    if bins < 1 or any(not 0 <= c <= 1 for c in confidence):
        raise ValueError("bins must be positive and confidence values must be in [0, 1]")
    n = len(confidence)
    if strategy == "equal_width":
        groups = [[] for _ in range(bins)]
        for i, c in enumerate(confidence):
            groups[min(bins - 1, int(c * bins))].append(i)
    elif strategy == "equal_frequency":
        order = sorted(range(n), key=lambda i: (confidence[i], i))
        chunk_size = (n + bins - 1) // bins
        groups = [list(order[start:start + chunk_size])
                  for start in range(0, n, chunk_size)]
    else:
        raise ValueError("strategy must be equal_width or equal_frequency")
    return sum(len(g) / n * abs(sum(confidence[i] for i in g) / len(g)
                                - sum(correct[i] for i in g) / len(g)) for g in groups if g)


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUC for positive label 1; ties receive half credit."""
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must have equal nonzero length")
    positives = sum(int(y == 1) for y in labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return math.nan
    ordered = sorted(zip(scores, labels), key=lambda pair: pair[0])
    positive_ranks = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        mean_rank = (start + 1 + end) / 2
        positive_ranks += mean_rank * sum(label == 1 for _, label in ordered[start:end])
        start = end
    return (positive_ranks - positives * (positives + 1) / 2) / (positives * negatives)


def spearman(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("score lists must have equal length of at least two")
    rank_a, rank_b = _ranks(first), _ranks(second)
    mean_a = sum(rank_a) / len(rank_a)
    mean_b = sum(rank_b) / len(rank_b)
    numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(rank_a, rank_b))
    denom_a = sum((a - mean_a) ** 2 for a in rank_a)
    denom_b = sum((b - mean_b) ** 2 for b in rank_b)
    return numerator / math.sqrt(denom_a * denom_b) if denom_a and denom_b else math.nan


def _ranks(values: Sequence[float]) -> list[float]:
    result = [0.0] * len(values)
    order = sorted(range(len(values)), key=lambda index: values[index])
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        mean_rank = (start + 1 + end) / 2
        for index in order[start:end]:
            result[index] = mean_rank
        start = end
    return result


def risk_coverage(scores: Sequence[float], correct: Sequence[int]) -> list[tuple[float, float]]:
    if len(scores) != len(correct) or not scores:
        raise ValueError("scores and correct must have equal nonzero length")
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    points = []
    errors = 0
    for count, index in enumerate(order, start=1):
        errors += int(not correct[index])
        points.append((count / len(order), errors / count))
    return points


def aurc(scores: Sequence[float], correct: Sequence[int]) -> float:
    """Mean selective risk over all nonzero coverages (discrete convention)."""
    points = risk_coverage(scores, correct)
    return sum(risk for _, risk in points) / len(points)


def risk_at_coverage(scores: Sequence[float], correct: Sequence[int], coverage: float) -> float:
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    points = risk_coverage(scores, correct)
    index = max(1, math.ceil(coverage * len(points))) - 1
    return points[index][1]


def order_agreement(first: Sequence[str], second: Sequence[str]) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("answer lists must have equal nonzero length")
    return sum(a == b for a, b in zip(first, second)) / len(first)

