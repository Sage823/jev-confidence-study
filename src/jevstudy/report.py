"""Evaluate fixed-answer confidence; fit calibration on calibration rows only."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from .calibration import apply_scalar_logistic, apply_temperature, fit_scalar_logistic, fit_temperature
from .metrics import accuracy, auroc, aurc, brier_multiclass, ece, nll, order_agreement, risk_at_coverage, risk_coverage, spearman
from .schema import ValidationError, validate_records


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _mean(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None and math.isfinite(value)]
    return sum(available) / len(available) if available else None


def _score_stats(scores: list[float], labels: list[int]) -> dict[str, Any]:
    bins = {f"{method}_{count}": ece(scores, labels, bins=count, strategy=method)
            for method in ("equal_width", "equal_frequency") for count in (10, 15)}
    reliability = []
    for bin_index in range(10):
        selected = [i for i, score in enumerate(scores) if min(9, int(score * 10)) == bin_index]
        if selected:
            reliability.append({"bin": bin_index, "n": len(selected),
                                "mean_confidence": sum(scores[i] for i in selected) / len(selected),
                                "accuracy": sum(labels[i] for i in selected) / len(selected)})
    return {"binary_brier": sum((score - label) ** 2 for score, label in zip(scores, labels)) / len(scores),
            "auroc_correct_positive": _finite(auroc(scores, labels)),
            "aurc_lower_is_better": aurc(scores, labels), "ece": bins,
            "risk_at_80pct_coverage": risk_at_coverage(scores, labels, 0.8),
            "reliability_equal_width_10": reliability,
            "risk_coverage": [{"coverage": coverage, "risk": risk}
                              for coverage, risk in risk_coverage(scores, labels)]}


def _calibration_threshold(train_scores: list[float], test_scores: list[float],
                           test_labels: list[int], target_coverage: float = 0.8) -> dict[str, Any]:
    if not train_scores:
        return {"status": "unavailable", "reason": "no calibration scores"}
    rank = max(1, math.ceil(target_coverage * len(train_scores))) - 1
    threshold = sorted(train_scores, reverse=True)[rank]
    selected = [label for score, label in zip(test_scores, test_labels) if score >= threshold]
    return {"threshold": threshold, "target_calibration_coverage": target_coverage,
            "test_coverage": len(selected) / len(test_scores),
            "test_error": 1 - accuracy(selected) if selected else None}


def _fit_sets(calibration: dict[tuple[str, str], list[dict[str, Any]]],
              model: str, domain: str) -> dict[str, list[dict[str, Any]]]:
    local = calibration.get((model, domain), [])
    other = [row for (candidate, source_domain), rows in calibration.items()
             if candidate == model and source_domain != domain for row in rows]
    return {"in_domain": local, "cross_domain": other, "joint": local + other}


def _calibrate_signal(test: list[dict[str, Any]], fit: list[dict[str, Any]], name: str) -> dict[str, Any]:
    fit = [row for row in fit if name in row.get("signals", {})]
    if not fit:
        return {"status": "unavailable", "reason": "no calibration rows with signal", "n_calibration": 0}
    try:
        parameters = fit_scalar_logistic(fit, name)
    except ValueError as exc:
        return {"status": "unavailable", "reason": str(exc), "n_calibration": len(fit)}
    scores = [apply_scalar_logistic(float(row["signals"][name]), parameters) for row in test]
    labels = [int(row["answer_key"] == row["gold_key"]) for row in test]
    train_scores = [apply_scalar_logistic(float(row["signals"][name]), parameters) for row in fit]
    train_labels = [int(row["answer_key"] == row["gold_key"]) for row in fit]
    return {"method": "monotone_logistic", "n_calibration": len(fit),
            "fit_domains": sorted({row["domain"] for row in fit}), "parameters": parameters,
            "in_sample_fit_diagnostic": {
                "binary_brier": sum((score - label) ** 2 for score, label in zip(train_scores, train_labels)) / len(fit),
                "ece_equal_width_10": ece(train_scores, train_labels, bins=10)},
            "calibration_fitted_threshold_80pct": _calibration_threshold(train_scores, scores, labels),
            **_score_stats(scores, labels)}


def _calibrate_distribution(test: list[dict[str, Any]], fit: list[dict[str, Any]]) -> dict[str, Any]:
    if not fit:
        return {"status": "unavailable", "reason": "no calibration rows", "n_calibration": 0}
    temperature = fit_temperature(fit)
    probabilities = [apply_temperature(row["option_probabilities"], temperature) for row in test]
    train_probabilities = [apply_temperature(row["option_probabilities"], temperature) for row in fit]
    return {"method": "temperature_scaling", "temperature": temperature,
            "n_calibration": len(fit), "fit_domains": sorted({row["domain"] for row in fit}),
            "in_sample_fit_diagnostic": {
                "multiclass_brier": brier_multiclass(train_probabilities, [row["gold_key"] for row in fit]),
                "multiclass_nll": nll(train_probabilities, [row["gold_key"] for row in fit])},
            "held_out_multiclass_brier": brier_multiclass(probabilities, [row["gold_key"] for row in test]),
            "held_out_multiclass_nll": nll(probabilities, [row["gold_key"] for row in test])}


def _audit_resources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for field in ("latency_ms", "input_tokens", "output_tokens", "cost_usd"):
        values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
        result[field] = {"n_available": len(values), "mean": _mean(values),
                         "total": sum(values) if values else None}
    return result


def _majority(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for budget in (5, 10, 20):
        valid = [row for row in rows if row.get("majority_vote_answers", {}).get(str(budget)) in row["options"]]
        result[str(budget)] = {"n": len(valid), "accuracy": accuracy([
            int(row["majority_vote_answers"][str(budget)] == row["gold_key"]) for row in valid]) if valid else None}
    return result


def _correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = sorted({name for row in rows for name in row.get("signals", {})})
    result = []
    for first, second in combinations(names, 2):
        paired = [row for row in rows if first in row.get("signals", {}) and second in row["signals"]]
        if len(paired) > 1:
            result.append({"first": first, "second": second, "n": len(paired),
                           "spearman": _finite(spearman([row["signals"][first] for row in paired],
                                                         [row["signals"][second] for row in paired]))})
    return result


def _paired_bootstrap(first: dict[str, dict[str, Any]], second: dict[str, dict[str, Any]],
                      draws: int = 500, seed: int = 4901) -> dict[str, Any] | None:
    shared = sorted(set(first) & set(second))
    if not shared:
        return None
    differences = [int(first[key]["answer_key"] == first[key]["gold_key"])
                   - int(second[key]["answer_key"] == second[key]["gold_key"]) for key in shared]
    rng = random.Random(seed)
    estimates = sorted(sum(rng.choice(differences) for _ in differences) / len(shared)
                       for _ in range(draws))
    return {"n_paired_items": len(shared), "n_first": len(first), "n_second": len(second),
            "first_minus_second": sum(differences) / len(shared),
            "ci_95_percentile": [estimates[int(draws * 0.025)], estimates[min(draws - 1, int(draws * 0.975))]],
            "draws": draws, "seed": seed}


def _paired_signal_bootstrap(first: list[float], second: list[float], labels: list[int],
                             draws: int, seed: int = 4901,
                             second_labels: list[int] | None = None) -> dict[str, Any]:
    if not first or len(first) != len(second) or len(first) != len(labels):
        raise ValueError("paired signal vectors must have equal nonzero length")
    second_labels = labels if second_labels is None else second_labels
    if len(second_labels) != len(labels):
        raise ValueError("second_labels must match paired items")
    methods = {"auroc": lambda x, y: auroc(x, y),
               "aurc": lambda x, y: aurc(x, y),
               "ece_equal_width_10": lambda x, y: ece(x, y, bins=10)}
    rng = random.Random(seed)
    estimates: dict[str, list[float]] = {name: [] for name in methods}
    for _ in range(draws):
        indices = [rng.randrange(len(labels)) for _ in labels]
        sample_first = [first[i] for i in indices]
        sample_second = [second[i] for i in indices]
        sample_labels = [labels[i] for i in indices]
        sample_second_labels = [second_labels[i] for i in indices]
        for name, metric in methods.items():
            difference = metric(sample_first, sample_labels) - metric(sample_second, sample_second_labels)
            if math.isfinite(difference):
                estimates[name].append(difference)
    result = {"n_paired_items": len(labels), "draws": draws, "seed": seed}
    for name, metric in methods.items():
        values = sorted(estimates[name])
        result[name] = {"first_minus_second": _finite(metric(first, labels) - metric(second, second_labels)),
                        "ci_95_percentile": [
                            values[int(0.025 * len(values))],
                            values[min(len(values) - 1, int(0.975 * len(values)))]] if values else None,
                        "valid_draws": len(values)}
    return result


def analyze_records(records: list[dict[str, Any]], *, allow_incomplete: bool = False,
                    bootstrap_draws: int = 0) -> dict[str, Any]:
    if bootstrap_draws < 0:
        raise ValueError("bootstrap_draws must be nonnegative")
    validate_records(records)
    source_kinds = {row["source_kind"] for row in records}
    if "simulation" in source_kinds and len(source_kinds) > 1:
        raise ValidationError("simulation and real model outputs cannot be analyzed together")
    identities: dict[str, tuple[str, str, str, str]] = {}
    seen: set[tuple[str, str, str]] = set()
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    calibration: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key = (row["model_id"], row["item_id"], row["condition_id"])
        if key in seen:
            raise ValidationError(f"duplicate model/item/condition across runs: {key}")
        seen.add(key)
        identity = (row["domain"], row["split"], row["gold_key"],
                    json.dumps([row.get("question"), row["options"]], sort_keys=True, ensure_ascii=False))
        if row["item_id"] in identities and identities[row["item_id"]] != identity:
            raise ValidationError(f"item content or gold changed across models: {row['item_id']}")
        identities[row["item_id"]] = identity
        if row["split"] == "test":
            groups[(row["model_id"], row["domain"], row["condition_id"])].append(row)
        elif row["condition_id"] == "canonical_order":
            calibration[(row["model_id"], row["domain"])].append(row)
    model_items: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for row in records:
        if row["condition_id"] == "canonical_order":
            model_items[row["model_id"]].add((row["domain"], row["split"], row["item_id"]))
    if len(model_items) > 1 and not allow_incomplete:
        first_set = next(iter(model_items.values()))
        if any(items != first_set for items in model_items.values()):
            raise ValidationError("model item sets differ; inspect failed runs or use --allow-incomplete for a pilot")

    results = []
    for (model, domain, condition), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda row: row["item_id"])
        labels = [int(row["answer_key"] == row["gold_key"]) for row in rows]
        fit_sets = _fit_sets(calibration, model, domain)
        signals = {}
        for name in sorted({name for row in rows for name in row.get("signals", {})}):
            available = [row for row in rows if name in row.get("signals", {})]
            scores = [float(row["signals"][name]) for row in available]
            if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
                raise ValidationError(f"invalid signal {name} in {model}/{domain}")
            y = [int(row["answer_key"] == row["gold_key"]) for row in available]
            fits = {mode: _calibrate_signal(available, fit, name) for mode, fit in fit_sets.items()}
            same_domain_scores = [float(row["signals"][name]) for row in fit_sets["in_domain"]
                                  if name in row.get("signals", {})]
            signals[name] = {"n_available": len(available), "n_missing": len(rows) - len(available),
                             **_score_stats(scores, y), "calibration": fits,
                             "raw_calibration_fitted_threshold_80pct":
                                 _calibration_threshold(same_domain_scores, scores, y),
                             "held_out_calibration": fits["in_domain"]}
        distribution = {mode: _calibrate_distribution(rows, fit) for mode, fit in fit_sets.items()}
        comparisons = []
        if bootstrap_draws and condition == "canonical_order":
            compared_names = [name for name in ("native_probability", "entropy_confidence",
                                                "self_report", "sample_consistency_20")
                              if name in signals]
            for first, second in combinations(compared_names, 2):
                paired = [row for row in rows if first in row.get("signals", {})
                          and second in row["signals"]]
                if paired:
                    comparisons.append({"first_signal": first, "second_signal": second,
                                        **_paired_signal_bootstrap(
                                            [row["signals"][first] for row in paired],
                                            [row["signals"][second] for row in paired],
                                            [int(row["answer_key"] == row["gold_key"]) for row in paired],
                                            bootstrap_draws)})
        results.append({"model_id": model, "domain": domain, "condition_id": condition,
                        "n_test": len(rows), "accuracy": accuracy(labels),
                        "multiclass_brier": brier_multiclass(
                            [row["option_probabilities"] for row in rows], [row["gold_key"] for row in rows]),
                        "multiclass_nll": nll(
                            [row["option_probabilities"] for row in rows], [row["gold_key"] for row in rows]),
                        "distribution_calibration": distribution, "signals": signals,
                        "signal_correlations": _correlations(rows),
                        "paired_signal_comparisons": comparisons,
                        "resource_use": _audit_resources(rows),
                        "majority_vote": _majority(rows),
                        "sample_audit": {
                            "total_samples": sum(len(row.get("sampled_answer_keys", [])) for row in rows),
                            "invalid_samples": sum(sum(key not in row["options"]
                                                       for key in row.get("sampled_answer_keys", [])) for row in rows)},
                        "case_ids": {
                            "confident_errors": [row["item_id"] for row in rows
                                                 if row["answer_key"] != row["gold_key"]
                                                 and row["signals"].get("native_probability", 0) >= 0.8][:100],
                            "uncertain_correct": [row["item_id"] for row in rows
                                                  if row["answer_key"] == row["gold_key"]
                                                  and row["signals"].get("native_probability", 1) <= 0.4][:100]}})

    paired_order: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    by_item: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        if row["split"] == "test":
            by_item[(row["model_id"], row["domain"], row["item_id"])][row["condition_id"]] = row
    for (model, domain, _), conditions in by_item.items():
        if "canonical_order" in conditions and "reversed_order" in conditions:
            paired_order[(model, domain)].append((conditions["canonical_order"], conditions["reversed_order"]))
    stability = [{"model_id": model, "domain": domain, "n_paired_test_items": len(pairs),
                  "canonical_vs_reversed_agreement": order_agreement(
                      [first["answer_key"] for first, _ in pairs], [second["answer_key"] for _, second in pairs]),
                  "mean_abs_fixed_option_probability_change": _mean([
                      abs(first["option_probabilities"][first["answer_key"]]
                          - second["option_probabilities"][first["answer_key"]]) for first, second in pairs])}
                 for (model, domain), pairs in sorted(paired_order.items())]

    macro_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        macro_groups[(row["model_id"], row["condition_id"])].append(row)
    macro = []
    for (model, condition), rows in sorted(macro_groups.items()):
        names = sorted({name for row in rows for name in row["signals"]})
        macro.append({"model_id": model, "condition_id": condition, "n_domains": len(rows),
                      "averaging": "equal_weight_domains",
                      **{metric: _mean([row[metric] for row in rows])
                         for metric in ("accuracy", "multiclass_brier", "multiclass_nll")},
                      "signals": {name: {metric: _mean([
                          row["signals"][name][metric] if name in row["signals"] else None for row in rows])
                          for metric in ("auroc_correct_positive", "aurc_lower_is_better", "risk_at_80pct_coverage")}
                          for name in names}})
    canonical: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        if row["split"] == "test" and row["condition_id"] == "canonical_order":
            canonical[(row["domain"], row["model_id"])][row["item_id"]] = row
    model_pairs = []
    disagreements: dict[str, list[str]] = defaultdict(list)
    for domain in sorted({domain for domain, _ in canonical}):
        models = sorted(model for candidate_domain, model in canonical if candidate_domain == domain)
        by_id: dict[str, set[str]] = defaultdict(set)
        for model in models:
            for item_id, row in canonical[(domain, model)].items():
                by_id[item_id].add(row["answer_key"])
        disagreements[domain] = [item_id for item_id, answers in sorted(by_id.items())
                                 if len(answers) > 1][:100]
        for first, second in combinations(models, 2):
            estimate = _paired_bootstrap(canonical[(domain, first)], canonical[(domain, second)])
            if estimate:
                if bootstrap_draws:
                    common = sorted(set(canonical[(domain, first)]) & set(canonical[(domain, second)]))
                    paired = [(canonical[(domain, first)][key], canonical[(domain, second)][key])
                              for key in common if "native_probability" in canonical[(domain, first)][key].get("signals", {})
                              and "native_probability" in canonical[(domain, second)][key].get("signals", {})]
                    if paired:
                        estimate["native_probability_comparison"] = _paired_signal_bootstrap(
                            [a["signals"]["native_probability"] for a, _ in paired],
                            [b["signals"]["native_probability"] for _, b in paired],
                            [int(a["answer_key"] == a["gold_key"]) for a, _ in paired],
                            bootstrap_draws,
                            second_labels=[int(b["answer_key"] == b["gold_key"]) for _, b in paired])
                model_pairs.append({"domain": domain, "first_model": first, "second_model": second, **estimate})

    synthetic = all(row["source_kind"] == "simulation" for row in records)
    report = {"source_kind": "simulation" if synthetic else "model_output",
              "groups": results, "domain_macro_averages": macro, "order_stability": stability,
              "paired_model_accuracy": model_pairs,
              "model_item_counts": {model: len(items) for model, items in sorted(model_items.items())},
              "alignment": "complete" if len({frozenset(items) for items in model_items.values()}) == 1
                           else "incomplete",
              "model_disagreement_case_ids": disagreements,
              "metric_conventions": {"target": "fixed answer correctness",
                                     "aurc": "mean selective risk over all nonzero coverages",
                                     "calibration_fit": "canonical calibration rows only"}}
    if synthetic:
        report["warning"] = "Synthetic output is for pipeline validation only; it is not empirical evidence."
    return report


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
