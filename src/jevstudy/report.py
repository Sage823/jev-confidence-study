"""Summarize simulated or normalized records by model and domain."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .calibration import (apply_scalar_logistic, apply_temperature,
                          fit_scalar_logistic, fit_temperature)
from .metrics import (accuracy, auroc, aurc, brier_multiclass, ece, nll,
                      order_agreement, risk_at_coverage)


def analyze_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    calibration_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row["split"] == "test":
            groups[(row["model_id"], row["domain"], row["condition_id"])].append(row)
        elif row["split"] == "calibration":
            calibration_groups[(row["model_id"], row["domain"], row["condition_id"])].append(row)
    results = []
    for (model_id, domain, condition_id), rows in sorted(groups.items()):
        y = [int(row["correct"]) for row in rows]
        full_probs = [row["option_probabilities"] for row in rows]
        gold = [row["gold_key"] for row in rows]
        signal_names = sorted(set.intersection(*(set(row["signals"]) for row in rows)))
        signal_stats = {}
        calibration_rows = calibration_groups.get((model_id, domain, condition_id), [])
        for name in signal_names:
            scores = [float(row["signals"][name]) for row in rows]
            stats = {
                "auroc_correct_positive": _finite(auroc(scores, y)),
                "aurc_lower_is_better": aurc(scores, y),
                "ece": {f"{strategy}_{bins}": ece(scores, y, bins=bins, strategy=strategy)
                        for strategy in ("equal_width", "equal_frequency") for bins in (10, 15)},
                "risk_at_80pct_coverage": risk_at_coverage(scores, y, 0.8),
            }
            try:
                fit_rows = [{"split": row["split"], "signals": row["signals"], "correct": row["correct"]}
                            for row in calibration_rows]
                params = fit_scalar_logistic(fit_rows, name)
                calibrated = [apply_scalar_logistic(score, params) for score in scores]
                stats["held_out_calibration"] = {
                    "method": "monotone_logistic",
                    "n_calibration": len(fit_rows),
                    "binary_brier": sum((p - target) ** 2 for p, target in zip(calibrated, y)) / len(y),
                    "ece": {f"{strategy}_{bins}": ece(calibrated, y, bins=bins, strategy=strategy)
                            for strategy in ("equal_width", "equal_frequency") for bins in (10, 15)},
                }
            except (ValueError, KeyError):
                stats["held_out_calibration"] = {"status": "unavailable", "reason": "insufficient calibration rows/classes"}
            signal_stats[name] = stats
        calibrated_distribution = None
        try:
            temperature = fit_temperature(calibration_rows)
            calibrated_distribution = [apply_temperature(row["option_probabilities"], temperature) for row in rows]
            distribution_calibration = {
                "method": "temperature_scaling",
                "temperature": temperature,
                "n_calibration": len(calibration_rows),
                "held_out_multiclass_brier": brier_multiclass(calibrated_distribution, gold),
                "held_out_multiclass_nll": nll(calibrated_distribution, gold),
            }
        except (ValueError, KeyError):
            distribution_calibration = {"status": "unavailable", "reason": "insufficient calibration rows"}
        results.append({
            "model_id": model_id,
            "domain": domain,
            "condition_id": condition_id,
            "n_test": len(rows),
            "accuracy": accuracy(y),
            "multiclass_brier": brier_multiclass(full_probs, gold),
            "multiclass_nll": nll(full_probs, gold),
            "distribution_calibration": distribution_calibration,
            "signals": signal_stats,
            "sample_audit": {
                "total_samples": sum(len(row.get("sampled_answer_keys", [])) for row in rows),
                "invalid_samples": sum(sum(key not in row["options"] for key in row.get("sampled_answer_keys", []))
                                        for row in rows),
            },
        })
    order_groups: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    for row in records:
        if row["split"] == "test":
            order_groups[(row["model_id"], row["domain"], row["item_id"])][row["condition_id"]] = row["answer_key"]
    stability = []
    by_model_domain: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for (model_id, domain, _item_id), answers in order_groups.items():
        if "canonical_order" in answers and "reversed_order" in answers:
            by_model_domain[(model_id, domain)].append((answers["canonical_order"], answers["reversed_order"]))
    for (model_id, domain), pairs in sorted(by_model_domain.items()):
        stability.append({"model_id": model_id, "domain": domain, "n_paired_test_items": len(pairs),
                          "canonical_vs_reversed_agreement": order_agreement(
                              [pair[0] for pair in pairs], [pair[1] for pair in pairs])})
    macro_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        macro_groups[(result["model_id"], result["condition_id"])].append(result)
    macro_averages = []
    for (model_id, condition_id), domain_results in sorted(macro_groups.items()):
        mean_metrics = {
            metric: _mean([row[metric] for row in domain_results])
            for metric in ("accuracy", "multiclass_brier", "multiclass_nll")
        }
        common_signals = set.intersection(*(set(row["signals"]) for row in domain_results))
        signal_means = {}
        for signal_name in sorted(common_signals):
            signal_means[signal_name] = {
                metric: _mean([row["signals"][signal_name][metric] for row in domain_results])
                for metric in ("auroc_correct_positive", "aurc_lower_is_better", "risk_at_80pct_coverage")
            }
            signal_means[signal_name]["ece"] = {
                bin_name: _mean([row["signals"][signal_name]["ece"][bin_name] for row in domain_results])
                for bin_name in domain_results[0]["signals"][signal_name]["ece"]
            }
        macro_averages.append({"model_id": model_id, "condition_id": condition_id,
                               "n_domains": len(domain_results), "averaging": "equal_weight_domains",
                               **mean_metrics, "signals": signal_means})
    return {
        "source_kind": "simulation" if all(row["source_kind"] == "simulation" for row in records) else "mixed_or_real",
        "warning": "Synthetic output is for pipeline validation only; it is not empirical evidence.",
        "groups": results,
        "domain_macro_averages": macro_averages,
        "order_stability": stability,
    }


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _mean(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(valid) / len(valid) if valid else None

