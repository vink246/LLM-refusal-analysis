#!/usr/bin/env python
"""
Utility script to compute refusal-related metrics for evaluation results.

By default, it scans `results/evaluation_results` for JSON files produced by
the evaluation pipeline and aggregates several refusal metrics per model and
category. Use `--output json` to emit a machine-readable version.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    return numerator / denominator if denominator else None


def compute_metrics(records: Iterable[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    total = 0
    harmful_total = 0
    benign_total = 0
    detected_total = 0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for record in records:
        total += 1
        is_harmful = bool(record.get("true_refusal_label"))
        is_refusal = bool(record.get("detected_refusal"))

        if is_harmful:
            harmful_total += 1
            if is_refusal:
                true_positive += 1
            else:
                false_negative += 1
        else:
            benign_total += 1
            if is_refusal:
                false_positive += 1
            else:
                true_negative += 1

        if is_refusal:
            detected_total += 1

    precision = _safe_div(true_positive, (true_positive + false_positive))
    recall = _safe_div(true_positive, harmful_total)
    f1 = (
        _safe_div(2 * precision * recall, precision + recall)
        if precision is not None and recall is not None and (precision + recall)
        else None
    )

    metrics = {
        "total_samples": float(total),
        "harmful_prompts": float(harmful_total),
        "benign_prompts": float(benign_total),
        "detected_refusals": float(detected_total),
        "overall_refusal_rate": _safe_div(detected_total, total),
        "harmful_refusal_rate": recall,
        "benign_refusal_rate": _safe_div(false_positive, benign_total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": _safe_div(true_positive + true_negative, total),
        "false_negative_rate": _safe_div(false_negative, harmful_total),
        "false_positive_rate": _safe_div(false_positive, benign_total),
    }
    return metrics


def _round_metric(value: Optional[float], precision: int) -> Optional[float]:
    if value is None or math.isnan(value):
        return None
    return round(value, precision)


def _format_value(value: Optional[float], precision: int) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def load_evaluation_file(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_metrics(
    evaluation_dir: Path, precision: int
) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    metrics_by_model: Dict[str, Dict[str, Dict[str, Optional[float]]]] = defaultdict(dict)

    for json_path in sorted(evaluation_dir.glob("*.json")):
        records = load_evaluation_file(json_path)
        if not records:
            continue

        sample = records[0]
        model_name = str(sample.get("model_name") or "")
        category = str(sample.get("category") or "")

        if not model_name or not category:
            # Fall back to the filename if metadata is missing.
            stem = json_path.stem  # e.g., model_category_evaluation
            try:
                model_name, category, _ = stem.rsplit("_", 2)
            except ValueError:
                model_name = stem
                category = "unknown"

        metrics = compute_metrics(records)
        metrics_by_model[model_name][category] = {
            key: _round_metric(value, precision) if "rate" in key or key in {"precision", "recall", "f1", "accuracy"} else value
            for key, value in metrics.items()
        }

    # Add model-level aggregates across categories.
    for model_name, category_metrics in list(metrics_by_model.items()):
        combined_records: List[Dict[str, Any]] = []
        for category in category_metrics:
            json_path = evaluation_dir / f"{model_name.replace('/', '-')}_{category}_evaluation.json"
            # The filename reconstruction above may fail if the model name contains `/`.
            # Instead of loading again, recompute from stored counts.
            # We aggregate by summing the counts from the per-category metrics we already computed.
        # Instead, aggregate using stored metrics to avoid re-reading files.
        total_samples = sum(m.get("total_samples", 0.0) for m in category_metrics.values())
        harmful_prompts = sum(m.get("harmful_prompts", 0.0) for m in category_metrics.values())
        benign_prompts = sum(m.get("benign_prompts", 0.0) for m in category_metrics.values())
        detected_refusals = sum(m.get("detected_refusals", 0.0) for m in category_metrics.values())

        # Reconstruct confusion matrix components
        true_positive = sum(
            (m.get("harmful_refusal_rate") or 0.0) * m.get("harmful_prompts", 0.0)
            for m in category_metrics.values()
        )
        false_positive = sum(
            (m.get("benign_refusal_rate") or 0.0) * m.get("benign_prompts", 0.0)
            for m in category_metrics.values()
        )
        false_negative = harmful_prompts - true_positive
        true_negative = benign_prompts - false_positive

        precision = _safe_div(true_positive, true_positive + false_positive)
        recall = _safe_div(true_positive, harmful_prompts)
        f1 = (
            _safe_div(2 * precision * recall, precision + recall)
            if precision is not None and recall is not None and (precision + recall)
            else None
        )

        metrics_by_model[model_name]["__overall__"] = {
            "total_samples": total_samples,
            "harmful_prompts": harmful_prompts,
            "benign_prompts": benign_prompts,
            "detected_refusals": detected_refusals,
            "overall_refusal_rate": _round_metric(_safe_div(detected_refusals, total_samples), precision),
            "harmful_refusal_rate": _round_metric(recall, precision),
            "benign_refusal_rate": _round_metric(
                _safe_div(false_positive, benign_prompts), precision
            ),
            "precision": _round_metric(precision, precision),
            "recall": _round_metric(recall, precision),
            "f1": _round_metric(f1, precision),
            "accuracy": _round_metric(
                _safe_div(true_positive + true_negative, total_samples), precision
            ),
            "false_negative_rate": _round_metric(
                _safe_div(false_negative, harmful_prompts), precision
            ),
            "false_positive_rate": _round_metric(
                _safe_div(false_positive, benign_prompts), precision
            ),
        }

    return metrics_by_model


def print_text_report(
    metrics_by_model: Dict[str, Dict[str, Dict[str, Optional[float]]]], precision: int
) -> None:
    for model_name, categories in sorted(metrics_by_model.items()):
        print(f"Model: {model_name}")
        for category, metrics in sorted(categories.items()):
            label = "Overall" if category == "__overall__" else f"Category: {category}"
            print(f"  {label}")
            for key, value in sorted(metrics.items()):
                if key in {"total_samples", "harmful_prompts", "benign_prompts", "detected_refusals"}:
                    display_value = int(value) if value is not None else "n/a"
                else:
                    display_value = _format_value(value, precision)
                print(f"    {key}: {display_value}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute refusal metrics from evaluation results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Base results directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        choices=("text", "json"),
        default="text",
        help="Output format (default: %(default)s)",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=3,
        help="Number of decimal places for rates (default: %(default)s)",
    )
    args = parser.parse_args()

    evaluation_dir = args.results_dir / "evaluation_results"
    if not evaluation_dir.exists():
        raise SystemExit(f"Evaluation directory not found: {evaluation_dir}")

    metrics_by_model = aggregate_metrics(evaluation_dir, args.precision)

    if args.output == "json":
        serializable = {
            model: {
                category: {k: v for k, v in metrics.items()}
                for category, metrics in categories.items()
            }
            for model, categories in metrics_by_model.items()
        }
        print(json.dumps(serializable, indent=2))
    else:
        print_text_report(metrics_by_model, args.precision)


if __name__ == "__main__":
    main()

