#!/usr/bin/env python3
"""Plot per-position accuracy curves for each condition in analysis JSON.

Usage:
    /root/work/venv/bin/python draw_per_position.py \
        --input gallery/constrained_gen_budget/lr0.0005_nce0.2_tau0.2_kl0.002_warm20_nce_mu_recon_anal.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _compute_trimmed_ylim(curves: Dict[str, List[Tuple[int, float]]]) -> Tuple[float, float]:
    all_acc = [acc for points in curves.values() for _, acc in points]
    if not all_acc:
        return 0.0, 1.0

    y_min = min(all_acc)
    y_max = max(all_acc)

    # Give a small visual margin while keeping the axis in valid accuracy range.
    span = y_max - y_min
    margin = max(0.005, span * 0.08)
    lo = max(0.0, y_min - margin)
    hi = min(1.0, y_max + margin)

    # If data is very flat, still provide a readable window.
    if hi - lo < 0.03:
        center = (hi + lo) / 2.0
        lo = max(0.0, center - 0.015)
        hi = min(1.0, center + 0.015)

    # Final fallback for pathological cases.
    if lo >= hi:
        return 0.0, 1.0

    return lo, hi


def _extract_curves(data: dict) -> Dict[str, List[Tuple[int, float]]]:
    report = data.get("report")
    if not isinstance(report, dict):
        raise ValueError("JSON must contain an object key 'report'.")

    curves: Dict[str, List[Tuple[int, float]]] = {}
    for condition, payload in report.items():
        if not isinstance(payload, dict):
            continue

        per_position = payload.get("per_position")
        if not isinstance(per_position, list):
            continue

        points: List[Tuple[int, float]] = []
        for row in per_position:
            if not isinstance(row, dict):
                continue

            position = row.get("position")
            total = row.get("total")
            correct = row.get("correct")

            if position is None or total in (None, 0) or correct is None:
                continue

            try:
                pos_i = int(position)
                acc = float(correct) / float(total)
            except (TypeError, ValueError, ZeroDivisionError):
                continue

            points.append((pos_i, acc))

        if points:
            points.sort(key=lambda x: x[0])
            curves[condition] = points

    if not curves:
        raise ValueError("No valid per-position data found under report.*.per_position")

    return curves


def plot_per_position_accuracy(input_path: Path, output_path: Path, title: str | None = None) -> None:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    curves = _extract_curves(data)

    plt.figure(figsize=(11, 6))
    for condition, points in curves.items():
        xs = [p for p, _ in points]
        ys = [a for _, a in points]
        plt.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.5, label=condition)

    plt.xlabel("position")
    plt.ylabel("accuracy (correct / total)")
    y_lo, y_hi = _compute_trimmed_ylim(curves)
    plt.ylim(y_lo, y_hi)
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if title:
        plt.title(title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()

    print(f"Saved plot: {output_path}")
    print(f"Conditions: {', '.join(curves.keys())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw per-position accuracy curves from analysis JSON.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("gallery/constrained_gen_budget/lr0.0005_nce0.2_tau0.2_kl0.002_warm20_nce_mu_recon_anal.json"),
        help="Path to analysis JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gallery/constrained_gen_budget/per_position_accuracy.png"),
        help="Path to output image file",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Per-position Accuracy by Condition",
        help="Plot title",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    plot_per_position_accuracy(args.input, args.output, args.title)


if __name__ == "__main__":
    main()
