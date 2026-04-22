"""Step 2 — test whether training-time mask widening is the culprit.

Take an hw_param=8-trained (baseline) checkpoint and run ``run_latent_walk``
with a widened generator (``max_vthread_extent=15``). All other settings
(model weights, cost ridge, record_json, walk hyperparams) match the
original exploration from training.

This is equivalent to "train hw=8, explore hw=15" without retraining:
with baseline training the recon gradient is already identical to what a
Step 2 retrain would produce (see report §3.3 for the argument).

Usage:
    python diag_step2_explore_hw15.py \\
        --checkpoint /abs/path/to/baseline_hw8.pt \\
        [--num-steps 30] [--step-size 0.25] [--top-k 1]

Compare ``best_measured_mean_cost`` with:
- baseline (train=8, explore=8): original training log
- Step 1 (train=15, explore=8): diag_step1_explore_hw8.py
- hw=15 experiment (train=15, explore=15): original hw=15 training log

If Step 2 ≈ baseline → training-time mask widening is the sole root cause.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from latent_model_budget.adapter import GeneratorRegistry
from tune_by_latent import (
    LoadedBundle,
    load_bundle,
    run_latent_walk,
)


DEFAULT_HW_OVERRIDE: Dict[str, Any] = {"max_vthread_extent": 15}


def _resolve_record_json(payload_config: Dict[str, Any]) -> Optional[str]:
    train_cfg = payload_config.get("train", {}) or {}
    explicit = train_cfg.get("latent_walk_record_json")
    if explicit:
        return str(explicit)
    data_cfg = payload_config.get("data", {}) or {}
    json_paths = list(data_cfg.get("json_paths", []) or [])
    return str(json_paths[0]) if json_paths else None


def _override_registry_hw_param(
    bundle: LoadedBundle, hw_param: Dict[str, Any]
) -> LoadedBundle:
    payload_config = bundle.checkpoint_payload.get("config", {}) or {}
    data_cfg = payload_config.get("data", {}) or {}
    network_info_folder = data_cfg.get("network_info_folder")
    generator_cfg = payload_config.get("generator", {}) or {}

    new_registry = GeneratorRegistry(
        network_info_folder,
        hw_param=hw_param or None,
        disable_constraint=generator_cfg.get("disable_constraint") or None,
    )
    bundle.registry = new_registry
    return bundle


def _summarize_records(records: List[Any]) -> Dict[str, float]:
    """Extract measured mean_cost stats from WalkRecord objects.

    WalkRecord exposes ``measurement`` as a dict-like with ``usable_measurement``
    and ``mean_cost`` — higher ``mean_cost`` is better (matches
    ``walk/best_measured_mean_cost`` used by training).
    """
    measured: List[float] = []
    for r in records:
        m = getattr(r, "measurement", None)
        if not isinstance(m, dict):
            continue
        if not m.get("usable_measurement"):
            continue
        val = m.get("mean_cost")
        if val is None:
            continue
        try:
            val = float(val)
        except (TypeError, ValueError):
            continue
        if val != val:  # NaN filter
            continue
        measured.append(val)

    summary: Dict[str, float] = {"num_records": float(len(records))}
    if measured:
        summary["num_measured"] = float(len(measured))
        summary["measured_max"] = max(measured)
        summary["measured_mean"] = sum(measured) / len(measured)
        summary["measured_min"] = min(measured)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--record-json", type=str, default=None,
        help="Override reference record JSON. Defaults to the one saved in "
             "the checkpoint config.",
    )
    parser.add_argument(
        "--hw-param-override", type=str, default=None,
        help=f"JSON dict; defaults to {json.dumps(DEFAULT_HW_OVERRIDE)}.",
    )
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--latent-gradient", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    bundle = load_bundle(
        checkpoint_path,
        device=args.device,
        use_latent_gradient=bool(args.latent_gradient),
    )
    payload_config = bundle.checkpoint_payload.get("config", {}) or {}
    original_hw = (payload_config.get("generator", {}) or {}).get("hw_param") or {}

    hw_override: Dict[str, Any]
    if args.hw_param_override is None:
        hw_override = dict(DEFAULT_HW_OVERRIDE)
    else:
        hw_override = json.loads(args.hw_param_override)

    print(f"[step2] checkpoint = {checkpoint_path}")
    print(f"[step2] original generator.hw_param = {original_hw}")
    print(f"[step2] override  generator.hw_param = {hw_override}")

    _override_registry_hw_param(bundle, hw_override)

    record_json_path = args.record_json or _resolve_record_json(payload_config)
    if not record_json_path:
        raise RuntimeError(
            "Could not resolve record_json_path. Pass --record-json explicitly."
        )
    print(f"[step2] record_json = {record_json_path}")

    train_cfg = payload_config.get("train", {}) or {}
    num_steps = args.num_steps if args.num_steps is not None else int(
        train_cfg.get("latent_walk_num_steps", 8) or 8
    )
    step_size = args.step_size if args.step_size is not None else float(
        train_cfg.get("latent_walk_step_size", 0.25) or 0.25
    )
    print(f"[step2] num_steps={num_steps} step_size={step_size} top_k={args.top_k}")

    from latent_model_budget.train import _select_topk_records_from_path

    ref_records = _select_topk_records_from_path(
        record_json_path, k=max(1, int(args.top_k))
    )
    if not ref_records:
        raise RuntimeError(f"no reference records found in {record_json_path}")

    output_root = Path(args.output) if args.output else (
        checkpoint_path.parent / "walk_records_step2_hw15"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    per_rank_summaries: List[Dict[str, float]] = []
    for rank, ref_record in enumerate(ref_records):
        rank_output = (
            output_root / f"rank{rank}" if len(ref_records) > 1 else output_root
        )
        rank_output.mkdir(parents=True, exist_ok=True)

        walk_records = run_latent_walk(
            checkpoint_path=str(checkpoint_path),
            record_json_path=str(record_json_path),
            device=str(args.device),
            num_steps=int(num_steps),
            step_size=float(step_size),
            latent_gradient=bool(args.latent_gradient),
            seed=int(args.seed),
            deterministic_start=True,
            preselected_record=ref_record,
            include_recon_predict=False,
            include_measurement=True,
            bundle=bundle,
            keep_bundle=True,
            output=str(rank_output),
        ) or []
        summary = _summarize_records(walk_records)
        summary["rank"] = rank
        per_rank_summaries.append(summary)
        print(f"[step2] rank={rank} summary = {summary}")

    aggregate = {
        "ranks": per_rank_summaries,
        "hw_param_override": hw_override,
        "original_hw_param": original_hw,
    }
    measured_maxes = [
        s["measured_max"] for s in per_rank_summaries if "measured_max" in s
    ]
    if measured_maxes:
        aggregate["best_measured_max_across_ranks"] = max(measured_maxes)

    summary_path = output_root / "step2_summary.json"
    summary_path.write_text(json.dumps(aggregate, indent=2))
    print(f"[step2] wrote summary -> {summary_path}")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
