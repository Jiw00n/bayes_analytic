"""Step 1 — test whether exploration-time domain is the culprit.

Take an hw_param=15-trained checkpoint and run ``run_latent_walk`` with a
narrowed generator (``max_vthread_extent=8``). All other settings (model
weights, cost ridge, record_json, walk hyperparams) match the original
exploration from training.

Usage:
    python diag_step1_explore_hw8.py \\
        --checkpoint /abs/path/to/checkpoint.pt \\
        [--num-steps 8] [--step-size 0.25] [--top-k 1]

Compare the resulting ``walk/best_measured_mean_cost`` (and per-record costs)
with what the same checkpoint produced under its original hw_param=15
exploration. If the hw_param=8 exploration recovers to baseline performance,
the bottleneck is exploration-time domain widening, not training-time (A).
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
    # LoadedBundle is typically a dataclass/namedtuple; mutate by attribute.
    bundle.registry = new_registry
    return bundle


def _summarize_records(records: List[Any]) -> Dict[str, float]:
    costs: List[float] = []
    measured: List[float] = []
    for r in records:
        cost = getattr(r, "cost", None)
        if cost is None:
            continue
        try:
            cost = float(cost)
        except (TypeError, ValueError):
            continue
        if not (cost == cost):  # NaN filter
            continue
        costs.append(cost)
        if getattr(r, "measured", False):
            measured.append(cost)
    summary: Dict[str, float] = {"num_records": float(len(records))}
    if costs:
        summary["cost_min"] = min(costs)
        summary["cost_mean"] = sum(costs) / len(costs)
    if measured:
        summary["measured_min"] = min(measured)
        summary["measured_mean"] = sum(measured) / len(measured)
        summary["num_measured"] = float(len(measured))
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
        help="JSON dict; defaults to {} meaning 'use generator defaults'. "
             "Example: '{\"max_vthread_extent\": 8}'",
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
        hw_override = {}  # fall back to generator defaults (i.e., hw=8)
    else:
        hw_override = json.loads(args.hw_param_override)

    print(f"[step1] checkpoint = {checkpoint_path}")
    print(f"[step1] original generator.hw_param = {original_hw}")
    print(f"[step1] override  generator.hw_param = {hw_override}")

    _override_registry_hw_param(bundle, hw_override)

    record_json_path = args.record_json or _resolve_record_json(payload_config)
    if not record_json_path:
        raise RuntimeError(
            "Could not resolve record_json_path. Pass --record-json explicitly."
        )
    print(f"[step1] record_json = {record_json_path}")

    train_cfg = payload_config.get("train", {}) or {}
    num_steps = args.num_steps if args.num_steps is not None else int(
        train_cfg.get("latent_walk_num_steps", 8) or 8
    )
    step_size = args.step_size if args.step_size is not None else float(
        train_cfg.get("latent_walk_step_size", 0.25) or 0.25
    )
    print(f"[step1] num_steps={num_steps} step_size={step_size} top_k={args.top_k}")

    from latent_model_budget.train import _select_topk_records_from_path

    ref_records = _select_topk_records_from_path(
        record_json_path, k=max(1, int(args.top_k))
    )
    if not ref_records:
        raise RuntimeError(f"no reference records found in {record_json_path}")

    output_root = Path(args.output) if args.output else (
        checkpoint_path.parent / "walk_records_step1_hw8"
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
        print(f"[step1] rank={rank} summary = {summary}")

    aggregate = {
        "ranks": per_rank_summaries,
        "hw_param_override": hw_override,
        "original_hw_param": original_hw,
    }
    measured_mins = [
        s["measured_min"] for s in per_rank_summaries if "measured_min" in s
    ]
    if measured_mins:
        aggregate["best_measured_min_across_ranks"] = min(measured_mins)

    summary_path = output_root / "step1_summary.json"
    summary_path.write_text(json.dumps(aggregate, indent=2))
    print(f"[step1] wrote summary -> {summary_path}")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
