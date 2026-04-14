from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence

import torch

if __package__ in (None, ""):
    _HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(_HERE))
    sys.path.insert(0, str(_HERE.parent))
    from latent_model_budget.config import build_config
    from latent_model_budget.adapter import JsonSampleRecord
    from result_csv_utils import make_sym_map_key
    from tune_by_latent import (
        compute_walk_direction,
        greedy_decode_from_z,
        load_bundle,
        make_shifted_zs,
        predict_score,
        prepare_record_context,
        resolve_start_z,
        _resolve_record_json_path,
        _select_record_from_path,
    )
else:
    from .latent_model_budget.config import build_config
    from .latent_model_budget.adapter import JsonSampleRecord
    from .result_csv_utils import make_sym_map_key
    from .tune_by_latent import (
        compute_walk_direction,
        greedy_decode_from_z,
        load_bundle,
        make_shifted_zs,
        predict_score,
        prepare_record_context,
        resolve_start_z,
        _resolve_record_json_path,
        _select_record_from_path,
    )


def _safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _jsonable_config_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = payload.get("config", {})
    data_cfg = cfg.get("data", {})
    default_model_cfg = vars(build_config().model)
    model_cfg = dict(default_model_cfg)
    model_cfg.update(cfg.get("model", {}))
    train_cfg = cfg.get("train", {})
    return {
        "json_paths": list(data_cfg.get("json_paths", [])),
        "network_info_folder": data_cfg.get("network_info_folder"),
        "budget": bool(data_cfg.get("budget", True)),
        "adaln": bool(model_cfg.get("adaln", True)),
        "learning_rate": _safe_float(train_cfg.get("learning_rate")),
        "lambda_nce": _safe_float(train_cfg.get("lambda_nce")),
        "tau_nce": _safe_float(train_cfg.get("tau_nce")),
        "beta_end": _safe_float(train_cfg.get("beta_end")),
        "beta_warmup_epochs": train_cfg.get("beta_warmup_epochs"),
        "order_nce": bool(train_cfg.get("order_nce", False)),
        "nce_mu": bool(train_cfg.get("nce_mu", False)),
        "lambda_latent_use": _safe_float(train_cfg.get("lambda_latent_use")),
        "latent_use_margin": _safe_float(train_cfg.get("latent_use_margin")),
        "latent_wrong_top1_margin": _safe_float(train_cfg.get("latent_wrong_top1_margin")),
    }


def _digest_sym_key(sym_key: Sequence[Sequence[Any]]) -> str:
    raw = json.dumps(list(sym_key), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _first_order_difference(left: Sequence[str], right: Sequence[str]) -> Optional[Dict[str, Any]]:
    limit = min(len(left), len(right))
    for idx in range(limit):
        if str(left[idx]) != str(right[idx]):
            return {
                "index": int(idx),
                "left": str(left[idx]),
                "right": str(right[idx]),
            }
    if len(left) != len(right):
        return {
            "index": int(limit),
            "left": None if len(left) <= limit else str(left[limit]),
            "right": None if len(right) <= limit else str(right[limit]),
        }
    return None


def _vector_similarity(left: torch.Tensor, right: torch.Tensor) -> Dict[str, Optional[float]]:
    if left.numel() != right.numel():
        return {"cosine": None, "l2": None}
    left = left.detach().to(dtype=torch.float32, device="cpu").view(-1)
    right = right.detach().to(dtype=torch.float32, device="cpu").view(-1)
    left_norm = float(left.norm().item())
    right_norm = float(right.norm().item())
    cosine = None
    if left_norm > 0.0 and right_norm > 0.0:
        cosine = float(torch.dot(left, right).item() / (left_norm * right_norm))
    l2 = float(torch.norm(left - right).item())
    return {"cosine": cosine, "l2": l2}


def _build_gold_path_trace(
    bundle,
    record: JsonSampleRecord,
    ordered_names: Sequence[str],
    ordered_values: Sequence[int],
    *,
    max_steps: int,
) -> Dict[str, Any]:
    oracle = bundle.registry.build_oracle_from_record(record)
    steps: List[Dict[str, Any]] = []
    first_failure = None
    for idx, (name, value) in enumerate(zip(ordered_names, ordered_values)):
        candidates = list(oracle.candidate_values(name))
        gold_ok = int(value) in candidates
        if idx < max_steps:
            steps.append(
                {
                    "step": int(idx),
                    "var_name": str(name),
                    "gold_value": int(value),
                    "candidate_count": int(len(candidates)),
                    "gold_in_candidates": bool(gold_ok),
                    "candidates": [int(v) for v in candidates],
                }
            )
        if not gold_ok:
            first_failure = {
                "step": int(idx),
                "var_name": str(name),
                "gold_value": int(value),
                "candidates": [int(v) for v in candidates],
            }
            break
        oracle.assign(name, int(value))
    return {
        "checked_steps": int(min(len(ordered_names), max_steps)),
        "first_failure": first_failure,
        "trace": steps,
    }


def _count_param_differences(
    ordered_names: Sequence[str],
    baseline_values: Sequence[int],
    decoded_params: Dict[str, int],
) -> Dict[str, Any]:
    diffs: List[Dict[str, Any]] = []
    for name, baseline in zip(ordered_names, baseline_values):
        pred = int(decoded_params[name])
        if pred == int(baseline):
            continue
        diffs.append(
            {
                "var_name": str(name),
                "baseline": int(baseline),
                "predicted": int(pred),
            }
        )
    return {
        "num_param_diffs": int(len(diffs)),
        "first_param_diff": diffs[0] if diffs else None,
        "param_diffs_preview": diffs[:10],
    }


def _build_walk_summary(
    bundle,
    checkpoint_path: str,
    record_json_path: Path,
    *,
    best_cost: bool,
    deterministic: bool,
    num_steps: int,
    step_size: float,
    normalize_direction: bool,
    gold_trace_steps: int,
) -> Dict[str, Any]:
    resolved_record_path = _resolve_record_json_path(record_json_path, bundle)
    record = _select_record_from_path(resolved_record_path, best_cost=best_cost)
    gen, ordered_names, ordered_values = prepare_record_context(bundle, record)
    start_ctx = resolve_start_z(
        bundle,
        record,
        random_z=False,
        seed=None,
        deterministic_start=deterministic,
    )
    z0 = start_ctx.z0
    direction = compute_walk_direction(bundle, z0)
    shifted_zs = make_shifted_zs(
        z0,
        direction,
        num_steps=num_steps,
        step_size=step_size,
        normalize_direction=normalize_direction,
    )

    walk_steps: List[Dict[str, Any]] = []
    unique_keys: List[tuple[tuple[str, int], ...]] = []
    seen_keys = set()
    repeat_from_previous = 0
    previous_key = None
    for step_index, alpha, z in shifted_zs:
        oracle = bundle.registry.build_oracle_from_record(record)
        decoded = greedy_decode_from_z(bundle, oracle, list(ordered_names), z)
        sym_key = make_sym_map_key(
            {str(name): int(value) for name, value in decoded.sym_map.items() if isinstance(value, int)}
        )
        if sym_key not in seen_keys:
            seen_keys.add(sym_key)
            unique_keys.append(sym_key)
        if previous_key is not None and sym_key == previous_key:
            repeat_from_previous += 1
        previous_key = sym_key
        diff_summary = _count_param_differences(ordered_names, ordered_values, decoded.params)
        walk_steps.append(
            {
                "step_index": int(step_index),
                "alpha": float(alpha),
                "predicted_score": float(predict_score(bundle, z)),
                "sym_map_digest": _digest_sym_key(sym_key),
                "num_sym_entries": int(len(sym_key)),
                "final_violations": list(decoded.final_violations),
                **diff_summary,
            }
        )

    full_order = list(gen.get_full_var_order_entries()["param_order"])
    tokenizer = bundle.tokenizer
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_name": Path(checkpoint_path).stem,
        "record_json_path": str(resolved_record_path),
        "sample_id": str(record.sample_id),
        "config_summary": _jsonable_config_summary(bundle.checkpoint_payload),
        "tokenizer": {
            "num_tokens": int(len(tokenizer.id_to_token)),
            "num_vars": int(len(tokenizer.id_to_var)),
        },
        "generator": {
            "full_order_len": int(len(full_order)),
            "model_order_len": int(len(ordered_names)),
            "full_order_head": [str(name) for name in full_order[:12]],
            "model_order_head": [str(name) for name in ordered_names[:12]],
        },
        "gold_path": _build_gold_path_trace(
            bundle,
            record,
            ordered_names,
            ordered_values,
            max_steps=gold_trace_steps,
        ),
        "latent": {
            "z0_norm": float(z0.detach().norm().item()),
            "direction_norm": float(direction.detach().norm().item()),
            "num_steps": int(num_steps),
            "step_size": float(step_size),
            "normalize_direction": bool(normalize_direction),
            "deterministic": bool(deterministic),
        },
        "walk": {
            "num_total_steps": int(len(walk_steps)),
            "num_unique_sym_maps": int(len(seen_keys)),
            "num_repeated_from_previous": int(repeat_from_previous),
            "unique_sym_map_digests": [_digest_sym_key(key) for key in unique_keys],
            "steps": walk_steps,
        },
        "artifacts": {
            "ordered_names": [str(name) for name in ordered_names],
            "ordered_values": [int(value) for value in ordered_values],
            "z0": [float(x) for x in z0.detach().cpu().tolist()],
            "direction": [float(x) for x in direction.detach().cpu().tolist()],
        },
    }


def _compare_walks(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    left_names = list(left["artifacts"]["ordered_names"])
    right_names = list(right["artifacts"]["ordered_names"])
    order_equal = left_names == right_names
    walk_left = list(left["walk"]["steps"])
    walk_right = list(right["walk"]["steps"])
    per_alpha: List[Dict[str, Any]] = []
    first_diff_alpha = None
    for left_step, right_step in zip(walk_left, walk_right):
        same_sym_map = str(left_step["sym_map_digest"]) == str(right_step["sym_map_digest"])
        same_first_diff = left_step.get("first_param_diff") == right_step.get("first_param_diff")
        entry = {
            "alpha": float(left_step["alpha"]),
            "same_sym_map": bool(same_sym_map),
            "left_sym_map_digest": str(left_step["sym_map_digest"]),
            "right_sym_map_digest": str(right_step["sym_map_digest"]),
            "left_num_param_diffs": int(left_step["num_param_diffs"]),
            "right_num_param_diffs": int(right_step["num_param_diffs"]),
            "left_first_param_diff": left_step.get("first_param_diff"),
            "right_first_param_diff": right_step.get("first_param_diff"),
            "same_first_param_diff": bool(same_first_diff),
            "score_gap": float(left_step["predicted_score"] - right_step["predicted_score"]),
        }
        if first_diff_alpha is None and not same_sym_map:
            first_diff_alpha = float(left_step["alpha"])
        per_alpha.append(entry)

    return {
        "same_record_json": str(left["record_json_path"]) == str(right["record_json_path"]),
        "same_sample_id": str(left["sample_id"]) == str(right["sample_id"]),
        "order_equal": bool(order_equal),
        "first_order_difference": _first_order_difference(left_names, right_names),
        "tokenizer_num_tokens": {
            "left": int(left["tokenizer"]["num_tokens"]),
            "right": int(right["tokenizer"]["num_tokens"]),
        },
        "tokenizer_num_vars": {
            "left": int(left["tokenizer"]["num_vars"]),
            "right": int(right["tokenizer"]["num_vars"]),
        },
        "latent_similarity": {
            "z0": _vector_similarity(
                torch.tensor(left["artifacts"]["z0"], dtype=torch.float32),
                torch.tensor(right["artifacts"]["z0"], dtype=torch.float32),
            ),
            "direction": _vector_similarity(
                torch.tensor(left["artifacts"]["direction"], dtype=torch.float32),
                torch.tensor(right["artifacts"]["direction"], dtype=torch.float32),
            ),
        },
        "gold_path": {
            "left_first_failure": left["gold_path"]["first_failure"],
            "right_first_failure": right["gold_path"]["first_failure"],
        },
        "unique_sym_maps": {
            "left": int(left["walk"]["num_unique_sym_maps"]),
            "right": int(right["walk"]["num_unique_sym_maps"]),
            "shared": int(
                len(
                    set(left["walk"]["unique_sym_map_digests"])
                    & set(right["walk"]["unique_sym_map_digests"])
                )
            ),
        },
        "first_alpha_where_sym_map_differs": first_diff_alpha,
        "per_alpha": per_alpha,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-a", type=str, required=True)
    parser.add_argument("--checkpoint-b", type=str, required=True)
    parser.add_argument("--record-json", type=str, default=None)
    parser.add_argument("--network-info-folder", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--step-size", type=float, default=0.25)
    parser.add_argument("--best-cost", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-normalize-direction", action="store_true")
    parser.add_argument("--gold-trace-steps", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_a = load_bundle(
        args.checkpoint_a,
        network_info_folder=args.network_info_folder,
        device=args.device,
    )
    bundle_b = load_bundle(
        args.checkpoint_b,
        network_info_folder=args.network_info_folder,
        device=args.device,
    )

    if args.record_json is not None:
        record_json_path = Path(args.record_json)
    else:
        record_json_path = _resolve_record_json_path(None, bundle_a)

    left = _build_walk_summary(
        bundle_a,
        args.checkpoint_a,
        record_json_path,
        best_cost=bool(args.best_cost),
        deterministic=bool(args.deterministic),
        num_steps=int(args.num_steps),
        step_size=float(args.step_size),
        normalize_direction=not bool(args.no_normalize_direction),
        gold_trace_steps=int(args.gold_trace_steps),
    )
    right = _build_walk_summary(
        bundle_b,
        args.checkpoint_b,
        record_json_path,
        best_cost=bool(args.best_cost),
        deterministic=bool(args.deterministic),
        num_steps=int(args.num_steps),
        step_size=float(args.step_size),
        normalize_direction=not bool(args.no_normalize_direction),
        gold_trace_steps=int(args.gold_trace_steps),
    )

    report = {
        "note": (
            "This compares two checkpoints under the current workspace code semantics. "
            "It does not re-run the historical code version."
        ),
        "comparison": _compare_walks(left, right),
        "left": left,
        "right": right,
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"[saved] {output_path}")
    print(text)


if __name__ == "__main__":
    main()
