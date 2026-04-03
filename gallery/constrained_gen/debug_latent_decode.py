from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

if __package__ in (None, ""):
    _HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(_HERE))
    sys.path.insert(0, str(_HERE.parent))
    from tune_by_latent import (
        load_bundle,
        load_json_sample,
        encode_record_to_z,
        compute_walk_direction,
        make_shifted_zs,
        predict_score,
        _resolve_decoded_value,
    )
else:
    from .tune_by_latent import (
        load_bundle,
        load_json_sample,
        encode_record_to_z,
        compute_walk_direction,
        make_shifted_zs,
        predict_score,
        _resolve_decoded_value,
    )


@dataclass
class StepDebugRecord:
    step_idx: int
    var_name: str
    candidate_count: int
    candidates: List[int]
    pred_token: str
    pred_value: int
    top2_margin: Optional[float]
    best_legal_logit: float
    second_legal_logit: Optional[float]
    zero_memory_same_pred: bool
    zero_memory_pred_value: int
    zero_memory_top2_margin: Optional[float]


@dataclass
class DecodeDebugSummary:
    tag: str
    predicted_params: Dict[str, int]
    final_violations: List[str]
    num_changed_from_reference: Optional[int]
    changed_param_names: List[str]
    step_records: List[StepDebugRecord]


@dataclass
class WalkDebugRecord:
    step_index: int
    alpha: float
    predicted_score: float
    decode: DecodeDebugSummary


def _topk_legal(masked_logits: torch.Tensor, k: int = 2) -> Tuple[List[int], List[float]]:
    finite_mask = torch.isfinite(masked_logits)
    finite_indices = torch.nonzero(finite_mask, as_tuple=False).flatten().tolist()
    if not finite_indices:
        return [], []
    ranked = sorted(
        ((int(idx), float(masked_logits[idx].item())) for idx in finite_indices),
        key=lambda x: x[1],
        reverse=True,
    )
    ranked = ranked[:k]
    return [idx for idx, _ in ranked], [val for _, val in ranked]


@torch.no_grad()
def greedy_decode_with_debug(
    bundle,
    ordered_names: Sequence[str],
    z: torch.Tensor,
    *,
    record,
    reference_params: Optional[Dict[str, int]] = None,
    tag: str = "decode",
) -> DecodeDebugSummary:
    model = bundle.model
    tokenizer = bundle.tokenizer
    device = bundle.device

    oracle = bundle.registry.build_oracle_from_record(record)
    z = z.to(device=device, dtype=torch.float32).view(1, -1)
    memory = model.latent_to_memory(z).view(1, model.cfg.latent_token_count, model.cfg.d_model)
    zero_memory = torch.zeros_like(memory)

    decoder_input_ids: List[int] = [tokenizer.bos_id]
    decoded_params: Dict[str, int] = {}
    step_records: List[StepDebugRecord] = []

    for step_idx, var_name in enumerate(ordered_names):
        candidate_values = list(oracle.candidate_values(var_name))
        if not candidate_values:
            raise RuntimeError(f"No legal candidates returned for {var_name}")

        token_mask = tokenizer.candidate_mask_from_values(var_name, candidate_values, device=device)
        if not bool(token_mask.any()):
            raise RuntimeError(
                f"Token vocab cannot represent any legal candidate for {var_name}. "
                f"candidates={candidate_values}"
            )

        step_input = torch.tensor([decoder_input_ids], dtype=torch.long, device=device)
        step_var_ids = torch.tensor(
            [[tokenizer.var_to_id[name] for name in ordered_names[: len(decoder_input_ids)]]],
            dtype=torch.long,
            device=device,
        )

        logits = model.decode(
            step_input,
            step_var_ids,
            memory,
            decoder_pad_mask=step_input.eq(tokenizer.pad_id),
        )
        step_logits = logits[0, -1].masked_fill(~token_mask, float("-inf"))
        pred_token_ids, pred_scores = _topk_legal(step_logits, k=2)
        if not pred_token_ids:
            raise RuntimeError(f"No finite legal logits for {var_name}")
        pred_token_id = int(pred_token_ids[0])
        pred_value = _resolve_decoded_value(tokenizer, var_name, pred_token_id, candidate_values)

        zero_logits = model.decode(
            step_input,
            step_var_ids,
            zero_memory,
            decoder_pad_mask=step_input.eq(tokenizer.pad_id),
        )
        zero_step_logits = zero_logits[0, -1].masked_fill(~token_mask, float("-inf"))
        zero_token_ids, zero_scores = _topk_legal(zero_step_logits, k=2)
        zero_pred_token_id = int(zero_token_ids[0]) if zero_token_ids else pred_token_id
        zero_pred_value = _resolve_decoded_value(tokenizer, var_name, zero_pred_token_id, candidate_values)

        oracle.assign(var_name, pred_value)
        decoded_params[var_name] = int(pred_value)
        decoder_input_ids.append(pred_token_id)

        step_records.append(
            StepDebugRecord(
                step_idx=int(step_idx),
                var_name=str(var_name),
                candidate_count=len(candidate_values),
                candidates=[int(v) for v in candidate_values],
                pred_token=str(tokenizer.id_to_token[pred_token_id]),
                pred_value=int(pred_value),
                top2_margin=None if len(pred_scores) < 2 else float(pred_scores[0] - pred_scores[1]),
                best_legal_logit=float(pred_scores[0]),
                second_legal_logit=None if len(pred_scores) < 2 else float(pred_scores[1]),
                zero_memory_same_pred=bool(zero_pred_value == pred_value),
                zero_memory_pred_value=int(zero_pred_value),
                zero_memory_top2_margin=None if len(zero_scores) < 2 else float(zero_scores[0] - zero_scores[1]),
            )
        )

    sym_map = dict(oracle.generator.s.sym_map)
    for name, value in decoded_params.items():
        sym_map[name] = int(value)
    del sym_map

    changed_names: List[str] = []
    changed_count: Optional[int] = None
    if reference_params is not None:
        changed_names = [
            name for name in ordered_names
            if int(reference_params[name]) != int(decoded_params[name])
        ]
        changed_count = len(changed_names)

    return DecodeDebugSummary(
        tag=str(tag),
        predicted_params={k: int(v) for k, v in decoded_params.items()},
        final_violations=list(oracle.final_violations()),
        num_changed_from_reference=changed_count,
        changed_param_names=changed_names,
        step_records=step_records,
    )


@torch.no_grad()
def inspect_walk(
    *,
    checkpoint_path: str,
    record_json_path: str,
    network_info_folder: Optional[str],
    device: str,
    num_steps: int,
    step_size: float,
    normalize_direction: bool,
    use_latent_gradient: bool,
    random_direction_seed: int,
    deterministic_start: bool,
) -> Dict[str, Any]:
    bundle = load_bundle(
        checkpoint_path,
        network_info_folder=network_info_folder,
        device=device,
        use_latent_gradient=use_latent_gradient,
    )
    record = load_json_sample(record_json_path)

    # encode_record_to_z currently samples z with deterministic=False inside tune.py.
    # For debugging, optionally override with deterministic mu.
    sampled_z0, gen, ordered_names, ordered_values = encode_record_to_z(bundle, record)
    if deterministic_start:
        enc_ids = torch.tensor(
            [bundle.tokenizer.encode_values(ordered_names, ordered_values)],
            dtype=torch.long,
            device=bundle.device,
        )
        enc_var_ids = torch.tensor(
            [bundle.tokenizer.encode_var_names(ordered_names)],
            dtype=torch.long,
            device=bundle.device,
        )
        enc_pad = enc_ids.eq(bundle.tokenizer.pad_id)
        _, _, z_mu, _ = bundle.model.encode(enc_ids, enc_var_ids, enc_pad, deterministic=True)
        z0 = z_mu[0].detach().clone()
        z0_source = "deterministic_mu"
    else:
        z0 = sampled_z0
        z0_source = "sampled_z"

    reference_params = {name: int(value) for name, value in zip(ordered_names, ordered_values)}

    cost_direction = compute_walk_direction(bundle, z0)
    random_gen = torch.Generator(device=bundle.device)
    random_gen.manual_seed(int(random_direction_seed))
    random_direction = torch.randn(z0.shape, generator=random_gen, device=bundle.device, dtype=torch.float32)

    cost_shifted = make_shifted_zs(
        z0,
        cost_direction,
        num_steps=num_steps,
        step_size=step_size,
        normalize_direction=normalize_direction,
    )
    random_shifted = make_shifted_zs(
        z0,
        random_direction,
        num_steps=num_steps,
        step_size=step_size,
        normalize_direction=normalize_direction,
    )

    baseline_decode = greedy_decode_with_debug(
        bundle,
        ordered_names,
        z0,
        record=record,
        reference_params=reference_params,
        tag="baseline",
    )

    cost_walk: List[WalkDebugRecord] = []
    for step_index, alpha, z in cost_shifted:
        decode = greedy_decode_with_debug(
            bundle,
            ordered_names,
            z,
            record=record,
            reference_params=baseline_decode.predicted_params,
            tag=f"cost_step_{step_index}",
        )
        cost_walk.append(
            WalkDebugRecord(
                step_index=int(step_index),
                alpha=float(alpha),
                predicted_score=float(predict_score(bundle, z)),
                decode=decode,
            )
        )

    random_walk: List[WalkDebugRecord] = []
    for step_index, alpha, z in random_shifted:
        decode = greedy_decode_with_debug(
            bundle,
            ordered_names,
            z,
            record=record,
            reference_params=baseline_decode.predicted_params,
            tag=f"random_step_{step_index}",
        )
        random_walk.append(
            WalkDebugRecord(
                step_index=int(step_index),
                alpha=float(alpha),
                predicted_score=float(predict_score(bundle, z)),
                decode=decode,
            )
        )

    baseline_steps = baseline_decode.step_records
    singleton_steps = [s.var_name for s in baseline_steps if s.candidate_count == 1]
    multi_steps = [s for s in baseline_steps if s.candidate_count >= 2]
    large_margin_steps = [
        s.var_name for s in baseline_steps
        if s.top2_margin is not None and s.top2_margin >= 2.0
    ]
    zero_same_ratio = (
        sum(1 for s in baseline_steps if s.zero_memory_same_pred) / max(len(baseline_steps), 1)
    )

    return {
        "meta": {
            "checkpoint_path": str(checkpoint_path),
            "record_json_path": str(record_json_path),
            "task_index": record.task_index,
            "workload_key": record.workload_key,
            "target_kind": record.target_kind,
            "sketch_index": record.sketch_index,
            "num_params": len(ordered_names),
            "z0_source": z0_source,
            "cost_source": bundle.cost_source,
            "normalize_direction": bool(normalize_direction),
            "num_steps": int(num_steps),
            "step_size": float(step_size),
        },
        "baseline": asdict(baseline_decode),
        "summary": {
            "singleton_step_count": len(singleton_steps),
            "singleton_steps": singleton_steps,
            "multi_candidate_step_count": len(multi_steps),
            "large_margin_step_count_ge_2.0": len(large_margin_steps),
            "large_margin_steps_ge_2.0": large_margin_steps,
            "zero_memory_same_prediction_ratio": float(zero_same_ratio),
            "baseline_changed_from_record": baseline_decode.num_changed_from_reference,
            "baseline_changed_param_names_from_record": baseline_decode.changed_param_names,
            "cost_walk_unique_decode_count": len({tuple(sorted(w.decode.predicted_params.items())) for w in cost_walk}),
            "random_walk_unique_decode_count": len({tuple(sorted(w.decode.predicted_params.items())) for w in random_walk}),
        },
        "cost_walk": [asdict(x) for x in cost_walk],
        "random_walk": [asdict(x) for x in random_walk],
    }


def _print_human_summary(report: Dict[str, Any]) -> None:
    meta = report["meta"]
    summary = report["summary"]
    baseline = report["baseline"]

    print("=" * 80)
    print("[meta]")
    print(json.dumps(meta, indent=2))
    print("=" * 80)
    print("[summary]")
    print(json.dumps(summary, indent=2))
    print("=" * 80)
    print("[baseline violations]", baseline["final_violations"])
    print("[baseline changed from record]", baseline["num_changed_from_reference"])
    if baseline["changed_param_names"]:
        print("[baseline changed param names]", baseline["changed_param_names"])
    print("=" * 80)
    print("[baseline step diagnostics]")
    for step in baseline["step_records"]:
        print(
            f"step={step['step_idx']:>2} var={step['var_name']:<14} "
            f"cand={step['candidate_count']:<3} pred={step['pred_value']:<6} "
            f"margin={step['top2_margin']} zero_same={step['zero_memory_same_pred']}"
        )
    print("=" * 80)
    print("[cost walk decode changes vs baseline]")
    for item in report["cost_walk"]:
        print(
            f"step={item['step_index']:>2} alpha={item['alpha']:.4f} "
            f"score={item['predicted_score']:.6f} "
            f"changed={item['decode']['num_changed_from_reference']} "
            f"viol={len(item['decode']['final_violations'])}"
        )
    print("=" * 80)
    print("[random walk decode changes vs baseline]")
    for item in report["random_walk"]:
        print(
            f"step={item['step_index']:>2} alpha={item['alpha']:.4f} "
            f"score={item['predicted_score']:.6f} "
            f"changed={item['decode']['num_changed_from_reference']} "
            f"viol={len(item['decode']['final_violations'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug latent-walk decode invariance.")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--record-json", required=True, type=str)
    parser.add_argument("--network-info-folder", default=None, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--num-steps", default=8, type=int)
    parser.add_argument("--step-size", default=0.25, type=float)
    parser.add_argument("--no-normalize-direction", action="store_true")
    parser.add_argument("--latent-gradient", action="store_true")
    parser.add_argument("--random-direction-seed", default=0, type=int)
    parser.add_argument("--sampled-start", action="store_true")
    parser.add_argument("--output", default=None, type=str)
    args = parser.parse_args()

    report = inspect_walk(
        checkpoint_path=args.checkpoint,
        record_json_path=args.record_json,
        network_info_folder=args.network_info_folder,
        device=args.device,
        num_steps=args.num_steps,
        step_size=args.step_size,
        normalize_direction=not args.no_normalize_direction,
        use_latent_gradient=bool(args.latent_gradient),
        random_direction_seed=int(args.random_direction_seed),
        deterministic_start=not bool(args.sampled_start),
    )
    _print_human_summary(report)

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
