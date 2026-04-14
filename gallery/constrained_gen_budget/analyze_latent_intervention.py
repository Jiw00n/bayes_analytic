from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

if __package__ in (None, ""):
    _HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(_HERE))
    sys.path.insert(0, str(_HERE.parent))

    from latent_model_budget.adapter import GeneratorRegistry, JsonSampleRecord, load_json_samples, split_records
    from latent_model_budget.config import (
        DataConfig,
        EvalConfig,
        ExperimentConfig,
        ModelConfig,
        TrainConfig,
        WandbConfig,
    )
    from latent_model_budget.dataset import (
        _apply_cached_order_metadata,
        _expand_json_paths,
        _extract_budget_specs,
        _get_generator_for_record,
        _normalize_param_signature,
        budget_enabled,
        get_model_param_order,
    )
    from latent_model_budget.model import LatentParamVAE
    from latent_model_budget.tokenizer import ParamTokenizer
    from latent_model_budget.runtime_utils import (
        configure_runtime,
        load_checkpoint,
        resolve_device,
        seed_everything,
    )
else:
    from .latent_model_budget.adapter import GeneratorRegistry, JsonSampleRecord, load_json_samples, split_records
    from .latent_model_budget.config import (
        DataConfig,
        EvalConfig,
        ExperimentConfig,
        ModelConfig,
        TrainConfig,
        WandbConfig,
    )
    from .latent_model_budget.dataset import (
        _apply_cached_order_metadata,
        _expand_json_paths,
        _extract_budget_specs,
        _get_generator_for_record,
        _normalize_param_signature,
        budget_enabled,
        get_model_param_order,
    )
    from .latent_model_budget.model import LatentParamVAE
    from .latent_model_budget.tokenizer import ParamTokenizer
    from .latent_model_budget.runtime_utils import (
        configure_runtime,
        load_checkpoint,
        resolve_device,
        seed_everything,
    )

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


DEFAULT_CHECKPOINT_PATH = "/root/work/tvm-ansor/gallery/constrained_gen_budget/checkpoints/last.pt"


@dataclass
class InterventionStats:
    exact_match_count: int = 0
    valid_count: int = 0
    changed_param_frac_sum: float = 0.0
    abs_cost_shift_sum: float = 0.0
    signed_cost_shift_sum: float = 0.0

    def to_dict(self, sample_count: int, base_exact_rate: float, base_valid_rate: float) -> dict:
        denom = max(int(sample_count), 1)
        intervened_exact_rate = self.exact_match_count / denom
        intervened_valid_rate = self.valid_count / denom
        return {
            "intervened_exact_rate": float(intervened_exact_rate),
            "exact_drop": float(base_exact_rate - intervened_exact_rate),
            "intervened_valid_rate": float(intervened_valid_rate),
            "valid_drop": float(base_valid_rate - intervened_valid_rate),
            "changed_param_frac": float(self.changed_param_frac_sum / denom),
            "abs_cost_shift": float(self.abs_cost_shift_sum / denom),
            "signed_cost_shift": float(self.signed_cost_shift_sum / denom),
        }


def _config_from_payload(payload: dict) -> ExperimentConfig:
    cfg = payload.get("config", {})
    return ExperimentConfig(
        data=DataConfig(**cfg.get("data", {})),
        model=ModelConfig(**cfg.get("model", {})),
        train=TrainConfig(**cfg.get("train", {})),
        eval=EvalConfig(**cfg.get("eval", {})),
        wandb=WandbConfig(**cfg.get("wandb", {})),
    )


def _load_records_from_config(config: ExperimentConfig) -> List[JsonSampleRecord]:
    records: List[JsonSampleRecord] = []
    for path in _expand_json_paths(config.data.json_paths):
        records.extend(load_json_samples(path))
    if not records:
        raise ValueError("No records were loaded from config.data.json_paths")
    return records


def _prepare_record_cache(
    records: Sequence[JsonSampleRecord],
    registry: GeneratorRegistry,
    include_budget: bool,
) -> Dict[int, tuple[List[str], List[int]]]:
    prepared_cache: Dict[int, tuple[List[str], List[int]]] = {}
    order_cache: Dict[tuple, Dict[str, object]] = {}

    for record in records:
        order_key = (
            record.workload_key,
            record.target_kind,
            record.task_index,
            record.sketch_index,
            _normalize_param_signature(record.param_signature),
        )
        cached_meta = order_cache.get(order_key)
        if cached_meta is None:
            gen = _get_generator_for_record(record, registry)
            order = get_model_param_order(gen, include_budget=include_budget)
            cached_meta = {
                "order": list(order),
                "budget_specs": list(_extract_budget_specs(gen)),
            }
            order_cache[order_key] = cached_meta

        order = list(cached_meta["order"])
        _apply_cached_order_metadata(record, order, cached_meta["budget_specs"])

        missing = [name for name in order if name not in record.params]
        if missing:
            raise ValueError(f"{record.sample_id} is missing ordered params: {missing}")

        prepared_cache[id(record)] = (
            list(order),
            [int(record.params[name]) for name in order],
        )

    return prepared_cache


def _filter_invalid_records_if_needed(
    records: Sequence[JsonSampleRecord],
    prepared_cache: Dict[int, tuple[List[str], List[int]]],
    registry: GeneratorRegistry,
    enabled: bool,
) -> List[JsonSampleRecord]:
    if not enabled:
        return list(records)

    filtered: List[JsonSampleRecord] = []
    for record in records:
        order, values = prepared_cache[id(record)]
        oracle = registry.build_oracle_from_record(record)
        if oracle.validate_assignment(order, values):
            filtered.append(record)

    if not filtered:
        raise ValueError("No valid records remain after legality filtering")
    return filtered


def _resolve_decoded_value(
    tokenizer: ParamTokenizer,
    var_name: str,
    token_id: int,
    candidate_values: Sequence[int],
) -> int:
    token = tokenizer.id_to_token[int(token_id)]
    value = tokenizer.token_to_value(var_name, token)
    if value is not None:
        return int(value)
    if not candidate_values:
        raise RuntimeError(f"No legal candidates for {var_name}")
    return int(candidate_values[0])


def _fixed_token_id_for_value(
    tokenizer: ParamTokenizer,
    var_name: str,
    value: int,
) -> int:
    token = tokenizer.value_to_token(var_name, int(value))
    return int(tokenizer.token_to_id.get(token, tokenizer.unk_id))


@torch.no_grad()
def _encode_record(
    record: JsonSampleRecord,
    ordered_names: Sequence[str],
    ordered_values: Sequence[int],
    model: LatentParamVAE,
    tokenizer: ParamTokenizer,
    device: torch.device,
) -> torch.Tensor:
    enc_ids = torch.tensor(
        [tokenizer.encode_values(ordered_names, ordered_values)],
        dtype=torch.long,
        device=device,
    )
    enc_var_ids = torch.tensor(
        [tokenizer.encode_var_names(ordered_names)],
        dtype=torch.long,
        device=device,
    )
    enc_pad = enc_ids.eq(tokenizer.pad_id)
    del record
    _, _, z, _ = model.encode(enc_ids, enc_var_ids, enc_pad, deterministic=True)
    return z[0].detach().clone()


@torch.no_grad()
def _greedy_decode_many_from_z(
    record: JsonSampleRecord,
    ordered_names: Sequence[str],
    z_batch: torch.Tensor,
    model: LatentParamVAE,
    tokenizer: ParamTokenizer,
    registry: GeneratorRegistry,
    device: torch.device,
) -> tuple[List[Dict[str, int]], List[List[str]]]:
    if z_batch.dim() == 1:
        z_batch = z_batch.unsqueeze(0)
    z_batch = z_batch.to(device=device, dtype=torch.float32)
    batch_size = int(z_batch.shape[0])

    if batch_size <= 0:
        return [], []

    memory = model.latent_to_memory(z_batch).view(
        batch_size,
        model.cfg.latent_token_count,
        model.cfg.d_model,
    )
    oracles = [registry.build_oracle_from_record(record) for _ in range(batch_size)]

    max_steps = len(ordered_names)
    ordered_var_ids = [int(tokenizer.var_to_id[name]) for name in ordered_names]
    decoder_input_ids = torch.full(
        (batch_size, max_steps + 1),
        tokenizer.pad_id,
        dtype=torch.long,
        device=device,
    )
    decoder_var_ids = torch.full(
        (batch_size, max_steps + 1),
        tokenizer.var_pad_id,
        dtype=torch.long,
        device=device,
    )
    decoder_input_ids[:, 0] = tokenizer.bos_id
    current_lengths = torch.ones(batch_size, dtype=torch.long, device=device)
    if ordered_var_ids:
        decoder_var_ids[:, 0] = ordered_var_ids[0]

    decoded_values: List[Dict[str, int]] = [dict() for _ in range(batch_size)]

    for step_idx, var_name in enumerate(ordered_names):
        variable_indices: List[int] = []
        candidate_lists: List[List[int]] = []

        for sample_idx in range(batch_size):
            candidate_values = list(oracles[sample_idx].candidate_values(var_name))
            if not candidate_values:
                raise RuntimeError(f"No legal candidates returned for {var_name}")
            if len(candidate_values) == 1:
                pred_value = int(candidate_values[0])
                pred_token_id = _fixed_token_id_for_value(tokenizer, var_name, pred_value)
                oracles[sample_idx].assign(var_name, pred_value)
                decoded_values[sample_idx][var_name] = int(pred_value)

                pos = int(current_lengths[sample_idx].item())
                decoder_input_ids[sample_idx, pos] = pred_token_id
                if pos < len(ordered_var_ids):
                    decoder_var_ids[sample_idx, pos] = ordered_var_ids[pos]
                current_lengths[sample_idx] = pos + 1
                continue

            variable_indices.append(sample_idx)
            candidate_lists.append(candidate_values)

        if not variable_indices:
            continue

        subset_indices = torch.tensor(variable_indices, dtype=torch.long, device=device)
        subset_lengths = current_lengths.index_select(0, subset_indices)
        current_width = int(subset_lengths.max().item())
        step_input = decoder_input_ids.index_select(0, subset_indices)[:, :current_width]
        step_var = decoder_var_ids.index_select(0, subset_indices)[:, :current_width]
        logits = model.decode(
            step_input,
            step_var,
            memory.index_select(0, subset_indices),
            z_batch.index_select(0, subset_indices),
            decoder_pad_mask=step_input.eq(tokenizer.pad_id),
        )
        gather_pos = (subset_lengths - 1).to(dtype=torch.long)
        step_logits = logits[torch.arange(len(variable_indices), device=device), gather_pos, :]
        step_masks = torch.zeros_like(step_logits, dtype=torch.bool)

        for local_idx, sample_idx in enumerate(variable_indices):
            step_masks[local_idx] = tokenizer.candidate_mask_from_values(
                var_name,
                candidate_lists[local_idx],
                device=device,
            )
            if not step_masks[local_idx].any():
                raise RuntimeError(
                    f"Token vocab cannot represent any legal candidate for {var_name}. "
                    f"candidates={candidate_lists[local_idx]}"
                )

        masked_logits = step_logits.masked_fill(~step_masks, float("-inf"))
        pred_token_ids = torch.argmax(masked_logits, dim=-1).tolist()

        for local_idx, sample_idx in enumerate(variable_indices):
            candidate_values = candidate_lists[local_idx]
            pred_token_id = int(pred_token_ids[local_idx])
            pred_value = _resolve_decoded_value(tokenizer, var_name, pred_token_id, candidate_values)

            oracles[sample_idx].assign(var_name, pred_value)
            decoded_values[sample_idx][var_name] = int(pred_value)

            pos = int(current_lengths[sample_idx].item())
            decoder_input_ids[sample_idx, pos] = pred_token_id
            if pos < len(ordered_var_ids):
                decoder_var_ids[sample_idx, pos] = ordered_var_ids[pos]
            current_lengths[sample_idx] = pos + 1

    final_violations = [list(oracle.final_violations()) for oracle in oracles]
    return decoded_values, final_violations


def _param_change_fraction(
    base_params: Dict[str, int],
    other_params: Dict[str, int],
    ordered_names: Sequence[str],
) -> float:
    total = 0
    changed = 0
    for name in ordered_names:
        if name not in base_params or name not in other_params:
            continue
        total += 1
        if int(base_params[name]) != int(other_params[name]):
            changed += 1
    if total == 0:
        return 0.0
    return changed / total


def _exact_match(
    pred_params: Dict[str, int],
    gold_values: Sequence[int],
    ordered_names: Sequence[str],
) -> bool:
    for name, gold in zip(ordered_names, gold_values):
        if int(pred_params[name]) != int(gold):
            return False
    return True


def _format_split_header(label: str, sample_count: int) -> str:
    return f"[analysis] split={label} samples={sample_count}"


def _print_running_status(
    label: str,
    sample_id: str,
    num_used: int,
    base_exact_count: int,
    base_valid_count: int,
    per_alpha_dim_stats: Dict[float, Dict[int, InterventionStats]],
) -> None:
    if num_used <= 0:
        return

    best_alpha = None
    best_dim = None
    best_changed = -1.0
    for alpha, dim_stats in per_alpha_dim_stats.items():
        for dim_idx, stats in dim_stats.items():
            changed_value = stats.changed_param_frac_sum / num_used
            if changed_value > best_changed:
                best_changed = changed_value
                best_alpha = alpha
                best_dim = dim_idx

    print(
        f"[intervention] split={label} sample={sample_id} "
        f"used={num_used} "
        f"base_exact={base_exact_count / num_used:.4f} "
        f"base_valid={base_valid_count / num_used:.4f} "
        f"top_alpha={best_alpha} "
        f"top_dim={best_dim} "
        f"top_changed={max(best_changed, 0.0):.4f}"
    )


def _print_analysis_summary(label: str, summary: dict, top_k: Optional[int]) -> None:
    limit = None if top_k is None or int(top_k) <= 0 else int(top_k)

    print(
        f"[analysis] {label} used_samples={summary['num_used']} "
        f"latent_dim={summary['latent_dim']} "
        f"base_exact={summary['base_exact_rate']:.4f} "
        f"base_valid={summary['base_valid_rate']:.4f}"
    )

    for alpha_text, rows in summary["top_changed_param_frac_by_alpha"].items():
        if not rows:
            continue
        print(f"[analysis] {label} alpha={alpha_text} top-changed_param_frac")
        show_rows = rows if limit is None else rows[:limit]
        for row in show_rows:
            print(
                "  "
                f"dim={row['dim']} "
                f"changed_param_frac={row['changed_param_frac']:.6f} "
                f"exact_drop={row['exact_drop']:.6f} "
                f"abs_cost_shift={row['abs_cost_shift']:.6f}"
            )

    for alpha_text, rows in summary["top_abs_cost_shift_by_alpha"].items():
        if not rows:
            continue
        print(f"[analysis] {label} alpha={alpha_text} top-abs_cost_shift")
        show_rows = rows if limit is None else rows[:limit]
        for row in show_rows:
            print(
                "  "
                f"dim={row['dim']} "
                f"abs_cost_shift={row['abs_cost_shift']:.6f} "
                f"changed_param_frac={row['changed_param_frac']:.6f} "
                f"exact_drop={row['exact_drop']:.6f}"
            )


@torch.no_grad()
def analyze_intervention_split(
    label: str,
    records: Sequence[JsonSampleRecord],
    prepared_cache: Dict[int, tuple[List[str], List[int]]],
    model: LatentParamVAE,
    registry: GeneratorRegistry,
    tokenizer: ParamTokenizer,
    device: torch.device,
    dims: Optional[Sequence[int]],
    alphas: Sequence[float],
    top_k: Optional[int],
    max_samples: Optional[int] = None,
) -> dict:
    records = list(records)
    if max_samples is not None and max_samples > 0:
        records = records[: int(max_samples)]

    if not records:
        print(f"[analysis] split={label} is empty")
        return {
            "label": label,
            "num_used": 0,
            "latent_dim": int(model.cfg.latent_dim),
            "base_exact_rate": 0.0,
            "base_valid_rate": 0.0,
            "base_cost_pred_mean": 0.0,
            "alphas": list(alphas),
            "all_dims_by_alpha": {},
            "top_changed_param_frac_by_alpha": {},
            "top_abs_cost_shift_by_alpha": {},
        }

    print(_format_split_header(label, len(records)))

    latent_dim = int(model.cfg.latent_dim)
    target_dims = list(range(latent_dim)) if dims is None else sorted(set(int(d) for d in dims))
    for dim_idx in target_dims:
        if dim_idx < 0 or dim_idx >= latent_dim:
            raise ValueError(f"Invalid dim index: {dim_idx}")

    alpha_list = [float(alpha) for alpha in alphas]
    if not alpha_list:
        raise ValueError("alphas must not be empty")

    per_alpha_dim_stats: Dict[float, Dict[int, InterventionStats]] = {
        alpha: {dim_idx: InterventionStats() for dim_idx in target_dims}
        for alpha in alpha_list
    }

    base_exact_count = 0
    base_valid_count = 0
    base_cost_preds: List[float] = []
    num_used = 0

    iterator = records
    if tqdm is not None:
        iterator = tqdm(records, desc=f"latent-intervention {label}")

    for record in iterator:
        try:
            ordered_names, ordered_values = prepared_cache[id(record)]
            z = _encode_record(
                record,
                ordered_names,
                ordered_values,
                model,
                tokenizer,
                device,
            )

            z_rows: List[torch.Tensor] = [z]
            intervention_keys: List[Tuple[float, int]] = []
            for alpha in alpha_list:
                for dim_idx in target_dims:
                    z_int = z.clone()
                    z_int[dim_idx] = z_int[dim_idx] + float(alpha)
                    z_rows.append(z_int)
                    intervention_keys.append((float(alpha), int(dim_idx)))

            z_batch = torch.stack(z_rows, dim=0)
            decoded_param_list, violation_list = _greedy_decode_many_from_z(
                record,
                ordered_names,
                z_batch,
                model,
                tokenizer,
                registry,
                device,
            )
            cost_pred_list = model.cost_head(z_batch).squeeze(-1).detach().cpu().tolist()

            base_params = decoded_param_list[0]
            base_violations = violation_list[0]
            base_exact = _exact_match(base_params, ordered_values, ordered_names)
            base_valid = len(base_violations) == 0
            base_cost_pred = float(cost_pred_list[0])

            base_exact_count += int(base_exact)
            base_valid_count += int(base_valid)
            base_cost_preds.append(base_cost_pred)

            for row_idx, (alpha, dim_idx) in enumerate(intervention_keys, start=1):
                int_params = decoded_param_list[row_idx]
                int_violations = violation_list[row_idx]
                int_exact = _exact_match(int_params, ordered_values, ordered_names)
                int_valid = len(int_violations) == 0
                int_cost_pred = float(cost_pred_list[row_idx])

                stats = per_alpha_dim_stats[alpha][dim_idx]
                stats.exact_match_count += int(int_exact)
                stats.valid_count += int(int_valid)
                stats.changed_param_frac_sum += _param_change_fraction(base_params, int_params, ordered_names)
                stats.abs_cost_shift_sum += abs(int_cost_pred - base_cost_pred)
                stats.signed_cost_shift_sum += (int_cost_pred - base_cost_pred)

            num_used += 1
            _print_running_status(
                label,
                str(record.sample_id),
                num_used,
                base_exact_count,
                base_valid_count,
                per_alpha_dim_stats,
            )

        except Exception as err:
            print(f"[analysis] skip sample={record.sample_id} error={type(err).__name__}: {err}")

    if num_used == 0:
        raise RuntimeError(f"No usable samples for split={label}")

    base_exact_rate = base_exact_count / max(num_used, 1)
    base_valid_rate = base_valid_count / max(num_used, 1)

    all_dims_by_alpha: Dict[str, List[dict]] = {}
    top_changed_param_frac_by_alpha: Dict[str, List[dict]] = {}
    top_abs_cost_shift_by_alpha: Dict[str, List[dict]] = {}

    for alpha in alpha_list:
        alpha_text = f"{alpha:g}"
        dim_rows: List[dict] = []
        for dim_idx in target_dims:
            row = {"dim": int(dim_idx), "alpha": float(alpha)}
            row.update(
                per_alpha_dim_stats[alpha][dim_idx].to_dict(
                    num_used,
                    base_exact_rate,
                    base_valid_rate,
                )
            )
            dim_rows.append(row)

        all_dims_by_alpha[alpha_text] = dim_rows
        top_changed_param_frac_by_alpha[alpha_text] = sorted(
            dim_rows,
            key=lambda row: row["changed_param_frac"],
            reverse=True,
        )
        top_abs_cost_shift_by_alpha[alpha_text] = sorted(
            dim_rows,
            key=lambda row: row["abs_cost_shift"],
            reverse=True,
        )

    summary = {
        "label": label,
        "num_used": int(num_used),
        "latent_dim": int(latent_dim),
        "base_exact_rate": float(base_exact_rate),
        "base_valid_rate": float(base_valid_rate),
        "base_cost_pred_mean": float(sum(base_cost_preds) / max(len(base_cost_preds), 1)),
        "alphas": list(alpha_list),
        "all_dims_by_alpha": all_dims_by_alpha,
        "top_changed_param_frac_by_alpha": top_changed_param_frac_by_alpha,
        "top_abs_cost_shift_by_alpha": top_abs_cost_shift_by_alpha,
    }

    _print_analysis_summary(label, summary, top_k)
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Per-dim latent intervention analysis. "
            "For each dim j and alpha, decode from z with z_j += alpha."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to checkpoint file such as last.pt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device from checkpoint config, e.g. cpu or cuda",
    )
    parser.add_argument(
        "--include-val",
        action="store_true",
        help="Also analyze the validation split",
    )
    parser.add_argument(
        "--val-only",
        action="store_true",
        help="Analyze only the validation split",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit records per split for faster analysis",
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="*",
        default=None,
        help="Only run intervention on selected dims. Omit for all dims.",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[-2.0, -1.0, 1.0, 2.0],
        help="Intervention strengths. Example: --alphas -1.0 -0.5 0.5 1.0",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="How many top entries to print for each alpha block. Omit for full output.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save full summary JSON",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    raw_payload = torch.load(checkpoint_path, map_location="cpu")
    config = _config_from_payload(raw_payload)
    if args.device is not None:
        config.train.device = str(args.device)

    seed_everything(config.data.seed)
    device = resolve_device(config.train.device)
    configure_runtime(config, device)

    tokenizer = ParamTokenizer.from_checkpoint_payload(raw_payload)
    model = LatentParamVAE(
        vocab_size=len(tokenizer.id_to_token),
        num_vars=len(tokenizer.id_to_var),
        cfg=config.model,
    )
    load_checkpoint(checkpoint_path, model)
    model.to(device)
    model.eval()

    registry = GeneratorRegistry(config.data.network_info_folder)
    records = _load_records_from_config(config)
    prepared_cache = _prepare_record_cache(records, registry, include_budget=budget_enabled(config))
    valid_records = _filter_invalid_records_if_needed(
        records,
        prepared_cache,
        registry,
        enabled=bool(config.data.filter_invalid_records),
    )

    train_records, val_records, _ = split_records(
        valid_records,
        config.data.train_ratio,
        config.data.val_ratio,
        config.data.test_ratio,
        config.data.seed,
    )

    print(f"[analysis] checkpoint={checkpoint_path}")
    print(f"[analysis] device={device}")
    print(f"[analysis] total_records={len(valid_records)}")
    print(f"[analysis] train_records={len(train_records)}")
    print(f"[analysis] val_records={len(val_records)}")
    print(f"[analysis] alphas={list(args.alphas)}")

    all_summaries: Dict[str, dict] = {}

    if not args.val_only:
        all_summaries["train"] = analyze_intervention_split(
            label="train",
            records=train_records,
            prepared_cache=prepared_cache,
            model=model,
            registry=registry,
            tokenizer=tokenizer,
            device=device,
            dims=args.dims,
            alphas=args.alphas,
            top_k=args.top_k,
            max_samples=args.max_samples,
        )

    if (args.include_val or args.val_only) and val_records:
        all_summaries["val"] = analyze_intervention_split(
            label="val",
            records=val_records,
            prepared_cache=prepared_cache,
            model=model,
            registry=registry,
            tokenizer=tokenizer,
            device=device,
            dims=args.dims,
            alphas=args.alphas,
            top_k=args.top_k,
            max_samples=args.max_samples,
        )
    elif args.val_only:
        raise ValueError("Validation split is empty, but --val-only was requested")

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(all_summaries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[analysis] saved_json={output_path}")


if __name__ == "__main__":
    main()
