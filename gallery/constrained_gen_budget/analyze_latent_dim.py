from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
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
class DimAblationStats:
    exact_match_count: int = 0
    valid_count: int = 0
    changed_param_frac_sum: float = 0.0
    abs_cost_shift_sum: float = 0.0
    signed_cost_shift_sum: float = 0.0

    def to_dict(self, sample_count: int, base_exact_rate: float, base_valid_rate: float) -> dict:
        denom = max(int(sample_count), 1)
        ablated_exact_rate = self.exact_match_count / denom
        ablated_valid_rate = self.valid_count / denom
        return {
            "ablated_exact_rate": float(ablated_exact_rate),
            "exact_drop": float(base_exact_rate - ablated_exact_rate),
            "ablated_valid_rate": float(ablated_valid_rate),
            "valid_drop": float(base_valid_rate - ablated_valid_rate),
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    mu, logvar, z, _ = model.encode(enc_ids, enc_var_ids, enc_pad, deterministic=True)
    return mu[0].detach().clone(), logvar[0].detach().clone(), z[0].detach().clone()


@torch.no_grad()
def _greedy_decode_from_z(
    record: JsonSampleRecord,
    ordered_names: Sequence[str],
    z: torch.Tensor,
    model: LatentParamVAE,
    tokenizer: ParamTokenizer,
    registry: GeneratorRegistry,
    device: torch.device,
) -> tuple[Dict[str, int], List[str]]:
    oracle = registry.build_oracle_from_record(record)

    z = z.to(device=device, dtype=torch.float32).view(1, -1)
    memory = model.latent_to_memory(z).view(1, model.cfg.latent_token_count, model.cfg.d_model)

    decoder_input_ids: List[int] = [tokenizer.bos_id]
    decoded_params: Dict[str, int] = {}

    for var_name in ordered_names:
        candidate_values = list(oracle.candidate_values(var_name))
        if not candidate_values:
            raise RuntimeError(f"No legal candidates returned for {var_name}")
        if len(candidate_values) == 1:
            pred_value = int(candidate_values[0])
            pred_token_id = _fixed_token_id_for_value(tokenizer, var_name, pred_value)
            oracle.assign(var_name, pred_value)
            decoded_params[var_name] = int(pred_value)
            decoder_input_ids.append(pred_token_id)
            continue

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
            z,
            decoder_pad_mask=step_input.eq(tokenizer.pad_id),
        )
        step_logits = logits[0, -1].masked_fill(~token_mask, float("-inf"))
        pred_token_id = int(torch.argmax(step_logits).item())
        pred_value = _resolve_decoded_value(tokenizer, var_name, pred_token_id, candidate_values)

        oracle.assign(var_name, pred_value)
        decoded_params[var_name] = int(pred_value)
        decoder_input_ids.append(pred_token_id)

    final_violations = list(oracle.final_violations())
    return decoded_params, final_violations


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


def _print_analysis_summary(label: str, summary: dict, top_k: Optional[int]) -> None:
    limit = None if top_k is None or int(top_k) <= 0 else int(top_k)

    header_parts = [
        f"[analysis] {label} used_samples={summary['num_used']}",
        f"latent_dim={summary['latent_dim']}",
    ]
    if summary.get("active_unit_count") is not None:
        header_parts.append(
            f"active_units={summary['active_unit_count']}/{summary['latent_dim']}"
        )
    if summary.get("base_exact_rate") is not None:
        header_parts.append(f"base_exact={summary['base_exact_rate']:.4f}")
    if summary.get("base_valid_rate") is not None:
        header_parts.append(f"base_valid={summary['base_valid_rate']:.4f}")
    print(" ".join(header_parts))

    def _show(rows: List[dict], title: str, keys: Sequence[str]) -> None:
        if not rows:
            return
        print(f"[analysis] {label} {title}")
        if limit is not None:
            rows = rows[:limit]
        for row in rows:
            parts = [f"dim={row['dim']}"]
            for key in keys:
                if key not in row:
                    continue
                value = row[key]
                if isinstance(value, bool):
                    parts.append(f"{key}={int(value)}")
                else:
                    parts.append(f"{key}={value:.6f}")
            print("  " + "  ".join(parts))

    _show(summary["top_var_mu"], "top-var_mu", ["var_mu", "mean_kl", "active_unit"])
    _show(summary["top_mean_kl"], "top-mean_kl", ["mean_kl", "var_mu", "active_unit"])
    _show(
        summary["top_changed_param_frac"],
        "top-changed_param_frac",
        ["changed_param_frac", "exact_drop", "abs_cost_shift"],
    )
    _show(
        summary["top_abs_cost_shift"],
        "top-abs_cost_shift",
        ["abs_cost_shift", "changed_param_frac", "exact_drop"],
    )


def _print_running_decoder_status(
    label: str,
    sample_id: str,
    num_used: int,
    base_exact_count: int,
    base_valid_count: int,
    per_dim_stats: Dict[int, DimAblationStats],
) -> None:
    if num_used <= 0:
        return

    top_changed_dim = None
    top_changed_value = -1.0
    top_cost_dim = None
    top_cost_value = -1.0
    for dim_idx, stats in per_dim_stats.items():
        changed_value = stats.changed_param_frac_sum / num_used
        if changed_value > top_changed_value:
            top_changed_value = changed_value
            top_changed_dim = dim_idx
        cost_value = stats.abs_cost_shift_sum / num_used
        if cost_value > top_cost_value:
            top_cost_value = cost_value
            top_cost_dim = dim_idx

    print(
        f"[decoder-live] split={label} sample={sample_id} "
        f"used={num_used} "
        f"base_exact={base_exact_count / num_used:.4f} "
        f"base_valid={base_valid_count / num_used:.4f} "
        f"top_changed_dim={top_changed_dim} "
        f"top_changed={max(top_changed_value, 0.0):.4f} "
        f"top_cost_dim={top_cost_dim} "
        f"top_cost_shift={max(top_cost_value, 0.0):.4f}"
    )


@torch.no_grad()
def analyze_latent_split(
    label: str,
    records: Sequence[JsonSampleRecord],
    prepared_cache: Dict[int, tuple[List[str], List[int]]],
    model: LatentParamVAE,
    registry: GeneratorRegistry,
    tokenizer: ParamTokenizer,
    device: torch.device,
    dims: Optional[Sequence[int]],
    top_k: Optional[int],
    au_threshold: float,
    run_encoder: bool,
    run_decoder: bool,
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
            "active_unit_count": None,
            "base_exact_rate": None,
            "base_valid_rate": None,
            "all_dims": [],
            "top_var_mu": [],
            "top_mean_kl": [],
            "top_changed_param_frac": [],
            "top_abs_cost_shift": [],
        }

    print(_format_split_header(label, len(records)))

    latent_dim = int(model.cfg.latent_dim)
    target_dims: List[int] = []
    if run_decoder:
        target_dims = list(range(latent_dim)) if dims is None else sorted(set(int(d) for d in dims))
        for dim_idx in target_dims:
            if dim_idx < 0 or dim_idx >= latent_dim:
                raise ValueError(f"Invalid dim index: {dim_idx}")

    mu_rows: List[np.ndarray] = []
    logvar_rows: List[np.ndarray] = []
    per_dim_stats: Dict[int, DimAblationStats] = {
        dim_idx: DimAblationStats() for dim_idx in target_dims
    }

    base_exact_count = 0
    base_valid_count = 0
    base_cost_preds: List[float] = []
    num_used = 0

    iterator = records
    if tqdm is not None:
        iterator = tqdm(records, desc=f"latent-analysis {label}")

    for record in iterator:
        try:
            ordered_names, ordered_values = prepared_cache[id(record)]

            mu, logvar, z = _encode_record(
                record,
                ordered_names,
                ordered_values,
                model,
                tokenizer,
                device,
            )
            if run_encoder:
                mu_rows.append(mu.detach().cpu().numpy())
                logvar_rows.append(logvar.detach().cpu().numpy())

            if run_decoder:
                z_batch = z.unsqueeze(0).repeat(len(target_dims) + 1, 1)
                for row_idx, dim_idx in enumerate(target_dims, start=1):
                    z_batch[row_idx, dim_idx] = 0.0

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

                for ablated_idx, dim_idx in enumerate(target_dims, start=1):
                    ab_params = decoded_param_list[ablated_idx]
                    ab_violations = violation_list[ablated_idx]
                    ab_exact = _exact_match(ab_params, ordered_values, ordered_names)
                    ab_valid = len(ab_violations) == 0
                    ab_cost_pred = float(cost_pred_list[ablated_idx])

                    stats = per_dim_stats[dim_idx]
                    stats.exact_match_count += int(ab_exact)
                    stats.valid_count += int(ab_valid)
                    stats.changed_param_frac_sum += _param_change_fraction(base_params, ab_params, ordered_names)
                    stats.abs_cost_shift_sum += abs(ab_cost_pred - base_cost_pred)
                    stats.signed_cost_shift_sum += (ab_cost_pred - base_cost_pred)

            num_used += 1
            if run_decoder:
                _print_running_decoder_status(
                    label,
                    str(record.sample_id),
                    num_used,
                    base_exact_count,
                    base_valid_count,
                    per_dim_stats,
                )

        except Exception as err:
            print(f"[analysis] skip sample={record.sample_id} error={type(err).__name__}: {err}")

    if num_used == 0:
        raise RuntimeError(f"No usable samples for split={label}")

    var_mu = None
    mean_abs_mu = None
    mean_kl = None
    active_units = None
    if run_encoder:
        mu_mat = np.stack(mu_rows, axis=0)
        logvar_mat = np.stack(logvar_rows, axis=0)
        kl_mat = 0.5 * (np.exp(logvar_mat) + mu_mat ** 2 - 1.0 - logvar_mat)

        var_mu = mu_mat.var(axis=0)
        mean_abs_mu = np.abs(mu_mat).mean(axis=0)
        mean_kl = kl_mat.mean(axis=0)
        active_units = (var_mu > float(au_threshold))

    base_exact_rate = None
    base_valid_rate = None
    if run_decoder:
        base_exact_rate = base_exact_count / max(num_used, 1)
        base_valid_rate = base_valid_count / max(num_used, 1)

    dim_rows: List[dict] = []
    dim_iter = range(latent_dim) if run_encoder else target_dims
    for dim_idx in dim_iter:
        row = {"dim": int(dim_idx)}
        if run_encoder:
            row.update(
                {
                    "var_mu": float(var_mu[dim_idx]),
                    "mean_abs_mu": float(mean_abs_mu[dim_idx]),
                    "mean_kl": float(mean_kl[dim_idx]),
                    "active_unit": bool(active_units[dim_idx]),
                }
            )
        if run_decoder and dim_idx in per_dim_stats:
            row.update(per_dim_stats[dim_idx].to_dict(num_used, base_exact_rate, base_valid_rate))
        dim_rows.append(row)

    top_var_mu = sorted(
        [row for row in dim_rows if "var_mu" in row],
        key=lambda row: row["var_mu"],
        reverse=True,
    )
    top_mean_kl = sorted(
        [row for row in dim_rows if "mean_kl" in row],
        key=lambda row: row["mean_kl"],
        reverse=True,
    )
    top_changed_param_frac = sorted(
        [row for row in dim_rows if "changed_param_frac" in row],
        key=lambda row: row["changed_param_frac"],
        reverse=True,
    )
    top_abs_cost_shift = sorted(
        [row for row in dim_rows if "abs_cost_shift" in row],
        key=lambda row: row["abs_cost_shift"],
        reverse=True,
    )

    summary = {
        "label": label,
        "num_used": int(num_used),
        "latent_dim": int(latent_dim),
        "run_encoder": bool(run_encoder),
        "run_decoder": bool(run_decoder),
        "active_unit_threshold": float(au_threshold),
        "active_unit_count": None if active_units is None else int(active_units.sum()),
        "base_exact_rate": None if base_exact_rate is None else float(base_exact_rate),
        "base_valid_rate": None if base_valid_rate is None else float(base_valid_rate),
        "base_cost_pred_mean": float(np.mean(base_cost_preds)) if base_cost_preds else 0.0,
        "all_dims": dim_rows,
        "top_var_mu": top_var_mu,
        "top_mean_kl": top_mean_kl,
        "top_changed_param_frac": top_changed_param_frac,
        "top_abs_cost_shift": top_abs_cost_shift,
    }

    _print_analysis_summary(label, summary, top_k)
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether latent dimensions are active and actually used by decoding. "
            "For each split, compute per-dim var(mu), mean KL, and zero-ablation effects."
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
        "--encoder",
        action="store_true",
        help="Run encoder-only statistics such as Var(mu) and mean KL",
    )
    parser.add_argument(
        "--decoder",
        action="store_true",
        help="Run decoder zero-ablation analysis",
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
        help="Only run zero-ablation on selected dims. Omit for all dims.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="How many top entries to print for each ranking block. Omit for full output.",
    )
    parser.add_argument(
        "--au-threshold",
        type=float,
        default=1e-2,
        help="Active-unit threshold on Var(mu_j)",
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

    run_encoder = bool(args.encoder or not args.decoder)
    run_decoder = bool(args.decoder or not args.encoder)

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
    print(f"[analysis] run_encoder={int(run_encoder)} run_decoder={int(run_decoder)}")
    print(f"[analysis] total_records={len(valid_records)}")
    print(f"[analysis] train_records={len(train_records)}")
    print(f"[analysis] val_records={len(val_records)}")

    all_summaries: Dict[str, dict] = {}

    if not args.val_only:
        all_summaries["train"] = analyze_latent_split(
            label="train",
            records=train_records,
            prepared_cache=prepared_cache,
            model=model,
            registry=registry,
            tokenizer=tokenizer,
            device=device,
            dims=args.dims,
            top_k=args.top_k,
            au_threshold=float(args.au_threshold),
            run_encoder=run_encoder,
            run_decoder=run_decoder,
            max_samples=args.max_samples,
        )

    if (args.include_val or args.val_only) and val_records:
        all_summaries["val"] = analyze_latent_split(
            label="val",
            records=val_records,
            prepared_cache=prepared_cache,
            model=model,
            registry=registry,
            tokenizer=tokenizer,
            device=device,
            dims=args.dims,
            top_k=args.top_k,
            au_threshold=float(args.au_threshold),
            run_encoder=run_encoder,
            run_decoder=run_decoder,
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
