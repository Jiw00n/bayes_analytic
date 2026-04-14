from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from latent_model_budget.adapter import GeneratorRegistry, JsonSampleRecord, load_json_samples, split_records
from latent_model_budget.config import DataConfig, EvalConfig, ExperimentConfig, ModelConfig, TrainConfig, WandbConfig
from latent_model_budget.dataset import (
    PreparedSample,
    _build_prepared_sample,
    _candidate_mask_cache_path_for_workload,
    _expand_json_paths,
    _get_generator_for_record,
    _normalize_param_signature,
    _extract_budget_specs,
    _apply_cached_order_metadata,
    budget_enabled,
    collate_prepared_samples,
    get_model_param_order,
)
from latent_model_budget.inference import greedy_decode_batch
from latent_model_budget.model import LatentParamVAE
from latent_model_budget.tokenizer import ParamTokenizer
from latent_model_budget.runtime_utils import (
    configure_runtime,
    load_checkpoint,
    resolve_device,
    seed_everything,
)
from latent_model_budget.train_eval import (
    _batch_to_device,
    _build_teacher_forcing_candidate_masks,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


DEFAULT_CHECKPOINT_PATH = "/root/work/tvm-ansor/gallery/constrained_gen_budget/checkpoints/last.pt"


@dataclass
class VariableStats:
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / max(self.total, 1)


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
            order,
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


def _build_samples_for_records(
    records: Sequence[JsonSampleRecord],
    prepared_cache: Dict[int, tuple[List[str], List[int]]],
    tokenizer: ParamTokenizer,
) -> List[PreparedSample]:
    samples: List[PreparedSample] = []
    for record in records:
        order, values = prepared_cache[id(record)]
        samples.append(
            _build_prepared_sample(
                record,
                order,
                values,
                tokenizer,
                registry=None,
                include_candidate_masks=False,
            )
        )
    return samples


def _format_split_header(label: str, sample_count: int) -> str:
    return f"[analysis] split={label} samples={sample_count}"


def _print_analysis_summary(
    label: str,
    samples: Sequence[PreparedSample],
    top_k: int | None,
    token_correct: int,
    token_total: int,
    exact_count: int,
    per_var_stats: Dict[str, VariableStats],
    first_error_pos_counts: Dict[int, int],
    first_error_var_counts: Dict[str, int],
) -> None:
    limit = None if top_k is None or int(top_k) <= 0 else int(top_k)
    print(
        f"[analysis] {label} token_acc={token_correct / max(token_total, 1):.4f} "
        f"exact={exact_count / max(len(samples), 1):.4f} ({exact_count}/{len(samples)})"
    )

    # worst_vars = sorted(
    #     per_var_stats.items(),
    #     key=lambda item: (item[1].accuracy, item[1].total, item[0]),
    # )

    worst_vars = per_var_stats.items()
    if limit is not None:
        worst_vars = worst_vars[:limit]
    print(f"[analysis] {label} lowest-accuracy vars")
    for name, stats in worst_vars:
        print(
            f"  {name}: acc={stats.accuracy:.4f} "
            f"({stats.correct}/{stats.total})"
        )

    print(f"[analysis] {label} first-error positions")
    first_error_positions = sorted(first_error_pos_counts.items(), key=lambda item: (-item[1], item[0]))
    if limit is not None:
        first_error_positions = first_error_positions[:limit]
    for pos, count in first_error_positions:
        print(f"  pos={pos}: count={count}")

    print(f"[analysis] {label} first-error vars")
    first_error_vars = sorted(first_error_var_counts.items(), key=lambda item: (-item[1], item[0]))
    if limit is not None:
        first_error_vars = first_error_vars[:limit]
    for name, count in first_error_vars:
        print(f"  {name}: count={count}")


def _load_candidate_mask_cache(
    config: ExperimentConfig,
    records: Sequence[JsonSampleRecord],
) -> Dict[str, torch.Tensor]:
    cached_masks: Dict[str, torch.Tensor] = {}
    workload_sigs = sorted(
        {
            (str(record.workload_key), str(record.target_kind))
            for record in records
            if record.workload_key and record.target_kind
        }
    )
    for workload_key, target_kind in workload_sigs:
        cache_path = _candidate_mask_cache_path_for_workload(config, workload_key, target_kind)
        if not cache_path.exists():
            continue
        payload = torch.load(cache_path, map_location="cpu")
        sample_masks = payload.get("sample_masks", {})
        for sample_id, mask in sample_masks.items():
            cached_masks[str(sample_id)] = mask.clone().to(dtype=torch.bool, device="cpu")
        print(
            f"[analysis] loaded candidate mask cache: path={cache_path} "
            f"count={len(sample_masks)}"
        )
    return cached_masks


@torch.no_grad()
def analyze_teacher_forcing_dataset(
    label: str,
    samples: Sequence[PreparedSample],
    model: LatentParamVAE,
    registry: GeneratorRegistry,
    tokenizer: ParamTokenizer,
    device: torch.device,
    batch_size: int,
    top_k: int | None,
    cached_candidate_masks: Dict[str, torch.Tensor] | None = None,
) -> None:
    samples = list(samples)
    if not samples:
        print(f"[analysis] split={label} is empty")
        return

    print(_format_split_header(label, len(samples)))
    per_var_stats: Dict[str, VariableStats] = {}
    first_error_pos_counts: Dict[int, int] = {}
    first_error_var_counts: Dict[str, int] = {}
    exact_count = 0
    token_correct = 0
    token_total = 0

    stride = max(int(batch_size), 1)
    iterator = range(0, len(samples), stride)
    if tqdm is not None:
        iterator = tqdm(iterator, desc=f"teacher-forcing {label}", total=(len(samples) + stride - 1) // stride)

    for start in iterator:
        batch_samples = samples[start:start + stride]
        batch = collate_prepared_samples(batch_samples, tokenizer)
        batch = _batch_to_device(batch, device)
        candidate_masks = None
        if cached_candidate_masks:
            cached_rows = []
            cache_ok = True
            for sample in batch_samples:
                mask = cached_candidate_masks.get(str(sample.sample_id))
                if mask is None or int(mask.shape[-1]) != len(tokenizer.id_to_token):
                    cache_ok = False
                    break
                cached_rows.append(mask)
            if cache_ok and cached_rows:
                candidate_masks = torch.stack(cached_rows, dim=0).to(device=device, non_blocking=device.type == "cuda")

        if candidate_masks is None:
            candidate_masks = _build_teacher_forcing_candidate_masks(
                batch,
                registry,
                tokenizer,
                device=device,
                debug_invalid_step=False,
            )

        out = model(
            batch["encoder_token_ids"],
            batch["encoder_var_ids"],
            batch["decoder_input_ids"],
            batch["decoder_var_ids"],
            pad_token_id=tokenizer.pad_id,
        )
        masked_logits = out.logits.masked_fill(~candidate_masks, float("-inf"))
        pred_ids = torch.argmax(masked_logits, dim=-1)
        targets = batch["target_ids"]
        valid_mask = targets.ne(tokenizer.pad_id)
        correct_mask = pred_ids.eq(targets) & valid_mask

        token_correct += int(correct_mask.sum().item())
        token_total += int(valid_mask.sum().item())

        ordered_names = batch["ordered_param_names"]
        for row_idx, sample_names in enumerate(ordered_names):
            first_error_found = False
            sample_exact = True
            for pos, var_name in enumerate(sample_names):
                if not bool(valid_mask[row_idx, pos].item()):
                    continue
                stats = per_var_stats.setdefault(str(var_name), VariableStats())
                stats.total += 1
                is_correct = bool(correct_mask[row_idx, pos].item())
                if is_correct:
                    stats.correct += 1
                elif not first_error_found:
                    first_error_found = True
                    sample_exact = False
                    first_error_pos_counts[pos] = first_error_pos_counts.get(pos, 0) + 1
                    first_error_var_counts[str(var_name)] = first_error_var_counts.get(str(var_name), 0) + 1
            if sample_exact:
                exact_count += 1

    _print_analysis_summary(
        label,
        samples,
        top_k,
        token_correct,
        token_total,
        exact_count,
        per_var_stats,
        first_error_pos_counts,
        first_error_var_counts,
    )


@torch.no_grad()
def analyze_autoregressive_dataset(
    label: str,
    samples: Sequence[PreparedSample],
    model: LatentParamVAE,
    registry: GeneratorRegistry,
    tokenizer: ParamTokenizer,
    device: torch.device,
    batch_size: int,
    top_k: int | None,
) -> None:
    samples = list(samples)
    if not samples:
        print(f"[analysis] split={label} is empty")
        return

    print(_format_split_header(label, len(samples)))
    per_var_stats: Dict[str, VariableStats] = {}
    first_error_pos_counts: Dict[int, int] = {}
    first_error_var_counts: Dict[str, int] = {}
    exact_count = 0
    token_correct = 0
    token_total = 0

    stride = max(int(batch_size), 1)
    iterator = range(0, len(samples), stride)
    if tqdm is not None:
        iterator = tqdm(iterator, desc=f"autoregressive {label}", total=(len(samples) + stride - 1) // stride)

    for start in iterator:
        batch_samples = samples[start:start + stride]
        results = greedy_decode_batch(model, batch_samples, registry, tokenizer, device)
        for sample, result in zip(batch_samples, results):
            sample_exact = True
            first_error_found = False
            for pos, (var_name, gold_value) in enumerate(zip(sample.ordered_param_names, sample.ordered_param_values)):
                pred_value = int(result.predicted_param_dict[var_name])
                stats = per_var_stats.setdefault(str(var_name), VariableStats())
                stats.total += 1
                token_total += 1
                if pred_value == int(gold_value):
                    stats.correct += 1
                    token_correct += 1
                else:
                    sample_exact = False
                    if not first_error_found:
                        first_error_found = True
                        first_error_pos_counts[pos] = first_error_pos_counts.get(pos, 0) + 1
                        first_error_var_counts[str(var_name)] = first_error_var_counts.get(str(var_name), 0) + 1
            if sample_exact:
                exact_count += 1

    _print_analysis_summary(
        label,
        samples,
        top_k,
        token_correct,
        token_total,
        exact_count,
        per_var_stats,
        first_error_pos_counts,
        first_error_var_counts,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze latent-model reconstruction quality for a checkpoint. "
            "Teacher forcing mode uses cached candidate masks when available; "
            "autoregressive mode ignores the cache and decodes on the fly."
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
        "--batch-size",
        type=int,
        default=None,
        help="Override evaluation batch size",
    )
    parser.add_argument(
        "--include-val",
        action="store_true",
        help="Also analyze the validation split",
    )
    parser.add_argument(
        "--teacher-forcing",
        action="store_true",
        help="Use teacher forcing analysis and read candidate-mask cache when available",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="How many worst variables / first-error entries to print. Omit for full output.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    raw_payload = torch.load(checkpoint_path, map_location="cpu")
    config = _config_from_payload(raw_payload)
    config.train.precompute_candidate_masks = False
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

    train_samples = _build_samples_for_records(train_records, prepared_cache, tokenizer)
    val_samples = _build_samples_for_records(val_records, prepared_cache, tokenizer)
    batch_size = int(args.batch_size or config.eval.batch_size or config.train.batch_size)
    analysis_records: List[JsonSampleRecord] = list(train_records)
    if args.include_val:
        analysis_records.extend(val_records)

    print(f"[analysis] checkpoint={checkpoint_path}")
    print(f"[analysis] device={device}")
    if args.teacher_forcing:
        cached_candidate_masks = _load_candidate_mask_cache(config, analysis_records)
        print(
            f"[analysis] mode=teacher_forcing "
            f"cache_entries={len(cached_candidate_masks)}"
        )
        analyze_teacher_forcing_dataset(
            "train",
            train_samples,
            model,
            registry,
            tokenizer,
            device,
            batch_size=batch_size,
            top_k=args.top_k,
            cached_candidate_masks=cached_candidate_masks,
        )
        if args.include_val and val_samples:
            analyze_teacher_forcing_dataset(
                "val",
                val_samples,
                model,
                registry,
                tokenizer,
                device,
                batch_size=batch_size,
                top_k=args.top_k,
                cached_candidate_masks=cached_candidate_masks,
            )
    else:
        print("[analysis] mode=autoregressive cache=disabled")
        analyze_autoregressive_dataset(
            "train",
            train_samples,
            model,
            registry,
            tokenizer,
            device,
            batch_size=batch_size,
            top_k=args.top_k,
        )
        if args.include_val and val_samples:
            analyze_autoregressive_dataset(
                "val",
                val_samples,
                model,
                registry,
                tokenizer,
                device,
                batch_size=batch_size,
                top_k=args.top_k,
            )


if __name__ == "__main__":
    main()
