from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .adapter import GeneratorRegistry, JsonSampleRecord, load_json_samples, split_records
from .shape_semantics import flatten_labels, semantic_labels_for_task
from .tokenizer import ParamTokenizer

# v12: shape prefix + dedicated PARAM_START token present in both encoder and
# decoder (replaces the old BOS). Encoder length = S + 1 + n; decoder length =
# S + n. candidate_masks align with decoder/target (shape S + n).
_CANDIDATE_MASK_CACHE_VERSION = "v12"
_CANDIDATE_MASK_CACHE_FLUSH_EVERY = 100


def _remap_cached_mask_to_tokenizer(
    mask: torch.Tensor,
    saved_id_to_token: Sequence[str],
    tokenizer: "ParamTokenizer",
) -> torch.Tensor:
    """Remap a cached candidate mask from its build-time vocab to the current
    tokenizer's vocab. Tokens absent from the current vocab are aggregated into
    ``tokenizer.unk_id`` (mirroring ``candidate_mask_from_values(allow_unk=True)``).

    The cached mask's last dim is indexed by token_id, which depends on the
    training split (and therefore ``data.seed``). This remapping lets a cache
    built under one seed be reused under another while keeping the tokenizer's
    train-only build order (for backward compatibility with existing runs).
    """
    saved_id_to_token = list(saved_id_to_token)
    current_vocab = list(tokenizer.id_to_token)
    if saved_id_to_token == current_vocab:
        return mask
    mask = mask.to(dtype=torch.bool, device="cpu")
    seq_len = int(mask.shape[0])
    new_mask = torch.zeros((seq_len, len(current_vocab)), dtype=torch.bool)
    unk_id = int(tokenizer.unk_id)
    for old_id, token in enumerate(saved_id_to_token):
        if old_id >= mask.shape[1]:
            break
        col = mask[:, old_id]
        if not col.any():
            continue
        new_id = tokenizer.token_to_id.get(token)
        if new_id is None:
            new_mask[:, unk_id] |= col
        else:
            new_mask[:, int(new_id)] |= col
    return new_mask


@dataclass
class PreparedSample:
    sample_id: str
    json_path: str
    sketch_index: int
    ordered_param_names: List[str]
    ordered_param_values: List[int]
    encoder_token_ids: List[int]
    encoder_var_ids: List[int]
    decoder_input_ids: List[int]
    decoder_var_ids: List[int]
    target_ids: List[int]
    cost: Optional[float]
    candidate_masks: Optional[torch.Tensor] = None
    workload_key: Optional[str] = None
    target_kind: Optional[str] = None
    target_model: Optional[str] = None
    task_desc: Optional[str] = None
    task_index: Optional[int] = None
    shape_token_ids: List[int] = field(default_factory=list)
    shape_var_ids: List[int] = field(default_factory=list)


@dataclass
class DatasetBundle:
    train_dataset: "LatentParamDataset"
    val_dataset: "LatentParamDataset"
    test_dataset: "LatentParamDataset"
    tokenizer: ParamTokenizer
    train_records: List[JsonSampleRecord]
    val_records: List[JsonSampleRecord]
    test_records: List[JsonSampleRecord]


class LatentParamDataset(Dataset):
    def __init__(self, samples: Sequence[PreparedSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> PreparedSample:
        return self.samples[idx]


# -----------------------------------------------------------------------------
# Generator lookup
# -----------------------------------------------------------------------------


def _get_generator_for_record(record: JsonSampleRecord, registry: GeneratorRegistry):
    """record의 workload signature를 우선 사용해 generator를 복원한다."""
    if hasattr(registry, "get_generator_from_record"):
        return registry.get_generator_from_record(record)

    get_generator = registry.get_generator
    try:
        return get_generator(
            workload_key=record.workload_key,
            target_kind=record.target_kind,
            task_index=record.task_index,
            sketch_index=record.sketch_index,
        )
    except TypeError:
        if record.task_index is None:
            raise ValueError(
                f"{record.sample_id} cannot be resolved: registry only supports "
                "legacy positional get_generator(task_index, sketch_index), but "
                "record.task_index is missing"
            )
        return get_generator(record.task_index, record.sketch_index)


# -----------------------------------------------------------------------------
# Sample preparation
# -----------------------------------------------------------------------------


def _prepare_single_sample(
    record: JsonSampleRecord,
    registry: GeneratorRegistry,
    tokenizer: ParamTokenizer | None,
    include_budget: bool = True,
    shape_tokens_and_vars: tuple[List[int], List[int]] | None = None,
) -> tuple[List[str], List[int], PreparedSample | None]:
    gen = _get_generator_for_record(record, registry)
    order = get_model_param_order(gen, include_budget=include_budget)

    missing = [name for name in order if name not in record.params]
    if missing:
        raise ValueError(f"{record.sample_id} is missing ordered params: {missing}")

    ordered_values = [int(record.params[name]) for name in order]

    if tokenizer is None:
        return order, ordered_values, None

    shape_token_ids: List[int] = []
    shape_var_ids: List[int] = []
    if shape_tokens_and_vars is not None:
        shape_token_ids, shape_var_ids = shape_tokens_and_vars
    return order, ordered_values, _build_prepared_sample(
        record,
        order,
        ordered_values,
        tokenizer,
        shape_token_ids=shape_token_ids,
        shape_var_ids=shape_var_ids,
    )


def _build_prepared_sample(
    record: JsonSampleRecord,
    order: Sequence[str],
    ordered_values: Sequence[int],
    tokenizer: ParamTokenizer,
    registry: GeneratorRegistry | None = None,
    include_candidate_masks: bool = False,
    oracle=None,
    prefix_len: int = 0,
    shape_token_ids: Sequence[int] = (),
    shape_var_ids: Sequence[int] = (),
) -> PreparedSample:
    param_token_ids = tokenizer.encode_values(order, ordered_values)
    param_var_ids = tokenizer.encode_var_names(order)
    shape_token_list = list(int(v) for v in shape_token_ids)
    shape_var_list = list(int(v) for v in shape_var_ids)
    if len(shape_token_list) != len(shape_var_list):
        raise ValueError(
            f"shape_token_ids ({len(shape_token_list)}) and shape_var_ids "
            f"({len(shape_var_list)}) length mismatch for {record.sample_id}"
        )
    shape_len = len(shape_token_list)

    # Encoder: [shape | PARAM_START | params]   (length S + 1 + n).
    encoder_token_ids = (
        shape_token_list + [tokenizer.param_start_id] + param_token_ids
    )
    encoder_var_ids = (
        shape_var_list + [tokenizer.param_start_var_id] + param_var_ids
    )
    # Decoder input: [shape | PARAM_START | params[:-1]]   (length S + n).
    # Decoder uses target-var-ids convention: decoder_var_ids[k] = var id of
    # the token being predicted at position k, so positions S..S+n-1 carry
    # v0..v_{n-1}. Position S (PARAM_START input) predicts param_0 → var v0.
    if param_token_ids:
        decoder_input_ids = (
            shape_token_list + [tokenizer.param_start_id] + param_token_ids[:-1]
        )
    else:
        decoder_input_ids = shape_token_list + [tokenizer.param_start_id]
    decoder_var_ids = shape_var_list + list(param_var_ids)
    if not param_token_ids:
        decoder_var_ids = shape_var_list + [tokenizer.var_pad_id]
    # Target: shape positions are pad_id (ignored by loss); param positions
    # carry gold token ids.
    target_ids = [tokenizer.pad_id] * shape_len + param_token_ids

    candidate_masks = None
    if include_candidate_masks:
        if registry is None:
            raise ValueError("registry is required when include_candidate_masks=True")
        if oracle is None:
            oracle = registry.build_oracle_from_record(record)
            prefix_len = 0
        vocab_size = len(tokenizer.id_to_token)
        # Candidate masks align with decoder/target positions (length S + n).
        candidate_masks = torch.zeros((shape_len + len(order), vocab_size), dtype=torch.bool)
        # Shape-prefix rows pinned to pad_id so `masked_fill(~mask, -inf)` is
        # not all-masked at those positions; loss ignores them via
        # ignore_index=pad_id.
        if shape_len:
            candidate_masks[:shape_len, tokenizer.pad_id] = True
        for t, (name, value) in enumerate(zip(order, ordered_values)):
            row = shape_len + t
            gold_token = tokenizer.value_to_token(name, value)
            gold_id = tokenizer.token_to_id.get(gold_token, tokenizer.unk_id)
            try:
                mask_key = (
                    tuple((order[idx], int(ordered_values[idx])) for idx in range(t)),
                    str(name),
                )
                cached_mask = oracle.generator._lpm_mask_cache.get(mask_key)
                if cached_mask is None:
                    if t < prefix_len:
                        raise KeyError(f"missing cached mask for restored prefix: {name}")
                    candidates = oracle.candidate_values(name)
                    cached_mask = tokenizer.candidate_mask_from_values(name, candidates)
                    oracle.generator._lpm_mask_cache[mask_key] = cached_mask.clone()
                candidate_masks[row] = cached_mask
                if not candidate_masks[row, gold_id]:
                    raise ValueError("gold value is outside oracle candidates")
                if t >= prefix_len:
                    oracle.assign(name, value)
            except Exception:  # pylint: disable=broad-except
                candidate_masks[row].zero_()
                candidate_masks[row, gold_id] = True
                for rem_t, rem_name, rem_value in zip(range(t + 1, len(order)), order[t + 1:], ordered_values[t + 1:]):
                    rem_gold_token = tokenizer.value_to_token(rem_name, rem_value)
                    rem_gold_id = tokenizer.token_to_id.get(rem_gold_token, tokenizer.unk_id)
                    rem_row = shape_len + rem_t
                    candidate_masks[rem_row].zero_()
                    candidate_masks[rem_row, rem_gold_id] = True
                break

    sample = PreparedSample(
        sample_id=record.sample_id,
        json_path=record.json_path,
        sketch_index=record.sketch_index,
        ordered_param_names=list(order),
        ordered_param_values=list(ordered_values),
        encoder_token_ids=encoder_token_ids,
        encoder_var_ids=encoder_var_ids,
        decoder_input_ids=decoder_input_ids,
        decoder_var_ids=decoder_var_ids,
        target_ids=target_ids,
        cost=record.cost,
        candidate_masks=candidate_masks,
        workload_key=record.workload_key,
        target_kind=record.target_kind,
        target_model=record.target_model,
        task_desc=record.task_desc,
        task_index=record.task_index,
        shape_token_ids=shape_token_list,
        shape_var_ids=shape_var_list,
    )
    return sample


# -----------------------------------------------------------------------------
# Collate
# -----------------------------------------------------------------------------


def collate_prepared_samples(
    batch: Sequence[PreparedSample],
    tokenizer: ParamTokenizer,
) -> Dict[str, object]:
    batch = list(batch)
    bsz = len(batch)
    # Encoder carries PARAM_START + params, decoder only has params[:-1] +
    # PARAM_START, so encoder length = decoder length + 1.
    enc_max_len = max(len(item.encoder_token_ids) for item in batch)
    dec_max_len = max(len(item.decoder_input_ids) for item in batch)
    pad_id = tokenizer.pad_id
    var_pad = tokenizer.var_pad_id

    enc_ids = torch.full((bsz, enc_max_len), pad_id, dtype=torch.long)
    enc_var_ids = torch.full((bsz, enc_max_len), var_pad, dtype=torch.long)
    dec_in_ids = torch.full((bsz, dec_max_len), pad_id, dtype=torch.long)
    dec_var_ids = torch.full((bsz, dec_max_len), var_pad, dtype=torch.long)
    tgt_ids = torch.full((bsz, dec_max_len), pad_id, dtype=torch.long)
    seq_lens = torch.zeros((bsz,), dtype=torch.long)
    candidate_masks = None
    if all(item.candidate_masks is not None for item in batch):
        candidate_masks = torch.zeros(
            (bsz, dec_max_len, len(tokenizer.id_to_token)),
            dtype=torch.bool,
        )

    costs = torch.zeros((bsz,), dtype=torch.float32)
    cost_mask = torch.zeros((bsz,), dtype=torch.bool)

    sample_ids: List[str] = []
    task_indices: List[Optional[int]] = []
    sketch_indices: List[int] = []
    workload_keys: List[Optional[str]] = []
    target_kinds: List[Optional[str]] = []
    target_models: List[Optional[str]] = []
    task_descs: List[Optional[str]] = []
    ordered_names: List[List[str]] = []
    ordered_values: List[List[int]] = []
    json_paths: List[str] = []

    for i, item in enumerate(batch):
        enc_n = len(item.encoder_token_ids)
        dec_n = len(item.decoder_input_ids)
        seq_lens[i] = dec_n
        enc_ids[i, :enc_n] = torch.tensor(item.encoder_token_ids, dtype=torch.long)
        enc_var_ids[i, :enc_n] = torch.tensor(item.encoder_var_ids, dtype=torch.long)
        dec_in_ids[i, :dec_n] = torch.tensor(item.decoder_input_ids, dtype=torch.long)
        dec_var_ids[i, :dec_n] = torch.tensor(item.decoder_var_ids, dtype=torch.long)
        tgt_ids[i, :dec_n] = torch.tensor(item.target_ids, dtype=torch.long)
        if candidate_masks is not None:
            candidate_masks[i, :dec_n] = item.candidate_masks

        if item.cost is not None and math.isfinite(float(item.cost)):
            costs[i] = float(item.cost)
            cost_mask[i] = True

        sample_ids.append(item.sample_id)
        task_indices.append(item.task_index)
        sketch_indices.append(item.sketch_index)
        workload_keys.append(item.workload_key)
        target_kinds.append(item.target_kind)
        target_models.append(item.target_model)
        task_descs.append(item.task_desc)
        ordered_names.append(list(item.ordered_param_names))
        ordered_values.append(list(item.ordered_param_values))
        json_paths.append(item.json_path)

    return {
        "encoder_token_ids": enc_ids,
        "encoder_var_ids": enc_var_ids,
        "decoder_input_ids": dec_in_ids,
        "decoder_var_ids": dec_var_ids,
        "target_ids": tgt_ids,
        "seq_lens": seq_lens,
        "candidate_masks": candidate_masks,
        "costs": costs,
        "cost_mask": cost_mask,
        "sample_ids": sample_ids,
        "task_indices": task_indices,
        "sketch_indices": sketch_indices,
        "workload_keys": workload_keys,
        "target_kinds": target_kinds,
        "target_models": target_models,
        "task_descs": task_descs,
        "ordered_param_names": ordered_names,
        "ordered_param_values": ordered_values,
        "json_paths": json_paths,
    }


# -----------------------------------------------------------------------------
# Bundle construction
# -----------------------------------------------------------------------------


def _expand_json_paths(entries: Sequence[str]) -> List[Path]:
    raw_paths: List[Path] = []
    for entry in entries:
        p = Path(entry)
        if p.is_dir():
            raw_paths.extend(sorted(p.glob("*.json")))
        else:
            raw_paths.append(p)
    return raw_paths


def _candidate_mask_cache_dir(config) -> Path:
    cache_dir = Path(config.train.checkpoint_dir).expanduser().resolve() / "candidate_mask_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def budget_enabled(config_or_payload=None) -> bool:
    if config_or_payload is None:
        return True

    data_cfg = getattr(config_or_payload, "data", None)
    if data_cfg is not None:
        return bool(getattr(data_cfg, "budget", True))

    if isinstance(config_or_payload, dict):
        data_payload = config_or_payload.get("data", {})
        if isinstance(data_payload, dict):
            return bool(data_payload.get("budget", True))

    return True


def filter_param_order(order: Sequence[str], *, include_budget: bool = True) -> List[str]:
    if include_budget:
        return [str(name) for name in order]
    return [
        str(name)
        for name in order
        if not (str(name).startswith("thread_budget") or str(name).startswith("vthread_budget"))
    ]


def get_model_param_order(gen, *, include_budget: bool = True) -> List[str]:
    full_order = list(gen.get_full_var_order_entries()["param_order"])
    return filter_param_order(full_order, include_budget=include_budget)


def _sanitize_workload_key(workload_key: str, target_kind: str) -> str:
    from modules.task_paths import clean_name

    safe_name = clean_name((workload_key, target_kind))
    alnum_count = sum(ch.isalnum() for ch in safe_name)
    if not safe_name or alnum_count < 8:
        digest = hashlib.sha256(f"{workload_key}|{target_kind}".encode("utf-8")).hexdigest()[:16]
        safe_name = f"taskkey_{digest}"
    return safe_name


_HW_PARAM_CACHE_TAGS: Dict[str, str] = {
    "max_vector_bytes": "vecBytes",
    "max_shared_memory_per_block": "shMem",
    "max_threads_per_block": "thrBlk",
    "max_thread_x": "thrX",
    "max_thread_y": "thrY",
    "max_thread_z": "thrZ",
    "max_vthread_extent": "vThread",
    "max_innermost_split_factor": "innerSplit",
    "warp_size": "warp",
}

_DISABLE_CONSTRAINT_CACHE_TAGS: Dict[str, str] = {
    "vectorize": "noVec",
    "shared_memory": "noShMem",
    "max_threads": "noThr",
    "max_vthread": "noVThr",
    "innermost_split": "noInner",
    "split_structure": "noSplit",
    "min_thread_extent": "noMinThr",
}


def _extract_generator_settings(config) -> tuple[Dict[str, object], List[str]]:
    gen_cfg = getattr(config, "generator", None)
    if gen_cfg is None:
        return {}, []
    hw_param = getattr(gen_cfg, "hw_param", None) or {}
    disable_constraint = getattr(gen_cfg, "disable_constraint", None) or []
    if isinstance(hw_param, dict):
        hw_param = dict(hw_param)
    else:
        hw_param = {}
    disable_constraint = [str(k) for k in disable_constraint]
    return hw_param, disable_constraint


def _generator_cache_suffix(config) -> str:
    """Suffix encoding hw_param/disable_constraint diffs from ScheduleGenerator defaults."""
    hw_param, disable_constraint = _extract_generator_settings(config)
    if not hw_param and not disable_constraint:
        return ""

    from modules.schedule_generator import ScheduleGenerator

    parts: List[str] = []
    defaults = ScheduleGenerator.DEFAULT_HW_PARAM
    for key in _HW_PARAM_CACHE_TAGS:
        if key not in hw_param:
            continue
        value = hw_param[key]
        if key in defaults and value == defaults[key]:
            continue
        parts.append(f"{_HW_PARAM_CACHE_TAGS[key]}{value}")

    default_enabled = set(ScheduleGenerator.DEFAULT_ENABLED_CONSTRAINT_KINDS)
    for kind in disable_constraint:
        if kind not in default_enabled:
            continue
        tag = _DISABLE_CONSTRAINT_CACHE_TAGS.get(kind)
        if tag is None:
            continue
        parts.append(tag)

    if not parts:
        return ""
    return "_" + "_".join(parts)


def _candidate_mask_cache_path_for_workload(config, workload_key: str, target_kind: str) -> Path:
    cache_dir = _candidate_mask_cache_dir(config)
    stem = _sanitize_workload_key(workload_key, target_kind)
    budget_tag = "with_budget" if budget_enabled(config) else "no_budget"
    compressed_tag = (
        "_compressed"
        if bool(getattr(getattr(config, "train", None), "use_compressed_teacher_forcing", False))
        else ""
    )
    generator_tag = _generator_cache_suffix(config)
    return cache_dir / (
        f"{stem}_{_CANDIDATE_MASK_CACHE_VERSION}_{budget_tag}"
        f"{compressed_tag}{generator_tag}.pt"
    )


def _normalize_param_signature(signature: Sequence[str] | None) -> tuple[str, ...]:
    if not signature:
        return tuple()
    normalized = []
    for name in signature:
        name = str(name)
        if name.startswith("thread_budget") or name.startswith("vthread_budget"):
            continue
        normalized.append(name)
    return tuple(normalized)


def _extend_domain_values(
    domain_values_by_name: Dict[str, Set[int]],
    name: str,
    domain,
) -> None:
    if isinstance(domain, list):
        lo, hi = int(domain[0]), int(domain[1])
        values = range(lo, hi + 1)
    else:
        values = (int(domain),)
    bucket = domain_values_by_name.setdefault(str(name), set())
    for value in values:
        bucket.add(int(value))


def _collect_generator_domain_values(
    gen,
    order: Sequence[str],
    *,
    include_budget: bool = True,
) -> Dict[str, List[int]]:
    domain_values_by_name: Dict[str, Set[int]] = {}
    split_domains = gen._build_split_domains()
    split_candidate_values: Dict[str, List[int]] = {}
    innermost_names = getattr(gen, "_innermost_names", set())
    innermost_limit = int(gen.hw.get("max_innermost_split_factor", 0)) or None

    for name in order:
        if name.startswith("ur_"):
            domain_values_by_name.setdefault(name, set()).update(
                int(value) for value in gen.pm.UNROLL_CANDIDATES
            )
            continue
        if name in split_domains:
            domain = split_domains[name]
            if isinstance(domain, list):
                extent = int(domain[1])
                values = [int(v) for v in gen._enumerate_split_candidates(name, extent)]
            else:
                values = [int(domain)]
            # Innermost split factors are capped at max_innermost_split_factor
            # by the sampler/constraints, so values exceeding it (e.g. the
            # full-extent divisor 100352 when extent=100352, limit=64) can
            # never appear as a parameter — drop them from vocab. We still
            # enumerate divisors of the real extent first, so legal values
            # like 7 (a divisor of 100352 that is ≤ 64) are preserved.
            if innermost_limit is not None and name in innermost_names:
                values = [v for v in values if v <= innermost_limit]
            split_candidate_values[name] = list(values)
            domain_values_by_name.setdefault(name, set()).update(values)

    if include_budget:
        for spec in getattr(gen, "_budget_specs", ()):
            budget_name = str(spec["name"])
            factor_lists: List[List[int]] = []
            for factor_name in spec.get("factor_names", ()):
                factor_values = split_candidate_values.get(str(factor_name))
                if not factor_values:
                    factor_lists = []
                    break
                factor_lists.append(list(factor_values))
            if factor_lists:
                reachable = gen.param_sampler._compute_reachable_products(
                    factor_lists,
                    int(spec["limit"]),
                )
                domain_values_by_name.setdefault(budget_name, set()).update(
                    int(value) for value in reachable
                )

    return {
        name: sorted(values)
        for name, values in domain_values_by_name.items()
    }


def _extract_budget_specs(gen) -> List[tuple[str, tuple[str, ...]]]:
    specs = []
    for spec in getattr(gen, "_budget_specs", ()):
        specs.append(
            (
                str(spec["name"]),
                tuple(str(name) for name in spec.get("factor_names", ())),
            )
        )
    return specs


def _apply_cached_order_metadata(
    record: JsonSampleRecord,
    order: Sequence[str],
    budget_specs: Sequence[tuple[str, tuple[str, ...]]],
) -> None:
    params = record.params
    for budget_name, factor_names in budget_specs:
        if budget_name in params:
            params[budget_name] = int(params[budget_name])
            continue
        if not factor_names or any(name not in params for name in factor_names):
            continue
        budget_value = 1
        for factor_name in factor_names:
            budget_value *= int(params[factor_name])
        params[budget_name] = int(budget_value)
    record.param_signature = tuple(str(name) for name in order if name in params)


def _save_candidate_mask_cache_files(
    config,
    subset: Sequence[JsonSampleRecord],
    cached_candidate_masks: Dict[str, torch.Tensor],
    cache_paths_by_workload: Dict[tuple[str, str], Path],
    tokenizer: "ParamTokenizer",
) -> None:
    sample_masks_by_workload: Dict[tuple[str, str], Dict[str, torch.Tensor]] = {}
    for record in subset:
        workload_key = record.workload_key
        target_kind = record.target_kind
        if not workload_key or not target_kind:
            continue
        mask = cached_candidate_masks.get(record.sample_id)
        if mask is None:
            continue
        workload_sig = (str(workload_key), str(target_kind))
        sample_masks_by_workload.setdefault(workload_sig, {})[record.sample_id] = (
            mask.clone().to(device="cpu")
        )

    id_to_token = list(tokenizer.id_to_token)
    for workload_sig, sample_masks in sample_masks_by_workload.items():
        cache_path = cache_paths_by_workload.get(workload_sig)
        if cache_path is None:
            cache_path = _candidate_mask_cache_path_for_workload(
                config,
                workload_sig[0],
                workload_sig[1],
            )
        torch.save(
            {
                "workload_key": workload_sig[0],
                "target_kind": workload_sig[1],
                "sample_masks": sample_masks,
                "id_to_token": id_to_token,
            },
            cache_path,
        )


def _validate_record_group_consistency(
    records: Sequence[JsonSampleRecord],
) -> None:
    """Records in one run must share task_desc, param count, and shape rank layout.

    Shape *values* can differ (that's the point); shape *structure* — number of
    tensors and rank per tensor — must not. Otherwise the encoder prefix
    layout becomes inconsistent and the caller almost certainly mixed
    different sketches by accident.
    """
    if not records:
        return
    ref = records[0]
    ref_desc = ref.task_desc
    ref_param_count = len(ref.param_signature) if ref.param_signature else None
    if ref.shapes is None:
        raise ValueError(
            f"Record {ref.sample_id} has no parsed shapes (workload_key missing?)"
        )
    ref_shape_layout = tuple(len(shape) for shape in ref.shapes)
    ref_ntensors = len(ref.shapes)
    for rec in records[1:]:
        if rec.task_desc != ref_desc:
            raise ValueError(
                f"task_desc mismatch across records: "
                f"{ref.sample_id}={ref_desc!r} vs {rec.sample_id}={rec.task_desc!r}"
            )
        rec_param_count = len(rec.param_signature) if rec.param_signature else None
        if rec_param_count != ref_param_count:
            raise ValueError(
                f"parameter count mismatch across records: "
                f"{ref.sample_id}={ref_param_count} vs "
                f"{rec.sample_id}={rec_param_count}"
            )
        if rec.shapes is None:
            raise ValueError(
                f"Record {rec.sample_id} has no parsed shapes (workload_key missing?)"
            )
        if len(rec.shapes) != ref_ntensors:
            raise ValueError(
                f"tensor count mismatch across records: "
                f"{ref.sample_id} has {ref_ntensors} tensors vs "
                f"{rec.sample_id} has {len(rec.shapes)}"
            )
        rec_shape_layout = tuple(len(shape) for shape in rec.shapes)
        if rec_shape_layout != ref_shape_layout:
            raise ValueError(
                f"shape rank layout mismatch across records: "
                f"{ref.sample_id}={ref_shape_layout} vs "
                f"{rec.sample_id}={rec_shape_layout}"
            )


def _collect_shape_tokens_and_labels(
    records: Sequence[JsonSampleRecord],
    registry: GeneratorRegistry,
) -> tuple[Dict[int, tuple[List[int], List[str]]], List[str], Set[int]]:
    """Returns (by_record_id, unique_flat_labels, unique_shape_values).

    ``by_record_id[id(record)]`` is the ``(shape_values, flat_labels)`` pair for
    that record. All records produce the same ``flat_labels`` as long as the
    group validation above has passed (same task_desc ⇒ same compute_dag
    structure ⇒ same semantic labels).
    """
    by_record: Dict[int, tuple[List[int], List[str]]] = {}
    canonical_labels: Optional[List[str]] = None
    unique_values: Set[int] = set()
    task_label_cache: Dict[Tuple[str, str], List[str]] = {}
    for record in records:
        if record.shapes is None:
            raise ValueError(
                f"Record {record.sample_id} has no parsed shapes"
            )
        shape_values: List[int] = []
        for shape in record.shapes:
            shape_values.extend(int(v) for v in shape)

        task_key = (str(record.workload_key), str(record.target_kind))
        labels = task_label_cache.get(task_key)
        if labels is None:
            task = registry._resolve_task(
                workload_key=record.workload_key,
                target_kind=record.target_kind,
                task_index=record.task_index,
            )
            nested = semantic_labels_for_task(task)
            # Structural check: labels must line up with record.shapes.
            if len(nested) != len(record.shapes):
                raise ValueError(
                    f"compute_dag tensor count ({len(nested)}) does not match "
                    f"parsed shapes ({len(record.shapes)}) for {record.sample_id}"
                )
            for tensor_idx, (tensor_labels, tensor_shape) in enumerate(zip(nested, record.shapes)):
                if len(tensor_labels) != len(tensor_shape):
                    raise ValueError(
                        f"compute_dag axis count mismatch at tensor {tensor_idx} "
                        f"for {record.sample_id}: labels={len(tensor_labels)} "
                        f"shape_rank={len(tensor_shape)}"
                    )
            labels = flatten_labels(nested)
            task_label_cache[task_key] = labels

        if len(labels) != len(shape_values):
            raise ValueError(
                f"semantic label count ({len(labels)}) does not match "
                f"flattened shape count ({len(shape_values)}) for {record.sample_id}"
            )
        if canonical_labels is None:
            canonical_labels = list(labels)
        elif labels != canonical_labels:
            raise ValueError(
                "semantic labels differ across records (compute_dag structure "
                f"is inconsistent): reference={canonical_labels} "
                f"got={labels} on {record.sample_id}"
            )
        by_record[id(record)] = (shape_values, list(labels))
        unique_values.update(shape_values)

    return by_record, (canonical_labels or []), unique_values


def build_dataset_bundle(config, registry: GeneratorRegistry) -> DatasetBundle:
    include_budget = budget_enabled(config)
    raw_paths = _expand_json_paths(config.data.json_paths)
    if not raw_paths:
        raise ValueError("No json_paths were provided")
    print(f"[dataset] expanding {len(raw_paths)} json path(s)")

    records: List[JsonSampleRecord] = []
    for path in raw_paths:
        print(f"[dataset] loading {path}")
        records.extend(load_json_samples(path))
    print(f"[dataset] loaded {len(records)} record(s)")

    _validate_record_group_consistency(records)
    shape_info_by_record, canonical_shape_labels, unique_shape_values = (
        _collect_shape_tokens_and_labels(records, registry)
    )
    print(
        f"[dataset] shape prefix: labels={canonical_shape_labels} "
        f"unique_values={len(unique_shape_values)}"
    )

    prepared_cache: Dict[int, tuple[List[str], List[int]]] = {}
    order_cache: Dict[tuple, Dict[str, object]] = {}
    domain_values_by_name: Dict[str, Set[int]] = {}
    print("[dataset] building ordered parameter cache")
    for idx, record in enumerate(records, start=1):
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
            budget_specs = _extract_budget_specs(gen)
            cached_meta = {
                "order": list(order),
                "budget_specs": list(budget_specs),
            }
            order_cache[order_key] = cached_meta
            generator_domain_values = _collect_generator_domain_values(
                gen,
                order,
                include_budget=include_budget,
            )
            # Skip dynamic-extent split steps (SplitStep immediately followed
            # by a vectorize AnnotationStep): their divisor-based initial
            # domain is meaningless because the extent varies per record.
            _vec_step_indices = getattr(gen, "_vectorize_split_step_indices", set())
            for name, values in generator_domain_values.items():
                if name.startswith("sp_"):
                    try:
                        _step_idx = int(name.split("_")[1])
                    except (ValueError, IndexError):
                        _step_idx = None
                    if _step_idx is not None and _step_idx in _vec_step_indices:
                        continue
                domain_values_by_name.setdefault(name, set()).update(int(v) for v in values)
        order = list(cached_meta["order"])
        budget_specs = list(cached_meta["budget_specs"])
        _apply_cached_order_metadata(record, order, budget_specs)
        missing = [name for name in order if name not in record.params]
        if missing:
            raise ValueError(f"{record.sample_id} is missing ordered params: {missing}")
        values = [int(record.params[name]) for name in order]
        prepared_cache[id(record)] = (order, values)
        if idx == len(records):
            print(
                f"[dataset] prepared {idx}/{len(records)} record(s); "
                f"unique_orders={len(order_cache)}"
            )

    # Records in the loaded set must share the *same parameter order* (not
    # just the same count) so that a single decoder var-id layout applies to
    # every sample.
    ref_id = id(records[0])
    ref_order, _ = prepared_cache[ref_id]
    for record in records[1:]:
        rec_order, _ = prepared_cache[id(record)]
        if rec_order != ref_order:
            raise ValueError(
                "parameter order differs across records: "
                f"{records[0].sample_id}={ref_order} vs "
                f"{record.sample_id}={rec_order}"
            )

    valid_records = list(records)
    if getattr(config.data, "filter_invalid_records", False):
        print("[dataset] filtering invalid records")
        filtered_records: List[JsonSampleRecord] = []
        invalid_count = 0
        for idx, record in enumerate(records, start=1):
            order, values = prepared_cache[id(record)]
            oracle = registry.build_oracle_from_record(record)
            if not oracle.validate_assignment(order, values):
                invalid_count += 1
                continue
            filtered_records.append(record)
            if idx % 500 == 0 or idx == len(records):
                print(
                    f"[dataset] validated {idx}/{len(records)} record(s); "
                    f"kept={len(filtered_records)} dropped={invalid_count}"
                )

        if invalid_count:
            print(
                f"[dataset] dropped {invalid_count} invalid records; "
                f"kept {len(filtered_records)} / {len(records)}"
            )
        if not filtered_records:
            raise ValueError("No valid records remain after legality filtering")
        valid_records = filtered_records

    train_records, val_records, test_records = split_records(
        valid_records,
        config.data.train_ratio,
        config.data.val_ratio,
        config.data.test_ratio,
        config.data.seed,
    )

    # Tokenizer build order is intentionally train-only (seed-dependent) to
    # stay compatible with existing checkpoints and wandb baselines. The cache
    # records its build-time ``id_to_token`` and is remapped on load when a
    # different seed produces a different token ordering.
    train_record_ids = {id(record) for record in train_records}
    train_orders: List[List[str]] = []
    train_values: List[List[int]] = []
    all_orders: List[List[str]] = []
    for record in valid_records:
        order, values = prepared_cache[id(record)]
        all_orders.append(order)
        if id(record) in train_record_ids:
            train_orders.append(order)
            train_values.append(values)

    # Register shape semantic labels as var-vocab entries and shape integer
    # values as tokens. Shape values share the numeric token namespace with
    # param values (``value_to_token`` returns str(int) for non-ur_ names).
    if canonical_shape_labels:
        all_orders = [list(canonical_shape_labels) + order for order in all_orders]
    domain_values_for_tokenizer = {
        name: sorted(values) for name, values in domain_values_by_name.items()
    }
    if unique_shape_values:
        domain_values_for_tokenizer["__shape__"] = sorted(int(v) for v in unique_shape_values)

    tokenizer = ParamTokenizer.build(
        train_ordered_names=train_orders,
        train_ordered_values=train_values,
        all_ordered_names=all_orders,
        domain_values_by_name=domain_values_for_tokenizer,
        pad_to_vocab_size=getattr(config.data, "pad_vocab_to", None),
    )
    print(
        f"[dataset] tokenizer built: vocab={len(tokenizer.id_to_token)} "
        f"vars={len(tokenizer.id_to_var)}"
    )

    precompute_candidate_masks = bool(getattr(config.train, "precompute_candidate_masks", False))
    cached_candidate_masks: Dict[str, torch.Tensor] = {}
    cache_paths_by_workload: Dict[tuple[str, str], Path] = {}
    # Precompute over ALL valid records (not just the current train+val split)
    # so the cache is seed-invariant: changing ``data.seed`` only reshuffles
    # which records land in train/val/test, never triggers recomputation.
    precompute_records = list(valid_records)
    if precompute_candidate_masks:
        print("[dataset] precomputing candidate masks for all valid records")
        precompute_workload_keys = sorted(
            {
                (record.workload_key, record.target_kind)
                for record in precompute_records
                if record.workload_key and record.target_kind
            }
        )
        for workload_key in precompute_workload_keys:
            cache_path = _candidate_mask_cache_path_for_workload(config, workload_key[0], workload_key[1])
            cache_paths_by_workload[workload_key] = cache_path
            if not cache_path.exists():
                continue
            print(f"[dataset] loading cached candidate masks from {cache_path}")
            payload = torch.load(cache_path, map_location="cpu")
            sample_masks = payload.get("sample_masks", {})
            saved_id_to_token = payload.get("id_to_token")
            if saved_id_to_token is None:
                print(
                    f"[dataset] cache at {cache_path} lacks id_to_token; "
                    f"ignoring (rebuild required)"
                )
                continue
            needs_remap = list(saved_id_to_token) != list(tokenizer.id_to_token)
            if needs_remap:
                print(
                    f"[dataset] remapping cached masks to current tokenizer "
                    f"(saved_vocab={len(saved_id_to_token)} "
                    f"current_vocab={len(tokenizer.id_to_token)})"
                )
            loaded_count = 0
            for sample_id, mask in sample_masks.items():
                m = mask.to(dtype=torch.bool, device="cpu")
                if needs_remap:
                    m = _remap_cached_mask_to_tokenizer(m, saved_id_to_token, tokenizer)
                cached_candidate_masks[str(sample_id)] = m.clone()
                loaded_count += 1
            print(
                f"[dataset] loaded cached masks for workload_key={workload_key} "
                f"count={loaded_count}"
            )

    def _shape_ids_for_record(record: JsonSampleRecord) -> tuple[List[int], List[int]]:
        shape_values, shape_labels = shape_info_by_record[id(record)]
        shape_token_ids = [
            tokenizer.token_to_id.get(str(int(v)), tokenizer.unk_id)
            for v in shape_values
        ]
        shape_var_ids = [tokenizer.var_to_id[label] for label in shape_labels]
        return shape_token_ids, shape_var_ids

    def build_samples(
        subset: Sequence[JsonSampleRecord],
        include_candidate_masks: bool,
        persist_cache: bool = False,
    ) -> List[PreparedSample]:
        if not include_candidate_masks:
            items: List[PreparedSample] = []
            total = len(subset)
            for idx, record in enumerate(subset, start=1):
                order, values = prepared_cache[id(record)]
                shape_token_ids, shape_var_ids = _shape_ids_for_record(record)
                items.append(
                    _build_prepared_sample(
                        record,
                        order,
                        values,
                        tokenizer,
                        registry=registry,
                        include_candidate_masks=False,
                        shape_token_ids=shape_token_ids,
                        shape_var_ids=shape_var_ids,
                    )
                )
                if idx % 1000 == 0 or idx == total:
                    print(f"[dataset] prepared samples {idx}/{total}")
            return items

        items_by_id: Dict[int, PreparedSample] = {}
        total = len(subset)
        sorted_subset = sorted(
            subset,
            key=lambda record: (
                record.workload_key or "",
                record.target_kind or "",
                -1 if record.task_index is None else int(record.task_index),
                int(record.sketch_index),
                tuple(record.param_signature or ()),
                tuple(prepared_cache[id(record)][1]),
            ),
        )
        current_group_key = None
        current_oracle = None
        current_order: List[str] = []
        current_values: List[int] = []
        computed_cache_count = 0
        reused_cache_count = 0

        for idx, record in enumerate(tqdm(sorted_subset), start=1):
            order, values = prepared_cache[id(record)]
            cached_mask = cached_candidate_masks.get(record.sample_id)
            shape_token_ids, shape_var_ids = _shape_ids_for_record(record)
            if cached_mask is not None:
                sample = _build_prepared_sample(
                    record,
                    order,
                    values,
                    tokenizer,
                    registry=registry,
                    include_candidate_masks=False,
                    shape_token_ids=shape_token_ids,
                    shape_var_ids=shape_var_ids,
                )
                sample.candidate_masks = cached_mask.clone()
                items_by_id[id(record)] = sample
                current_group_key = None
                current_oracle = None
                current_order = []
                current_values = []
                reused_cache_count += 1
                if idx % 500 == 0 or idx == total:
                    print(
                        f"[dataset] precomputed masks {idx}/{total} "
                        f"(reused={reused_cache_count} computed={computed_cache_count})"
                    )
                continue

            group_key = (
                record.workload_key,
                record.target_kind,
                record.task_index,
                record.sketch_index,
                tuple(record.param_signature or ()),
            )
            prefix_len = 0

            if current_group_key != group_key:
                current_group_key = group_key
                current_oracle = registry.build_oracle_from_record(record)
                current_order = list(order)
                current_values = []
            else:
                limit = min(len(current_order), len(current_values), len(order), len(values))
                while (
                    prefix_len < limit
                    and current_order[prefix_len] == order[prefix_len]
                    and current_values[prefix_len] == int(values[prefix_len])
                ):
                    prefix_len += 1

                prefix_key = tuple(
                    (order[pos], int(values[pos]))
                    for pos in range(prefix_len)
                )
                snapshot = current_oracle.generator._lpm_prefix_state_cache.get(prefix_key)
                if snapshot is None:
                    current_oracle = registry.build_oracle_from_record(record)
                    prefix_len = 0
                else:
                    current_oracle.assignment.clear()
                    current_oracle.assignment.update(snapshot[0])
                    current_oracle._domains = current_oracle.generator.param_sampler._copy_domains(snapshot[1])
                    current_oracle._group_remaining = current_oracle.generator.param_sampler._copy_group_remaining(snapshot[2])
                    current_oracle._budget_remaining = current_oracle.generator.param_sampler._copy_budget_remaining(snapshot[3])
                    current_oracle._sym_map = dict(snapshot[4])
                    current_oracle.generator.param_sampler._restore_sym_map(snapshot[4])
                    current_oracle.last_report = None

            items_by_id[id(record)] = _build_prepared_sample(
                record,
                order,
                values,
                tokenizer,
                registry=registry,
                include_candidate_masks=True,
                oracle=current_oracle,
                prefix_len=prefix_len,
                shape_token_ids=shape_token_ids,
                shape_var_ids=shape_var_ids,
            )
            cached_candidate_masks[record.sample_id] = items_by_id[id(record)].candidate_masks.clone()
            current_order = list(order)
            current_values = list(values)
            computed_cache_count += 1

            if persist_cache and (idx % _CANDIDATE_MASK_CACHE_FLUSH_EVERY == 0 or idx == total):
                _save_candidate_mask_cache_files(
                    config,
                    subset,
                    cached_candidate_masks,
                    cache_paths_by_workload,
                    tokenizer,
                )
            if idx == total:
                print(
                    f"[dataset] precomputed masks {idx}/{total} "
                    f"(reused={reused_cache_count} computed={computed_cache_count})"
                )

        return [items_by_id[id(record)] for record in subset]

    if precompute_candidate_masks and cached_candidate_masks:
        missing_precompute_masks = sum(
            1 for record in precompute_records if record.sample_id not in cached_candidate_masks
        )
        if missing_precompute_masks == 0:
            print("[dataset] candidate mask cache covers entire train+val splits")

    bundle = DatasetBundle(
        train_dataset=LatentParamDataset(
            build_samples(
                train_records,
                include_candidate_masks=precompute_candidate_masks,
                persist_cache=precompute_candidate_masks,
            )
        ),
        val_dataset=LatentParamDataset(
            build_samples(
                val_records,
                include_candidate_masks=precompute_candidate_masks,
                persist_cache=precompute_candidate_masks,
            )
        ),
        test_dataset=LatentParamDataset(
            build_samples(
                test_records,
                include_candidate_masks=precompute_candidate_masks,
                persist_cache=precompute_candidate_masks,
            )
        ),
        tokenizer=tokenizer,
        train_records=list(train_records),
        val_records=list(val_records),
        test_records=list(test_records),
    )
    if precompute_candidate_masks:
        _save_candidate_mask_cache_files(
            config,
            precompute_records,
            cached_candidate_masks,
            cache_paths_by_workload,
            tokenizer,
        )
        for workload_sig in sorted(cache_paths_by_workload):
            cache_path = cache_paths_by_workload[workload_sig]
            count = sum(
                1
                for record in precompute_records
                if (record.workload_key, record.target_kind) == workload_sig
                and record.sample_id in cached_candidate_masks
            )
            print(
                f"[dataset] saving candidate masks to {cache_path} "
                f"(workload_key={workload_sig}, count={count})"
            )
    print(
        f"[dataset] ready: train={len(bundle.train_dataset)} "
        f"val={len(bundle.val_dataset)} test={len(bundle.test_dataset)}"
    )
    return bundle
