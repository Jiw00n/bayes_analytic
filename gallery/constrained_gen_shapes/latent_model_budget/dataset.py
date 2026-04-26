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

from .adapter import (
    GeneratorRegistry,
    JsonSampleRecord,
    compute_task_min_costs,
    cost_raw_to_label,
    load_json_samples,
    split_records,
)
from .shape_semantics import flatten_labels, semantic_labels_for_task
from .tokenizer import ParamTokenizer

# v12: shape prefix + dedicated PARAM_START token present in both encoder and
# decoder (replaces the old BOS). Encoder length = S + 1 + n; decoder length =
# S + n. candidate_masks align with decoder/target (shape S + n).
# When ``data.extent_token=True`` an additional extent prefix is inserted
# between shape and PARAM_START — that variant gets a ``_withExtent`` tag in
# the cache filename so the no-extent caches stay reusable under the same
# version.
_CANDIDATE_MASK_CACHE_VERSION = "v12"
_CANDIDATE_MASK_CACHE_FLUSH_EVERY = 100


def _build_remap_index(
    saved_id_to_token: Sequence[str],
    tokenizer: "ParamTokenizer",
) -> Optional[torch.Tensor]:
    """Build the ``(old_vocab,)`` long tensor mapping each old token id to its
    new id under ``tokenizer``. Returns ``None`` if no remap is needed (vocab
    identical) so the caller can skip the OR-aggregation entirely."""
    saved_id_to_token = list(saved_id_to_token)
    current_vocab = list(tokenizer.id_to_token)
    if saved_id_to_token == current_vocab:
        return None
    unk_id = int(tokenizer.unk_id)
    indices = torch.empty((len(saved_id_to_token),), dtype=torch.long)
    token_to_id = tokenizer.token_to_id
    for old_id, token in enumerate(saved_id_to_token):
        new_id = token_to_id.get(token)
        indices[old_id] = int(new_id) if new_id is not None else unk_id
    return indices


def _remap_stacked_masks(
    masks: torch.Tensor,
    remap_index: torch.Tensor,
    new_vocab_size: int,
) -> torch.Tensor:
    """Vectorized remap of a stack of bool masks ``(N, seq_len, old_vocab)``
    via ``remap_index: (old_vocab,)``. Replaces the per-sample / per-token
    Python loop that previously dominated mask-cache load time. Multiple old
    tokens can collapse into the same new id (e.g., absent tokens to
    ``unk_id``); we OR-aggregate by summing int8 values and thresholding.
    """
    if masks.dim() != 3:
        raise ValueError(f"expected (N, seq_len, old_vocab), got {tuple(masks.shape)}")
    n, seq_len, old_vocab = masks.shape
    if int(remap_index.shape[0]) != old_vocab:
        # Truncate or pad — matches the legacy single-mask break-on-overflow.
        usable = min(int(remap_index.shape[0]), old_vocab)
        masks = masks[:, :, :usable]
        remap_index = remap_index[:usable]
        old_vocab = usable
    masks_int = masks.to(dtype=torch.int8)
    accum = torch.zeros((n, seq_len, new_vocab_size), dtype=torch.int8)
    accum.index_add_(-1, remap_index.to(dtype=torch.long), masks_int)
    return accum > 0


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
    extent_token_ids: List[int] = field(default_factory=list)
    extent_var_ids: List[int] = field(default_factory=list)


@dataclass
class DatasetBundle:
    train_dataset: "LatentParamDataset"
    val_dataset: "LatentParamDataset"
    test_dataset: "LatentParamDataset"
    tokenizer: ParamTokenizer
    train_records: List[JsonSampleRecord]
    val_records: List[JsonSampleRecord]
    test_records: List[JsonSampleRecord]
    cost_target: str = "neg_log"
    # Populated only when cost_target == "norm_throughput": maps
    # (workload_key, target_kind) → the raw ``min_cost`` used for normalization.
    task_min_costs: Dict[Tuple[Optional[str], Optional[str]], float] = field(
        default_factory=dict
    )

    def task_min_cost_for(
        self, workload_key: Optional[str], target_kind: Optional[str]
    ) -> Optional[float]:
        return self.task_min_costs.get((workload_key, target_kind))


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
    extent_token_ids: Sequence[int] = (),
    extent_var_ids: Sequence[int] = (),
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
    extent_token_list = list(int(v) for v in extent_token_ids)
    extent_var_list = list(int(v) for v in extent_var_ids)
    if len(extent_token_list) != len(extent_var_list):
        raise ValueError(
            f"extent_token_ids ({len(extent_token_list)}) and extent_var_ids "
            f"({len(extent_var_list)}) length mismatch for {record.sample_id}"
        )
    extent_len = len(extent_token_list)
    # Combined non-param prefix laid out as [shape | extent].
    prefix_token_list = shape_token_list + extent_token_list
    prefix_var_list = shape_var_list + extent_var_list
    prefix_total_len = shape_len + extent_len

    # Encoder: [shape | extent | PARAM_START | params]   (length P + 1 + n).
    encoder_token_ids = (
        prefix_token_list + [tokenizer.param_start_id] + param_token_ids
    )
    encoder_var_ids = (
        prefix_var_list + [tokenizer.param_start_var_id] + param_var_ids
    )
    # Decoder input: [shape | extent | PARAM_START | params[:-1]]   (length P + n).
    # Decoder uses target-var-ids convention: decoder_var_ids[k] = var id of
    # the token being predicted at position k, so positions P..P+n-1 carry
    # v0..v_{n-1}. Position P (PARAM_START input) predicts param_0 → var v0.
    if param_token_ids:
        decoder_input_ids = (
            prefix_token_list + [tokenizer.param_start_id] + param_token_ids[:-1]
        )
    else:
        decoder_input_ids = prefix_token_list + [tokenizer.param_start_id]
    decoder_var_ids = prefix_var_list + list(param_var_ids)
    if not param_token_ids:
        decoder_var_ids = prefix_var_list + [tokenizer.var_pad_id]
    # Target: shape/extent positions are pad_id (ignored by loss); param
    # positions carry gold token ids.
    target_ids = [tokenizer.pad_id] * prefix_total_len + param_token_ids

    candidate_masks = None
    if include_candidate_masks:
        if registry is None:
            raise ValueError("registry is required when include_candidate_masks=True")
        if oracle is None:
            oracle = registry.build_oracle_from_record(record)
            prefix_len = 0
        vocab_size = len(tokenizer.id_to_token)
        # Candidate masks align with decoder/target positions (length P + n).
        candidate_masks = torch.zeros(
            (prefix_total_len + len(order), vocab_size), dtype=torch.bool
        )
        # Shape/extent-prefix rows pinned to pad_id so
        # ``masked_fill(~mask, -inf)`` is not all-masked at those positions;
        # loss ignores them via ignore_index=pad_id.
        if prefix_total_len:
            candidate_masks[:prefix_total_len, tokenizer.pad_id] = True
        for t, (name, value) in enumerate(zip(order, ordered_values)):
            row = prefix_total_len + t
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
                    rem_row = prefix_total_len + rem_t
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
        extent_token_ids=extent_token_list,
        extent_var_ids=extent_var_list,
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


def _family_from_config(config) -> Optional[str]:
    """Derive the dataset family (e.g.
    ``nn_contrib_conv2d_winograd_without_weight_transform``) from the first
    configured ``json_paths`` entry. The family is the second-to-last directory
    on those paths
    (``.../{family}/{target}/{N}_*.json``). Returns ``None`` when the path
    layout is too shallow to identify it."""
    paths = getattr(getattr(config, "data", None), "json_paths", None) or []
    for p in paths:
        if not p:
            continue
        parents = Path(p).parents
        if len(parents) >= 2:
            return parents[1].name
    return None


def _per_family_cache_dir(config, leaf: str) -> Path:
    """``{checkpoint_dir}/[{family}/]{leaf}/`` (mkdir -p). Shared base for
    every per-family preprocessing artifact (mask cache, dataset cache, ...).
    """
    cache_dir = Path(config.train.checkpoint_dir).expanduser().resolve()
    family = _family_from_config(config)
    if family and cache_dir.name != family:
        cache_dir = cache_dir / family
    cache_dir = cache_dir / leaf
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _candidate_mask_cache_dir(config) -> Path:
    return _per_family_cache_dir(config, "candidate_mask_cache")


def _dataset_cache_dir(config) -> Path:
    """Disk cache for parsed records and ordered-parameter metadata. Sibling
    of ``candidate_mask_cache`` so all per-family preprocessing artifacts live
    next to each other."""
    return _per_family_cache_dir(config, "dataset_cache")


def _short_hash(payload) -> str:
    """16-hex-char SHA256 over a JSON-serializable payload, used as a stable
    fingerprint for cache invalidation."""
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _records_cache_fingerprint(json_paths: Sequence[Path]) -> str:
    """Hash over (path, size, mtime_ns). Renaming or rewriting any source
    JSON yields a fresh fingerprint and invalidates the cache."""
    payload: List[tuple] = []
    for p in sorted(json_paths, key=str):
        try:
            st = Path(p).stat()
            payload.append((str(p), int(st.st_size), int(st.st_mtime_ns)))
        except OSError:
            payload.append((str(p), -1, -1))
    return _short_hash(payload)


def _order_cache_fingerprint(records_fingerprint: str, config) -> str:
    """Order-cache contents are determined by which records exist (encoded
    via ``records_fingerprint``) plus the generator settings and the
    budget/extent flags driving how generators get built."""
    return _short_hash({
        "records": records_fingerprint,
        "generator": _generator_cache_suffix(config),
        "budget": budget_enabled(config),
        "extent": extent_token_enabled(config),
    })


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


def extent_token_enabled(config_or_payload=None) -> bool:
    if config_or_payload is None:
        return False

    data_cfg = getattr(config_or_payload, "data", None)
    if data_cfg is not None:
        return bool(getattr(data_cfg, "extent_token", False))

    if isinstance(config_or_payload, dict):
        data_payload = config_or_payload.get("data", {})
        if isinstance(data_payload, dict):
            return bool(data_payload.get("extent_token", False))

    return False


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


def _candidate_mask_cache_tag(config) -> str:
    """Compose the trailing tag (everything past the stem) that identifies a
    cache variant: ``{cache_version}_{budget_tag}{compressed?}{extent?}{generator?}``.

    This is what previously rode along on the filename as a suffix; promoting
    it to a subdirectory keeps each variant's per-task ``.pt`` files grouped
    so it's straightforward to delete/inspect/swap a whole variant at once.
    """
    budget_tag = "with_budget" if budget_enabled(config) else "no_budget"
    generator_tag = _generator_cache_suffix(config)
    extent_tag = "_withExtent" if extent_token_enabled(config) else ""
    return (
        f"{_CANDIDATE_MASK_CACHE_VERSION}_{budget_tag}"
        f"{extent_tag}{generator_tag}"
    )


def _candidate_mask_cache_path_for_workload(config, workload_key: str, target_kind: str) -> Path:
    cache_dir = _candidate_mask_cache_dir(config)
    stem = _sanitize_workload_key(workload_key, target_kind)
    tag = _candidate_mask_cache_tag(config)
    variant_dir = cache_dir / tag
    variant_dir.mkdir(parents=True, exist_ok=True)
    return variant_dir / f"{stem}.pt"


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
    restrict_to_workloads: Optional[Set[Tuple[str, str]]] = None,
) -> None:
    sample_masks_by_workload: Dict[tuple[str, str], Dict[str, torch.Tensor]] = {}
    for record in subset:
        workload_key = record.workload_key
        target_kind = record.target_kind
        if not workload_key or not target_kind:
            continue
        workload_sig = (str(workload_key), str(target_kind))
        if restrict_to_workloads is not None and workload_sig not in restrict_to_workloads:
            continue
        mask = cached_candidate_masks.get(record.sample_id)
        if mask is None:
            continue
        sample_masks_by_workload.setdefault(workload_sig, {})[record.sample_id] = (
            mask.to(device="cpu")
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


def _precompute_masks_worker(args: tuple) -> Dict[str, torch.Tensor]:
    """Compute candidate masks for one (workload_key, target_kind) group.

    Module-level so it pickles cleanly for ``multiprocessing.Pool``. Each
    worker rebuilds its own ``GeneratorRegistry`` (so generator caches and TVM
    C-extension state stay process-local) and runs the same prefix-snapshot
    logic as the sequential path, preserving ``_lpm_mask_cache`` hit rate
    inside the group. Workers report progress in chunks via a shared Queue
    and write the workload's pt cache file directly, avoiding a
    main-process I/O bottleneck.
    """
    (
        workload_key,
        target_kind,
        records,
        record_param_data,
        tokenizer_state,
        registry_init_args,
        existing_masks,
        cache_path,
        cached_id_to_token,
        progress_queue,
        progress_chunk,
    ) = args

    from .adapter import GeneratorRegistry
    from .tokenizer import ParamTokenizer

    registry = GeneratorRegistry(**registry_init_args)
    tokenizer = ParamTokenizer.from_state_dict(tokenizer_state)

    indexed = list(enumerate(records))
    indexed.sort(
        key=lambda ir: (
            -1 if ir[1].task_index is None else int(ir[1].task_index),
            int(ir[1].sketch_index),
            tuple(ir[1].param_signature or ()),
            tuple(record_param_data[ir[0]][1]),
        )
    )

    current_group_key = None
    current_oracle = None
    current_order: List[str] = []
    current_values: List[int] = []
    new_masks: Dict[str, torch.Tensor] = {}
    pending_progress = 0

    def _flush_progress(final: bool = False) -> None:
        nonlocal pending_progress
        if progress_queue is None or pending_progress <= 0:
            return
        if final or pending_progress >= progress_chunk:
            try:
                progress_queue.put(int(pending_progress))
            except Exception:  # pylint: disable=broad-except
                pass
            pending_progress = 0

    for orig_idx, record in indexed:
        rec_data = record_param_data[orig_idx]
        if len(rec_data) >= 6:
            (
                order,
                values,
                shape_token_ids,
                shape_var_ids,
                extent_token_ids,
                extent_var_ids,
            ) = rec_data
        else:
            order, values, shape_token_ids, shape_var_ids = rec_data
            extent_token_ids, extent_var_ids = [], []
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
            limit = min(
                len(current_order), len(current_values), len(order), len(values)
            )
            while (
                prefix_len < limit
                and current_order[prefix_len] == order[prefix_len]
                and current_values[prefix_len] == int(values[prefix_len])
            ):
                prefix_len += 1
            prefix_key = tuple(
                (order[pos], int(values[pos])) for pos in range(prefix_len)
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

        sample = _build_prepared_sample(
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
            extent_token_ids=extent_token_ids,
            extent_var_ids=extent_var_ids,
        )
        new_masks[record.sample_id] = sample.candidate_masks
        current_order = list(order)
        current_values = list(values)
        pending_progress += 1
        _flush_progress(final=False)

    _flush_progress(final=True)

    if cache_path is not None:
        merged = dict(existing_masks)
        merged.update(new_masks)
        if merged:
            torch.save(
                {
                    "workload_key": str(workload_key),
                    "target_kind": str(target_kind),
                    "sample_masks": merged,
                    "id_to_token": cached_id_to_token,
                },
                cache_path,
            )

    # Return numpy arrays instead of torch tensors so the IPC pickle path
    # uses numpy's plain buffer reducer rather than torch's shared-memory /
    # file-descriptor reducer. With tens of thousands of mask tensors per
    # group, the torch reducer exhausts system fd/shm limits during result
    # collection (RuntimeError: unable to mmap ... Cannot allocate memory).
    return {sid: mask.numpy() for sid, mask in new_masks.items()}


def _resolve_pool_workers(requested: int | None, n_items: int) -> int:
    """Clamp ``requested`` against ``cpu_count()`` and item count. ``requested
    <= 0`` means "auto" (use all CPUs up to n_items)."""
    import os as _os

    req = int(requested) if requested is not None else 0
    if n_items <= 0:
        return 0
    if req <= 0:
        return min(_os.cpu_count() or 1, n_items)
    return min(req, n_items)


def _imap_unordered_pool(worker_fn, args_iter, *, n_workers: int, total: int, desc: str):
    """Yield results of ``worker_fn`` over ``args_iter`` from a fork-context
    Pool with a tqdm progress bar. Caller drives result reduction.
    """
    import multiprocessing as _mp

    ctx = _mp.get_context("fork")
    with ctx.Pool(processes=n_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(worker_fn, args_iter),
            total=total,
            desc=desc,
        ):
            yield result


def _load_json_worker(args: tuple) -> List[JsonSampleRecord]:
    """Module-level worker for parallel JSON loading.

    Returns parsed records for a single path so the main process can extend
    its accumulator. Costs are loaded with ``cost_target="norm_throughput"``
    (raw); the actual cost_target transform happens in the main process.
    """
    path, cost_target = args
    return load_json_samples(path, cost_target=cost_target)


def _run_parallel_json_load(
    raw_paths: Sequence[Path],
    *,
    cost_target: str,
    requested_workers: int,
) -> List[JsonSampleRecord]:
    """Load N json files in parallel processes; falls back to sequential when
    requested_workers <= 1 or there's only one file. Order is *not*
    preserved (we use ``imap_unordered`` for a smoother tqdm), but downstream
    code does not depend on file order.
    """
    n_workers = _resolve_pool_workers(requested_workers, len(raw_paths))
    if n_workers < 2 or len(raw_paths) < 2:
        records: List[JsonSampleRecord] = []
        for path in tqdm(raw_paths, desc="[dataset] loading json"):
            records.extend(load_json_samples(path, cost_target=cost_target))
        return records

    print(f"[dataset] launching json load: workers={n_workers} files={len(raw_paths)}")
    args_iter = [(path, cost_target) for path in raw_paths]
    records: List[JsonSampleRecord] = []
    for batch in _imap_unordered_pool(
        _load_json_worker,
        args_iter,
        n_workers=n_workers,
        total=len(raw_paths),
        desc="[dataset] loading json",
    ):
        records.extend(batch)
    return records


def _build_registry_init_args(config) -> Dict[str, object]:
    args: Dict[str, object] = {
        "network_info_folder": config.data.network_info_folder,
    }
    gen_cfg = getattr(config, "generator", None)
    if gen_cfg is not None:
        hw_param = getattr(gen_cfg, "hw_param", None)
        disable_constraint = getattr(gen_cfg, "disable_constraint", None)
        if hw_param:
            args["hw_param"] = dict(hw_param)
        if disable_constraint:
            args["disable_constraint"] = list(disable_constraint)
    return args


def _run_parallel_mask_precompute(
    *,
    config,
    precompute_records: Sequence[JsonSampleRecord],
    cached_candidate_masks: Dict[str, torch.Tensor],
    cache_paths_by_workload: Dict[Tuple[str, str], Path],
    tokenizer: "ParamTokenizer",
    prepared_cache: Dict[int, tuple],
    shape_ids_for_record,
    extent_ids_for_record=None,
) -> None:
    """Compute cache-miss candidate masks across (workload_key, target_kind)
    groups in parallel processes. Populates ``cached_candidate_masks`` in
    place; each worker also persists its own pt cache file so an interrupted
    run keeps progress.

    Falls back to no-op when there are no missing records or when
    ``config.train.precompute_workers`` resolves to <= 1 (callers handle the
    sequential path inside ``build_samples``).
    """
    miss_by_workload: Dict[Tuple[str, str], List[JsonSampleRecord]] = {}
    for record in precompute_records:
        if record.sample_id in cached_candidate_masks:
            continue
        if not (record.workload_key and record.target_kind):
            continue
        sig = (str(record.workload_key), str(record.target_kind))
        miss_by_workload.setdefault(sig, []).append(record)

    if not miss_by_workload:
        return

    n_groups = len(miss_by_workload)
    n_workers = _resolve_pool_workers(
        getattr(getattr(config, "train", None), "precompute_workers", None),
        n_groups,
    )
    if n_workers < 2:
        # Caller's sequential build_samples path will compute these.
        return

    print(
        f"[dataset] launching mask precompute: "
        f"workers={n_workers} groups={n_groups} "
        f"missing={sum(len(r) for r in miss_by_workload.values())}"
    )

    registry_init_args = _build_registry_init_args(config)
    tokenizer_state = tokenizer.to_state_dict()
    cached_id_to_token = list(tokenizer.id_to_token)

    per_group_args: List[tuple] = []
    for sig, recs in miss_by_workload.items():
        workload_key, target_kind = sig
        cache_path = cache_paths_by_workload.get(sig)
        if cache_path is None:
            cache_path = _candidate_mask_cache_path_for_workload(
                config, workload_key, target_kind
            )
            cache_paths_by_workload[sig] = cache_path

        record_param_data: List[tuple] = []
        for rec in recs:
            order, values = prepared_cache[id(rec)]
            stoks, svars = shape_ids_for_record(rec)
            if extent_ids_for_record is not None:
                etoks, evars = extent_ids_for_record(rec)
            else:
                etoks, evars = [], []
            record_param_data.append(
                (
                    list(order),
                    [int(v) for v in values],
                    list(stoks),
                    list(svars),
                    list(etoks),
                    list(evars),
                )
            )

        existing_for_workload = {
            rec.sample_id: cached_candidate_masks[rec.sample_id]
            for rec in precompute_records
            if (str(rec.workload_key or ""), str(rec.target_kind or "")) == sig
            and rec.sample_id in cached_candidate_masks
        }

        per_group_args.append((
            workload_key,
            target_kind,
            recs,
            record_param_data,
            tokenizer_state,
            registry_init_args,
            existing_for_workload,
            cache_path,
            cached_id_to_token,
            None,  # progress_queue, filled below
            _CANDIDATE_MASK_CACHE_FLUSH_EVERY,
        ))

    import multiprocessing as _mp
    from queue import Empty as _Empty

    ctx = _mp.get_context("fork")
    manager = ctx.Manager()
    progress_queue = manager.Queue()
    per_group_args = [
        (*args[:9], progress_queue, args[10]) for args in per_group_args
    ]

    total_misses = sum(len(g[2]) for g in per_group_args)
    pbar = tqdm(total=total_misses, desc="[dataset] candidate masks")
    try:
        with ctx.Pool(processes=n_workers) as pool:
            futures = [
                pool.apply_async(_precompute_masks_worker, (args,))
                for args in per_group_args
            ]
            while True:
                all_done = all(f.ready() for f in futures)
                try:
                    n = progress_queue.get(timeout=0.5)
                    pbar.update(int(n))
                    continue
                except _Empty:
                    pass
                if all_done:
                    while True:
                        try:
                            n = progress_queue.get_nowait()
                            pbar.update(int(n))
                        except _Empty:
                            break
                    break

            for f in futures:
                group_new_masks = f.get()
                for sample_id, mask_np in group_new_masks.items():
                    cached_candidate_masks[sample_id] = torch.from_numpy(mask_np)
    finally:
        pbar.close()

    print(
        f"[dataset] precompute done: "
        f"new_masks={total_misses} groups={n_groups}"
    )


def _order_cache_worker(args: tuple) -> Dict[str, object]:
    """Build a single ScheduleGenerator for one ``order_key`` and extract the
    metadata the order_cache loop needs. Module-level so it pickles cleanly
    for ``multiprocessing.Pool``.

    The worker rebuilds its own ``GeneratorRegistry`` so generator caches and
    TVM C-extension state stay process-local. Returns plain Python types only
    (lists / dicts of ints + strings) so the result IPC trivially.
    """
    (
        order_key,
        record_serialized,
        registry_init_args,
        include_budget,
        use_extent_tokens,
    ) = args

    import pickle as _pickle

    from .adapter import GeneratorRegistry

    registry = GeneratorRegistry(**registry_init_args)
    record = _pickle.loads(record_serialized)

    gen = _get_generator_for_record(record, registry)
    order = get_model_param_order(gen, include_budget=include_budget)
    budget_specs = _extract_budget_specs(gen)
    generator_domain_values = _collect_generator_domain_values(
        gen, order, include_budget=include_budget
    )
    vec_step_indices = list(getattr(gen, "_vectorize_split_step_indices", set()))
    sp_extents_dict: Dict[int, int] = {}
    if use_extent_tokens:
        sp_extents = dict(getattr(gen, "_sp_extents", {}))
        sp_extents_dict = {int(k): int(v) for k, v in sp_extents.items()}

    return {
        "order_key": order_key,
        "order": list(order),
        "budget_specs": list(budget_specs),
        "generator_domain_values": {
            str(k): [int(x) for x in v]
            for k, v in generator_domain_values.items()
        },
        "vec_step_indices": [int(i) for i in vec_step_indices],
        "sp_extents": sp_extents_dict,
    }


def _run_parallel_order_cache_build(
    *,
    config,
    records: Sequence[JsonSampleRecord],
    include_budget: bool,
    use_extent_tokens: bool,
) -> Optional[tuple]:
    """Pre-populate the order_cache by building one generator per unique
    ``order_key`` across worker processes. Returns the populated
    ``(order_cache, domain_values_by_name, extent_values_by_order_key,
    canonical_extent_indices, canonical_extent_labels, unique_extent_values)``
    tuple; falls back to ``None`` when fewer than 2 workers would be used so
    the caller can run the existing sequential path.
    """
    import pickle as _pickle

    sample_record_by_key: Dict[tuple, JsonSampleRecord] = {}
    for record in records:
        order_key = (
            record.workload_key,
            record.target_kind,
            record.task_index,
            record.sketch_index,
            _normalize_param_signature(record.param_signature),
        )
        sample_record_by_key.setdefault(order_key, record)

    n_unique = len(sample_record_by_key)
    n_workers = _resolve_pool_workers(
        getattr(getattr(config, "train", None), "precompute_workers", None),
        n_unique,
    )
    if n_workers < 2:
        return None

    print(
        f"[dataset] launching parallel order-cache build: "
        f"workers={n_workers} unique_orders={n_unique}"
    )

    registry_init_args = _build_registry_init_args(config)
    per_group_args = [
        (
            order_key,
            _pickle.dumps(record),
            registry_init_args,
            include_budget,
            use_extent_tokens,
        )
        for order_key, record in sample_record_by_key.items()
    ]

    order_cache: Dict[tuple, Dict[str, object]] = {}
    domain_values_by_name: Dict[str, Set[int]] = {}
    extent_values_by_order_key: Dict[tuple, List[int]] = {}
    canonical_extent_indices: List[int] = []
    canonical_extent_labels: List[str] = []
    unique_extent_values: Set[int] = set()

    for result in _imap_unordered_pool(
        _order_cache_worker,
        per_group_args,
        n_workers=n_workers,
        total=n_unique,
        desc="[dataset] order cache",
    ):
        order_key = tuple(result["order_key"])
        order_cache[order_key] = {
            "order": list(result["order"]),
            "budget_specs": list(result["budget_specs"]),
        }
        vec_step_indices = set(result["vec_step_indices"])
        # Same dynamic-extent SplitStep skip as the sequential loop:
        # vectorize-attached splits have meaningless divisor-based
        # initial domains.
        for name, values in result["generator_domain_values"].items():
            if name.startswith("sp_"):
                try:
                    step_idx = int(name.split("_")[1])
                except (ValueError, IndexError):
                    step_idx = None
                if step_idx is not None and step_idx in vec_step_indices:
                    continue
            domain_values_by_name.setdefault(name, set()).update(
                int(v) for v in values
            )
        if use_extent_tokens:
            sp_extents = result["sp_extents"]
            sorted_indices = sorted(int(s) for s in sp_extents.keys())
            if not canonical_extent_indices:
                canonical_extent_indices = list(sorted_indices)
                canonical_extent_labels = [
                    f"sp_extent_{s}" for s in canonical_extent_indices
                ]
            elif sorted_indices != canonical_extent_indices:
                raise ValueError(
                    "SplitStep step-index layout differs across order_keys: "
                    f"reference={canonical_extent_indices} got={sorted_indices}"
                )
            extents = [int(sp_extents[s]) for s in canonical_extent_indices]
            extent_values_by_order_key[order_key] = extents
            unique_extent_values.update(extents)

    return (
        order_cache,
        domain_values_by_name,
        extent_values_by_order_key,
        canonical_extent_indices,
        canonical_extent_labels,
        unique_extent_values,
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

    cost_target = getattr(config.data, "cost_target", "neg_log")
    print(f"[dataset] cost_target={cost_target!r}")

    # Load raw costs first (regardless of cost_target) so task_min_costs can
    # always be computed; the cost_target transform is applied below. This
    # keeps ``task_min_costs`` available even when ``cost_target='neg_log'``,
    # which lets ``cost_target_regression`` pick a throughput variant.
    load_workers = getattr(config.data, "load_workers", None)
    load_workers = int(load_workers) if load_workers is not None else 0
    records_fingerprint = _records_cache_fingerprint(raw_paths)
    records_cache_path = _dataset_cache_dir(config) / f"records_{records_fingerprint}.pt"
    records: Optional[List[JsonSampleRecord]] = None
    if records_cache_path.exists():
        try:
            cached_records = torch.load(records_cache_path, map_location="cpu")
        except Exception as err:  # pylint: disable=broad-except
            print(
                f"[dataset] records cache unreadable ({type(err).__name__}: "
                f"{err}); will re-parse JSONs"
            )
            cached_records = None
        if cached_records is not None:
            records = list(cached_records)
            print(
                f"[dataset] loaded {len(records)} record(s) from cache "
                f"{records_cache_path.name}"
            )
    if records is None:
        records = _run_parallel_json_load(
            raw_paths,
            cost_target="norm_throughput",
            requested_workers=load_workers,
        )
        print(f"[dataset] loaded {len(records)} record(s)")
        # Drop the original JSON payload before persisting. ``raw`` is only
        # consumed by ``GeneratorRegistry.get_generator_from_record`` when
        # ``record.raw`` carries a TVM measure-record dict; the
        # param_signature fallback handles the ``raw=None`` case.
        for record in records:
            record.raw = None
        try:
            torch.save(records, records_cache_path)
            print(f"[dataset] saved records cache to {records_cache_path}")
        except Exception as err:  # pylint: disable=broad-except
            print(
                f"[dataset] failed to save records cache ({type(err).__name__}: "
                f"{err}); continuing without it"
            )

    task_min_costs = compute_task_min_costs(records)
    # if task_min_costs:
    #     print(
    #         f"[dataset] task_min_costs="
    #         f"{ {k: float(f'{v:.6g}') for k, v in task_min_costs.items()} }"
    #     )

    for record in records:
        if record.cost is None:
            continue
        key = (record.workload_key, record.target_kind)
        record.cost = cost_raw_to_label(
            record.cost, cost_target, task_min_cost=task_min_costs.get(key)
        )

    _validate_record_group_consistency(records)
    shape_info_by_record, canonical_shape_labels, unique_shape_values = (
        _collect_shape_tokens_and_labels(records, registry)
    )
    print(
        f"[dataset] shape prefix: labels={canonical_shape_labels} "
        f"unique_values={len(unique_shape_values)}"
    )

    use_extent_tokens = extent_token_enabled(config)
    # Populated lazily by the prepared_cache loop below (one entry per unique
    # order_key, not per record) — extents are static loop bounds tied to
    # (workload_key, target_kind, sketch_index, param_signature) so we can
    # extract them once from the canonical generator and reuse across all
    # records sharing that order_key.
    extent_values_by_order_key: Dict[tuple, List[int]] = {}
    canonical_extent_indices: List[int] = []
    canonical_extent_labels: List[str] = []
    unique_extent_values: Set[int] = set()

    prepared_cache: Dict[int, tuple[List[str], List[int]]] = {}
    order_cache: Dict[tuple, Dict[str, object]] = {}
    domain_values_by_name: Dict[str, Set[int]] = {}
    # When extent_token is enabled we look up per-record extent values via
    # the order_key; recording it once per record avoids rebuilding the tuple
    # later.
    record_order_key: Dict[int, tuple] = {}

    # Try loading the order-cache snapshot from disk. The per-record loop
    # below short-circuits its slow ``_get_generator_for_record`` branch on
    # every iteration when ``order_cache`` is already populated for the
    # requested key, eliminating the 33+ TVM generator builds that dominate
    # cold startup.
    order_cache_fp = _order_cache_fingerprint(records_fingerprint, config)
    order_cache_path = _dataset_cache_dir(config) / f"orders_{order_cache_fp}.pt"
    order_cache_loaded = False
    if order_cache_path.exists():
        try:
            cached = torch.load(order_cache_path, map_location="cpu")
            order_cache = {
                tuple(k): dict(v) for k, v in cached["order_cache"].items()
            }
            domain_values_by_name = {
                str(k): set(int(x) for x in v)
                for k, v in cached["domain_values_by_name"].items()
            }
            extent_values_by_order_key = {
                tuple(k): list(v)
                for k, v in cached["extent_values_by_order_key"].items()
            }
            canonical_extent_indices = list(cached["canonical_extent_indices"])
            canonical_extent_labels = list(cached["canonical_extent_labels"])
            unique_extent_values = set(int(x) for x in cached["unique_extent_values"])
            order_cache_loaded = True
            print(
                f"[dataset] loaded order cache from {order_cache_path.name} "
                f"(unique_orders={len(order_cache)})"
            )
        except Exception as err:  # pylint: disable=broad-except
            print(
                f"[dataset] order cache unreadable ({type(err).__name__}: "
                f"{err}); will rebuild"
            )
            order_cache = {}
            domain_values_by_name = {}
            extent_values_by_order_key = {}
            canonical_extent_indices = []
            canonical_extent_labels = []
            unique_extent_values = set()
    if not order_cache_loaded:
        # Parallelize the slow ScheduleGenerator builds across worker processes
        # before falling into the sequential loop. With the order_cache
        # populated up-front the loop below never enters its
        # ``_get_generator_for_record`` branch.
        parallel_result = _run_parallel_order_cache_build(
            config=config,
            records=records,
            include_budget=include_budget,
            use_extent_tokens=use_extent_tokens,
        )
        if parallel_result is not None:
            (
                order_cache,
                domain_values_by_name,
                extent_values_by_order_key,
                canonical_extent_indices,
                canonical_extent_labels,
                unique_extent_values,
            ) = parallel_result
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
            # Capture SplitStep extents from this canonical generator. Static
            # loop bounds depend only on (workload_key, target_kind,
            # sketch_index) so reusing them across all records sharing the
            # same order_key avoids per-record generator builds.
            if use_extent_tokens:
                sp_extents = dict(getattr(gen, "_sp_extents", {}))
                sorted_indices = sorted(int(s) for s in sp_extents.keys())
                if not canonical_extent_indices:
                    canonical_extent_indices = list(sorted_indices)
                    canonical_extent_labels = [
                        f"sp_extent_{s}" for s in canonical_extent_indices
                    ]
                elif sorted_indices != canonical_extent_indices:
                    raise ValueError(
                        "SplitStep step-index layout differs across order_keys: "
                        f"reference={canonical_extent_indices} got={sorted_indices} "
                        f"on {record.sample_id}"
                    )
                extents = [int(sp_extents[s]) for s in canonical_extent_indices]
                extent_values_by_order_key[order_key] = extents
                unique_extent_values.update(extents)
        order = list(cached_meta["order"])
        budget_specs = list(cached_meta["budget_specs"])
        _apply_cached_order_metadata(record, order, budget_specs)
        missing = [name for name in order if name not in record.params]
        if missing:
            raise ValueError(f"{record.sample_id} is missing ordered params: {missing}")
        values = [int(record.params[name]) for name in order]
        prepared_cache[id(record)] = (order, values)
        if use_extent_tokens:
            record_order_key[id(record)] = order_key
        if idx == len(records):
            print(
                f"[dataset] prepared {idx}/{len(records)} record(s); "
                f"unique_orders={len(order_cache)}"
            )
    if use_extent_tokens:
        print(
            f"[dataset] extent prefix: labels={canonical_extent_labels} "
            f"unique_values={len(unique_extent_values)}"
        )

    # Persist the order cache when it was newly built. ``Set[int]`` and tuple
    # keys are normalized to lists so the snapshot survives torch.save's
    # default pickling without depending on internal set ordering.
    if not order_cache_loaded and order_cache:
        try:
            torch.save(
                {
                    "order_cache": {
                        tuple(k): dict(v) for k, v in order_cache.items()
                    },
                    "domain_values_by_name": {
                        str(k): sorted(int(x) for x in v)
                        for k, v in domain_values_by_name.items()
                    },
                    "extent_values_by_order_key": {
                        tuple(k): list(v)
                        for k, v in extent_values_by_order_key.items()
                    },
                    "canonical_extent_indices": list(canonical_extent_indices),
                    "canonical_extent_labels": list(canonical_extent_labels),
                    "unique_extent_values": sorted(int(x) for x in unique_extent_values),
                },
                order_cache_path,
            )
            print(f"[dataset] saved order cache to {order_cache_path}")
        except Exception as err:  # pylint: disable=broad-except
            print(
                f"[dataset] failed to save order cache "
                f"({type(err).__name__}: {err}); continuing without it"
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
        mode=getattr(config.data, "split_mode", None),
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

    # Register shape (and extent) semantic labels as var-vocab entries and
    # shape/extent integer values as tokens. Both share the numeric token
    # namespace with param values (``value_to_token`` returns str(int) for
    # non-ur_ names).
    extra_var_labels: List[str] = list(canonical_shape_labels) + list(canonical_extent_labels)
    if extra_var_labels:
        all_orders = [list(extra_var_labels) + order for order in all_orders]
    domain_values_for_tokenizer = {
        name: sorted(values) for name, values in domain_values_by_name.items()
    }
    if unique_shape_values:
        domain_values_for_tokenizer["__shape__"] = sorted(int(v) for v in unique_shape_values)
    if unique_extent_values:
        domain_values_for_tokenizer["__extent__"] = sorted(int(v) for v in unique_extent_values)

    tokenizer = ParamTokenizer.build(
        train_ordered_names=train_orders,
        train_ordered_values=train_values,
        all_ordered_names=all_orders,
        domain_values_by_name=domain_values_for_tokenizer,
    )
    print(
        f"[dataset] tokenizer built: vocab={len(tokenizer.id_to_token)} "
        f"vars={len(tokenizer.id_to_var)}"
    )

    precompute_candidate_masks = bool(getattr(config.train, "precompute_candidate_masks", False))
    cached_candidate_masks: Dict[str, torch.Tensor] = {}
    cache_paths_by_workload: Dict[tuple[str, str], Path] = {}
    # Workloads with masks computed since the last flush. Intermediate flushes
    # only re-save these, avoiding the previous O(N_flush × N_workload)
    # torch.save storm where every workload's pt file was rewritten on every
    # flush tick.
    dirty_workloads: Set[Tuple[str, str]] = set()
    # Union of every workload that ever became dirty across train+val+test
    # build_samples passes. Drives the post-loop final save so a fully-cached
    # startup performs zero ``torch.save`` calls instead of rewriting every
    # workload's pt file.
    ever_dirty_workloads: Set[Tuple[str, str]] = set()
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
        new_vocab_size = len(tokenizer.id_to_token)
        n_loaded_workloads = 0
        n_loaded_samples = 0
        n_remapped_workloads = 0
        progress = tqdm(
            precompute_workload_keys,
            desc="[dataset] mask cache",
            total=len(precompute_workload_keys),
        )
        for workload_key in progress:
            cache_path = _candidate_mask_cache_path_for_workload(config, workload_key[0], workload_key[1])
            cache_paths_by_workload[workload_key] = cache_path
            if not cache_path.exists():
                continue
            payload = torch.load(cache_path, map_location="cpu")
            sample_masks = payload.get("sample_masks", {})
            saved_id_to_token = payload.get("id_to_token")
            if saved_id_to_token is None:
                progress.write(
                    f"[dataset] cache at {cache_path} lacks id_to_token; "
                    f"ignoring (rebuild required)"
                )
                continue
            if not sample_masks:
                continue
            remap_index = _build_remap_index(saved_id_to_token, tokenizer)
            if remap_index is None:
                # Identical vocab — drop straight in. No per-tensor work.
                cached_candidate_masks.update(
                    {str(k): v for k, v in sample_masks.items()}
                )
            else:
                # Vectorized remap across all masks for this workload at once.
                # Per-sample seq_len is identical (samples in a workload share
                # the same generator / param order), so they stack cleanly.
                sample_ids = list(sample_masks.keys())
                stacked = torch.stack(
                    [sample_masks[sid].to(dtype=torch.bool) for sid in sample_ids],
                    dim=0,
                )
                remapped = _remap_stacked_masks(stacked, remap_index, new_vocab_size)
                for i, sid in enumerate(sample_ids):
                    cached_candidate_masks[str(sid)] = remapped[i]
                n_remapped_workloads += 1
            n_loaded_workloads += 1
            n_loaded_samples += len(sample_masks)
            progress.set_postfix(
                workloads=n_loaded_workloads,
                samples=n_loaded_samples,
                remapped=n_remapped_workloads,
            )
        print(
            f"[dataset] loaded cached masks: workloads={n_loaded_workloads} "
            f"samples={n_loaded_samples} remapped={n_remapped_workloads}"
        )

    def _shape_ids_for_record(record: JsonSampleRecord) -> tuple[List[int], List[int]]:
        shape_values, shape_labels = shape_info_by_record[id(record)]
        shape_token_ids = [
            tokenizer.token_to_id.get(str(int(v)), tokenizer.unk_id)
            for v in shape_values
        ]
        shape_var_ids = [tokenizer.var_to_id[label] for label in shape_labels]
        return shape_token_ids, shape_var_ids

    def _extent_ids_for_record(record: JsonSampleRecord) -> tuple[List[int], List[int]]:
        if not use_extent_tokens or not canonical_extent_labels:
            return [], []
        order_key = record_order_key.get(id(record))
        if order_key is None:
            raise ValueError(
                f"missing extent order_key for {record.sample_id}; "
                "extent_token=True requires every record to be processed by "
                "the prepared_cache loop"
            )
        extents = extent_values_by_order_key.get(order_key)
        if extents is None:
            raise ValueError(
                f"missing extent values for order_key={order_key} "
                f"(record {record.sample_id})"
            )
        extent_token_ids = [
            tokenizer.token_to_id.get(str(int(v)), tokenizer.unk_id)
            for v in extents
        ]
        extent_var_ids = [
            tokenizer.var_to_id[label] for label in canonical_extent_labels
        ]
        return extent_token_ids, extent_var_ids

    if precompute_candidate_masks:
        _run_parallel_mask_precompute(
            config=config,
            precompute_records=precompute_records,
            cached_candidate_masks=cached_candidate_masks,
            cache_paths_by_workload=cache_paths_by_workload,
            tokenizer=tokenizer,
            prepared_cache=prepared_cache,
            shape_ids_for_record=_shape_ids_for_record,
            extent_ids_for_record=_extent_ids_for_record,
        )

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
                extent_token_ids, extent_var_ids = _extent_ids_for_record(record)
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
                        extent_token_ids=extent_token_ids,
                        extent_var_ids=extent_var_ids,
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
            extent_token_ids, extent_var_ids = _extent_ids_for_record(record)
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
                    extent_token_ids=extent_token_ids,
                    extent_var_ids=extent_var_ids,
                )
                sample.candidate_masks = cached_mask
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
                extent_token_ids=extent_token_ids,
                extent_var_ids=extent_var_ids,
            )
            cached_candidate_masks[record.sample_id] = items_by_id[id(record)].candidate_masks
            if record.workload_key and record.target_kind:
                workload_sig = (str(record.workload_key), str(record.target_kind))
                dirty_workloads.add(workload_sig)
                ever_dirty_workloads.add(workload_sig)
            current_order = list(order)
            current_values = list(values)
            computed_cache_count += 1

            if (
                persist_cache
                and dirty_workloads
                and (idx % _CANDIDATE_MASK_CACHE_FLUSH_EVERY == 0 or idx == total)
            ):
                _save_candidate_mask_cache_files(
                    config,
                    subset,
                    cached_candidate_masks,
                    cache_paths_by_workload,
                    tokenizer,
                    restrict_to_workloads=set(dirty_workloads),
                )
                dirty_workloads.clear()
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
        cost_target=cost_target,
        task_min_costs=dict(task_min_costs),
    )
    if precompute_candidate_masks and ever_dirty_workloads:
        _save_candidate_mask_cache_files(
            config,
            precompute_records,
            cached_candidate_masks,
            cache_paths_by_workload,
            tokenizer,
            restrict_to_workloads=set(ever_dirty_workloads),
        )
        for workload_sig in sorted(ever_dirty_workloads):
            cache_path = cache_paths_by_workload.get(workload_sig)
            if cache_path is None:
                continue
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
