from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .adapter import GeneratorRegistry, JsonSampleRecord, load_json_samples, split_records
from .tokenizer import ParamTokenizer


_CANDIDATE_MASK_CACHE_VERSION = "v7"
_CANDIDATE_MASK_CACHE_FLUSH_EVERY = 100


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
) -> tuple[List[str], List[int], PreparedSample | None]:
    gen = _get_generator_for_record(record, registry)
    order = get_model_param_order(gen, include_budget=include_budget)

    missing = [name for name in order if name not in record.params]
    if missing:
        raise ValueError(f"{record.sample_id} is missing ordered params: {missing}")

    ordered_values = [int(record.params[name]) for name in order]

    if tokenizer is None:
        return order, ordered_values, None

    return order, ordered_values, _build_prepared_sample(record, order, ordered_values, tokenizer)


def _build_prepared_sample(
    record: JsonSampleRecord,
    order: Sequence[str],
    ordered_values: Sequence[int],
    tokenizer: ParamTokenizer,
    registry: GeneratorRegistry | None = None,
    include_candidate_masks: bool = False,
    precomputed_candidates: Optional[Sequence[Sequence[int]]] = None,
) -> PreparedSample:
    target_ids = tokenizer.encode_values(order, ordered_values)
    var_ids = tokenizer.encode_var_names(order)
    decoder_input_ids = [tokenizer.bos_id] + target_ids[:-1]
    candidate_masks = None

    if include_candidate_masks:
        if precomputed_candidates is None:
            raise ValueError(
                "include_candidate_masks=True requires precomputed_candidates "
                "(produced by _collect_record_step_candidates during bundle build)"
            )
        if len(precomputed_candidates) != len(order):
            raise ValueError(
                f"precomputed_candidates length {len(precomputed_candidates)} does not "
                f"match order length {len(order)} for {record.sample_id}"
            )
        candidate_masks = torch.zeros((len(order), len(tokenizer.id_to_token)), dtype=torch.bool)
        for t, (name, value, cands) in enumerate(
            zip(order, ordered_values, precomputed_candidates)
        ):
            gold_token = tokenizer.value_to_token(name, value)
            gold_id = tokenizer.token_to_id.get(gold_token, tokenizer.unk_id)
            candidate_masks[t] = tokenizer.candidate_mask_from_values(name, cands)
            if not candidate_masks[t, gold_id]:
                candidate_masks[t].zero_()
                candidate_masks[t, gold_id] = True

    sample = PreparedSample(
        sample_id=record.sample_id,
        json_path=record.json_path,
        sketch_index=record.sketch_index,
        ordered_param_names=list(order),
        ordered_param_values=list(ordered_values),
        encoder_token_ids=target_ids,
        encoder_var_ids=var_ids,
        decoder_input_ids=decoder_input_ids,
        decoder_var_ids=var_ids,
        target_ids=target_ids,
        cost=record.cost,
        candidate_masks=candidate_masks,
        workload_key=record.workload_key,
        target_kind=record.target_kind,
        target_model=record.target_model,
        task_desc=record.task_desc,
        task_index=record.task_index,
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
    max_len = max(len(item.target_ids) for item in batch)
    pad_id = tokenizer.pad_id
    var_pad = tokenizer.var_pad_id

    enc_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    enc_var_ids = torch.full((bsz, max_len), var_pad, dtype=torch.long)
    dec_in_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    dec_var_ids = torch.full((bsz, max_len), var_pad, dtype=torch.long)
    tgt_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    seq_lens = torch.zeros((bsz,), dtype=torch.long)
    candidate_masks = None
    if all(item.candidate_masks is not None for item in batch):
        candidate_masks = torch.zeros(
            (bsz, max_len, len(tokenizer.id_to_token)),
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
        n = len(item.target_ids)
        seq_lens[i] = n
        enc_ids[i, :n] = torch.tensor(item.encoder_token_ids, dtype=torch.long)
        enc_var_ids[i, :n] = torch.tensor(item.encoder_var_ids, dtype=torch.long)
        dec_in_ids[i, :n] = torch.tensor(item.decoder_input_ids, dtype=torch.long)
        dec_var_ids[i, :n] = torch.tensor(item.decoder_var_ids, dtype=torch.long)
        tgt_ids[i, :n] = torch.tensor(item.target_ids, dtype=torch.long)
        if candidate_masks is not None:
            candidate_masks[i, :n] = item.candidate_masks

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


def _candidate_mask_cache_path_for_workload(config, workload_key: str, target_kind: str) -> Path:
    cache_dir = _candidate_mask_cache_dir(config)
    stem = _sanitize_workload_key(workload_key, target_kind)
    budget_tag = "with_budget" if budget_enabled(config) else "no_budget"
    return cache_dir / f"{stem}_{_CANDIDATE_MASK_CACHE_VERSION}_{budget_tag}.pt"


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


def _collect_record_step_candidates(
    record: JsonSampleRecord,
    order: Sequence[str],
    oracle,
    *,
    prefix_len: int = 0,
    prior_step_candidates: Optional[Sequence[Sequence[int]]] = None,
) -> tuple[List[List[int]], bool]:
    """Walk the oracle for one record and collect propagation-aware candidate
    sets at each AR position, starting from ``prefix_len``.

    The caller is responsible for constructing the oracle and restoring it to
    the correct prefix state (via ``_lpm_prefix_state_cache`` snapshot) before
    invoking this helper. Positions ``[0:prefix_len]`` are filled from
    ``prior_step_candidates``; positions ``[prefix_len:]`` are walked live.

    Returns (step_candidates, valid). ``valid`` is True iff every position's
    ground-truth value appeared inside the oracle's candidate set. Whenever
    the oracle fails to enumerate or the gold value is missing, the
    corresponding position (and every subsequent one) collapses to ``[gold]``
    so downstream mask construction stays consistent with the existing
    gold-only fallback.
    """
    order = list(order)
    step_candidates: List[List[int]] = []
    valid = True
    fallback = False

    if prefix_len > 0:
        if prior_step_candidates is None or len(prior_step_candidates) < prefix_len:
            raise ValueError(
                "prior_step_candidates must cover prefix_len positions"
            )
        for pos in range(prefix_len):
            step_candidates.append([int(v) for v in prior_step_candidates[pos]])

    for idx in range(prefix_len, len(order)):
        name = order[idx]
        gold_present = name in record.params
        gold = int(record.params[name]) if gold_present else None
        if fallback:
            step_candidates.append([gold] if gold is not None else [])
            continue
        try:
            cands = sorted({int(v) for v in oracle.candidate_values(name)})
        except Exception:  # pylint: disable=broad-except
            valid = False
            fallback = True
            step_candidates.append([gold] if gold is not None else [])
            continue
        if not cands:
            valid = False
            fallback = True
            step_candidates.append([gold] if gold is not None else [])
            continue
        if gold is None or gold not in cands:
            valid = False
            step_candidates.append([gold] if gold is not None else cands)
            fallback = True
            continue
        step_candidates.append(cands)
        try:
            oracle.assign(name, gold)
        except Exception:  # pylint: disable=broad-except
            valid = False
            fallback = True
    return step_candidates, valid


def _collect_generator_domain_values(
    gen,
    order: Sequence[str],
    *,
    include_budget: bool = True,
) -> Dict[str, List[int]]:
    domain_values_by_name: Dict[str, Set[int]] = {}
    split_domains = gen._build_split_domains()
    split_candidate_values: Dict[str, List[int]] = {}

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
    *,
    step_candidates_by_sample_id: Dict[str, List[List[int]]],
    validity_by_sample_id: Dict[str, bool],
    prepared_cache: Dict[int, tuple[List[str], List[int]]],
) -> None:
    per_workload: Dict[tuple[str, str], Dict[str, object]] = {}
    for record in subset:
        workload_key = record.workload_key
        target_kind = record.target_kind
        if not workload_key or not target_kind:
            continue
        sig = (str(workload_key), str(target_kind))
        entry = per_workload.setdefault(
            sig,
            {
                "masks": {},
                "step_cands": {},
                "validity": {},
                "domain_sets": {},
            },
        )
        sample_id = record.sample_id
        mask = cached_candidate_masks.get(sample_id)
        if mask is not None:
            entry["masks"][sample_id] = mask.clone().to(device="cpu")
        step_cands = step_candidates_by_sample_id.get(sample_id)
        if step_cands is not None:
            entry["step_cands"][sample_id] = [list(map(int, col)) for col in step_cands]
            order, _ = prepared_cache[id(record)]
            for name, col in zip(order, step_cands):
                entry["domain_sets"].setdefault(str(name), set()).update(int(v) for v in col)
        if sample_id in validity_by_sample_id:
            entry["validity"][sample_id] = bool(validity_by_sample_id[sample_id])

    for workload_sig, entry in per_workload.items():
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
                "sample_masks": entry["masks"],
                "sample_step_candidates": entry["step_cands"],
                "sample_validity": entry["validity"],
                "domain_values": {
                    name: sorted(vs) for name, vs in entry["domain_sets"].items()
                },
            },
            cache_path,
        )


def _load_walk_cache_files(
    config,
    records: Sequence[JsonSampleRecord],
    cache_paths_by_workload: Dict[tuple[str, str], Path],
    step_candidates_by_sample_id: Dict[str, List[List[int]]],
    validity_by_sample_id: Dict[str, bool],
    domain_values_by_name: Dict[str, Set[int]],
    cached_candidate_masks: Dict[str, torch.Tensor],
) -> Set[str]:
    """Load cached walk outputs (step candidates, validity, domain values, masks)
    for any workload with an existing cache file. Returns the set of sample_ids
    whose step candidates came from cache (walk can be skipped for them)."""
    by_workload: Dict[tuple[str, str], List[JsonSampleRecord]] = {}
    for record in records:
        if record.workload_key and record.target_kind:
            sig = (str(record.workload_key), str(record.target_kind))
            by_workload.setdefault(sig, []).append(record)

    cache_hit_sample_ids: Set[str] = set()
    for workload_sig, workload_records in by_workload.items():
        cache_path = cache_paths_by_workload.get(workload_sig)
        if cache_path is None:
            cache_path = _candidate_mask_cache_path_for_workload(
                config, workload_sig[0], workload_sig[1]
            )
            cache_paths_by_workload[workload_sig] = cache_path
        if not cache_path.exists():
            continue
        print(f"[dataset] loading walk/mask cache from {cache_path}")
        payload = torch.load(cache_path, map_location="cpu")
        sample_step_cands = payload.get("sample_step_candidates", {}) or {}
        sample_validity = payload.get("sample_validity", {}) or {}
        domain_values = payload.get("domain_values", {}) or {}
        sample_masks = payload.get("sample_masks", {}) or {}

        for name, values in domain_values.items():
            domain_values_by_name.setdefault(str(name), set()).update(
                int(v) for v in values
            )

        for sample_id, mask in sample_masks.items():
            cached_candidate_masks[str(sample_id)] = mask.clone().to(
                dtype=torch.bool, device="cpu"
            )

        loaded = 0
        for record in workload_records:
            sid = record.sample_id
            if sid in sample_step_cands:
                step_candidates_by_sample_id[sid] = [
                    list(map(int, col)) for col in sample_step_cands[sid]
                ]
                if sid in sample_validity:
                    validity_by_sample_id[sid] = bool(sample_validity[sid])
                cache_hit_sample_ids.add(sid)
                loaded += 1
        print(
            f"[dataset] walk cache hit workload={workload_sig} "
            f"records={loaded}/{len(workload_records)} masks={len(sample_masks)}"
        )

    return cache_hit_sample_ids


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

    prepared_cache: Dict[int, tuple[List[str], List[int]]] = {}
    order_cache: Dict[tuple, Dict[str, object]] = {}
    domain_values_by_name: Dict[str, Set[int]] = {}
    record_step_candidates_by_sample_id: Dict[str, List[List[int]]] = {}
    record_validity_by_sample_id: Dict[str, bool] = {}
    cache_paths_by_workload: Dict[tuple[str, str], Path] = {}
    cached_candidate_masks: Dict[str, torch.Tensor] = {}

    print("[dataset] building ordered parameter cache")
    record_order_keys: List[tuple] = []
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
            # Split steps followed by a vectorize annotation have a dynamic
            # extent that varies per record.  Divisor-based domain computation
            # is meaningless for them; instead we collect every gold value
            # from the dataset.
            _vec_step_indices = gen._vectorize_split_step_indices
            dynamic_sp_names = set()
            for _name in order:
                if _name.startswith("sp_"):
                    _step_idx = int(_name.split("_")[1])
                    if _step_idx in _vec_step_indices:
                        dynamic_sp_names.add(_name)
            cached_meta = {
                "order": list(order),
                "budget_specs": list(budget_specs),
                "dynamic_sp_names": dynamic_sp_names,
            }
            order_cache[order_key] = cached_meta
            # Union the maximal per-variable domain so tokenizer vocab covers
            # candidate values that no record's gold walk would exercise
            # (e.g. the factor=1 path when every record has factor>1 in a
            # split group). Gold-path walks alone can miss div(E) entries.
            # Dynamic-extent names are excluded — they are handled below
            # by collecting gold values directly.
            initial_domains = _collect_generator_domain_values(
                gen, order, include_budget=include_budget
            )
            for name, values in initial_domains.items():
                if name not in dynamic_sp_names:
                    domain_values_by_name.setdefault(str(name), set()).update(
                        int(v) for v in values
                    )
        order = list(cached_meta["order"])
        budget_specs = list(cached_meta["budget_specs"])
        _apply_cached_order_metadata(record, order, budget_specs)
        missing = [name for name in order if name not in record.params]
        if missing:
            raise ValueError(f"{record.sample_id} is missing ordered params: {missing}")
        values = [int(record.params[name]) for name in order]
        prepared_cache[id(record)] = (order, values)
        record_order_keys.append(order_key)
        # For dynamic-extent split names, union the gold value from every
        # record so the tokenizer vocab covers all observed factor values.
        for dyn_name in cached_meta.get("dynamic_sp_names", ()):
            if dyn_name in record.params:
                domain_values_by_name.setdefault(str(dyn_name), set()).add(
                    int(record.params[dyn_name])
                )

        # if idx % 2000 == 0 or idx == len(records):
        #     print(
        #         f"[dataset] prepared {idx}/{len(records)} record(s); "
        #         f"unique_orders={len(order_cache)}"
        #     )

    # Try to short-circuit the propagation walk by loading cached outputs from
    # prior runs (step candidates + validity + domain_values per workload).
    print("[dataset] checking walk cache")
    cache_hit_sample_ids = _load_walk_cache_files(
        config,
        records,
        cache_paths_by_workload,
        record_step_candidates_by_sample_id,
        record_validity_by_sample_id,
        domain_values_by_name,
        cached_candidate_masks,
    )
    if cache_hit_sample_ids:
        print(
            f"[dataset] walk cache covers {len(cache_hit_sample_ids)}/{len(records)} record(s)"
        )

    # Per-record propagation-aware candidate walk with prefix sharing across
    # records that share the same (order_key, gold-value prefix). Sorting by
    # (order_key, values) maximises the common prefix between consecutive
    # records; for each record we reuse the previous group's oracle and
    # restore it from ``_lpm_prefix_state_cache`` to the longest common
    # prefix, then walk only the remaining positions.
    print("[dataset] collecting propagation candidates with prefix sharing")
    walk_indices = [
        i for i, record in enumerate(records)
        if record.sample_id not in cache_hit_sample_ids
    ]
    sorted_indices = sorted(
        walk_indices,
        key=lambda i: (
            record_order_keys[i],
            tuple(prepared_cache[id(records[i])][1]),
        ),
    )

    current_group_key = None
    current_oracle = None
    current_order: List[str] = []
    current_values: List[int] = []
    current_step_cands: List[List[int]] = []
    reused_prefix_positions = 0
    walked_positions = 0

    for cnt, i in enumerate(tqdm(sorted_indices), start=1):
        record = records[i]
        order, values = prepared_cache[id(record)]
        group_key = record_order_keys[i]

        if current_group_key != group_key or current_oracle is None:
            current_group_key = group_key
            current_oracle = registry.build_oracle_from_record(record)
            current_order = list(order)
            current_values = []
            current_step_cands = []
            prefix_len = 0
        else:
            limit = min(len(current_order), len(current_values), len(order), len(values))
            prefix_len = 0
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
                current_step_cands = []
            else:
                ps = current_oracle.generator.param_sampler
                current_oracle.assignment.clear()
                current_oracle.assignment.update(snapshot[0])
                current_oracle._domains = ps._copy_domains(snapshot[1])
                current_oracle._group_remaining = ps._copy_group_remaining(snapshot[2])
                current_oracle._budget_remaining = ps._copy_budget_remaining(snapshot[3])
                current_oracle._sym_map = dict(snapshot[4])
                ps._restore_sym_map(snapshot[4])
                current_oracle.last_report = None

        step_cands, valid = _collect_record_step_candidates(
            record,
            order,
            current_oracle,
            prefix_len=prefix_len,
            prior_step_candidates=current_step_cands if prefix_len > 0 else None,
        )
        record_step_candidates_by_sample_id[record.sample_id] = step_cands
        record_validity_by_sample_id[record.sample_id] = valid
        # Positions [0:prefix_len] were already unioned when a prior record in
        # the same group walked them; union only the newly-walked suffix.
        for pos in range(prefix_len, len(step_cands)):
            domain_values_by_name.setdefault(order[pos], set()).update(
                int(v) for v in step_cands[pos]
            )

        current_order = list(order)
        current_values = list(values)
        current_step_cands = list(step_cands)
        reused_prefix_positions += prefix_len
        walked_positions += len(step_cands) - prefix_len

        if cnt % 500 == 0 or cnt == len(sorted_indices):
            print(
                f"[dataset] propagation {cnt}/{len(sorted_indices)} "
                f"(reused_positions={reused_prefix_positions} "
                f"walked_positions={walked_positions})"
            )

    valid_records = list(records)
    if getattr(config.data, "filter_invalid_records", False):
        print("[dataset] filtering invalid records (gold not in oracle candidates)")
        filtered_records = [
            record
            for record in records
            if record_validity_by_sample_id.get(record.sample_id, False)
        ]
        invalid_count = len(records) - len(filtered_records)
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

    tokenizer = ParamTokenizer.build(
        train_ordered_names=train_orders,
        train_ordered_values=train_values,
        all_ordered_names=all_orders,
        domain_values_by_name={
            name: sorted(values)
            for name, values in domain_values_by_name.items()
        },
    )
    print(
        f"[dataset] tokenizer built: vocab={len(tokenizer.id_to_token)} "
        f"vars={len(tokenizer.id_to_var)}"
    )

    precompute_candidate_masks = bool(getattr(config.train, "precompute_candidate_masks", False))
    precompute_records = list(train_records) + list(val_records)
    if precompute_candidate_masks:
        print("[dataset] precomputing candidate masks for train+val splits")

    def build_samples(
        subset: Sequence[JsonSampleRecord],
        include_candidate_masks: bool,
        persist_cache: bool = False,
    ) -> List[PreparedSample]:
        items: List[PreparedSample] = []
        total = len(subset)
        computed_cache_count = 0
        reused_cache_count = 0
        for idx, record in enumerate(subset, start=1):
            order, values = prepared_cache[id(record)]
            if not include_candidate_masks:
                items.append(
                    _build_prepared_sample(
                        record,
                        order,
                        values,
                        tokenizer,
                        registry=registry,
                        include_candidate_masks=False,
                    )
                )
                if idx % 1000 == 0 or idx == total:
                    print(f"[dataset] prepared samples {idx}/{total}")
                continue

            cached_mask = cached_candidate_masks.get(record.sample_id)
            if cached_mask is not None:
                sample = _build_prepared_sample(
                    record,
                    order,
                    values,
                    tokenizer,
                    registry=registry,
                    include_candidate_masks=False,
                )
                sample.candidate_masks = cached_mask.clone()
                items.append(sample)
                reused_cache_count += 1
            else:
                step_cands = record_step_candidates_by_sample_id.get(record.sample_id)
                if step_cands is None:
                    raise ValueError(
                        f"missing precomputed step candidates for {record.sample_id}; "
                        "bundle build did not populate record_step_candidates_by_sample_id"
                    )
                sample = _build_prepared_sample(
                    record,
                    order,
                    values,
                    tokenizer,
                    registry=registry,
                    include_candidate_masks=True,
                    precomputed_candidates=step_cands,
                )
                cached_candidate_masks[record.sample_id] = sample.candidate_masks.clone()
                items.append(sample)
                computed_cache_count += 1

                if persist_cache and (idx % _CANDIDATE_MASK_CACHE_FLUSH_EVERY == 0 or idx == total):
                    _save_candidate_mask_cache_files(
                        config,
                        subset,
                        cached_candidate_masks,
                        cache_paths_by_workload,
                        step_candidates_by_sample_id=record_step_candidates_by_sample_id,
                        validity_by_sample_id=record_validity_by_sample_id,
                        prepared_cache=prepared_cache,
                    )

            if idx % 500 == 0 or idx == total:
                print(
                    f"[dataset] mask samples {idx}/{total} "
                    f"(reused={reused_cache_count} computed={computed_cache_count})"
                )

        return items

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
        test_dataset=LatentParamDataset(build_samples(test_records, include_candidate_masks=False)),
        tokenizer=tokenizer,
        train_records=list(train_records),
        val_records=list(val_records),
        test_records=list(test_records),
    )
    _save_candidate_mask_cache_files(
        config,
        list(records),
        cached_candidate_masks,
        cache_paths_by_workload,
        step_candidates_by_sample_id=record_step_candidates_by_sample_id,
        validity_by_sample_id=record_validity_by_sample_id,
        prepared_cache=prepared_cache,
    )
    for workload_sig in sorted(cache_paths_by_workload):
        cache_path = cache_paths_by_workload[workload_sig]
        mask_count = sum(
            1
            for record in precompute_records
            if (record.workload_key, record.target_kind) == workload_sig
            and record.sample_id in cached_candidate_masks
        )
        walk_count = sum(
            1
            for record in records
            if (record.workload_key, record.target_kind) == workload_sig
            and record.sample_id in record_step_candidates_by_sample_id
        )
        print(
            f"[dataset] saved walk cache to {cache_path} "
            f"(workload_key={workload_sig}, walk={walk_count} masks={mask_count})"
        )
    print(
        f"[dataset] ready: train={len(bundle.train_dataset)} "
        f"val={len(bundle.val_dataset)} test={len(bundle.test_dataset)}"
    )
    return bundle
