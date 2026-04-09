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
) -> tuple[List[str], List[int], PreparedSample | None]:
    gen = _get_generator_for_record(record, registry)
    order = list(gen.get_full_var_order_entries()["param_order"])

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
    oracle=None,
    prefix_len: int = 0,
) -> PreparedSample:
    target_ids = tokenizer.encode_values(order, ordered_values)
    var_ids = tokenizer.encode_var_names(order)
    decoder_input_ids = [tokenizer.bos_id] + target_ids[:-1]
    candidate_masks = None

    if include_candidate_masks:
        if registry is None:
            raise ValueError("registry is required when include_candidate_masks=True")
        if oracle is None:
            oracle = registry.build_oracle_from_record(record)
            prefix_len = 0
        candidate_masks = torch.zeros((len(order), len(tokenizer.id_to_token)), dtype=torch.bool)
        for t, (name, value) in enumerate(zip(order, ordered_values)):
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
                candidate_masks[t] = cached_mask
                if not candidate_masks[t, gold_id]:
                    raise ValueError("gold value is outside oracle candidates")
                if t >= prefix_len:
                    oracle.assign(name, value)
            except Exception:  # pylint: disable=broad-except
                candidate_masks[t].zero_()
                candidate_masks[t, gold_id] = True
                for rem_t, rem_name, rem_value in zip(range(t + 1, len(order)), order[t + 1:], ordered_values[t + 1:]):
                    rem_gold_token = tokenizer.value_to_token(rem_name, rem_value)
                    rem_gold_id = tokenizer.token_to_id.get(rem_gold_token, tokenizer.unk_id)
                    candidate_masks[rem_t].zero_()
                    candidate_masks[rem_t, rem_gold_id] = True
                break

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
    # breakpoint()
    return cache_dir / f"{_sanitize_workload_key(workload_key, target_kind)}.pt"


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


def _collect_generator_domain_values(gen, order: Sequence[str]) -> Dict[str, List[int]]:
    domain_values_by_name: Dict[str, Set[int]] = {}
    split_domains = gen._build_split_domains()

    for name in order:
        if name.startswith("ur_"):
            domain_values_by_name.setdefault(name, set()).update(
                int(value) for value in gen.pm.UNROLL_CANDIDATES
            )
            continue
        if name in split_domains:
            _extend_domain_values(domain_values_by_name, name, split_domains[name])

    return {
        name: sorted(values)
        for name, values in domain_values_by_name.items()
    }


def build_dataset_bundle(config, registry: GeneratorRegistry) -> DatasetBundle:
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
    order_cache: Dict[tuple, List[str]] = {}
    domain_values_by_name: Dict[str, Set[int]] = {}
    print("[dataset] building ordered parameter cache")
    for idx, record in enumerate(records, start=1):
        order_key = (
            record.workload_key,
            record.target_kind,
            record.task_index,
            record.sketch_index,
            tuple(record.param_signature or ()),
        )
        order = order_cache.get(order_key)
        if order is None:
            gen = _get_generator_for_record(record, registry)
            order = list(gen.get_full_var_order_entries()["param_order"])
            order_cache[order_key] = list(order)
            generator_domain_values = _collect_generator_domain_values(gen, order)
            for name, values in generator_domain_values.items():
                domain_values_by_name.setdefault(name, set()).update(int(v) for v in values)
        missing = [name for name in order if name not in record.params]
        if missing:
            raise ValueError(f"{record.sample_id} is missing ordered params: {missing}")
        values = [int(record.params[name]) for name in order]
        prepared_cache[id(record)] = (order, values)
        if idx % 1000 == 0 or idx == len(records):
            print(
                f"[dataset] prepared {idx}/{len(records)} record(s); "
                f"unique_orders={len(order_cache)}"
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
    cached_candidate_masks: Dict[str, torch.Tensor] = {}
    cache_paths_by_workload: Dict[str, Path] = {}
    if precompute_candidate_masks:
        print("[dataset] precomputing candidate masks for train split")
        train_workload_keys = sorted({(record.workload_key, record.target_kind) for record in train_records if record.workload_key})
        for workload_key in train_workload_keys:
            cache_path = _candidate_mask_cache_path_for_workload(config, workload_key[0], workload_key[1])
            cache_paths_by_workload[workload_key] = cache_path
            if not cache_path.exists():
                continue
            print(f"[dataset] loading cached candidate masks from {cache_path}")
            payload = torch.load(cache_path, map_location="cpu")
            sample_masks = payload.get("sample_masks", {})
            loaded_count = 0
            for sample_id, mask in sample_masks.items():
                cached_candidate_masks[str(sample_id)] = mask.clone().to(dtype=torch.bool, device="cpu")
                loaded_count += 1
            print(
                f"[dataset] loaded cached masks for workload_key={workload_key} "
                f"count={loaded_count}"
            )

    def build_samples(subset: Sequence[JsonSampleRecord], include_candidate_masks: bool) -> List[PreparedSample]:
        if not include_candidate_masks:
            items: List[PreparedSample] = []
            total = len(subset)
            for idx, record in enumerate(subset, start=1):
                order, values = prepared_cache[id(record)]
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
                    current_oracle._sym_map = dict(snapshot[3])
                    current_oracle.generator.param_sampler._restore_sym_map(snapshot[3])
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
            )
            cached_candidate_masks[record.sample_id] = items_by_id[id(record)].candidate_masks.clone()
            current_order = list(order)
            current_values = list(values)
            computed_cache_count += 1

            if idx % 500 == 0 or idx == total:
                print(
                    f"[dataset] precomputed masks {idx}/{total} "
                    f"(reused={reused_cache_count} computed={computed_cache_count})"
                )

        return [items_by_id[id(record)] for record in subset]

    if precompute_candidate_masks and cached_candidate_masks:
        missing_train_masks = sum(1 for record in train_records if record.sample_id not in cached_candidate_masks)
        if missing_train_masks == 0:
            print("[dataset] candidate mask cache covers entire train split")

    bundle = DatasetBundle(
        train_dataset=LatentParamDataset(
            build_samples(train_records, include_candidate_masks=precompute_candidate_masks)
        ),
        val_dataset=LatentParamDataset(build_samples(val_records, include_candidate_masks=False)),
        test_dataset=LatentParamDataset(build_samples(test_records, include_candidate_masks=False)),
        tokenizer=tokenizer,
        train_records=list(train_records),
        val_records=list(val_records),
        test_records=list(test_records),
    )
    if precompute_candidate_masks:
        sample_masks_by_workload: Dict[str, Dict[str, torch.Tensor]] = {}
        for sample in bundle.train_dataset.samples:
            if sample.candidate_masks is None:
                continue
            workload_key = sample.workload_key
            if not workload_key:
                continue
            sample_masks_by_workload.setdefault(workload_key, {})[sample.sample_id] = (
                sample.candidate_masks.clone().to(device="cpu")
            )

        for workload_key, sample_masks in sample_masks_by_workload.items():
            cache_path = cache_paths_by_workload.get(workload_key)
            if cache_path is None:
                cache_path = _candidate_mask_cache_path_for_workload(config, workload_key[0], workload_key[1])
            print(
                f"[dataset] saving candidate masks to {cache_path} "
                f"(workload_key={workload_key}, count={len(sample_masks)})"
            )
            torch.save(
                {
                    "workload_key": workload_key,
                    "sample_masks": sample_masks,
                },
                cache_path,
            )
    print(
        f"[dataset] ready: train={len(bundle.train_dataset)} "
        f"val={len(bundle.val_dataset)} test={len(bundle.test_dataset)}"
    )
    return bundle
