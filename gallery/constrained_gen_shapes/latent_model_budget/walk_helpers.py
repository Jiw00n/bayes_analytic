"""Training-side walk plumbing.

Aggregates and ingests :class:`WalkRecord` outputs from
:mod:`latent_walk`, manages the per-task measurement_lookup files, and
exposes :func:`_run_periodic_latent_walk` — the in-memory adapter
``train_main`` calls each walk-cadence epoch.

Moved out of ``latent_walk.py`` so that file stays focused on the walk
algorithm itself (z encoding/decoding, score prediction, walk loop).
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from .dataset import DatasetBundle

from .adapter import (
    GeneratorRegistry,
    JsonSampleRecord,
    cost_label_to_raw,
    cost_raw_to_label,
    load_json_samples,
)
from .dataset import (
    _build_prepared_sample,
    _get_generator_for_record,
    budget_enabled,
    get_model_param_order,
)
from .inference import SamplingOptions
from .latent_walk import (
    _get_reference_best_seconds,
    make_bundle,
    run_latent_walk,
)
from .recon_predict_gp import WalkSampleBuffer, make_task_sym_map_key
from .tokenizer import ParamTokenizer



# ---------------------------------------------------------------------------
# Per-epoch walk orchestration helpers (formerly in train.py)
# ---------------------------------------------------------------------------

def _resolve_walk_record_jsons(config) -> List[str]:
    """Return the list of record JSONs to walk per epoch/final.

    Order of resolution:
    1. Explicit ``config.latent_walk.record_json``: a single file or a
       directory (directory → all ``*.json`` inside, sorted).
    2. Otherwise: every entry of ``config.data.json_paths`` (one task each).

    If ``config.latent_walk.task_indices`` is set (tuple/list of ints), the
    resolved list is filtered to keep only jsons whose leading-digit task
    index (per ``_task_lookup_key_from_json``) is in the selection, AND the
    output is reordered to match the order given in ``task_indices``.
    """
    explicit = getattr(config.latent_walk, "record_json", None)
    if explicit:
        p = Path(str(explicit))
        if p.is_dir():
            paths = [str(x) for x in sorted(p.glob("*.json"))]
        else:
            paths = [str(p)]
    else:
        json_paths = list(getattr(config.data, "json_paths", []) or [])
        paths = [str(p) for p in json_paths]

    task_indices = getattr(config.latent_walk, "task_indices", None)
    if task_indices:
        ordered_keys = [str(int(i)) for i in task_indices]
        by_key: Dict[str, str] = {}
        for p in paths:
            k = _task_lookup_key_from_json(p)
            by_key.setdefault(k, p)  # first match wins on duplicates
        missing = [k for k in ordered_keys if k not in by_key]
        if missing:
            print(
                f"[train] latent_walk.task_indices not found in resolved jsons: "
                f"{missing}"
            )
        paths = [by_key[k] for k in ordered_keys if k in by_key]
    return paths


def _task_lookup_key_from_json(json_path: str | Path) -> str:
    """Stable per-task identifier for naming the task's measurement_lookup
    file. Mirrors the heuristic used elsewhere: leading digits of the json
    filename (the task_index prefix) when present, otherwise the full stem.
    """
    import re

    p = Path(json_path)
    m = re.match(r"^(\d+)", p.stem)
    return m.group(1) if m else p.stem


def _probe_workload_key_from_json(json_path: str | Path) -> Optional[str]:
    """Read the first sample of ``json_path`` and return its ``workload_key``.
    Used to filter the in-memory walk_buffer when seeding a per-task cache so
    only the task's own entries are folded in (and ultimately saved to its
    own lookup file)."""
    try:
        probes = load_json_samples(json_path)
    except Exception:  # pylint: disable=broad-except
        return None
    if not probes:
        return None
    return getattr(probes[0], "workload_key", None)


def _resolve_walk_task_min_cost(
    record_json_path: str,
    *,
    bundle: DatasetBundle,
    fallback: Optional[float],
) -> Optional[float]:
    """Look up the per-task ``min_cost`` for the workload of the given record
    JSON. Probes one record (the first) to read its ``(workload_key,
    target_kind)`` and queries ``bundle.task_min_cost_for``. Falls back to
    ``fallback`` (typically the bundle's first-task min_cost) if the lookup
    fails — keeps single-task behavior unchanged.
    """
    try:
        probes = load_json_samples(record_json_path)
    except Exception:  # pylint: disable=broad-except
        return fallback
    if not probes:
        return fallback
    probe = probes[0]
    found = bundle.task_min_cost_for(probe.workload_key, probe.target_kind)
    return found if found is not None else fallback


def _select_topk_records_from_path(
    record_json_path: str | Path,
    *,
    k: int,
) -> list[JsonSampleRecord]:
    if k <= 0:
        return []
    records = load_json_samples(record_json_path)
    if not records:
        return []
    records_with_cost = [
        record for record in records
        if record.cost is not None and math.isfinite(float(record.cost))
    ]
    if not records_with_cost:
        return []
    records_with_cost.sort(key=lambda record: float(record.cost), reverse=True)
    return records_with_cost[: int(k)]


def _summarize_walk_records(
    records,
    *,
    reference_params: Optional[Dict[str, int]] = None,
    reference_best_seconds: Optional[float] = None,
) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    if not records:
        return summary

    ref = (
        {str(k): int(v) for k, v in reference_params.items()}
        if reference_params
        else None
    )

    def _is_reference(params) -> bool:
        if ref is None or not params:
            return False
        shared = set(ref) & set(params)
        if not shared:
            return False
        return all(int(params[k]) == ref[k] for k in shared)

    pred_costs = [float(r.predicted_score) for r in records]
    summary["walk/num_steps"] = float(len(records))
    if pred_costs:
        summary["walk/best_predicted_cost"] = float(max(pred_costs))
        summary["walk/mean_predicted_cost"] = float(sum(pred_costs) / len(pred_costs))

    recon_records = [
        r for r in records if getattr(r, "recon_predict_cost", None) is not None
    ]
    if recon_records:
        recon_costs = [float(r.recon_predict_cost) for r in recon_records]
        summary["walk/best_recon_predict_cost"] = float(max(recon_costs))
        summary["walk/mean_recon_predict_cost"] = float(sum(recon_costs) / len(recon_costs))
        best_recon_record = max(
            recon_records, key=lambda r: float(r.recon_predict_cost)
        )
        best_std = getattr(best_recon_record, "recon_predict_std", None)
        if best_std is not None:
            summary["walk/best_recon_predict_std"] = float(best_std)

    measured_novel: list[tuple[float, float]] = []
    measured_novel_raw_seconds: list[float] = []
    true_cost_at_alpha0: Optional[float] = None
    for record in records:
        meas = record.measurement or {}
        if meas.get("ok") and meas.get("usable_measurement"):
            is_novel = (
                abs(float(record.alpha)) >= 1e-12
                and not _is_reference(record.params)
            )
            mean_cost = meas.get("mean_cost")
            if mean_cost is not None and is_novel:
                measured_novel.append((float(record.alpha), float(mean_cost)))
            if is_novel:
                raw_costs = meas.get("costs") or []
                if raw_costs:
                    raw_mean = float(
                        sum(float(c) for c in raw_costs) / len(raw_costs)
                    )
                    if math.isfinite(raw_mean) and raw_mean > 0:
                        measured_novel_raw_seconds.append(raw_mean)
        if abs(float(record.alpha)) < 1e-12:
            true_mc = meas.get("true_mean_cost") if meas else None
            if true_mc is not None:
                true_cost_at_alpha0 = float(true_mc)

    summary["walk/num_unique_sym_map"] = float(
        len({frozenset((str(k), int(v)) for k, v in r.sym_map.items()) for r in records})
    )
    if measured_novel:
        best_alpha, best_mc = max(measured_novel, key=lambda x: x[1])
        summary["walk/best_measured_mean_cost"] = best_mc
        summary["walk/alpha_at_best"] = best_alpha
    if measured_novel_raw_seconds:
        best_raw = min(measured_novel_raw_seconds)
        summary["walk/best_measured_raw_seconds"] = best_raw
        if (
            reference_best_seconds is not None
            and math.isfinite(float(reference_best_seconds))
            and float(reference_best_seconds) > 0
            and best_raw > 0
        ):
            summary["walk/reference_best_seconds"] = float(reference_best_seconds)
            summary["walk/speedup_vs_reference"] = (
                float(reference_best_seconds) / best_raw
            )
    if true_cost_at_alpha0 is not None:
        summary["walk/true_cost_at_alpha0"] = true_cost_at_alpha0
    return summary


def _merge_walk_summaries(
    summaries: list[Dict[str, float]],
    *,
    walk_key_prefix: str = "",
) -> Dict[str, float]:
    """Aggregate per-task walk summaries into the keys the wandb ``walk/``
    section displays:

    - ``walk/num_steps``: total decoded steps across tasks.
    - ``walk/num_records``: number of tasks walked.
    - ``walk/mean_measured_best_cost``: mean over tasks of each task's best
      measured cost in this walk.

    Per-task breakdowns (best/alpha/num_unique_sym_map/etc.) are emitted
    separately by ``_augment_summary_with_per_task``.

    When ``walk_key_prefix`` is non-empty (e.g. ``"cost_head_"``), the
    per-task summaries are read with that prefix on the ``walk/`` keys (since
    :func:`_run_periodic_latent_walk` rewrites ``walk/X`` to
    ``{walk_key_prefix}walk/X`` for prefixed walks) and only the prefixed
    aggregate ``walk/{walk_key_prefix}mean_measured_best_cost`` is emitted.
    """
    merged: Dict[str, float] = {}
    if not summaries:
        return merged

    bm_key = f"{walk_key_prefix}walk/best_measured_mean_cost"
    steps_key = f"{walk_key_prefix}walk/num_steps"

    num_records = 0
    total_steps = 0.0
    best_measured_per_task: list[float] = []
    for summary in summaries:
        if not summary:
            continue
        num_records += 1
        total_steps += float(summary.get(steps_key, 0.0))
        bm = summary.get(bm_key)
        if bm is not None:
            best_measured_per_task.append(float(bm))

    if walk_key_prefix:
        # Minimal aggregate for prefixed walks: only the per-task mean (this
        # is what the user is comparing across walks). The unprefixed walk
        # already emits walk/num_records / walk/num_steps so we don't
        # duplicate them with a prefix.
        if best_measured_per_task:
            merged[f"walk/{walk_key_prefix}mean_measured_best_cost"] = (
                sum(best_measured_per_task) / len(best_measured_per_task)
            )
        return merged

    merged["walk/num_records"] = float(num_records)
    merged["walk/num_steps"] = total_steps
    if best_measured_per_task:
        merged["walk/mean_measured_best_cost"] = (
            sum(best_measured_per_task) / len(best_measured_per_task)
        )
    return merged


def _augment_summary_with_per_task(
    summary_out: Dict[str, float],
    per_task_summaries_keyed: list[tuple[str, Dict[str, float]]],
    *,
    epoch: int,
    running_best: Dict[str, float],
    running_best_epoch: Dict[str, int],
    running_best_alpha: Dict[str, float],
    key_prefix: str = "",
    reference_label: Optional[str] = None,
    walk_key_prefix: str = "",
) -> None:
    """Emit per-task wandb panels and update the all-time-best trackers.

    Sections produced (each per task_index, suffixed by ``key_prefix`` when
    needed for e.g. ``final_`` runs):

    - ``walk_measured/{task}`` — best measured cost in *this* walk
    - ``walk_num_unique_sym_map/{task}``
    - ``walk_best_cost/{task}`` — running max across walks
    - ``walk_best_epoch/{task}`` — epoch at which ``walk_best_cost`` improved
    - ``walk_alpha_at_best/{task}`` — alpha at the all-time best

    ``running_best*`` dicts are mutated in place so the running max persists
    across walks. Callers should pass a *separate* dict triple per
    ``walk_key_prefix`` so that cost_head walk's running best does not
    pollute cost_vec walk's tracking and vice versa.

    When ``walk_key_prefix`` is non-empty (e.g. ``"cost_head_"``), the
    per-task summaries are read with that prefix on the ``walk/`` keys and
    only the prefixed ``{walk_key_prefix}walk_measured_{reference_label}/{task}``
    and ``{walk_key_prefix}walk_alpha_at_best/{task}`` panels are emitted —
    the other panels (walk_measured / walk_best_cost / walk_best_epoch /
    walk_num_unique_sym_map) are intentionally skipped to keep the wandb
    dashboard focused on the two metrics the user wanted for prefixed walks.
    """
    bm_key = f"{walk_key_prefix}walk/best_measured_mean_cost"
    nu_key = f"{walk_key_prefix}walk/num_unique_sym_map"
    sp_key = f"{walk_key_prefix}walk/speedup_vs_reference"
    alpha_key = f"{walk_key_prefix}walk/alpha_at_best"
    is_prefixed = bool(walk_key_prefix)

    for tkey, s in per_task_summaries_keyed:
        if not s:
            continue
        bm = s.get(bm_key)
        if not is_prefixed and bm is not None:
            summary_out[f"{key_prefix}walk_measured/{tkey}"] = float(bm)
        if not is_prefixed:
            nu = s.get(nu_key)
            if nu is not None:
                summary_out[f"{key_prefix}walk_num_unique_sym_map/{tkey}"] = float(nu)
        if reference_label:
            sp = s.get(sp_key)
            if sp is not None:
                summary_out[
                    f"{key_prefix}{walk_key_prefix}walk_measured_{reference_label}/{tkey}"
                ] = float(sp)
        if bm is not None:
            prev = running_best.get(tkey)
            if prev is None or float(bm) > prev:
                running_best[tkey] = float(bm)
                running_best_epoch[tkey] = int(epoch)
                aab = s.get(alpha_key)
                if aab is not None:
                    running_best_alpha[tkey] = float(aab)
                elif tkey in running_best_alpha:
                    # Drop stale alpha when the new best lacks one.
                    running_best_alpha.pop(tkey, None)
        if tkey in running_best:
            if not is_prefixed:
                summary_out[f"{key_prefix}walk_best_cost/{tkey}"] = running_best[tkey]
                summary_out[f"{key_prefix}walk_best_epoch/{tkey}"] = float(
                    running_best_epoch[tkey]
                )
            if tkey in running_best_alpha:
                summary_out[
                    f"{key_prefix}{walk_key_prefix}walk_alpha_at_best/{tkey}"
                ] = running_best_alpha[tkey]


def _ingest_walk_records_into_buffer(
    walk_buffer: WalkSampleBuffer,
    *,
    walk_records,
    ref_record: JsonSampleRecord,
    registry: GeneratorRegistry,
    tokenizer: ParamTokenizer,
    config,
) -> None:
    """Add measured walk records into the GP-augmentation buffer.

    Each walk record contributes one entry keyed by its sym_map; entries are
    PreparedSamples encoded with the same param order the dataset uses, with
    `cost` set to the measured negative-log mean cost.
    """
    include_budget = budget_enabled(config)
    try:
        gen = _get_generator_for_record(ref_record, registry)
        order = get_model_param_order(gen, include_budget=include_budget)
    except Exception as err:  # pragma: no cover
        print(f"[walk-buffer] cannot resolve param order: {type(err).__name__}: {err}")
        return

    added = 0
    skipped_dup = 0
    for record in walk_records:
        if not getattr(record, "state_build_ok", False):
            continue
        meas = getattr(record, "measurement", None) or {}
        if not (meas.get("ok") and meas.get("usable_measurement")):
            continue
        mean_cost = meas.get("mean_cost")
        if mean_cost is None or not math.isfinite(float(mean_cost)):
            continue

        sym_key = make_task_sym_map_key(
            getattr(ref_record, "workload_key", None),
            {str(k): int(v) for k, v in record.sym_map.items() if isinstance(v, int)},
        )
        if sym_key in walk_buffer:
            skipped_dup += 1
            continue

        try:
            ordered_values = [int(record.params[name]) for name in order]
        except KeyError:
            continue

        sample_id = f"{ref_record.sample_id}_walk_{abs(hash(sym_key)) & 0xFFFFFFFF:08x}"
        synthesized = dataclasses.replace(
            ref_record,
            sample_id=sample_id,
            params=dict(record.params),
            cost=float(mean_cost),
        )
        try:
            prepared = _build_prepared_sample(
                synthesized,
                order,
                ordered_values,
                tokenizer,
                registry=registry,
                include_candidate_masks=False,
            )
        except Exception as err:  # pragma: no cover
            print(f"[walk-buffer] prepare failed: {type(err).__name__}: {err}")
            continue
        walk_buffer.add(sym_key, prepared)
        added += 1

    if added or skipped_dup:
        print(
            f"[walk-buffer] added={added} dup_skipped={skipped_dup} "
            f"buffer_size={len(walk_buffer)}"
        )


def _run_periodic_latent_walk(
    *,
    model,
    device,
    checkpoint_path,
    record_json_path: str,
    walk_output_dir: str,
    network_info_folder: Optional[str],
    epoch_label: str,
    config,
    registry,
    tokenizer,
    latent_cost_ridge,
    timestamp: Optional[str] = None,
    top_k: int = 1,
    num_steps: int = 8,
    step_size: float = 0.25,
    use_latent_gradient: bool = False,
    include_recon_predict: bool = False,
    include_measurement: bool = True,
    recon_predictor=None,
    reencode_predictor=None,
    reencode_predictor_name: str = "cost_head",
    walk_buffer: Optional[WalkSampleBuffer] = None,
    walk_key_prefix: str = "",
    measurement_cache: Optional[dict] = None,
    sampling_options: Optional[SamplingOptions] = None,
    cost_target: str = "neg_log",
    task_min_cost: Optional[float] = None,
    sort_by: str = "re_pred",
    show_neg_log: bool = False,
    reference_best_dir: Optional[str] = None,
) -> Dict[str, float]:
    ref_records = _select_topk_records_from_path(record_json_path, k=max(1, int(top_k)))
    if not ref_records:
        print(f"[train] no reference records available for latent walk: {record_json_path}")
        return {}

    print(
        f"[train] latent walk ({epoch_label}) reusing in-memory model"
    )

    # Reuse in-memory model + cached registry to skip checkpoint round-trip
    # and GeneratorRegistry rebuild on every walk. The model stays on the
    # training device — no CPU/GPU shuffling.
    was_training = model.training
    model.eval()
    bundle = make_bundle(
        model=model,
        tokenizer=tokenizer,
        registry=registry,
        config_payload=config.to_dict(),
        latent_cost_ridge=latent_cost_ridge,
        device=device,
        use_latent_gradient=bool(use_latent_gradient),
        timestamp=timestamp,
        recon_predictor=recon_predictor,
        reencode_predictor=reencode_predictor,
        reencode_predictor_name=reencode_predictor_name,
    )

    per_rank_summaries: list[Dict[str, float]] = []
    base_output_dir = Path(walk_output_dir)
    try:
        for rank, ref_record in enumerate(ref_records):
            # ``ref_record`` is the *training-data* record that anchors this
            # walk; the sibling-hardware (e.g. a6000) best is used downstream
            # only as a comparison value in ``walk/speedup_vs_reference``.
            reference_params = {
                str(k): int(v) for k, v in (ref_record.params or {}).items()
            }
            rank_output_dir = (
                base_output_dir / f"rank{rank}" if len(ref_records) > 1 else base_output_dir
            )
            try:
                walk_records = run_latent_walk(
                    checkpoint_path=str(checkpoint_path),
                    record_json_path=str(record_json_path),
                    network_info_folder=network_info_folder,
                    device=str(device),
                    output=str(rank_output_dir),
                    num_steps=int(num_steps),
                    step_size=float(step_size),
                    latent_gradient=bool(use_latent_gradient),
                    deterministic_start=True,
                    preselected_record=ref_record,
                    include_recon_predict=bool(include_recon_predict),
                    include_measurement=bool(include_measurement),
                    bundle=bundle,
                    keep_bundle=True,
                    measurement_cache=measurement_cache,
                    sampling_options=sampling_options,
                    cost_target=cost_target,
                    task_min_cost=task_min_cost,
                    sort_by=sort_by,
                    show_neg_log=show_neg_log,
                    reference_best_dir=reference_best_dir,
                ) or []
            except Exception as err:  # pragma: no cover
                print(f"[train] latent walk rank={rank} failed: {type(err).__name__}: {err}")
                walk_records = []
            ref_best_secs = _get_reference_best_seconds(
                reference_best_dir,
                getattr(ref_record, "workload_key", None),
            )
            per_rank_summaries.append(
                _summarize_walk_records(
                    walk_records,
                    reference_params=reference_params,
                    reference_best_seconds=ref_best_secs,
                )
            )
            if walk_buffer is not None and walk_records:
                _ingest_walk_records_into_buffer(
                    walk_buffer,
                    walk_records=walk_records,
                    ref_record=ref_record,
                    registry=registry,
                    tokenizer=tokenizer,
                    config=config,
                )
    finally:
        if was_training:
            model.train()

    combined: Dict[str, float] = {}
    for rank, rank_summary in enumerate(per_rank_summaries):
        if not rank_summary:
            continue
        prefix = f"top{rank + 1}_"
        for key, value in rank_summary.items():
            if key.startswith("walk/"):
                new_key = prefix + key
            else:
                new_key = prefix + "walk/" + key
            combined[new_key] = value
    combined.update(_merge_walk_summaries(per_rank_summaries))
    # Per-task aggregates across ranks. ``_augment_summary_with_per_task``
    # reads these to emit the ``walk_measured/{task}`` /
    # ``walk_alpha_at_best/{task}`` / ``walk_num_unique_sym_map/{task}`` panels;
    # without them every lookup returned ``None`` and no per-task lines were
    # logged to wandb.
    best_measured: Optional[float] = None
    best_alpha: Optional[float] = None
    total_unique = 0.0
    for rank_summary in per_rank_summaries:
        if not rank_summary:
            continue
        bm = rank_summary.get("walk/best_measured_mean_cost")
        if bm is not None and (best_measured is None or float(bm) > best_measured):
            best_measured = float(bm)
            alpha = rank_summary.get("walk/alpha_at_best")
            best_alpha = float(alpha) if alpha is not None else None
        nu = rank_summary.get("walk/num_unique_sym_map")
        if nu is not None:
            total_unique += float(nu)
    if best_measured is not None:
        combined["walk/best_measured_mean_cost"] = best_measured
        if best_alpha is not None:
            combined["walk/alpha_at_best"] = best_alpha
    combined["walk/num_unique_sym_map"] = total_unique
    best_speedup: Optional[float] = None
    best_raw_secs: Optional[float] = None
    ref_best_secs_value: Optional[float] = None
    for rank_summary in per_rank_summaries:
        if not rank_summary:
            continue
        sp = rank_summary.get("walk/speedup_vs_reference")
        if sp is not None and (best_speedup is None or float(sp) > best_speedup):
            best_speedup = float(sp)
        raw_s = rank_summary.get("walk/best_measured_raw_seconds")
        if raw_s is not None and (best_raw_secs is None or float(raw_s) < best_raw_secs):
            best_raw_secs = float(raw_s)
        ref_s = rank_summary.get("walk/reference_best_seconds")
        if ref_s is not None and ref_best_secs_value is None:
            ref_best_secs_value = float(ref_s)
    if best_speedup is not None:
        combined["walk/speedup_vs_reference"] = best_speedup
    if best_raw_secs is not None:
        combined["walk/best_measured_raw_seconds"] = best_raw_secs
    if ref_best_secs_value is not None:
        combined["walk/reference_best_seconds"] = ref_best_secs_value
    if walk_key_prefix:
        target = f"{walk_key_prefix}walk/"
        combined = {
            key.replace("walk/", target, 1) if "walk/" in key else key: value
            for key, value in combined.items()
        }
    return combined




# ---------------------------------------------------------------------------
# Measurement-lookup persistence (formerly in train.py)
# ---------------------------------------------------------------------------

def _seed_measurement_cache_from_buffer(
    walk_buffer: Optional[WalkSampleBuffer],
    disk_cache: Optional[dict] = None,
    cost_target: str = "neg_log",
    task_min_cost: Optional[float] = None,
    workload_key_filter: Optional[str] = None,
) -> dict:
    """Pre-populate the measurement cache with prior measurements so that
    sym_maps already measured in earlier epochs skip re-measurement.

    If ``disk_cache`` is provided, its entries are merged in first (walk-buffer
    entries from the current run take precedence on conflict).

    ``sample.cost`` lives in ``cost_target`` label space; we also back-compute
    the raw-seconds value and stash it in ``costs=[raw]`` so display paths
    (log_grouped_candidate_result) can show ``measured=`` for buffer-hit
    entries.

    When ``workload_key_filter`` is given, walk_buffer entries whose
    task-aware key (``(workload_key, sym_tuple)``) does not match the filter
    are skipped. This keeps each task's cache (and its on-disk lookup file)
    free of cross-task entries even though the buffer is shared."""
    cache: dict = {}
    if disk_cache:
        for sym_key, entry in disk_cache.items():
            cache[sym_key] = copy.deepcopy(entry)
    if walk_buffer is None:
        return cache
    target_wk: Optional[str] = (
        str(workload_key_filter) if workload_key_filter is not None else None
    )
    for sym_key, sample in walk_buffer.items():
        if target_wk is not None:
            if not (isinstance(sym_key, tuple) and len(sym_key) == 2):
                continue
            if sym_key[0] != target_wk:
                continue
        cost = getattr(sample, "cost", None)
        if cost is None or not math.isfinite(float(cost)):
            continue
        entry = {
            "ok": True,
            "usable_measurement": True,
            "mean_cost": float(cost),
            "from_walk_buffer": True,
        }
        raw = cost_label_to_raw(cost, cost_target, task_min_cost=task_min_cost)
        if raw is not None and math.isfinite(raw):
            entry["costs"] = [float(raw)]
        cache[sym_key] = entry
    return cache


def _load_measurement_lookup(
    path: Path,
    cost_target: str = "neg_log",
    task_min_cost: Optional[float] = None,
) -> dict:
    """Load the persisted (workload_key, sym_map)→cost table into a
    measurement_cache-compatible dict. Each JSONL line stores
    ``{"workload_key": str, "sym_map": {name: int, ...}, "cost": float}``
    where ``cost`` is the raw (seconds) mean cost; it is converted to the
    in-memory ``mean_cost`` (``cost_target``-space) on load via
    :func:`cost_raw_to_label`. Missing files return an empty dict.

    Legacy entries without ``workload_key`` (older format that keyed by
    ``sym_map`` only) are dropped: they conflated measurements across distinct
    workloads and re-using them would re-introduce that contamination."""
    cache: dict = {}
    if not path.exists():
        return cache
    skipped_legacy = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            workload_key = entry.get("workload_key")
            if not workload_key:
                skipped_legacy += 1
                continue
            sym_map = entry.get("sym_map") or {}
            cost = entry.get("cost")
            mean_cost = cost_raw_to_label(
                cost, cost_target, task_min_cost=task_min_cost
            )
            if mean_cost is None:
                continue
            try:
                normalized = {str(k): int(v) for k, v in sym_map.items()}
            except (TypeError, ValueError):
                continue
            sym_key = make_task_sym_map_key(workload_key, normalized)
            # Stash the raw (seconds) cost in ``costs`` too so display paths
            # (e.g. log_grouped_candidate_result) can report raw-seconds
            # ``measured=`` values for lookup-hit cache entries just like for
            # freshly-measured ones.
            cache[sym_key] = {
                "ok": True,
                "usable_measurement": True,
                "mean_cost": mean_cost,
                "costs": [float(cost)],
                "from_lookup": True,
            }
    # if skipped_legacy:
    #     print(
    #         f"[train] measurement lookup: dropped {skipped_legacy} legacy "
    #         f"entries without workload_key from {path}"
    #     )
    return cache


def _save_measurement_lookup(
    path: Path,
    cache: dict,
    cost_target: str = "neg_log",
    task_min_cost: Optional[float] = None,
) -> int:
    """Persist all usable measurements in ``cache`` to ``path`` (JSONL, one
    ``{"workload_key": str, "sym_map": ..., "cost": ...}`` per line). ``cost``
    on disk is always the raw (seconds) mean cost, inverted from the in-memory
    ``mean_cost`` (``cost_target``-space) via :func:`cost_label_to_raw`.
    Returns the number of entries written. Non-usable or non-finite entries
    are skipped, as are entries whose key lacks a workload_key (i.e. legacy
    sym_map-only keys still in memory)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    written = 0
    with tmp.open("w", encoding="utf-8") as f:
        for sym_key, entry in cache.items():
            if not entry.get("ok") or not entry.get("usable_measurement"):
                continue
            # Task-aware key shape: (workload_key, ((name, value), ...)).
            if (
                not isinstance(sym_key, tuple)
                or len(sym_key) != 2
                or not isinstance(sym_key[0], str)
                or not sym_key[0]
            ):
                continue
            workload_key, sym_tuple = sym_key
            raw_cost = cost_label_to_raw(
                entry.get("mean_cost"), cost_target, task_min_cost=task_min_cost
            )
            if raw_cost is None:
                continue
            sym_map = {str(name): int(value) for name, value in sym_tuple}
            f.write(
                json.dumps(
                    {
                        "workload_key": workload_key,
                        "sym_map": sym_map,
                        "cost": raw_cost,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    tmp.replace(path)
    return written


def _merge_cache_into_lookup(persistent: dict, shared: dict) -> int:
    """Copy new successful measurements from ``shared`` into ``persistent``.
    Existing persistent entries are not overwritten. Returns the count of
    newly added entries."""
    added = 0
    for sym_key, entry in shared.items():
        if sym_key in persistent:
            continue
        if not entry.get("ok") or not entry.get("usable_measurement"):
            continue
        cost = entry.get("mean_cost")
        if cost is None:
            continue
        try:
            cost_f = float(cost)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(cost_f):
            continue
        new_entry = {
            "ok": True,
            "usable_measurement": True,
            "mean_cost": cost_f,
            "from_lookup": True,
        }
        raw_costs = entry.get("costs")
        if raw_costs:
            new_entry["costs"] = list(raw_costs)
        persistent[sym_key] = new_entry
        added += 1
    return added


def _iter_walk_ridges(latent_cost_ridges, config):
    """Yield (ridge_payload, walk_key_prefix, use_cost_head_gradient) for each
    walk to run. Order: cost_head (if enabled) → cost_vec → cost_vec_weighted."""
    use_cost_head = bool(getattr(config.latent_walk, "use_cost_head", False))
    if use_cost_head:
        yield None, "cost_head_", True

    if not latent_cost_ridges:
        return
    unweighted = next(
        (p for p in latent_cost_ridges if not bool(p.get("weighted", False))),
        None,
    )
    weighted = next(
        (p for p in latent_cost_ridges if bool(p.get("weighted", False))),
        None,
    )
    if unweighted is not None:
        yield unweighted, "", False
    if weighted is not None:
        yield weighted, "w_ridge_", False


