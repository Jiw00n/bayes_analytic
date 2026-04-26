from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Dict, Optional

import torch

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

import copy
import dataclasses
import math

from .adapter import (
    GeneratorRegistry,
    JsonSampleRecord,
    cost_label_to_raw,
    cost_raw_to_label,
    load_json_samples,
)
from .dataset import (
    DatasetBundle,
    LatentParamDataset,
    _build_prepared_sample,
    _generator_cache_suffix,
    _get_generator_for_record,
    budget_enabled,
    build_dataset_bundle,
    get_model_param_order,
)
from .inference import SamplingOptions, greedy_decode_sample, pretty_print_reconstruction
from .model import LatentParamVAE
from .runtime_utils import (
    configure_runtime,
    load_checkpoint,
    prepare_loader,
    resolve_device,
    save_checkpoint,
    save_training_artifacts,
    seed_everything,
)
from .recon_predict_gp import (
    WalkSampleBuffer,
    fit_gp_recon_predictor,
    make_sym_map_key,
    make_task_sym_map_key,
)
from .tokenizer import ParamTokenizer
from .train_epoch import train_one_epoch
from .train_eval import (
    _alpha_metric_suffix,
    _build_named_latent_cost_ridges,
    _build_reencode_predictor,
    _concat_encoded,
    _resolve_ridge_alphas,
    encode_dataset,
    evaluate_autoregressive,
    evaluate_cost_ranking,
    evaluate_teacher_forcing,
    fit_latent_cost_ridges,
)


def _wandb_section(key: str) -> str:
    """Group metric keys into wandb sections (train/, val/, walk/, ...)."""
    if "/" in key:
        return key
    if key == "epoch":
        return key
    section_prefixes = (
        ("train_", "train/"),
        ("val_", "val/"),
        ("eval_val_", "eval_val/"),
        ("eval_test_", "eval_test/"),
        ("test_", "test/"),
        ("final_", "final/"),
    )
    for old, new in section_prefixes:
        if key.startswith(old):
            return new + key[len(old):]
    return f"train/{key}"


def _remap_for_wandb(metrics: Dict) -> Dict:
    return {_wandb_section(k): v for k, v in metrics.items()}


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
        if record.cost is not None and torch.isfinite(torch.tensor(float(record.cost)))
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


def _merge_walk_summaries(summaries: list[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate per-task walk summaries into the keys the wandb ``walk/``
    section displays:

    - ``walk/num_steps``: total decoded steps across tasks.
    - ``walk/num_records``: number of tasks walked.
    - ``walk/mean_measured_best_cost``: mean over tasks of each task's best
      measured cost in this walk.

    Per-task breakdowns (best/alpha/num_unique_sym_map/etc.) are emitted
    separately by ``_augment_summary_with_per_task``.
    """
    merged: Dict[str, float] = {}
    if not summaries:
        return merged

    num_records = 0
    total_steps = 0.0
    best_measured_per_task: list[float] = []
    for summary in summaries:
        if not summary:
            continue
        num_records += 1
        total_steps += float(summary.get("walk/num_steps", 0.0))
        bm = summary.get("walk/best_measured_mean_cost")
        if bm is not None:
            best_measured_per_task.append(float(bm))

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
    across walks.
    """
    for tkey, s in per_task_summaries_keyed:
        if not s:
            continue
        bm = s.get("walk/best_measured_mean_cost")
        if bm is not None:
            summary_out[f"{key_prefix}walk_measured/{tkey}"] = float(bm)
        nu = s.get("walk/num_unique_sym_map")
        if nu is not None:
            summary_out[f"{key_prefix}walk_num_unique_sym_map/{tkey}"] = float(nu)
        if reference_label:
            sp = s.get("walk/speedup_vs_reference")
            if sp is not None:
                summary_out[
                    f"{key_prefix}walk_measured_{reference_label}/{tkey}"
                ] = float(sp)
        if bm is not None:
            prev = running_best.get(tkey)
            if prev is None or float(bm) > prev:
                running_best[tkey] = float(bm)
                running_best_epoch[tkey] = int(epoch)
                aab = s.get("walk/alpha_at_best")
                if aab is not None:
                    running_best_alpha[tkey] = float(aab)
                elif tkey in running_best_alpha:
                    # Drop stale alpha when the new best lacks one.
                    running_best_alpha.pop(tkey, None)
        if tkey in running_best:
            summary_out[f"{key_prefix}walk_best_cost/{tkey}"] = running_best[tkey]
            summary_out[f"{key_prefix}walk_best_epoch/{tkey}"] = float(
                running_best_epoch[tkey]
            )
            if tkey in running_best_alpha:
                summary_out[f"{key_prefix}walk_alpha_at_best/{tkey}"] = (
                    running_best_alpha[tkey]
                )


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
    here = Path(__file__).resolve().parent.parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    try:
        from tune_by_latent import (
            make_bundle,
            run_latent_walk,
            _get_reference_best_seconds,
        )
    except Exception as err:  # pragma: no cover
        print(f"[train] latent walk unavailable: {type(err).__name__}: {err}")
        return {}

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
                reference_best_dir, getattr(ref_record, "workload_key", None)
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


def build_everything(config):
    print(f"[build] loading registry from {config.data.network_info_folder}")
    gen_cfg = getattr(config, "generator", None)
    registry = GeneratorRegistry(
        config.data.network_info_folder,
        hw_param=getattr(gen_cfg, "hw_param", None),
        disable_constraint=getattr(gen_cfg, "disable_constraint", None),
    )
    print("[build] building dataset bundle")
    bundle = build_dataset_bundle(config, registry)
    tokenizer = bundle.tokenizer
    print(
        f"[build] tokenizer ready: vocab={len(tokenizer.id_to_token)} "
        f"vars={len(tokenizer.id_to_var)}"
    )
    model_seed = getattr(config.model, "seed", None)
    if model_seed is not None:
        print(f"[build] overriding torch RNG with model.seed={int(model_seed)} "
              f"(decouples model init from data.seed={config.data.seed})")
        torch.manual_seed(int(model_seed))
        torch.cuda.manual_seed_all(int(model_seed))

    print("[build] constructing model")
    model = LatentParamVAE(
        vocab_size=len(tokenizer.id_to_token),
        num_vars=len(tokenizer.id_to_var),
        cfg=config.model,
    )
    return registry, bundle, tokenizer, model


def _family_from_json_path(json_path: str | Path) -> str:
    """Extract the dataset family name from a measure-record JSON path.

    Layout: ``.../measure_tenset_filtered_family/{family}/{target}/{N}_*.json``
    so the family is ``parents[1].name``. Returns "na" if the layout is too
    shallow.
    """
    p = Path(json_path)
    if len(p.parents) >= 2:
        return p.parents[1].name
    return "na"


def _resolve_run_family(bundle: DatasetBundle) -> str:
    """Family identifier for run-level naming (wandb / checkpoints).
    Per-task / per-workload artifacts (e.g. the candidate_mask_cache) and the
    measurement lookup file intentionally keep task-specific names and do
    NOT use this helper.
    """
    all_records = (
        list(bundle.train_records) + list(bundle.val_records) + list(bundle.test_records)
    )
    for record in all_records:
        json_path = getattr(record, "json_path", None)
        if not json_path:
            continue
        family = _family_from_json_path(json_path)
        if family != "na":
            return family
    return "na"


def _resolve_run_task_index(bundle: DatasetBundle) -> str:
    """Task-index identifier kept for the measurement_lookup filename, where
    measurements are inherently per-task and the file should not be shared
    across tasks within the same family.
    """
    import re

    all_records = (
        list(bundle.train_records) + list(bundle.val_records) + list(bundle.test_records)
    )
    for record in all_records:
        if record.task_index is not None:
            return str(int(record.task_index))
    for record in all_records:
        json_path = getattr(record, "json_path", None)
        if not json_path:
            continue
        m = re.match(r"^(\d+)", Path(json_path).stem)
        if m:
            return m.group(1)
    return "na"


def _resolve_run_family_from_config(config) -> str:
    for p in getattr(config.data, "json_paths", []) or []:
        family = _family_from_json_path(p)
        if family != "na":
            return family
    return "na"


def _build_wandb_project_name(config, bundle: DatasetBundle | None = None) -> str:
    if bundle is not None:
        family = _resolve_run_family(bundle)
    else:
        family = _resolve_run_family_from_config(config)
    project_suffix = getattr(config.wandb, "project", None) or "single_v1"
    return f"{family}_{project_suffix}"


def _build_wandb_run_name(config, bundle: DatasetBundle | None = None) -> str:
    name = ""

    if config.model.num_encoder_layers != 4:
        name += f"_enc{config.model.num_encoder_layers}"
    if config.model.num_decoder_layers != 4:
        name += f"_dec{config.model.num_decoder_layers}"
    if config.model.nhead != 4:
        name += f"_head{config.model.nhead}"
    
    if config.model.latent_dim != 64:
        name += f"_zdim{config.model.latent_dim}"
    if config.model.dim_feedforward != 384:
        name += f"_fdim{config.model.dim_feedforward}"
    if config.model.cost_hidden_dim != 128:
        name += f"_cdim{config.model.cost_hidden_dim}"
    if config.model.latent_token_count != 4:
        name += f"_ztok{config.model.latent_token_count}"

    if config.train.num_epochs != 100:
        name += f"_ep{config.train.num_epochs}"
    name += (
        f"_lr{config.train.learning_rate}"
        f"_nce{config.train.lambda_nce}"
        f"_tau{config.train.tau_nce}"
        f"_kl{config.train.beta_end}"
        f"_bw{config.train.beta_warmup_epochs}"
    )

    if config.train.lambda_recon != 1.0:
        name += f"_lamr{config.train.lambda_recon}"
    if config.train.lambda_cost != 0.01:
        name += f"_lamc{config.train.lambda_cost}"
    if bool(config.train.order_nce):
        name += "_order"
    if bool(getattr(config.train, "nce_mu", False)):
        name += "_nce_mu"
    if bool(config.model.adaln):
        name += "_adaln"
    if bool(getattr(config.train, "cobo_sample_weighting", False)):
        cobo_tag = getattr(config.train, "cobo_apply_to", [])
        name += (
            f"_cobo{float(config.train.weight_quantile):.1f}"
            f"_{float(config.train.weight_sigma):.1f}"
            f"_{cobo_tag}"
        )
    if bool(getattr(config.train, "cost_ridge_weighted", False)):
        name += "_wridge"
        if not bool(getattr(config.train, "cobo_sample_weighting", False)):
            name += (
                f"{float(config.train.weight_quantile):.1f}"
                f"_{float(config.train.weight_sigma):.1f}"
            )
    if bool(getattr(config.train, "use_compressed_teacher_forcing", False)):
        name += "_comp"
    ls = float(getattr(config.train, "label_smoothing", 0.0))
    if ls > 0.0:
        name += f"_ls{ls}"
    if bool(getattr(config.train, "order_nce_pos_weight_by_percentile", False)):
        name += f"_pos{float(config.train.order_nce_pos_weight_sigma):.1f}"
    if getattr(config.sampling, "strategy", "greedy") != "greedy":
        name += f"_{config.sampling.strategy}"
        if config.sampling.strategy == "sampling":
            name += (
                f"_t{config.sampling.temperature}"
                f"_k{config.sampling.top_k}"
            )
            if config.sampling.top_p != 1.0:
                name += f"_p{config.sampling.top_p}"
            name += f"_sseed{config.sampling.seed}"
    name += f"_dseed{config.data.seed}"
    name += f"_mseed{config.model.seed}"
            
    name += _generator_cache_suffix(config)
    if getattr(config.data, "pad_vocab_to", None) is not None:
        name += f"_vocab{config.data.pad_vocab_to}"
    if getattr(config.model, "vocab_align_to", None) is not None:
        name += f"_vocab_align{config.model.vocab_align_to}"
    return name


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


def _fit_epoch_ridges(
    model, bundle, tokenizer, config, device,
    *,
    ridge_dataset=None,
    ridge_loader=None,
    encoded=None,
):
    if not bool(getattr(config.train, "cost_ridge_vec", False)):
        return [], None, {}

    if ridge_dataset is None:
        include_val = bool(getattr(config.train, "cost_ridge_include_val", False))
        if include_val and len(bundle.val_dataset.samples) > 0:
            ridge_dataset = LatentParamDataset(
                list(bundle.train_dataset.samples) + list(bundle.val_dataset.samples)
            )
        else:
            ridge_dataset = bundle.train_dataset

    ridge_alphas = _resolve_ridge_alphas(config)
    _ridge_cost_target = str(getattr(config.data, "cost_target", "neg_log"))
    _ridge_cost_target_regression = getattr(config.data, "cost_target_regression", None)
    _ridge_mins = list(bundle.task_min_costs.values())
    _ridge_task_min_cost = float(_ridge_mins[0]) if _ridge_mins else None
    _ridge_fit_target = _ridge_cost_target_regression or _ridge_cost_target
    print(
        f"[ridge] fit_target={_ridge_fit_target!r} output_target={_ridge_cost_target!r} "
        f"task_min_cost={_ridge_task_min_cost!r}"
    )
    latent_cost_ridges = fit_latent_cost_ridges(
        model,
        ridge_dataset,
        tokenizer,
        device,
        alphas=ridge_alphas,
        batch_size=config.eval.batch_size,
        cost_target=_ridge_cost_target,
        cost_target_regression=_ridge_cost_target_regression,
        task_min_cost=_ridge_task_min_cost,
        loader=ridge_loader,
        encoded=encoded,
    )
    ridge_metrics = {}
    for ridge_payload in latent_cost_ridges:
        alpha = float(ridge_payload["alpha"])
        alpha_suffix = _alpha_metric_suffix(alpha)
        if alpha == float(ridge_alphas[0]):
            ridge_metrics["train_ridge_mse"] = float(ridge_payload["train_mse"])
        ridge_metrics[f"train_ridge_alpha_{alpha_suffix}_mse"] = float(ridge_payload["train_mse"])

    if bool(getattr(config.train, "cost_ridge_weighted", False)):
        weighted_ridges = fit_latent_cost_ridges(
            model,
            ridge_dataset,
            tokenizer,
            device,
            alphas=ridge_alphas,
            batch_size=config.eval.batch_size,
            sample_weight_quantile=float(getattr(config.train, "weight_quantile", 0.85)),
            sample_weight_sigma=float(getattr(config.train, "weight_sigma", 0.25)),
            cost_target=_ridge_cost_target,
            cost_target_regression=_ridge_cost_target_regression,
            task_min_cost=_ridge_task_min_cost,
            loader=ridge_loader,
            encoded=encoded,
        )
        for ridge_payload in weighted_ridges:
            alpha = float(ridge_payload["alpha"])
            alpha_suffix = _alpha_metric_suffix(alpha)
            if alpha == float(ridge_alphas[0]):
                ridge_metrics["train_ridge_weighted_mse"] = float(ridge_payload["train_mse"])
            ridge_metrics[f"train_ridge_weighted_alpha_{alpha_suffix}_mse"] = float(
                ridge_payload["train_mse"]
            )
        latent_cost_ridges = list(latent_cost_ridges) + list(weighted_ridges)

    return latent_cost_ridges, ridge_metrics


def _evaluate_validation_epoch(
    model, bundle, registry, tokenizer, config, device, epoch, latent_cost_ridges,
    *,
    val_loader=None,
    encoded_val=None,
):
    summary: Dict[str, float] = {}
    if not bundle.val_dataset.samples:
        return summary

    # print(f"[train] evaluating validation split with teacher forcing after epoch {epoch}")
    val_tf_metrics = evaluate_teacher_forcing(
        model,
        bundle.val_dataset,
        registry,
        tokenizer,
        device,
        batch_size=config.eval.batch_size,
        use_compressed=bool(getattr(config.train, "use_compressed_teacher_forcing", False)),
        loader=val_loader,
    )
    summary.update({f"val_{k}": float(v) for k, v in val_tf_metrics.items()})
    print(
        f"[epoch {epoch}] val_tok_acc={summary['val_token_accuracy']:.4f} "
        f"val_exact={summary['val_full_sequence_exact_match']:.4f}"
    )

    # print("[train] evaluating validation cost ranking")
    cost_metrics = evaluate_cost_ranking(
        model,
        bundle.val_dataset,
        tokenizer,
        device,
        batch_size=config.eval.batch_size,
        latent_cost_ridges=_build_named_latent_cost_ridges(latent_cost_ridges),
        loader=val_loader,
        encoded=encoded_val,
    )
    summary.update({f"val_{k}": v for k, v in cost_metrics.items()})

    if "cost_head_actual_top1_pred_rank" in cost_metrics:
        print(
            f"val_cost_head_actual_top1_pred_rank : {int(cost_metrics['cost_head_actual_top1_pred_rank'])}\n"
            f"val_cost_head_pred_top1_actual_cost : {cost_metrics['cost_head_pred_top1_actual_cost']:.6f}\n"
            f"val_cost_head_pred_top10_mean_actual_cost : {cost_metrics['cost_head_pred_top10_mean_actual_cost']:.6f}\n"
            # f"val_cost_head_pred_top20_mean_actual_cost : {cost_metrics['cost_head_pred_top20_mean_actual_cost']:.6f}\n"
        )
    if "cost_vec_actual_top1_pred_rank" in cost_metrics:
        print(
            f"val_cost_vec_actual_top1_pred_rank : {int(cost_metrics['cost_vec_actual_top1_pred_rank'])}\n"
            f"val_cost_vec_pred_top1_actual_cost : {cost_metrics['cost_vec_pred_top1_actual_cost']:.6f}\n"
            f"val_cost_vec_pred_top10_mean_actual_cost : {cost_metrics['cost_vec_pred_top10_mean_actual_cost']:.6f}\n"
            # f"val_cost_vec_pred_top20_mean_actual_cost : {cost_metrics['cost_vec_pred_top20_mean_actual_cost']:.6f}\n"
        )
    if "cost_vec_weighted_actual_top1_pred_rank" in cost_metrics:
        print(
            f"val_cost_vec_weighted_actual_top1_pred_rank : {int(cost_metrics['cost_vec_weighted_actual_top1_pred_rank'])}\n"
            f"val_cost_vec_weighted_pred_top1_actual_cost : {cost_metrics['cost_vec_weighted_pred_top1_actual_cost']:.6f}\n"
            f"val_cost_vec_weighted_pred_top10_mean_actual_cost : {cost_metrics['cost_vec_weighted_pred_top10_mean_actual_cost']:.6f}\n"
        )
    for key, value in sorted(cost_metrics.items()):
        if key.startswith("cost_vec_alpha_") and key.endswith("_actual_top1_pred_rank"):
            prefix = key[: -len("_actual_top1_pred_rank")]
            top1_cost_key = f"{prefix}_pred_top1_actual_cost"
            top10_key = f"{prefix}_pred_top10_mean_actual_cost"
            top20_key = f"{prefix}_pred_top20_mean_actual_cost"
            print(
                f"{'val_' + key} : {int(value)}\n"
                f"{'val_' + top1_cost_key} : {cost_metrics[top1_cost_key]:.6f}\n"
                f"{'val_' + top10_key} : {cost_metrics[top10_key]:.6f}\n"
                f"{'val_' + top20_key} : {cost_metrics[top20_key]:.6f}\n"
            )

    return summary


def _evaluate_final_checkpoint(model, bundle, registry, tokenizer, config, device, latent_cost_ridges):
    summary: Dict[str, float] = {}
    run_full_ar = bool(getattr(config.eval, "final_full_autoregressive", True))

    if bundle.val_dataset.samples:
        print("[train] evaluating best checkpoint on val split")
        val_tf_metrics = evaluate_teacher_forcing(
            model,
            bundle.val_dataset,
            registry,
            tokenizer,
            device,
            batch_size=config.eval.batch_size,
            use_compressed=bool(getattr(config.train, "use_compressed_teacher_forcing", False)),
        )
        summary.update({f"eval_val_{k}": float(v) for k, v in val_tf_metrics.items()})

        val_cost_metrics = evaluate_cost_ranking(
            model,
            bundle.val_dataset,
            tokenizer,
            device,
            batch_size=config.eval.batch_size,
            latent_cost_ridges=_build_named_latent_cost_ridges(latent_cost_ridges),
        )
        summary.update({f"eval_val_{k}": float(v) for k, v in val_cost_metrics.items()})

        if run_full_ar:
            val_ar_metrics = evaluate_autoregressive(
                model,
                bundle.val_dataset,
                registry,
                tokenizer,
                device,
                batch_size=config.eval.batch_size,
            )
            summary.update({f"val_autoregressive_{k}": float(v) for k, v in val_ar_metrics.items()})
        else:
            print("[train] skipping val full autoregressive eval (config.eval.final_full_autoregressive=False)")

    if bundle.test_dataset.samples:
        print("[train] evaluating best checkpoint on test split")
        test_tf_metrics = evaluate_teacher_forcing(
            model,
            bundle.test_dataset,
            registry,
            tokenizer,
            device,
            batch_size=config.eval.batch_size,
            use_compressed=bool(getattr(config.train, "use_compressed_teacher_forcing", False)),
        )
        summary.update({f"eval_test_{k}": float(v) for k, v in test_tf_metrics.items()})

        test_cost_metrics = evaluate_cost_ranking(
            model,
            bundle.test_dataset,
            tokenizer,
            device,
            batch_size=config.eval.batch_size,
            latent_cost_ridges=_build_named_latent_cost_ridges(latent_cost_ridges),
        )
        summary.update({f"eval_test_{k}": float(v) for k, v in test_cost_metrics.items()})

        if run_full_ar:
            test_ar_metrics = evaluate_autoregressive(
                model,
                bundle.test_dataset,
                registry,
                tokenizer,
                device,
                batch_size=config.eval.batch_size,
            )
            summary.update({f"eval_test_autoregressive_{k}": float(v) for k, v in test_ar_metrics.items()})
        else:
            print("[train] skipping test full autoregressive eval (config.eval.final_full_autoregressive=False)")

    return summary


def _resolve_pt_dir(config) -> Path:
    """Mirror of the inline ``pt_dir`` derivation that runs once early so we
    can peek at ``last.pt`` (for resume) before ``wandb.init``. Uses the
    config-only family resolver — equivalent to the bundle-based one because
    bundle records originate from ``config.data.json_paths``.
    """
    base = Path(config.train.checkpoint_dir).expanduser().resolve()
    family = _resolve_run_family_from_config(config)
    if family and family != "na" and base.name != family:
        base = base / family
    base = base / "checkpoints"
    project = getattr(config.wandb, "project", None)
    if project:
        base = base / str(project)
    name = _build_wandb_run_name(config)
    if name:
        base = base / str(name)
    return base


def _config_for_resume_compare(payload: dict) -> dict:
    """Strip resume-related infra fields before comparing two config dumps.
    These are control flags that legitimately differ between original run
    and resume; everything else must match exactly.
    """
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    train = dict(out.get("train") or {})
    train.pop("resume", None)
    train.pop("resume_from", None)
    out["train"] = train
    return out


def train_main(config) -> Dict[str, float]:
    seed_everything(config.data.seed)
    device = resolve_device(config.train.device)
    configure_runtime(config, device)
    print(f"[train] resolved device: requested={config.train.device} actual={device}")
    print(
        f"[train] runtime config: amp={bool(config.train.use_amp)} "
        f"tf32={bool(getattr(config.train, 'allow_tf32', True))}"
    )

    # Pre-compute pt_dir so we can decide whether to resume an existing run
    # (model + wandb) BEFORE wandb.init.
    early_pt_dir = _resolve_pt_dir(config)
    early_last_ckpt = early_pt_dir / "last.pt"
    resume_payload: Optional[dict] = None
    auto_resume_path: Optional[str] = None
    if bool(getattr(config.train, "resume", False)):
        if early_last_ckpt.exists():
            try:
                peek = torch.load(early_last_ckpt, map_location="cpu")
            except Exception as err:  # pragma: no cover
                print(
                    f"[train] resume: failed to read {early_last_ckpt}: "
                    f"{type(err).__name__}: {err}"
                )
                peek = None
            if peek is not None:
                saved = _config_for_resume_compare(peek.get("config") or {})
                current = _config_for_resume_compare(config.to_dict())
                if saved != current:
                    raise RuntimeError(
                        f"[train] resume aborted: config in {early_last_ckpt} "
                        f"does not match current config"
                    )
                resume_payload = peek
                auto_resume_path = str(early_last_ckpt)
                print(
                    f"[train] resume: matched config; will resume from "
                    f"{early_last_ckpt} (epoch={peek.get('epoch')})"
                )
        else:
            print(
                f"[train] resume: no checkpoint at {early_last_ckpt}; "
                "starting fresh"
            )

    wandb_run = None
    wandb_project = getattr(config.wandb, "project", None)
    run_name = _build_wandb_run_name(config)
    if wandb_project:
        if wandb is None:
            print("[train] wandb project is set but wandb is not installed; skipping wandb logging")
        else:
            project_name = _build_wandb_project_name(config)
            # Co-locate wandb run logs with the model checkpoints by pointing
            # ``wandb.init(dir=...)`` at the project subdir. wandb auto-creates
            # ``{dir}/wandb/run-<id>/`` inside it, matching
            # ``{checkpoint_dir}/{family}/checkpoints/{wandb_project}/wandb/run-…``.
            wandb_log_root = Path(config.train.checkpoint_dir).expanduser().resolve()
            family_for_log = _resolve_run_family_from_config(config)
            if family_for_log and family_for_log != "na":
                wandb_log_root = wandb_log_root / family_for_log
            wandb_log_root = wandb_log_root / "checkpoints" / str(wandb_project)
            wandb_log_root.mkdir(parents=True, exist_ok=True)
            print(
                f"[train] initializing wandb: project={project_name} run={run_name} "
                f"dir={wandb_log_root}"
            )
            wandb_init_kwargs = dict(
                project=project_name,
                name=run_name,
                config=config.to_dict(),
                dir=str(wandb_log_root),
            )
            resume_run_id = (
                resume_payload.get("wandb_run_id") if resume_payload else None
            )
            if resume_run_id:
                wandb_init_kwargs["id"] = str(resume_run_id)
                wandb_init_kwargs["resume"] = "allow"
                print(f"[train] resume: continuing wandb run id={resume_run_id}")
            wandb_run = wandb.init(**wandb_init_kwargs)

    registry, bundle, tokenizer, model = build_everything(config)
    model.to(device)
    print(f"[train] model moved to {device}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] model/total_params: {total_params:,}")

    if wandb_run is not None:
        wandb_run.summary["model/architecture"] = str(model)
        wandb_run.summary["model/total_params"] = total_params
        wandb_run.summary["model/trainable_params"] = trainable_params


    train_loader = prepare_loader(
        bundle.train_dataset,
        tokenizer,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=config.train.pin_memory and device.type == "cuda",
        persistent_workers=config.train.persistent_workers,
        prefetch_factor=config.train.prefetch_factor,
    )
    print(
        f"[train] data loader ready: batches={len(train_loader)} "
        f"batch_size={config.train.batch_size} "
        f"num_workers={config.train.num_workers} "
        f"pin_memory={bool(config.train.pin_memory and device.type == 'cuda')}"
    )

    # Persistent eval / ridge loaders. Building them once here (rather than
    # inside ``_evaluate_validation_epoch`` / ``_fit_epoch_ridges`` per epoch)
    # amortizes worker fork over the whole run; the previous per-epoch
    # rebuild was the dominant cause of validation/ridge being slower than
    # the no-worker baseline.
    eval_num_workers = int(config.train.num_workers)
    eval_pin_memory = bool(config.train.pin_memory and device.type == "cuda")
    val_loader = None
    if bundle.val_dataset.samples:
        val_loader = prepare_loader(
            bundle.val_dataset,
            tokenizer,
            batch_size=config.eval.batch_size,
            shuffle=False,
            num_workers=eval_num_workers,
            pin_memory=eval_pin_memory,
            persistent_workers=config.train.persistent_workers,
            prefetch_factor=config.train.prefetch_factor,
        )
    # ``train_eval_loader`` covers the per-epoch ``evaluate_teacher_forcing`` /
    # ``evaluate_cost_ranking`` calls on ``bundle.train_dataset`` (~9× the val
    # split). Without this, those two calls fell back to internal
    # ``num_workers=0`` loaders and held GPU at <20% util while main-thread
    # collate ran ~1000 batches sequentially.
    train_eval_loader = prepare_loader(
        bundle.train_dataset,
        tokenizer,
        batch_size=config.eval.batch_size,
        shuffle=False,
        num_workers=eval_num_workers,
        pin_memory=eval_pin_memory,
        persistent_workers=config.train.persistent_workers,
        prefetch_factor=config.train.prefetch_factor,
    )
    if bool(getattr(config.train, "cost_ridge_vec", False)):
        if (
            bool(getattr(config.train, "cost_ridge_include_val", False))
            and bundle.val_dataset.samples
        ):
            ridge_dataset = LatentParamDataset(
                list(bundle.train_dataset.samples) + list(bundle.val_dataset.samples)
            )
        else:
            ridge_dataset = bundle.train_dataset
        ridge_loader = prepare_loader(
            ridge_dataset,
            tokenizer,
            batch_size=config.eval.batch_size,
            shuffle=False,
            num_workers=eval_num_workers,
            pin_memory=eval_pin_memory,
            persistent_workers=config.train.persistent_workers,
            prefetch_factor=config.train.prefetch_factor,
        )
    else:
        ridge_dataset = None
        ridge_loader = None
    print(
        f"[train] eval loaders: train_batches={len(train_eval_loader)} "
        f"val_batches="
        f"{len(val_loader) if val_loader is not None else 0} "
        f"ridge_batches={len(ridge_loader) if ridge_loader is not None else 0} "
        f"num_workers={eval_num_workers}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    best_metric_name = str(getattr(config.train, "best_metric_name", "val_full_sequence_exact_match"))
    best_metric_mode = str(getattr(config.train, "best_metric_mode", "max")).lower()
    early_stop_patience = int(getattr(config.train, "early_stop_patience", 15))
    early_stop_min_delta = float(getattr(config.train, "early_stop_min_delta", 1e-4))

    scheduler_name = str(getattr(config.train, "scheduler_name", "none")).lower()
    warmup_epochs = max(0, int(getattr(config.train, "warmup_epochs", 0) or 0))
    warmup_start_factor = float(getattr(config.train, "warmup_start_factor", 0.1))
    if scheduler_name == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(getattr(config.train, "scheduler_milestones", [20])),
            gamma=float(getattr(config.train, "scheduler_gamma", 1.0 / 3.0)),
        )
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=best_metric_mode,
            factor=float(getattr(config.train, "plateau_factor", 0.5)),
            patience=int(getattr(config.train, "plateau_patience", 5)),
            threshold=float(getattr(config.train, "plateau_threshold", 1e-4)),
            threshold_mode="abs",
            min_lr=float(getattr(config.train, "plateau_min_lr", 1e-5)),
        )
    elif scheduler_name == "cosine":
        user_t_max = int(getattr(config.train, "cosine_t_max", 0) or 0)
        if user_t_max > 0:
            cosine_t_max = user_t_max
        else:
            cosine_t_max = max(1, int(config.train.num_epochs) - warmup_epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_t_max,
            eta_min=float(getattr(config.train, "cosine_eta_min", 0.0)),
        )
    else:
        scheduler = None

    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=max(1e-8, warmup_start_factor),
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        print(
            f"[train] warmup: epochs={warmup_epochs} "
            f"start_factor={warmup_start_factor:.4g}"
        )
    else:
        warmup_scheduler = None
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.train.use_amp and device.type == "cuda"))

    checkpoint_dir = Path(config.train.checkpoint_dir).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] checkpoint dir: {checkpoint_dir}")

    # ``pt_dir`` was precomputed via ``_resolve_pt_dir(config)`` before
    # wandb.init so we could peek at last.pt for resume. Reuse that here.
    pt_dir = early_pt_dir
    # ``config.json`` and ``tokenizer.json`` live alongside ``best.pt`` /
    # ``last.pt`` in the per-run directory so each run is fully
    # self-describing.
    save_training_artifacts(pt_dir, config, tokenizer)
    print(f"[train] checkpoint pt dir: {pt_dir}")

    start_epoch = 1
    best_exact_match = float("-inf")
    best_val_acc = float("-inf")
    best_checkpoint_path = pt_dir / "best.pt"
    last_checkpoint_path = pt_dir / "last.pt"
    if best_metric_mode == "max":
        best_metric_value = float("-inf")
    else:
        best_metric_value = float("inf")
    epochs_without_improve = 0
    best_metric_epoch: Optional[int] = None
    last_summary: Dict[str, float] = {}

    # Explicit ``resume_from`` path takes precedence; otherwise fall back to
    # the auto-detected ``last.pt`` from ``config.train.resume=True``.
    resume_path = config.train.resume_from or auto_resume_path
    if resume_path:
        print(f"[train] resuming from {resume_path}")
        payload = load_checkpoint(resume_path, model, optimizer, scheduler)
        start_epoch = int(payload["epoch"]) + 1
        best_exact_match = float(payload.get("best_exact_match", best_exact_match))
        best_val_acc = float(payload.get("best_val_acc", best_val_acc))
        best_metric_value = float(payload.get("best_metric_value", best_metric_value))
        epochs_without_improve = int(payload.get("epochs_without_improve", epochs_without_improve))
        best_metric_epoch = payload.get("best_metric_epoch", None)
        if best_metric_epoch is not None:
            best_metric_epoch = int(best_metric_epoch)

    best_metrics: Dict[str, float] = {}
    latent_cost_ridges: list[dict] = []
    walk_sample_buffer = WalkSampleBuffer()
    # Running maxima per task across all walks (periodic + final). Power the
    # ``walk_best_cost`` / ``walk_best_epoch`` / ``walk_alpha_at_best`` panels.
    walk_running_best: Dict[str, float] = {}
    walk_running_best_epoch: Dict[str, int] = {}
    walk_running_best_alpha: Dict[str, float] = {}
    timestamp = time.strftime("%m%d%H%M")
    latent_walk_every_n = int(getattr(config.latent_walk, "every_n_epochs", 0) or 0)
    latent_walk_on_final = bool(getattr(config.latent_walk, "on_final", False))
    latent_walk_top_k = int(getattr(config.latent_walk, "top_k", 1) or 1)
    latent_walk_num_steps = int(getattr(config.latent_walk, "num_steps", 8) or 8)
    latent_walk_step_size = float(getattr(config.latent_walk, "step_size", 0.25) or 0.25)
    latent_walk_use_latent_gradient = not bool(getattr(config.train, "cost_ridge_vec", False))
    walk_recon_predict_every_n = int(getattr(config.latent_walk, "predict_every_n_epochs", 0) or 0)
    walk_recon_predict_enabled = walk_recon_predict_every_n > 0
    walk_recon_predict_use_gp = bool(getattr(config.latent_walk, "predict_use_gp", False))
    walk_recon_predict_gp_top_k = int(getattr(config.latent_walk, "predict_gp_top_k", 0) or 0)
    walk_recon_predict_gp_random_n = int(getattr(config.latent_walk, "predict_gp_random_n", 0) or 0)
    latent_walk_record_jsons = _resolve_walk_record_jsons(config)
    latent_walk_output_dir = (
        getattr(config.latent_walk, "output_dir", None)
        or str(checkpoint_dir)
    )
    latent_walk_network_info = getattr(config.data, "network_info_folder", None)
    latent_walk_sampling_options = SamplingOptions.from_config(
        getattr(config, "sampling", None)
    )
    if latent_walk_sampling_options.strategy != "greedy":
        print(
            f"[train] latent walk decoding: strategy={latent_walk_sampling_options.strategy} "
            f"temperature={latent_walk_sampling_options.temperature} "
            f"top_k={latent_walk_sampling_options.top_k} "
            f"top_p={latent_walk_sampling_options.top_p}"
        )
    if (latent_walk_every_n > 0 or latent_walk_on_final) and not latent_walk_record_jsons:
        print(
            "[train] latent walk requested but no record JSON resolvable from "
            "config.latent_walk.record_json or config.data.json_paths; disabling"
        )
        latent_walk_every_n = 0
        latent_walk_on_final = False
    elif len(latent_walk_record_jsons) > 1:
        print(
            f"[train] latent walk will iterate {len(latent_walk_record_jsons)} "
            f"task json(s) per trigger"
        )

    # A bundle aggregates one or more tasks; ``task_min_costs`` is keyed by
    # ``(workload_key, target_kind)``. The bundle-level fallback below is only
    # used when a per-task probe fails (e.g. malformed JSON).
    _mins = list(bundle.task_min_costs.values())
    _lookup_task_min_cost: Optional[float] = float(_mins[0]) if _mins else None

    # Per-task measurement-lookup files. One JSONL per task, named by the
    # task_index extracted from the record-JSON filename. Each file holds
    # ``{"workload_key", "sym_map", "cost"}`` entries — workload_key in the
    # cache key already prevents cross-task contamination, but splitting the
    # files keeps each task's history independently inspectable, deletable,
    # and auditable.
    measurement_lookup_paths_by_task: Dict[str, Path] = {}
    persistent_measurement_cache_by_task: Dict[str, dict] = {}
    task_min_cost_by_task: Dict[str, Optional[float]] = {}
    workload_key_by_task: Dict[str, Optional[str]] = {}
    # ``bundle.{train,val,test}_records`` already carry every record's
    # ``(workload_key, target_kind)``; build a json_path → record index so
    # the per-task probes below can read those fields without re-parsing the
    # source JSON files (33 tasks × ~1 MB each, formerly loaded twice each).
    records_by_json_path: Dict[str, JsonSampleRecord] = {}
    for record in (
        list(bundle.train_records)
        + list(bundle.val_records)
        + list(bundle.test_records)
    ):
        json_path = getattr(record, "json_path", None)
        if json_path:
            records_by_json_path.setdefault(str(json_path), record)
    # Single canonical lookup file per task at
    # ``{checkpoint_dir}/{family}/lookup_sym_maps/{task_index}_*.jsonl``.
    # ``build_measurement_lookup.py`` and the training-time
    # ``_save_measurement_lookup`` write to and read from the same file
    # (previously the runtime path lived as a sibling
    # ``{family}/{task_index}_measurement_lookup.jsonl`` separate from the
    # build-script seeds, which forced a two-source merge every startup).
    try:
        from modules.task_paths import clean_name as _clean_name
    except ImportError:  # pragma: no cover
        _clean_name = lambda x: str(x)  # noqa: E731
    for record_json_path in latent_walk_record_jsons:
        key = _task_lookup_key_from_json(record_json_path)
        family = _family_from_json_path(record_json_path)
        lookup_dir = Path(checkpoint_dir)
        if family and family != "na" and lookup_dir.name != family:
            lookup_dir = lookup_dir / family
        lookup_dir = lookup_dir / "lookup_sym_maps"
        existing_matches = (
            sorted(lookup_dir.glob(f"{key}_*.jsonl")) if lookup_dir.is_dir() else []
        )
        probe_record = records_by_json_path.get(str(record_json_path))
        if probe_record is not None:
            workload_key_by_task[key] = getattr(probe_record, "workload_key", None)
            target_kind = getattr(probe_record, "target_kind", None)
            looked_up = bundle.task_min_cost_for(
                getattr(probe_record, "workload_key", None), target_kind
            )
            per_task_min_cost = (
                looked_up if looked_up is not None else _lookup_task_min_cost
            )
        else:
            # Fallback: only when the walk JSON wasn't part of bundle.{train,val,test}.
            workload_key_by_task[key] = _probe_workload_key_from_json(record_json_path)
            per_task_min_cost = _resolve_walk_task_min_cost(
                record_json_path, bundle=bundle, fallback=_lookup_task_min_cost
            )
        task_min_cost_by_task[key] = per_task_min_cost
        if existing_matches:
            path = existing_matches[0]
        else:
            workload_key = workload_key_by_task.get(key)
            target_kind_for_name = (
                getattr(probe_record, "target_kind", None) if probe_record is not None else None
            )
            if workload_key:
                stem = f"{key}_{_clean_name((workload_key, target_kind_for_name or ''))}"
            else:
                stem = f"{key}_measurement_lookup"
            lookup_dir.mkdir(parents=True, exist_ok=True)
            path = lookup_dir / f"{stem}.jsonl"
        measurement_lookup_paths_by_task[key] = path
        cache = _load_measurement_lookup(
            path,
            cost_target=bundle.cost_target,
            task_min_cost=per_task_min_cost,
        )
        persistent_measurement_cache_by_task[key] = cache
        # if cache:
        #     print(
        #         f"[train] measurement lookup loaded: {len(cache)} entries from {path}"
        #     )
        # else:
        #     print(f"[train] measurement lookup: none at {path}")

    for epoch in range(start_epoch, config.train.num_epochs + 1):
        print(f"[train] starting epoch {epoch}/{config.train.num_epochs}")
        _phase_t0 = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            registry,
            tokenizer,
            config,
            device,
            epoch,
            task_min_cost=_lookup_task_min_cost,
        )
        print(f"[timing] train={time.time() - _phase_t0:.1f}s")

        # train_metrics already carries ``token_accuracy`` and
        # ``full_sequence_exact_match`` collected during the training loop's
        # forward passes — the previous ``evaluate_teacher_forcing`` call on
        # ``bundle.train_dataset`` was a redundant 30+ s second forward over
        # the same data and has been removed.
        summary = dict(train_metrics)
        print(
            f"[epoch {epoch}] "
            f"loss={summary['loss']:.4f} recon={summary['recon_loss']:.4f} "
            f"kl={summary['kl_loss']:.4f} "
            f"tok_acc={summary['token_accuracy']:.4f} "
            f"exact={summary['full_sequence_exact_match']:.4f}"
        )

        # Encode every cost-ranking / ridge consumer's data ONCE per epoch.
        # Old order ran three encoder passes (ridge over train+val,
        # cost_ranking on train, cost_ranking on val); now they all share
        # ``encoded_train`` / ``encoded_val`` outputs.
        _phase_t0 = time.time()
        encoded_train = encode_dataset(
            model,
            bundle.train_dataset,
            tokenizer,
            device,
            batch_size=config.eval.batch_size,
            loader=train_eval_loader,
        )
        encoded_val = None
        if bundle.val_dataset.samples and val_loader is not None:
            encoded_val = encode_dataset(
                model,
                bundle.val_dataset,
                tokenizer,
                device,
                batch_size=config.eval.batch_size,
                loader=val_loader,
            )
        print(f"[timing] encode={time.time() - _phase_t0:.1f}s")

        _phase_t0 = time.time()
        if (
            bool(getattr(config.train, "cost_ridge_include_val", False))
            and encoded_val is not None
        ):
            ridge_encoded = _concat_encoded([encoded_train, encoded_val])
        else:
            ridge_encoded = encoded_train
        latent_cost_ridges, ridge_metrics = _fit_epoch_ridges(
            model,
            bundle,
            tokenizer,
            config,
            device,
            ridge_dataset=ridge_dataset,
            ridge_loader=ridge_loader,
            encoded=ridge_encoded,
        )
        summary.update(ridge_metrics)
        print(f"[timing] ridge={time.time() - _phase_t0:.1f}s")

        _phase_t0 = time.time()
        train_cost_metrics = evaluate_cost_ranking(
            model,
            bundle.train_dataset,
            tokenizer,
            device,
            batch_size=config.eval.batch_size,
            latent_cost_ridges=_build_named_latent_cost_ridges(latent_cost_ridges),
            loader=train_eval_loader,
            encoded=encoded_train,
        )
        summary.update({f"train_{k}": float(v) for k, v in train_cost_metrics.items()})
        print(f"[timing] train_cost={time.time() - _phase_t0:.1f}s")

        _phase_t0 = time.time()
        summary.update(
            _evaluate_validation_epoch(
                model,
                bundle,
                registry,
                tokenizer,
                config,
                device,
                epoch,
                latent_cost_ridges,
                val_loader=val_loader,
                encoded_val=encoded_val,
            )
        )
        print(f"[timing] val={time.time() - _phase_t0:.1f}s")

        def _fmt(value) -> str:
            return f"{float(value):+.4f}" if value is not None and math.isfinite(float(value)) else "nan"

        for source in ("cost_head", "cost_vec", "cost_vec_weighted"):
            tr_all = summary.get(f"train_{source}_spearman")
            tr_top = summary.get(f"train_{source}_spearman_top5pct")
            va_all = summary.get(f"val_{source}_spearman")
            va_top = summary.get(f"val_{source}_spearman_top5pct")
            if tr_all is None and va_all is None:
                continue
            print(
                f"[epoch {epoch}] {source.replace('cost_', '')} spearman "
                f"train={_fmt(tr_all)} (top5%={_fmt(tr_top)}) "
                f"val={_fmt(va_all)} (top5%={_fmt(va_top)})"
            )

        if "val_full_sequence_exact_match" in summary:
            best_exact_match = max(best_exact_match, float(summary["val_full_sequence_exact_match"]))
        if "val_token_accuracy" in summary:
            best_val_acc = max(best_val_acc, float(summary["val_token_accuracy"]))

        # Run latent walk BEFORE improvement check so walk-based best metrics
        # (e.g. walk/best_measured_mean_cost) actually influence best.pt save.
        walk_summary: Dict[str, float] = {}
        is_final_epoch = epoch == int(config.train.num_epochs)
        recon_predict_due = walk_recon_predict_enabled and (
            (epoch % walk_recon_predict_every_n == 0) or is_final_epoch
        )
        walk_due = latent_walk_every_n > 0 and (
            (epoch % latent_walk_every_n == 0) or is_final_epoch
        )
        # Snapshot ``last.pt`` BEFORE the walk so a crash or interruption mid-
        # walk still leaves a resumable checkpoint at this epoch's
        # post-training state. ``best_metric_value`` / ``epochs_without_improve``
        # carry their pre-walk (= last finalized) values; the walk-driven
        # bookkeeping update + ``best.pt`` / ``epoch_N.pt`` saves still happen
        # after the walk completes.
        save_checkpoint(
            last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_exact_match=best_exact_match,
            best_val_acc=best_val_acc,
            best_metric_name=best_metric_name,
            best_metric_value=best_metric_value,
            best_metric_epoch=best_metric_epoch,
            epochs_without_improve=epochs_without_improve,
            config=config,
            tokenizer=tokenizer,
            latent_cost_ridge=next(
                (p for p in latent_cost_ridges if not bool(p.get("weighted", False))),
                latent_cost_ridges[0] if latent_cost_ridges else None,
            ),
            latent_cost_ridges=latent_cost_ridges,
            timestamp=timestamp,
            wandb_run_id=(wandb_run.id if wandb_run is not None else None),
        )
        # Re-encode is computed inside the walk, so the walk must run whenever
        # re-encode is due, even off the regular walk cadence.
        if walk_due or recon_predict_due:
            include_recon_predict = recon_predict_due
            include_measurement = walk_due
            walk_recon_predictor = None
            if include_recon_predict and walk_recon_predict_use_gp:
                walk_recon_predictor = fit_gp_recon_predictor(
                    model=model,
                    dataset=bundle.train_dataset,
                    tokenizer=tokenizer,
                    device=device,
                    top_k=walk_recon_predict_gp_top_k,
                    random_n=walk_recon_predict_gp_random_n,
                    batch_size=config.eval.batch_size,
                    seed=int(getattr(config.data, "seed", 0)),
                    walk_buffer=walk_sample_buffer,
                )
            walk_summary = {}
            ridge_walks = list(_iter_walk_ridges(latent_cost_ridges, config)) or [(None, "", False)]
            # One shared cache per task. Each task's cache is seeded from its
            # own on-disk lookup file plus the matching subset of the in-memory
            # walk_buffer (filtered by workload_key). The walk subsequently
            # populates that same dict with new measurements, which we merge
            # back into the task's persistent cache and write to its own
            # ``{task}_measurement_lookup.jsonl`` after the walks for this
            # epoch finish.
            shared_measurement_cache_by_task: Dict[str, dict] = {}
            for record_json_path in latent_walk_record_jsons:
                tkey = _task_lookup_key_from_json(record_json_path)
                shared_measurement_cache_by_task[tkey] = (
                    _seed_measurement_cache_from_buffer(
                        walk_sample_buffer,
                        disk_cache=persistent_measurement_cache_by_task.get(tkey),
                        cost_target=bundle.cost_target,
                        task_min_cost=task_min_cost_by_task.get(tkey, _lookup_task_min_cost),
                        workload_key_filter=workload_key_by_task.get(tkey),
                    )
                )
            reencode_predictor_name = str(
                getattr(config.train, "re_encode_predictor", "cost_head")
            )
            reencode_predictor = _build_reencode_predictor(
                name=reencode_predictor_name,
                model=model,
                bundle=bundle,
                tokenizer=tokenizer,
                device=device,
                latent_cost_ridges=latent_cost_ridges,
                config=config,
            )
            for walk_ridge, walk_prefix, force_cost_head in ridge_walks:
                per_task_summaries_keyed: List[tuple[str, Dict[str, float]]] = []
                for record_json_path in latent_walk_record_jsons:
                    tkey = _task_lookup_key_from_json(record_json_path)
                    task_label = Path(record_json_path).stem
                    label = (
                        f"epoch {epoch}"
                        + (f" [{walk_prefix.rstrip('_')}]" if walk_prefix else "")
                        + (f" task={task_label}" if len(latent_walk_record_jsons) > 1 else "")
                    )
                    # All tasks share the same ``{checkpoint_dir}/{family}``
                    # output root; per-task disambiguation happens via the
                    # ``{task_index}_{...}.json`` filename.
                    per_task_output_dir = latent_walk_output_dir
                    per_task_min_cost = task_min_cost_by_task.get(
                        tkey, _lookup_task_min_cost
                    )
                    sub_summary = _run_periodic_latent_walk(
                        model=model,
                        device=device,
                        checkpoint_path=last_checkpoint_path,
                        record_json_path=record_json_path,
                        walk_output_dir=per_task_output_dir,
                        network_info_folder=latent_walk_network_info,
                        epoch_label=label,
                        config=config,
                        registry=registry,
                        tokenizer=tokenizer,
                        latent_cost_ridge=walk_ridge,
                        timestamp=timestamp,
                        top_k=latent_walk_top_k,
                        num_steps=latent_walk_num_steps,
                        step_size=latent_walk_step_size,
                        use_latent_gradient=force_cost_head or latent_walk_use_latent_gradient,
                        include_recon_predict=include_recon_predict,
                        include_measurement=include_measurement,
                        recon_predictor=walk_recon_predictor,
                        reencode_predictor=reencode_predictor,
                        reencode_predictor_name=reencode_predictor_name,
                        walk_buffer=walk_sample_buffer if include_measurement else None,
                        walk_key_prefix=walk_prefix,
                        measurement_cache=shared_measurement_cache_by_task[tkey],
                        sampling_options=latent_walk_sampling_options,
                        cost_target=bundle.cost_target,
                        task_min_cost=per_task_min_cost,
                        sort_by=str(getattr(config.latent_walk, "sort_by", "re_pred")),
                        show_neg_log=bool(getattr(config.latent_walk, "show_neg_log", False)),
                        reference_best_dir=getattr(
                            config.latent_walk, "reference_best_dir", None
                        ),
                    )
                    if sub_summary:
                        per_task_summaries_keyed.append((tkey, sub_summary))
                merged = _merge_walk_summaries(
                    [s for _, s in per_task_summaries_keyed]
                )
                if merged:
                    walk_summary.update(merged)
                _ref_dir = getattr(
                    config.latent_walk, "reference_best_dir", None
                )
                _ref_label = Path(_ref_dir).name if _ref_dir else None
                _augment_summary_with_per_task(
                    walk_summary,
                    per_task_summaries_keyed,
                    epoch=int(epoch),
                    running_best=walk_running_best,
                    running_best_epoch=walk_running_best_epoch,
                    running_best_alpha=walk_running_best_alpha,
                    reference_label=_ref_label,
                )
            if walk_summary:
                summary.update(walk_summary)
            for record_json_path in latent_walk_record_jsons:
                tkey = _task_lookup_key_from_json(record_json_path)
                persistent = persistent_measurement_cache_by_task.setdefault(tkey, {})
                shared = shared_measurement_cache_by_task.get(tkey, {})
                added = _merge_cache_into_lookup(persistent, shared)
                if not added:
                    continue
                lookup_path = measurement_lookup_paths_by_task[tkey]
                _save_measurement_lookup(
                    lookup_path,
                    persistent,
                    cost_target=bundle.cost_target,
                    task_min_cost=task_min_cost_by_task.get(tkey, _lookup_task_min_cost),
                )
                # print(
                #     f"[train] measurement lookup: +{added} new "
                #     f"(total={len(persistent)}) → {lookup_path}"
                # )

        # Snapshot AFTER walk so the latest walk metrics survive into final.
        last_summary = dict(summary)

        current_metric = summary.get(best_metric_name)
        improved = False
        can_early_stop = current_metric is not None
        if current_metric is not None:
            current_metric = float(current_metric)
            if best_metric_mode == "max":
                improved = current_metric > (best_metric_value + early_stop_min_delta)
            else:
                improved = current_metric < (best_metric_value - early_stop_min_delta)
        elif not best_checkpoint_path.exists():
            improved = True

        if warmup_scheduler is not None and epoch <= warmup_epochs:
            warmup_scheduler.step()
        elif scheduler is not None:
            if scheduler_name == "plateau":
                if current_metric is not None:
                    scheduler.step(float(current_metric))
            else:
                scheduler.step()

        print(f"[train] lr={optimizer.param_groups[0]['lr']:.6g}")

        # Update bookkeeping (best_metric_value, epochs_without_improve, best_metric_epoch)
        # BEFORE the checkpoint pickle so wandb.log can fire right after the
        # walk without waiting on disk I/O.
        if improved:
            if current_metric is not None:
                best_metric_value = float(current_metric)
            epochs_without_improve = 0
            best_metric_epoch = int(epoch)
            best_metrics = dict(summary)
            print(
                f"[train] best updated: {best_metric_name}={best_metric_value:.6f} "
                f"@ epoch {best_metric_epoch}"
            )
        elif can_early_stop:
            epochs_without_improve += 1

        if wandb_run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "early_stop/best_metric_value": best_metric_value,
                    "early_stop/epochs_without_improve": epochs_without_improve,
                    **_remap_for_wandb(summary),
                },
                step=epoch,
            )

        checkpoint_kwargs = dict(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_exact_match=best_exact_match,
            best_val_acc=best_val_acc,
            best_metric_name=best_metric_name,
            best_metric_value=best_metric_value,
            best_metric_epoch=best_metric_epoch,
            epochs_without_improve=epochs_without_improve,
            config=config,
            tokenizer=tokenizer,
            latent_cost_ridge=next(
                (p for p in latent_cost_ridges if not bool(p.get("weighted", False))),
                latent_cost_ridges[0] if latent_cost_ridges else None,
            ),
            latent_cost_ridges=latent_cost_ridges,
            timestamp=timestamp,
            wandb_run_id=(wandb_run.id if wandb_run is not None else None),
        )

        if improved:
            save_checkpoint(best_checkpoint_path, **checkpoint_kwargs)
        # ``last.pt`` is saved earlier (pre-walk) for crash safety. The
        # ``best.pt`` / ``epoch_N.pt`` saves below intentionally keep the
        # post-walk state.
        # Per-walk-epoch snapshot. Only triggered on walk-cadence epochs so we
        # don't dump a checkpoint every epoch.
        if walk_due:
            epoch_checkpoint_path = pt_dir / f"epoch_{epoch}.pt"
            save_checkpoint(epoch_checkpoint_path, **checkpoint_kwargs)

        if can_early_stop and epochs_without_improve >= early_stop_patience:
            print(
                f"[train] early stop at epoch {epoch}: "
                f"no improvement in {best_metric_name} for {epochs_without_improve} epochs"
            )
            break

    # Run a post-training walk *before* loading best.pt so the recorded
    # metrics reflect the actual last-epoch model. The block below is then
    # repeated after best.pt is loaded; the two calls are tagged with
    # disjoint key prefixes (``last_`` and ``best_``) so the wandb panels
    # don't collide.
    def _do_post_training_walk(
        *,
        label_prefix: str,
        ckpt_path: Path,
        ridges,
    ) -> Dict[str, float]:
        """Run the full post-training walk pipeline once. ``label_prefix`` is
        prepended to ``epoch_label`` for log lines and to every emitted key.
        """
        if not (latent_walk_on_final and latent_walk_record_jsons):
            return {}
        recon_predictor_local = None
        if walk_recon_predict_enabled and walk_recon_predict_use_gp:
            recon_predictor_local = fit_gp_recon_predictor(
                model=model,
                dataset=bundle.train_dataset,
                tokenizer=tokenizer,
                device=device,
                top_k=walk_recon_predict_gp_top_k,
                random_n=walk_recon_predict_gp_random_n,
                batch_size=config.eval.batch_size,
                seed=int(getattr(config.data, "seed", 0)),
                walk_buffer=walk_sample_buffer,
            )
        ridge_walks_local = list(_iter_walk_ridges(ridges, config)) or [(None, "", False)]
        cache_by_task_local: Dict[str, dict] = {}
        for record_json_path in latent_walk_record_jsons:
            tkey = _task_lookup_key_from_json(record_json_path)
            cache_by_task_local[tkey] = _seed_measurement_cache_from_buffer(
                walk_sample_buffer,
                disk_cache=persistent_measurement_cache_by_task.get(tkey),
                cost_target=bundle.cost_target,
                task_min_cost=task_min_cost_by_task.get(tkey, _lookup_task_min_cost),
                workload_key_filter=workload_key_by_task.get(tkey),
            )
        reencode_predictor_name_local = str(
            getattr(config.train, "re_encode_predictor", "cost_head")
        )
        reencode_predictor_local = _build_reencode_predictor(
            name=reencode_predictor_name_local,
            model=model,
            bundle=bundle,
            tokenizer=tokenizer,
            device=device,
            latent_cost_ridges=ridges,
            config=config,
        )
        walk_summary_local: Dict[str, float] = {}
        for walk_ridge, walk_prefix, force_cost_head in ridge_walks_local:
            keyed_summaries: List[tuple[str, Dict[str, float]]] = []
            for record_json_path in latent_walk_record_jsons:
                tkey = _task_lookup_key_from_json(record_json_path)
                task_label = Path(record_json_path).stem
                label = (
                    label_prefix
                    + (f" [{walk_prefix.rstrip('_')}]" if walk_prefix else "")
                    + (f" task={task_label}" if len(latent_walk_record_jsons) > 1 else "")
                )
                per_task_min_cost = task_min_cost_by_task.get(
                    tkey, _lookup_task_min_cost
                )
                sub_summary = _run_periodic_latent_walk(
                    model=model,
                    device=device,
                    checkpoint_path=ckpt_path,
                    record_json_path=record_json_path,
                    walk_output_dir=latent_walk_output_dir,
                    network_info_folder=latent_walk_network_info,
                    epoch_label=label,
                    config=config,
                    registry=registry,
                    tokenizer=tokenizer,
                    latent_cost_ridge=walk_ridge,
                    timestamp=timestamp,
                    top_k=latent_walk_top_k,
                    num_steps=latent_walk_num_steps,
                    step_size=latent_walk_step_size,
                    use_latent_gradient=force_cost_head or latent_walk_use_latent_gradient,
                    include_recon_predict=walk_recon_predict_enabled,
                    recon_predictor=recon_predictor_local,
                    reencode_predictor=reencode_predictor_local,
                    reencode_predictor_name=reencode_predictor_name_local,
                    walk_buffer=walk_sample_buffer,
                    walk_key_prefix=walk_prefix,
                    measurement_cache=cache_by_task_local[tkey],
                    sampling_options=latent_walk_sampling_options,
                    cost_target=bundle.cost_target,
                    task_min_cost=per_task_min_cost,
                    sort_by=str(getattr(config.latent_walk, "sort_by", "re_pred")),
                    show_neg_log=bool(getattr(config.latent_walk, "show_neg_log", False)),
                    reference_best_dir=getattr(
                        config.latent_walk, "reference_best_dir", None
                    ),
                )
                if sub_summary:
                    keyed_summaries.append((tkey, sub_summary))
            merged = _merge_walk_summaries([s for _, s in keyed_summaries])
            if merged:
                walk_summary_local.update(merged)
            _ref_dir_final = getattr(
                config.latent_walk, "reference_best_dir", None
            )
            _ref_label_final = (
                Path(_ref_dir_final).name if _ref_dir_final else None
            )
            _augment_summary_with_per_task(
                walk_summary_local,
                keyed_summaries,
                epoch=int(config.train.num_epochs),
                running_best=walk_running_best,
                running_best_epoch=walk_running_best_epoch,
                running_best_alpha=walk_running_best_alpha,
                reference_label=_ref_label_final,
            )
        for record_json_path in latent_walk_record_jsons:
            tkey = _task_lookup_key_from_json(record_json_path)
            persistent = persistent_measurement_cache_by_task.setdefault(tkey, {})
            shared = cache_by_task_local.get(tkey, {})
            added = _merge_cache_into_lookup(persistent, shared)
            if not added:
                continue
            lookup_path = measurement_lookup_paths_by_task[tkey]
            _save_measurement_lookup(
                lookup_path,
                persistent,
                cost_target=bundle.cost_target,
                task_min_cost=task_min_cost_by_task.get(tkey, _lookup_task_min_cost),
            )
            # print(
            #     f"[train] measurement lookup: +{added} new "
            #     f"(total={len(persistent)}) → {lookup_path}"
            # )
        return walk_summary_local

    # Post-training walk runs on the model's current (last-epoch) state.
    # ``best.pt`` is kept on disk as an archival snapshot only — we don't
    # reload it for evaluation here.
    final_walk_summary = _do_post_training_walk(
        label_prefix="final",
        ckpt_path=last_checkpoint_path,
        ridges=latent_cost_ridges,
    )

    final_metrics = dict(last_summary) if last_summary else {}
    final_metrics.update(
        _evaluate_final_checkpoint(
            model,
            bundle,
            registry,
            tokenizer,
            config,
            device,
            latent_cost_ridges,
        )
    )
    if final_walk_summary:
        final_metrics.update(
            {f"final_{k}": v for k, v in final_walk_summary.items()}
        )

    print("[final]", json.dumps(final_metrics, indent=2))
    print("[train] checkpoint dir:", pt_dir)

    if bundle.val_dataset.samples:
        sample = bundle.val_dataset.samples[0]
        decoded = greedy_decode_sample(model, sample, registry, tokenizer, device)
        print(pretty_print_reconstruction(sample, decoded))
    elif bundle.test_dataset.samples:
        sample = bundle.test_dataset.samples[0]
        decoded = greedy_decode_sample(model, sample, registry, tokenizer, device)
        print(pretty_print_reconstruction(sample, decoded))

    if wandb_run is not None:
        final_log_metrics = {
            key: value
            for key, value in final_metrics.items()
            if isinstance(value, (int, float))
        }
        if final_log_metrics:
            wandb.log(_remap_for_wandb(final_log_metrics), step=epoch)
        wandb_run.summary.update(_remap_for_wandb(final_metrics))
        wandb_run.finish()

    return final_metrics
