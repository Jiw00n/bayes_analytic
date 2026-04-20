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

import dataclasses
import math

from .adapter import GeneratorRegistry, JsonSampleRecord, load_json_samples
from .dataset import (
    DatasetBundle,
    _build_prepared_sample,
    _get_generator_for_record,
    budget_enabled,
    build_dataset_bundle,
    get_model_param_order,
)
from .inference import greedy_decode_sample, pretty_print_reconstruction
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
)
from .tokenizer import ParamTokenizer
from .train_epoch import train_one_epoch
from .train_eval import (
    _alpha_metric_suffix,
    _build_named_latent_cost_ridges,
    _build_reencode_predictor,
    _resolve_ridge_alphas,
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


def _resolve_walk_record_json(config) -> Optional[str]:
    explicit = getattr(config.train, "latent_walk_record_json", None)
    if explicit:
        return str(explicit)
    json_paths = list(getattr(config.data, "json_paths", []) or [])
    return str(json_paths[0]) if json_paths else None


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
    true_cost_at_alpha0: Optional[float] = None
    for record in records:
        meas = record.measurement or {}
        if meas.get("ok") and meas.get("usable_measurement"):
            mean_cost = meas.get("mean_cost")
            if mean_cost is not None:
                entry = (float(record.alpha), float(mean_cost))
                if abs(entry[0]) >= 1e-12 and not _is_reference(record.params):
                    measured_novel.append(entry)
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
    if true_cost_at_alpha0 is not None:
        summary["walk/true_cost_at_alpha0"] = true_cost_at_alpha0
    return summary


def _merge_walk_summaries(summaries: list[Dict[str, float]]) -> Dict[str, float]:
    merged: Dict[str, float] = {}
    if not summaries:
        return merged

    num_records = 0
    total_steps = 0.0
    total_unique_sym = 0.0
    pred_weighted_sum = 0.0
    pred_weight = 0.0
    best_pred: Optional[float] = None
    best_measured: Optional[float] = None
    best_measured_alpha: Optional[float] = None
    best_true_alpha0: Optional[float] = None
    recon_weighted_sum = 0.0
    recon_weight = 0.0
    best_recon: Optional[float] = None
    best_recon_std: Optional[float] = None

    for summary in summaries:
        if not summary:
            continue
        num_records += 1
        n_steps = float(summary.get("walk/num_steps", 0.0))
        total_steps += n_steps
        total_unique_sym += float(summary.get("walk/num_unique_sym_map", 0.0))
        mean_pred = summary.get("walk/mean_predicted_cost")
        if mean_pred is not None and n_steps > 0:
            pred_weighted_sum += float(mean_pred) * n_steps
            pred_weight += n_steps
        best_predicted = summary.get("walk/best_predicted_cost")
        if best_predicted is not None and (best_pred is None or best_predicted > best_pred):
            best_pred = float(best_predicted)
        best_measured_summary = summary.get("walk/best_measured_mean_cost")
        if best_measured_summary is not None and (
            best_measured is None or best_measured_summary > best_measured
        ):
            best_measured = float(best_measured_summary)
            alpha_at_best = summary.get("walk/alpha_at_best")
            best_measured_alpha = float(alpha_at_best) if alpha_at_best is not None else None
        true_alpha0 = summary.get("walk/true_cost_at_alpha0")
        if true_alpha0 is not None and (
            best_true_alpha0 is None or true_alpha0 > best_true_alpha0
        ):
            best_true_alpha0 = float(true_alpha0)
        mean_recon = summary.get("walk/mean_recon_predict_cost")
        if mean_recon is not None and n_steps > 0:
            recon_weighted_sum += float(mean_recon) * n_steps
            recon_weight += n_steps
        best_recon_summary = summary.get("walk/best_recon_predict_cost")
        if best_recon_summary is not None and (
            best_recon is None or best_recon_summary > best_recon
        ):
            best_recon = float(best_recon_summary)
            best_recon_std_summary = summary.get("walk/best_recon_predict_std")
            best_recon_std = (
                float(best_recon_std_summary)
                if best_recon_std_summary is not None
                else None
            )

    merged["walk/num_records"] = float(num_records)
    merged["walk/num_steps"] = total_steps
    merged["walk/num_unique_sym_map"] = total_unique_sym
    if pred_weight > 0:
        merged["walk/mean_predicted_cost"] = pred_weighted_sum / pred_weight
    if best_pred is not None:
        merged["walk/best_predicted_cost"] = best_pred
    if best_measured is not None:
        merged["walk/best_measured_mean_cost"] = best_measured
        if best_measured_alpha is not None:
            merged["walk/alpha_at_best"] = best_measured_alpha
    if best_true_alpha0 is not None:
        merged["walk/true_cost_at_alpha0"] = best_true_alpha0
    if recon_weight > 0:
        merged["walk/mean_recon_predict_cost"] = recon_weighted_sum / recon_weight
    if best_recon is not None:
        merged["walk/best_recon_predict_cost"] = best_recon
    if best_recon_std is not None:
        merged["walk/best_recon_predict_std"] = best_recon_std
    return merged


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

        sym_key = make_sym_map_key(
            {str(k): int(v) for k, v in record.sym_map.items() if isinstance(v, int)}
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
) -> Dict[str, float]:
    here = Path(__file__).resolve().parent.parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    try:
        from tune_by_latent import make_bundle, run_latent_walk
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
                ) or []
            except Exception as err:  # pragma: no cover
                print(f"[train] latent walk rank={rank} failed: {type(err).__name__}: {err}")
                walk_records = []
            per_rank_summaries.append(
                _summarize_walk_records(walk_records, reference_params=reference_params)
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
    if walk_key_prefix:
        target = f"{walk_key_prefix}walk/"
        combined = {
            key.replace("walk/", target, 1) if "walk/" in key else key: value
            for key, value in combined.items()
        }
    return combined


def build_everything(config):
    print(f"[build] loading registry from {config.data.network_info_folder}")
    registry = GeneratorRegistry(config.data.network_info_folder)
    print("[build] building dataset bundle")
    bundle = build_dataset_bundle(config, registry)
    tokenizer = bundle.tokenizer
    print(
        f"[build] tokenizer ready: vocab={len(tokenizer.id_to_token)} "
        f"vars={len(tokenizer.id_to_var)}"
    )
    print("[build] constructing model")
    model = LatentParamVAE(
        vocab_size=len(tokenizer.id_to_token),
        num_vars=len(tokenizer.id_to_var),
        cfg=config.model,
    )
    return registry, bundle, tokenizer, model


def _resolve_run_task_index(bundle: DatasetBundle) -> str:
    import re

    all_records = (
        list(bundle.train_records) + list(bundle.val_records) + list(bundle.test_records)
    )
    for record in all_records:
        if record.task_index is not None:
            return str(int(record.task_index))
    # Fall back: leading digits of the record JSON filename (e.g.
    # "1490_([...],cuda).json" → "1490"). Measure-record JSONs don't carry a
    # task_index field, so this keeps wandb project names meaningful.
    for record in all_records:
        json_path = getattr(record, "json_path", None)
        if not json_path:
            continue
        m = re.match(r"^(\d+)", Path(json_path).stem)
        if m:
            return m.group(1)
    return "na"


def _build_wandb_project_name(config, bundle: DatasetBundle) -> str:
    task_index = _resolve_run_task_index(bundle)
    project_suffix = getattr(config.wandb, "project", None) or "single_v1"
    return f"Task{task_index}_{project_suffix}"


def _build_wandb_run_name(config, bundle: DatasetBundle) -> str:
    name = ""

    if config.model.num_encoder_layers != 3:
        name += f"_enc{config.model.num_encoder_layers}"
    if config.model.num_decoder_layers != 3:
        name += f"_dec{config.model.num_decoder_layers}"
    if config.model.nhead != 4:
        name += f"_head{config.model.nhead}"
    
    if config.model.latent_dim != 64:
        name += f"_zdim{config.model.latent_dim}"
    if config.model.dim_feedforward != 192:
        name += f"_fdim{config.model.dim_feedforward}"
    if config.model.cost_hidden_dim != 128:
        name += f"_cdim{config.model.cost_hidden_dim}"
    if config.model.latent_token_count != 4:
        name += f"_ztok{config.model.latent_token_count}"

    if config.train.num_epochs != 100:
        name += f"_ep{config.train.num_epochs}"
    name = (
        f"_lr{config.train.learning_rate}"
        f"_nce{config.train.lambda_nce}"
        f"_t{config.train.tau_nce}"
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
    if bool(getattr(config.train, "latent_walk_use_cost_head", False)):
        name += "_ch"
    if bool(getattr(config.train, "use_compressed_teacher_forcing", False)):
        name += "_comp"
    ls = float(getattr(config.train, "label_smoothing", 0.0))
    if ls > 0.0:
        name += f"_ls{ls}"
    if bool(getattr(config.train, "order_nce_pos_weight_by_percentile", False)):
        name += f"_pos{float(config.train.order_nce_pos_weight_sigma):.1f}"
    return name


def _seed_measurement_cache_from_buffer(
    walk_buffer: Optional[WalkSampleBuffer],
) -> dict:
    """Pre-populate the measurement cache with prior measurements so that
    sym_maps already measured in earlier epochs skip re-measurement."""
    cache: dict = {}
    if walk_buffer is None:
        return cache
    for sym_key, sample in walk_buffer.items():
        cost = getattr(sample, "cost", None)
        if cost is None or not math.isfinite(float(cost)):
            continue
        cache[sym_key] = {
            "ok": True,
            "usable_measurement": True,
            "mean_cost": float(cost),
            "from_walk_buffer": True,
        }
    return cache


def _iter_walk_ridges(latent_cost_ridges, config):
    """Yield (ridge_payload, walk_key_prefix, use_cost_head_gradient) for each
    walk to run. Order: cost_head (if enabled) → cost_vec → cost_vec_weighted."""
    use_cost_head = bool(getattr(config.train, "latent_walk_use_cost_head", False))
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


def _fit_epoch_ridges(model, bundle, tokenizer, config, device):
    if not bool(getattr(config.train, "cost_ridge_vec", False)):
        return [], None, {}

    ridge_alphas = _resolve_ridge_alphas(config)
    latent_cost_ridges = fit_latent_cost_ridges(
        model,
        bundle.train_dataset,
        tokenizer,
        device,
        alphas=ridge_alphas,
        batch_size=config.eval.batch_size,
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
            bundle.train_dataset,
            tokenizer,
            device,
            alphas=ridge_alphas,
            batch_size=config.eval.batch_size,
            sample_weight_quantile=float(getattr(config.train, "weight_quantile", 0.85)),
            sample_weight_sigma=float(getattr(config.train, "weight_sigma", 0.25)),
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


def _evaluate_validation_epoch(model, bundle, registry, tokenizer, config, device, epoch, latent_cost_ridges):
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


def train_main(config) -> Dict[str, float]:
    seed_everything(config.data.seed)
    device = resolve_device(config.train.device)
    configure_runtime(config, device)
    print(f"[train] resolved device: requested={config.train.device} actual={device}")
    print(
        f"[train] runtime config: amp={bool(config.train.use_amp)} "
        f"tf32={bool(getattr(config.train, 'allow_tf32', True))}"
    )

    registry, bundle, tokenizer, model = build_everything(config)
    model.to(device)
    print(f"[train] model moved to {device}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] model/total_params: {total_params:,}")

    wandb_run = None
    wandb_project = getattr(config.wandb, "project", None)
    run_name = _build_wandb_run_name(config, bundle)
    if wandb_project:
        if wandb is None:
            print("[train] wandb project is set but wandb is not installed; skipping wandb logging")
        else:
            project_name = _build_wandb_project_name(config, bundle)
            print(f"[train] initializing wandb: project={project_name} run={run_name}")
            wandb_run = wandb.init(
                project=project_name,
                name=run_name,
                config=config.to_dict(),
            )
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

    checkpoint_dir = save_training_artifacts(config.train.checkpoint_dir, config, tokenizer)
    print(f"[train] checkpoint dir: {checkpoint_dir}")

    start_epoch = 1
    best_exact_match = float("-inf")
    best_val_acc = float("-inf")
    best_checkpoint_path = checkpoint_dir / "best.pt"
    last_checkpoint_path = checkpoint_dir / "last.pt"
    checkpoint_path = checkpoint_dir / f"{run_name}.pt"
    if checkpoint_path.exists():
        timestamp = time.strftime("%m%d%H%M")
        checkpoint_path = checkpoint_dir / f"{run_name}_{timestamp}.pt"
        print(f"[train] checkpoint already exists; using timestamped path: {checkpoint_path}")
    if best_metric_mode == "max":
        best_metric_value = float("-inf")
    else:
        best_metric_value = float("inf")
    epochs_without_improve = 0
    best_metric_epoch: Optional[int] = None
    last_summary: Dict[str, float] = {}

    if config.train.resume_from:
        print(f"[train] resuming from {config.train.resume_from}")
        payload = load_checkpoint(config.train.resume_from, model, optimizer, scheduler)
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
    timestamp = time.strftime("%m%d%H%M")
    latent_walk_every_n = int(getattr(config.train, "latent_walk_every_n_epochs", 0) or 0)
    latent_walk_on_final = bool(getattr(config.train, "latent_walk_on_final", False))
    latent_walk_top_k = int(getattr(config.train, "latent_walk_top_k", 1) or 1)
    latent_walk_num_steps = int(getattr(config.train, "latent_walk_num_steps", 8) or 8)
    latent_walk_step_size = float(getattr(config.train, "latent_walk_step_size", 0.25) or 0.25)
    latent_walk_use_latent_gradient = not bool(getattr(config.train, "cost_ridge_vec", False))
    walk_recon_predict_every_n = int(getattr(config.train, "latent_walk_predict_every_n_epochs", 0) or 0)
    walk_recon_predict_enabled = walk_recon_predict_every_n > 0
    walk_recon_predict_use_gp = bool(getattr(config.train, "latent_walk_predict_use_gp", False))
    walk_recon_predict_gp_top_k = int(getattr(config.train, "latent_walk_predict_gp_top_k", 0) or 0)
    walk_recon_predict_gp_random_n = int(getattr(config.train, "latent_walk_predict_gp_random_n", 0) or 0)
    latent_walk_record_json = _resolve_walk_record_json(config)
    latent_walk_output_dir = (
        getattr(config.train, "latent_walk_output_dir", None)
        or str(checkpoint_dir)
    )
    latent_walk_network_info = getattr(config.data, "network_info_folder", None)
    if (latent_walk_every_n > 0 or latent_walk_on_final) and not latent_walk_record_json:
        print(
            "[train] latent walk requested but no record JSON resolvable from "
            "config.train.latent_walk_record_json or config.data.json_paths; disabling"
        )
        latent_walk_every_n = 0
        latent_walk_on_final = False

    for epoch in range(start_epoch, config.train.num_epochs + 1):
        print(f"[train] starting epoch {epoch}/{config.train.num_epochs}")
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
        )
        # print(f"[train] evaluating train split with offline teacher forcing after epoch {epoch}")
        train_eval_metrics = evaluate_teacher_forcing(
            model,
            bundle.train_dataset,
            registry,
            tokenizer,
            device,
            batch_size=config.eval.batch_size,
            use_compressed=bool(getattr(config.train, "use_compressed_teacher_forcing", False)),
        )

        summary = {
            **train_metrics,
            "token_accuracy": float(train_eval_metrics["token_accuracy"]),
            "full_sequence_exact_match": float(train_eval_metrics["full_sequence_exact_match"]),
        }
        print(
            f"[epoch {epoch}] "
            f"loss={summary['loss']:.4f} recon={summary['recon_loss']:.4f} "
            f"kl={summary['kl_loss']:.4f} "
            f"tok_acc={summary['token_accuracy']:.4f} "
            f"exact={summary['full_sequence_exact_match']:.4f}"
        )

        latent_cost_ridges, ridge_metrics = _fit_epoch_ridges(
            model,
            bundle,
            tokenizer,
            config,
            device,
        )
        summary.update(ridge_metrics)

        train_cost_metrics = evaluate_cost_ranking(
            model,
            bundle.train_dataset,
            tokenizer,
            device,
            batch_size=config.eval.batch_size,
            latent_cost_ridges=_build_named_latent_cost_ridges(latent_cost_ridges),
        )
        summary.update({f"train_{k}": float(v) for k, v in train_cost_metrics.items()})

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
            )
        )

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
            shared_measurement_cache: dict = _seed_measurement_cache_from_buffer(
                walk_sample_buffer
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
                label = f"epoch {epoch}" + (f" [{walk_prefix.rstrip('_')}]" if walk_prefix else "")
                sub_summary = _run_periodic_latent_walk(
                    model=model,
                    device=device,
                    checkpoint_path=last_checkpoint_path,
                    record_json_path=latent_walk_record_json,
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
                    include_recon_predict=include_recon_predict,
                    include_measurement=include_measurement,
                    recon_predictor=walk_recon_predictor,
                    reencode_predictor=reencode_predictor,
                    reencode_predictor_name=reencode_predictor_name,
                    walk_buffer=walk_sample_buffer if include_measurement else None,
                    walk_key_prefix=walk_prefix,
                    measurement_cache=shared_measurement_cache,
                )
                if sub_summary:
                    walk_summary.update(sub_summary)
            if walk_summary:
                summary.update(walk_summary)

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
        )

        if improved:
            if current_metric is not None:
                best_metric_value = float(current_metric)
            epochs_without_improve = 0
            best_metric_epoch = int(epoch)
            best_metrics = dict(summary)
            checkpoint_kwargs["best_metric_value"] = best_metric_value
            checkpoint_kwargs["epochs_without_improve"] = epochs_without_improve
            checkpoint_kwargs["best_metric_epoch"] = best_metric_epoch
            save_checkpoint(best_checkpoint_path, **checkpoint_kwargs)
            save_checkpoint(checkpoint_path, **checkpoint_kwargs)
            print(
                f"[train] best updated: {best_metric_name}={best_metric_value:.6f} "
                f"@ epoch {best_metric_epoch}"
            )
        else:
            if can_early_stop:
                epochs_without_improve += 1
            checkpoint_kwargs["best_metric_value"] = best_metric_value
            checkpoint_kwargs["epochs_without_improve"] = epochs_without_improve

        save_checkpoint(last_checkpoint_path, **checkpoint_kwargs)

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

        if can_early_stop and epochs_without_improve >= early_stop_patience:
            print(
                f"[train] early stop at epoch {epoch}: "
                f"no improvement in {best_metric_name} for {epochs_without_improve} epochs"
            )
            break

    if best_checkpoint_path.exists():
        best_payload = torch.load(best_checkpoint_path, map_location="cpu")
        model.load_state_dict(best_payload["model_state"])
        latent_cost_ridges = list(best_payload.get("latent_cost_ridges") or [])
        model.to(device)
        print(f"[train] reloaded best checkpoint from {best_checkpoint_path}")

    # Start from best_metrics (snapshot at the best epoch) but overlay the
    # latest epoch's walk-related metrics so the very last walk (e.g. epoch
    # num_epochs) is reflected in the final report instead of being shadowed
    # by the walk recorded during the best epoch.
    final_metrics = dict(best_metrics if best_metrics else last_summary)
    if best_metrics and last_summary:
        for key, value in last_summary.items():
            if key.startswith("walk/") or "walk/" in key:
                final_metrics[key] = value
    if best_metric_epoch is not None:
        final_metrics["walk/best_epoch"] = float(best_metric_epoch)
    final_metrics["walk/best_cost"] = float(best_metric_value)
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
    print("[final]", json.dumps(final_metrics, indent=2))
    print("[train] checkpoint :", checkpoint_path)

    if latent_walk_on_final and latent_walk_record_json:
        final_walk_checkpoint = (
            best_checkpoint_path if best_checkpoint_path.exists() else last_checkpoint_path
        )
        final_walk_recon_predictor = None
        if walk_recon_predict_enabled and walk_recon_predict_use_gp:
            final_walk_recon_predictor = fit_gp_recon_predictor(
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
        final_walk_summary: Dict[str, float] = {}
        final_ridge_walks = list(_iter_walk_ridges(latent_cost_ridges, config)) or [(None, "", False)]
        final_measurement_cache: dict = _seed_measurement_cache_from_buffer(
            walk_sample_buffer
        )
        final_reencode_predictor_name = str(
            getattr(config.train, "re_encode_predictor", "cost_head")
        )
        final_reencode_predictor = _build_reencode_predictor(
            name=final_reencode_predictor_name,
            model=model,
            bundle=bundle,
            tokenizer=tokenizer,
            device=device,
            latent_cost_ridges=latent_cost_ridges,
            config=config,
        )
        for walk_ridge, walk_prefix, force_cost_head in final_ridge_walks:
            label = "final" + (f" [{walk_prefix.rstrip('_')}]" if walk_prefix else "")
            sub_summary = _run_periodic_latent_walk(
                model=model,
                device=device,
                checkpoint_path=final_walk_checkpoint,
                record_json_path=latent_walk_record_json,
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
                recon_predictor=final_walk_recon_predictor,
                reencode_predictor=final_reencode_predictor,
                reencode_predictor_name=final_reencode_predictor_name,
                walk_buffer=walk_sample_buffer,
                walk_key_prefix=walk_prefix,
                measurement_cache=final_measurement_cache,
            )
            if sub_summary:
                final_walk_summary.update(sub_summary)
        if final_walk_summary:
            final_metrics.update(
                {f"final_{k}": v for k, v in final_walk_summary.items()}
            )

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
