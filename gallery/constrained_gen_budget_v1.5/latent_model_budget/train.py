from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
import time
from typing import Dict, Optional

import torch

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

from .adapter import GeneratorRegistry
from .dataset import DatasetBundle, build_dataset_bundle
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
from .tokenizer import ParamTokenizer
from .train_epoch import train_one_epoch
from .train_eval import (
    _alpha_metric_suffix,
    _build_named_latent_cost_ridges,
    _resolve_ridge_alphas,
    evaluate_autoregressive,
    evaluate_cost_ranking,
    evaluate_teacher_forcing,
    fit_latent_cost_ridges,
)


def _wandb_section(key: str) -> str:
    """Group metric keys into wandb sections (train/, val/, walk/, ...).

    wandb groups panels on the Charts page by the prefix before the first '/'.
    Training-phase metrics like ``loss``/``recon_loss`` come from
    ``train_one_epoch`` without any prefix, so we route them to ``train/``.
    """
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
    """Pick the record JSON used for the periodic latent walk. Fall back to the
    first data.json_paths entry when the dedicated field is unset."""
    explicit = getattr(config.train, "latent_walk_record_json", None)
    if explicit:
        return str(explicit)
    json_paths = list(getattr(config.data, "json_paths", []) or [])
    return str(json_paths[0]) if json_paths else None


def _summarize_walk_records(
    records,
    *,
    reference_params: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """Reduce a list of WalkRecord into scalar metrics for wandb logging.

    ``mean_cost`` and ``predicted_score`` are both ``-log(latency)``, so higher
    is better (faster kernel). ``best_measured_mean_cost`` excludes the
    trivial alpha=0 step and any step whose decoded params match
    ``reference_params`` (walk did not move away from the starting point).
    """
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

    measured_all: list[tuple[float, float]] = []
    measured_novel: list[tuple[float, float]] = []
    true_cost_at_alpha0: Optional[float] = None
    for r in records:
        meas = r.measurement or {}
        if meas.get("ok") and meas.get("usable_measurement"):
            mc = meas.get("mean_cost")
            if mc is not None:
                entry = (float(r.alpha), float(mc))
                measured_all.append(entry)
                if abs(entry[0]) >= 1e-12 and not _is_reference(r.params):
                    measured_novel.append(entry)
        if abs(float(r.alpha)) < 1e-12:
            true_mc = meas.get("true_mean_cost") if meas else None
            if true_mc is not None:
                true_cost_at_alpha0 = float(true_mc)

    summary["walk/num_usable"] = float(len(measured_all))
    summary["walk/num_novel"] = float(len(measured_novel))
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


def _merge_walk_summaries(summaries: list) -> Dict[str, float]:
    """Aggregate per-rank walk summaries into a single dict for wandb.

    - ``best_*`` costs take the max across ranks (higher ``-log(latency)`` = faster).
    - ``alpha_at_best`` is carried from whichever rank produced the overall best
      ``best_measured_mean_cost``.
    - Count metrics sum across ranks.
    - ``mean_predicted_cost`` is re-averaged by ``num_steps`` so the aggregate
      reflects all steps equally regardless of rank.
    """
    merged: Dict[str, float] = {}
    if not summaries:
        return merged

    num_records = 0
    total_steps = 0.0
    total_usable = 0.0
    total_novel = 0.0
    total_unique_sym = 0.0
    pred_weighted_sum = 0.0
    pred_weight = 0.0
    best_pred: Optional[float] = None
    best_measured: Optional[float] = None
    best_measured_alpha: Optional[float] = None
    best_true_alpha0: Optional[float] = None

    for s in summaries:
        if not s:
            continue
        num_records += 1
        n_steps = float(s.get("walk/num_steps", 0.0))
        total_steps += n_steps
        total_usable += float(s.get("walk/num_usable", 0.0))
        total_novel += float(s.get("walk/num_novel", 0.0))
        total_unique_sym += float(s.get("walk/num_unique_sym_map", 0.0))
        mp = s.get("walk/mean_predicted_cost")
        if mp is not None and n_steps > 0:
            pred_weighted_sum += float(mp) * n_steps
            pred_weight += n_steps
        bp = s.get("walk/best_predicted_cost")
        if bp is not None and (best_pred is None or bp > best_pred):
            best_pred = float(bp)
        bm = s.get("walk/best_measured_mean_cost")
        if bm is not None and (best_measured is None or bm > best_measured):
            best_measured = float(bm)
            ab = s.get("walk/alpha_at_best")
            best_measured_alpha = float(ab) if ab is not None else None
        ta0 = s.get("walk/true_cost_at_alpha0")
        if ta0 is not None and (best_true_alpha0 is None or ta0 > best_true_alpha0):
            best_true_alpha0 = float(ta0)

    merged["walk/num_records"] = float(num_records)
    merged["walk/num_steps"] = total_steps
    merged["walk/num_usable"] = total_usable
    merged["walk/num_novel"] = total_novel
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
    return merged


def _run_periodic_latent_walk(
    *,
    model,
    device,
    checkpoint_path,
    record_json_path: str,
    walk_output_dir: str,
    network_info_folder: Optional[str],
    epoch_label: str,
    top_k: int = 1,
    num_steps: int = 8,
    step_size: float = 0.25,
) -> Dict[str, float]:
    """Run the tune.sh-equivalent latent walk against ``checkpoint_path``.

    Training model is moved to CPU for the duration of the walk so the GPU is
    free both for the walk's own forward passes and for the TVM measurement
    subprocess, which needs exclusive GPU access for accurate timings. When
    ``top_k > 1`` the walk is repeated per top-k reference record and the
    per-rank summaries are merged. Returns summary metrics for wandb logging.
    """
    here = Path(__file__).resolve().parent.parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    try:
        from tune_by_latent import run_latent_walk, _select_topk_records_from_path
    except Exception as err:  # pragma: no cover
        print(f"[train] latent walk unavailable: {type(err).__name__}: {err}")
        return {}

    try:
        ref_records = _select_topk_records_from_path(record_json_path, k=max(1, int(top_k)))
    except Exception as err:  # pragma: no cover
        print(f"[train] could not resolve reference records: {err}")
        return {}
    if not ref_records:
        return {}

    print(
        f"[train] latent walk ({epoch_label}) using checkpoint={checkpoint_path} "
        f"top_k={len(ref_records)}"
    )
    model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    walk_device = "cuda" if (device.type == "cuda" and torch.cuda.is_available()) else "cpu"
    per_rank_summaries: list = []
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
                    device=walk_device,
                    output=str(rank_output_dir),
                    num_steps=int(num_steps),
                    step_size=float(step_size),
                    deterministic_start=True,
                    preselected_record=ref_record,
                ) or []
            except Exception as err:  # pragma: no cover
                print(f"[train] latent walk rank={rank} failed: {type(err).__name__}: {err}")
                walk_records = []
            per_rank_summaries.append(
                _summarize_walk_records(walk_records, reference_params=reference_params)
            )
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        model.to(device)

    return _merge_walk_summaries(per_rank_summaries)


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
    name = (
        f"lr{config.train.learning_rate}"
        f"_nce{config.train.lambda_nce}"
        f"_tau{config.train.tau_nce}"
        f"_kl{config.train.beta_end}"
        f"_warm{config.train.beta_warmup_epochs}"
    )
    if bool(config.train.order_nce):
        name += "_order"
    if bool(getattr(config.train, "nce_mu", False)):
        name += "_nce_mu"
    if bool(config.model.adaln):
        name += "_adaln"
    return f"{name}"


def _fit_epoch_ridges(model, bundle, tokenizer, config, device):
    if not bool(getattr(config.train, "cost_ridge_vec", False)):
        return [], None, {}

    ridge_alphas = _resolve_ridge_alphas(config)
    print(f"[train] fitting latent cost ridge on train split for alphas={ridge_alphas}")
    latent_cost_ridges = fit_latent_cost_ridges(
        model,
        bundle.train_dataset,
        tokenizer,
        device,
        alphas=ridge_alphas,
        batch_size=config.eval.batch_size,
    )
    latent_cost_ridge = latent_cost_ridges[0] if latent_cost_ridges else None
    ridge_metrics = {}
    for ridge_payload in latent_cost_ridges:
        alpha = float(ridge_payload["alpha"])
        alpha_suffix = _alpha_metric_suffix(alpha)
        print(
            f"[train] latent cost ridge ready: samples={ridge_payload['num_samples']} "
            f"alpha={alpha:.2e} train_mse={ridge_payload['train_mse']:.6f}"
        )
        if alpha == float(ridge_alphas[0]):
            ridge_metrics["train_ridge_mse"] = float(ridge_payload["train_mse"])
        ridge_metrics[f"train_ridge_alpha_{alpha_suffix}_mse"] = float(ridge_payload["train_mse"])
    return latent_cost_ridges, latent_cost_ridge, ridge_metrics


def _evaluate_validation_epoch(model, bundle, registry, tokenizer, config, device, epoch, latent_cost_ridges):
    summary: Dict[str, float] = {}
    if not bundle.val_dataset.samples:
        return summary

    print(f"[train] evaluating validation split with teacher forcing after epoch {epoch}")
    val_tf_metrics = evaluate_teacher_forcing(
        model,
        bundle.val_dataset,
        registry,
        tokenizer,
        device,
        batch_size=config.eval.batch_size,
    )
    summary.update({f"val_{k}": float(v) for k, v in val_tf_metrics.items()})
    print(
        f"val_tok_acc={summary['val_token_accuracy']:.4f} "
        f"val_exact={summary['val_full_sequence_exact_match']:.4f}"
    )

    print("[train] evaluating validation cost ranking")
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
        )
    if "cost_vec_actual_top1_pred_rank" in cost_metrics:
        print(
            f"val_cost_vec_actual_top1_pred_rank : {int(cost_metrics['cost_vec_actual_top1_pred_rank'])}\n"
            f"val_cost_vec_pred_top1_actual_cost : {cost_metrics['cost_vec_pred_top1_actual_cost']:.6f}\n"
            f"val_cost_vec_pred_top10_mean_actual_cost : {cost_metrics['cost_vec_pred_top10_mean_actual_cost']:.6f}\n"
        )
    for key, value in sorted(cost_metrics.items()):
        if key.startswith("cost_vec_alpha_") and key.endswith("_actual_top1_pred_rank"):
            prefix = key[: -len("_actual_top1_pred_rank")]
            top1_cost_key = f"{prefix}_pred_top1_actual_cost"
            top10_key = f"{prefix}_pred_top10_mean_actual_cost"
            print(
                f"{'val_' + key} : {int(value)}\n"
                f"{'val_' + top1_cost_key} : {cost_metrics[top1_cost_key]:.6f}\n"
                f"{'val_' + top10_key} : {cost_metrics[top10_key]:.6f}\n"
            )

    return summary


def _evaluate_final_checkpoint(model, bundle, registry, tokenizer, config, device, latent_cost_ridges):
    summary: Dict[str, float] = {}

    if bundle.val_dataset.samples:
        print("[train] evaluating best checkpoint on val split")
        val_tf_metrics = evaluate_teacher_forcing(
            model,
            bundle.val_dataset,
            registry,
            tokenizer,
            device,
            batch_size=config.eval.batch_size,
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

        val_ar_metrics = evaluate_autoregressive(
            model,
            bundle.val_dataset,
            registry,
            tokenizer,
            device,
            batch_size=config.eval.batch_size,
        )
        summary.update({f"val_autoregressive_{k}": float(v) for k, v in val_ar_metrics.items()})

    if bundle.test_dataset.samples:
        print("[train] evaluating best checkpoint on test split")
        test_tf_metrics = evaluate_teacher_forcing(
            model,
            bundle.test_dataset,
            registry,
            tokenizer,
            device,
            batch_size=config.eval.batch_size,
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

        test_ar_metrics = evaluate_autoregressive(
            model,
            bundle.test_dataset,
            registry,
            tokenizer,
            device,
            batch_size=config.eval.batch_size,
        )
        summary.update({f"eval_test_autoregressive_{k}": float(v) for k, v in test_ar_metrics.items()})

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
    else:
        scheduler = None
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.train.use_amp and device.type == "cuda"))

    checkpoint_dir = save_training_artifacts(config.train.checkpoint_dir, config, tokenizer)
    print(f"[train] checkpoint dir: {checkpoint_dir}")

    start_epoch = 1
    best_exact_match = float("-inf")
    best_val_acc = float("-inf")
    best_checkpoint_path = checkpoint_dir / "best.pt"
    last_checkpoint_path = checkpoint_dir / "last.pt"
    checkpoint_path = checkpoint_dir / f"{run_name}.pt"
    if best_metric_mode == "max":
        best_metric_value = float("-inf")
    else:
        best_metric_value = float("inf")
    epochs_without_improve = 0
    last_summary: Dict[str, float] = {}

    if config.train.resume_from:
        print(f"[train] resuming from {config.train.resume_from}")
        payload = load_checkpoint(config.train.resume_from, model, optimizer, scheduler)
        start_epoch = int(payload["epoch"]) + 1
        best_exact_match = float(payload.get("best_exact_match", best_exact_match))
        best_val_acc = float(payload.get("best_val_acc", best_val_acc))
        best_metric_value = float(payload.get("best_metric_value", best_metric_value))
        epochs_without_improve = int(payload.get("epochs_without_improve", epochs_without_improve))

    best_metrics: Dict[str, float] = {}
    latent_cost_ridges: list[dict] = []
    timestamp = time.strftime("%m%d%H%M")

    latent_walk_every_n = int(getattr(config.train, "latent_walk_every_n_epochs", 0) or 0)
    latent_walk_on_final = bool(getattr(config.train, "latent_walk_on_final", False))
    latent_walk_top_k = int(getattr(config.train, "latent_walk_top_k", 1) or 1)
    latent_walk_num_steps = int(getattr(config.train, "latent_walk_num_steps", 8) or 8)
    latent_walk_step_size = float(getattr(config.train, "latent_walk_step_size", 0.25) or 0.25)
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
        print(f"[train] evaluating train split with offline teacher forcing after epoch {epoch}")
        train_eval_metrics = evaluate_teacher_forcing(
            model,
            bundle.train_dataset,
            registry,
            tokenizer,
            device,
            batch_size=config.eval.batch_size,
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

        latent_cost_ridges, latent_cost_ridge, ridge_metrics = _fit_epoch_ridges(
            model,
            bundle,
            tokenizer,
            config,
            device,
        )
        summary.update(ridge_metrics)
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

        last_summary = dict(summary)
        if "val_full_sequence_exact_match" in summary:
            best_exact_match = max(best_exact_match, float(summary["val_full_sequence_exact_match"]))
        if "val_token_accuracy" in summary:
            best_val_acc = max(best_val_acc, float(summary["val_token_accuracy"]))

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

        if scheduler is not None:
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
            epochs_without_improve=epochs_without_improve,
            config=config,
            tokenizer=tokenizer,
            latent_cost_ridge=latent_cost_ridge,
            latent_cost_ridges=latent_cost_ridges,
            timestamp=timestamp,
        )

        if improved:
            if current_metric is not None:
                best_metric_value = float(current_metric)
            epochs_without_improve = 0
            best_metrics = dict(summary)
            checkpoint_kwargs["best_metric_value"] = best_metric_value
            checkpoint_kwargs["epochs_without_improve"] = epochs_without_improve
            save_checkpoint(best_checkpoint_path, **checkpoint_kwargs)
            save_checkpoint(checkpoint_path, **checkpoint_kwargs)
            print(f"[train] best updated: {best_metric_name}={best_metric_value:.6f}")
        else:
            if can_early_stop:
                epochs_without_improve += 1
            checkpoint_kwargs["best_metric_value"] = best_metric_value
            checkpoint_kwargs["epochs_without_improve"] = epochs_without_improve

        save_checkpoint(last_checkpoint_path, **checkpoint_kwargs)

        walk_summary: Dict[str, float] = {}
        if latent_walk_every_n > 0 and (epoch % latent_walk_every_n == 0):
            walk_summary = _run_periodic_latent_walk(
                model=model,
                device=device,
                checkpoint_path=last_checkpoint_path,
                record_json_path=latent_walk_record_json,
                walk_output_dir=latent_walk_output_dir,
                network_info_folder=latent_walk_network_info,
                epoch_label=f"epoch {epoch}",
                top_k=latent_walk_top_k,
                num_steps=latent_walk_num_steps,
                step_size=latent_walk_step_size,
            )
            if walk_summary:
                summary.update(walk_summary)

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

    final_metrics = dict(best_metrics if best_metrics else last_summary)
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
        final_walk_summary = _run_periodic_latent_walk(
            model=model,
            device=device,
            checkpoint_path=final_walk_checkpoint,
            record_json_path=latent_walk_record_json,
            walk_output_dir=latent_walk_output_dir,
            network_info_folder=latent_walk_network_info,
            epoch_label="final",
            top_k=latent_walk_top_k,
            num_steps=latent_walk_num_steps,
            step_size=latent_walk_step_size,
        )
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
