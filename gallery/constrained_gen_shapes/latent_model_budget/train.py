from __future__ import annotations

from pathlib import Path
import time
from typing import Dict, List, Optional

import torch

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

import math

from .adapter import GeneratorRegistry, JsonSampleRecord
from .dataset import (
    LatentParamDataset,
    build_dataset_bundle,
)
from .inference import SamplingOptions
from .model import LatentParamVAE
from .runtime_utils import (
    TaskBalancedBatchSampler,
    _build_wandb_project_name,
    _build_wandb_run_name,
    _config_for_resume_compare,
    _family_from_json_path,
    _remap_for_wandb,
    _resolve_pt_dir,
    _resolve_run_family_from_config,
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
)
from .train_epoch import train_one_epoch
from .train_eval import (
    _build_named_latent_cost_ridges,
    _build_reencode_predictor,
    _concat_encoded,
    _evaluate_validation_epoch,
    _fit_epoch_ridges,
    encode_dataset,
    evaluate_cost_ranking,
    evaluate_teacher_forcing_with_encoded,
)
from .walk_helpers import (
    _augment_summary_with_per_task,
    _iter_walk_ridges,
    _load_measurement_lookup,
    _merge_cache_into_lookup,
    _merge_walk_summaries,
    _probe_workload_key_from_json,
    _resolve_walk_record_jsons,
    _resolve_walk_task_min_cost,
    _run_periodic_latent_walk,
    _save_measurement_lookup,
    _seed_measurement_cache_from_buffer,
    _task_lookup_key_from_json,
)


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


    tasks_per_batch = getattr(config.train, "tasks_per_batch", None)
    train_batch_sampler = None
    if tasks_per_batch is not None:
        # Prefer the explicit ``task_index`` when JSONs carry it; otherwise
        # fall back to grouping by (workload_key, target_kind) since that is
        # the natural task identity for tenset-style measurement records,
        # which omit the meta.task_index field.
        samples = bundle.train_dataset.samples
        if any(s.task_index is not None for s in samples):
            sample_task_indices = [s.task_index for s in samples]
        else:
            key_to_id: Dict[tuple, int] = {}
            sample_task_indices = []
            for s in samples:
                key = (s.workload_key or "", s.target_kind or "")
                if key not in key_to_id:
                    key_to_id[key] = len(key_to_id)
                sample_task_indices.append(key_to_id[key])
        n_distinct_tasks = len({-1 if t is None else int(t) for t in sample_task_indices})
        if n_distinct_tasks < int(tasks_per_batch):
            print(
                f"[train] tasks_per_batch={tasks_per_batch} but dataset has only "
                f"{n_distinct_tasks} distinct task(s); falling back to random shuffle"
            )
        else:
            train_batch_sampler = TaskBalancedBatchSampler(
                task_indices=sample_task_indices,
                batch_size=int(config.train.batch_size),
                tasks_per_batch=int(tasks_per_batch),
                shuffle=True,
                seed=int(config.data.seed),
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
        batch_sampler=train_batch_sampler,
    )
    if train_batch_sampler is not None:
        print(
            f"[train] task-balanced loader: batches={len(train_loader)} "
            f"batch_size={config.train.batch_size} "
            f"tasks_per_batch={tasks_per_batch} "
            f"samples_per_task={config.train.batch_size // int(tasks_per_batch)} "
            f"num_workers={config.train.num_workers}"
        )
    else:
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
    # Nested by walk_key_prefix ("" for cost_vec walk, "cost_head_" for
    # cost_head-gradient walk, "w_ridge_" for weighted-ridge walk) so each
    # walk tracks its own best independently — without this separation, the
    # second walk overwrote the first walk's running max in shared dicts.
    walk_running_best: Dict[str, Dict[str, float]] = {}
    walk_running_best_epoch: Dict[str, Dict[str, int]] = {}
    walk_running_best_alpha: Dict[str, Dict[str, float]] = {}
    timestamp = time.strftime("%m%d%H%M")
    latent_walk_every_n = int(getattr(config.latent_walk, "every_n_epochs", 0) or 0)
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
    if latent_walk_every_n > 0 and not latent_walk_record_jsons:
        print(
            "[train] latent walk requested but no record JSON resolvable from "
            "config.latent_walk.record_json or config.data.json_paths; disabling"
        )
        latent_walk_every_n = 0
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

        # ``token_accuracy`` / ``full_sequence_exact_match`` are overwritten
        # below with an eval-mode (dropout-off) decoder-only pass that reuses
        # ``encoded_train["z"]`` — no extra encoder forward.
        summary = dict(train_metrics)

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

        train_tf_metrics = evaluate_teacher_forcing_with_encoded(
            model,
            bundle.train_dataset,
            registry,
            tokenizer,
            device,
            encoded=encoded_train,
            batch_size=config.eval.batch_size,
            loader=train_eval_loader,
        )
        summary["token_accuracy"] = float(train_tf_metrics["token_accuracy"])
        summary["full_sequence_exact_match"] = float(
            train_tf_metrics["full_sequence_exact_match"]
        )
        print(
            f"[epoch {epoch}] "
            f"loss={summary['loss']:.4f} recon={summary['recon_loss']:.4f} "
            f"kl={summary['kl_loss']:.4f} "
            f"tok_acc={summary['token_accuracy']:.4f} "
            f"exact={summary['full_sequence_exact_match']:.4f}"
        )

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
            _ref_dir = getattr(config.latent_walk, "reference_best_dir", None)
            _ref_label = Path(_ref_dir).name if _ref_dir else None
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
                        if wandb_run is not None:
                            partial: Dict[str, float] = {}
                            _augment_summary_with_per_task(
                                partial,
                                [(tkey, sub_summary)],
                                epoch=int(epoch),
                                running_best=walk_running_best.setdefault(walk_prefix, {}),
                                running_best_epoch=walk_running_best_epoch.setdefault(walk_prefix, {}),
                                running_best_alpha=walk_running_best_alpha.setdefault(walk_prefix, {}),
                                reference_label=_ref_label,
                                walk_key_prefix=walk_prefix,
                            )
                            if partial:
                                wandb.log(
                                    _remap_for_wandb(partial), step=int(epoch)
                                )
                merged = _merge_walk_summaries(
                    [s for _, s in per_task_summaries_keyed],
                    walk_key_prefix=walk_prefix,
                )
                if merged:
                    walk_summary.update(merged)
                _augment_summary_with_per_task(
                    walk_summary,
                    per_task_summaries_keyed,
                    epoch=int(epoch),
                    running_best=walk_running_best.setdefault(walk_prefix, {}),
                    running_best_epoch=walk_running_best_epoch.setdefault(walk_prefix, {}),
                    running_best_alpha=walk_running_best_alpha.setdefault(walk_prefix, {}),
                    reference_label=_ref_label,
                    walk_key_prefix=walk_prefix,
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

    print("[train] checkpoint dir:", pt_dir)

    if wandb_run is not None:
        if last_summary:
            wandb_run.summary.update(_remap_for_wandb(last_summary))
        wandb_run.finish()

    return dict(last_summary)
