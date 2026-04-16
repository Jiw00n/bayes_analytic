from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Dict

import torch

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

import math
from .adapter import GeneratorRegistry, JsonSampleRecord
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
# from .numeric_side_features import install_numeric_side_features
# from .decoder_only_dynamic_side_features import install_decoder_only_dynamic_side_features


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
    # from latent_model_budget.constraint_state_features import (
    #     install_decoder_only_constraint_state_features,
    #     StepStateAdapter,
    # )
    # install_decoder_only_constraint_state_features(
    #     model,
    #     feature_dim=8,
    #     hidden_dim=64,
    #     dropout=0.0,
    #     scale_init=0.10,
    # )
    return registry, bundle, tokenizer, model


def _resolve_run_task_index(registry: GeneratorRegistry, bundle: DatasetBundle) -> str:
    for record in list(bundle.train_records) + list(bundle.val_records) + list(bundle.test_records):
        resolved = _resolve_record_task_index(registry, record)
        if resolved is not None:
            return str(resolved)
    return "na"


def _resolve_record_task_index(
    registry: GeneratorRegistry,
    record: JsonSampleRecord,
) -> int | None:
    try:
        return registry.get_task_index(
            workload_key=record.workload_key,
            target_kind=record.target_kind,
            task_index=record.task_index,
        )
    except (KeyError, ValueError):
        return None


def _build_wandb_project_name(config, registry: GeneratorRegistry, bundle: DatasetBundle) -> str:
    task_index = _resolve_run_task_index(registry, bundle)
    project_suffix = getattr(config.wandb, "project", None) or "single_v1"
    return f"Task{task_index}_{project_suffix}"


def _build_wandb_run_name(timestamp, config, bundle: DatasetBundle) -> str:
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


def _evaluate_validation_epoch(
    model,
    bundle,
    registry,
    tokenizer,
    config,
    device,
    epoch,
    latent_cost_ridges,
    *,
    include_cost_metrics: bool = True,
):
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

    if include_cost_metrics:
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


def _evaluate_final_checkpoint(
    model,
    bundle,
    registry,
    tokenizer,
    config,
    device,
    latent_cost_ridges,
    *,
    include_cost_metrics: bool = True,
    include_autoregressive: bool = True,
):
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

        if include_cost_metrics:
            val_cost_metrics = evaluate_cost_ranking(
                model,
                bundle.val_dataset,
                tokenizer,
                device,
                batch_size=config.eval.batch_size,
                latent_cost_ridges=_build_named_latent_cost_ridges(latent_cost_ridges),
            )
            summary.update({f"eval_val_{k}": float(v) for k, v in val_cost_metrics.items()})

        if include_autoregressive:
            val_ar_metrics = evaluate_autoregressive(
                model,
                bundle.val_dataset,
                registry,
                tokenizer,
                device,
                batch_size=config.eval.batch_size,
            )
            summary.update({f"val_autoregressive_{k}": float(v) for k, v in val_ar_metrics.items()})
        # summary.update({f"eval_val_autoregressive_{k}": float(v) for k, v in val_ar_metrics.items()})

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

        if include_cost_metrics:
            test_cost_metrics = evaluate_cost_ranking(
                model,
                bundle.test_dataset,
                tokenizer,
                device,
                batch_size=config.eval.batch_size,
                latent_cost_ridges=_build_named_latent_cost_ridges(latent_cost_ridges),
            )
            summary.update({f"eval_test_{k}": float(v) for k, v in test_cost_metrics.items()})

        if include_autoregressive:
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
    timestamp = time.strftime("%m%d%H%M")
    run_name = _build_wandb_run_name(timestamp, config, bundle)
    if wandb_project:
        if wandb is None:
            print("[train] wandb project is set but wandb is not installed; skipping wandb logging")
        else:
            project_name = _build_wandb_project_name(config, registry, bundle)
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

    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=config.train.learning_rate,
    #     weight_decay=config.train.weight_decay,
    # )
    # scheduler = None
    best_metric_name = str(getattr(config.train, "best_metric_name", "val_cost_head_pred_top10_mean_actual_cost"))
    best_metric_mode = str(getattr(config.train, "best_metric_mode", "max")).lower()
    early_stop_patience = int(getattr(config.train, "early_stop_patience", 15))
    early_stop_min_delta = float(getattr(config.train, "early_stop_min_delta", 1e-4))


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    scheduler_name = str(getattr(config.train, "scheduler_name", "none")).lower()

    if scheduler_name == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(getattr(config.train, "scheduler_milestones", [70])),
            gamma=float(getattr(config.train, "scheduler_gamma", 1.0 / 3.0)),
        )
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=best_metric_mode,  # "max" or "min"
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
    evaluate_train_tf = bool(getattr(config.train, "evaluate_train_teacher_forcing_each_epoch", True))
    evaluate_cost_metrics = bool(getattr(config.train, "evaluate_cost_metrics_each_epoch", True))
    evaluate_final_checkpoint_metrics = bool(
        getattr(config.train, "evaluate_final_checkpoint_metrics", True)
    )
    print_reconstruction_after_train = bool(
        getattr(config.train, "print_reconstruction_after_train", True)
    )
    evaluate_autoregressive_epochs = getattr(config.train, "evaluate_autoregressive_each_epoch", 10)
    

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

        summary = dict(train_metrics)
        if evaluate_train_tf:
            print(f"[train] evaluating train split with offline teacher forcing after epoch {epoch}")
            train_eval_metrics = evaluate_teacher_forcing(
                model,
                bundle.train_dataset,
                registry,
                tokenizer,
                device,
                batch_size=config.eval.batch_size,
            )
            summary["token_accuracy"] = float(train_eval_metrics["token_accuracy"])
            summary["full_sequence_exact_match"] = float(
                train_eval_metrics["full_sequence_exact_match"]
            )
        print(
            f"[epoch {epoch}] "
            f"loss={summary['loss']:.4f} recon={summary['recon_loss']:.4f} "
            f"kl={summary['kl_loss']:.4f} "
            f"tok_acc={summary['token_accuracy']:.4f} "
            f"exact={summary['full_sequence_exact_match']:.4f}"
        )

        latent_cost_ridge = None
        ridge_metrics = {}
        if evaluate_cost_metrics:
            latent_cost_ridges, latent_cost_ridge, ridge_metrics = _fit_epoch_ridges(
                model,
                bundle,
                tokenizer,
                config,
                device,
            )
        else:
            latent_cost_ridges = []
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
                include_cost_metrics=evaluate_cost_metrics,
            )
        )

        if evaluate_autoregressive_epochs != 0 and epoch % evaluate_autoregressive_epochs == 0:
            print(f"[train] evaluating validation split with autoregressive decoding after epoch {epoch}")
            ar_metrics = evaluate_autoregressive(
                model,
                bundle.val_dataset,
                registry,
                tokenizer,
                device,
                batch_size=config.eval.batch_size,
            )
            summary.update({f"val_autoregressive_{k}": float(v) for k, v in ar_metrics.items()})
            print(
                f"val_autoregressive_tok_acc={summary['val_autoregressive_token_accuracy']:.4f} "
                f"val_autoregressive_exact={summary['val_autoregressive_full_sequence_exact_match']:.4f}"
            )



        # if "val_full_sequence_exact_match" in summary:
        #     best_exact_match = max(best_exact_match, float(summary["val_full_sequence_exact_match"]))
        # if "val_token_accuracy" in summary:
        #     best_val_acc = max(best_val_acc, float(summary["val_token_accuracy"]))
        # if wandb_run is not None:
        #     wandb.log({"epoch": epoch, **summary}, step=epoch)

        # if epoch == config.train.num_epochs:
        #     best_metrics = dict(summary)
        #     checkpoint_acc = float(summary.get("val_token_accuracy", summary["token_accuracy"]))
        #     checkpoint_kwargs = dict(
        #         model=model,
        #         optimizer=optimizer,
        #         scheduler=scheduler,
        #         epoch=epoch,
        #         best_exact_match=best_exact_match,
        #         config=config,
        #         tokenizer=tokenizer,
        #         latent_cost_ridge=latent_cost_ridge,
        #         latent_cost_ridges=latent_cost_ridges,
        #         timestamp=timestamp,
        #     )
        #     save_checkpoint(
        #         checkpoint_dir / f"{timestamp}_acc_{checkpoint_acc:.2f}.pt",
        #         **checkpoint_kwargs,
        #     )
        #     save_checkpoint(checkpoint_dir / "last.pt", **checkpoint_kwargs)


        last_summary = dict(summary)

        if "val_full_sequence_exact_match" in summary:
            best_exact_match = max(best_exact_match, float(summary["val_full_sequence_exact_match"]))
        if "val_token_accuracy" in summary:
            best_val_acc = max(best_val_acc, float(summary["val_token_accuracy"]))

        current_metric = summary.get(best_metric_name)
        improved = False
        if current_metric is not None:
            current_metric = float(current_metric)
            if best_metric_mode == "max":
                improved = current_metric > (best_metric_value + early_stop_min_delta)
            else:
                improved = current_metric < (best_metric_value - early_stop_min_delta)

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
            best_metric_value = float(current_metric)
            epochs_without_improve = 0
            best_metrics = dict(summary)

            checkpoint_kwargs["best_metric_value"] = best_metric_value
            checkpoint_kwargs["epochs_without_improve"] = epochs_without_improve

            save_checkpoint(best_checkpoint_path, **checkpoint_kwargs)
            save_checkpoint(
                checkpoint_dir / f"{run_name}.pt",
                **checkpoint_kwargs,
            )
            print(f"[train] best updated: {best_metric_name}={best_metric_value:.4f}")
        else:
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
                    **summary,
                },
                step=epoch,
            )

        if epochs_without_improve >= early_stop_patience:
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
    if evaluate_final_checkpoint_metrics:
        final_metrics.update(
            _evaluate_final_checkpoint(
                model,
                bundle,
                registry,
                tokenizer,
                config,
                device,
                latent_cost_ridges,
                include_cost_metrics=evaluate_cost_metrics,
                include_autoregressive=evaluate_cost_metrics,
            )
        )
    print("[final]", json.dumps(final_metrics, indent=2))

    if print_reconstruction_after_train and bundle.val_dataset.samples:
        sample = bundle.val_dataset.samples[0]
        decoded = greedy_decode_sample(model, sample, registry, tokenizer, device)
        print(pretty_print_reconstruction(sample, decoded))
    elif print_reconstruction_after_train and bundle.test_dataset.samples:
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
            wandb.log(final_log_metrics, step=epoch)
        wandb_run.summary.update(final_metrics)
        wandb_run.finish()

    return final_metrics
