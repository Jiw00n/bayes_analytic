from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from .adapter import GeneratorRegistry
from .model import LatentParamVAE
from .tokenizer import ParamTokenizer
from .train_eval import (
    _batch_to_device,
    _build_early_param_position_weights,
    _build_singleton_position_mask,
    _build_teacher_forcing_candidate_masks,
    _compress_teacher_forcing_batch,
    _teacher_forcing_accuracy_stats,
)
from .train_losses import (
    beta_by_epoch,
    compute_cobo_sample_weights,
    kl_divergence,
    latent_use_margin_loss,
    masked_cross_entropy,
    ordered_infonce_loss,
    soft_infonce_loss,
    weighted_cost_loss,
)


def _convert_cost_tensor_space(
    costs: torch.Tensor,
    src: str,
    dst: str,
    task_min_cost: Optional[float],
) -> torch.Tensor:
    """Convert a cost-label tensor from ``src`` space to ``dst`` space by
    routing through raw seconds. Throughput variants require ``task_min_cost``.
    """
    if src == dst:
        return costs
    eps = 1e-30
    # label → raw
    if src == "neg_log":
        raw = torch.exp(-costs)
    elif src == "norm_throughput":
        if task_min_cost is None:
            raise ValueError("task_min_cost required for norm_throughput")
        raw = task_min_cost / costs.clamp_min(eps)
    elif src == "log_norm_throughput":
        if task_min_cost is None:
            raise ValueError("task_min_cost required for log_norm_throughput")
        raw = task_min_cost * torch.exp(-costs)
    else:
        raise ValueError(f"Unknown cost_target: {src!r}")
    # raw → label
    if dst == "neg_log":
        return -torch.log(raw.clamp_min(eps))
    if dst == "norm_throughput":
        if task_min_cost is None:
            raise ValueError("task_min_cost required for norm_throughput")
        return task_min_cost / raw.clamp_min(eps)
    if dst == "log_norm_throughput":
        if task_min_cost is None:
            raise ValueError("task_min_cost required for log_norm_throughput")
        return math.log(task_min_cost) - torch.log(raw.clamp_min(eps))
    raise ValueError(f"Unknown cost_target: {dst!r}")


def train_one_epoch(
    model: LatentParamVAE,
    loader: DataLoader,
    optimizer,
    scaler,
    registry: GeneratorRegistry,
    tokenizer: ParamTokenizer,
    cfg,
    device: torch.device,
    epoch: int,
    task_min_cost: Optional[float] = None,
) -> Dict[str, float]:
    cost_target = str(getattr(cfg.data, "cost_target", "neg_log"))
    cost_target_regression = (
        getattr(cfg.data, "cost_target_regression", None) or cost_target
    )
    model.train()
    # Accumulate scalar losses + accuracy counters as device tensors so the
    # GPU never has to flush mid-epoch. Each ``.item()`` was a sync that
    # serialized the loop; doing them once at the end keeps the train loop
    # GPU-bound.
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_recon = torch.zeros((), device=device, dtype=torch.float32)
    total_kl = torch.zeros((), device=device, dtype=torch.float32)
    total_cost = torch.zeros((), device=device, dtype=torch.float32)
    total_nce = torch.zeros((), device=device, dtype=torch.float32)
    total_latent_use = torch.zeros((), device=device, dtype=torch.float32)
    total_latent_use_rank = torch.zeros((), device=device, dtype=torch.float32)
    total_latent_use_top1_drop = torch.zeros((), device=device, dtype=torch.float32)
    total_token_correct = torch.zeros((), device=device, dtype=torch.long)
    total_token_count = torch.zeros((), device=device, dtype=torch.long)
    total_exact_count = torch.zeros((), device=device, dtype=torch.long)
    total_sample_count = torch.zeros((), device=device, dtype=torch.long)
    total_batches = 0

    iterator = tqdm(loader, desc=f"train epoch {epoch}") if tqdm is not None else loader
    beta = beta_by_epoch(cfg.train, epoch)

    for batch in iterator:
        batch = _batch_to_device(batch, device)
        candidate_masks = _build_teacher_forcing_candidate_masks(
            batch,
            registry,
            tokenizer,
            device=device,
            debug_invalid_step=cfg.train.debug_invalid_step,
        )
        use_compressed = bool(getattr(cfg.train, "use_compressed_teacher_forcing", False))
        if use_compressed:
            compressed = _compress_teacher_forcing_batch(batch, candidate_masks, tokenizer)
            decoder_input_ids = compressed["decoder_input_ids"]
            decoder_var_ids = compressed["decoder_var_ids"]
            target_ids = compressed["target_ids"]
            cand_masks_eff = compressed["candidate_masks"]
            position_weights = _build_early_param_position_weights(
                compressed["seq_lens"],
                max_len=int(target_ids.shape[1]),
                max_weight=float(getattr(cfg.train, "early_param_weight_max", 1.0)),
                power=float(getattr(cfg.train, "early_param_weight_power", 1.0)),
                device=device,
            )
        else:
            decoder_input_ids = batch["decoder_input_ids"]
            decoder_var_ids = batch["decoder_var_ids"]
            target_ids = batch["target_ids"]
            cand_masks_eff = candidate_masks
            singleton_mask = _build_singleton_position_mask(
                target_ids, cand_masks_eff, tokenizer.pad_id
            )
            position_weights = _build_early_param_position_weights(
                batch["seq_lens"],
                max_len=int(target_ids.shape[1]),
                max_weight=float(getattr(cfg.train, "early_param_weight_max", 1.0)),
                power=float(getattr(cfg.train, "early_param_weight_power", 1.0)),
                device=device,
            )
            # Singleton positions are already determined by the oracle; keep them in
            # the sequence so the decoder's causal context matches inference, but
            # zero their loss contribution.
            position_weights = position_weights * (~singleton_mask).to(position_weights.dtype)

        optimizer.zero_grad(set_to_none=True)
        use_amp = bool(cfg.train.use_amp and device.type == "cuda")
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(
                batch["encoder_token_ids"],
                batch["encoder_var_ids"],
                decoder_input_ids,
                decoder_var_ids,
                pad_token_id=tokenizer.pad_id,
            )
            # CoBO-style sample weighting
            cobo_sw = None
            cobo_apply = set()
            if getattr(cfg.train, "cobo_sample_weighting", False):
                cobo_sw = compute_cobo_sample_weights(
                    batch["costs"],
                    batch["cost_mask"],
                    quantile=float(getattr(cfg.train, "weight_quantile", 0.95)),
                    sigma=float(getattr(cfg.train, "weight_sigma", 0.1)),
                )
                cobo_apply = {
                    str(x).lower()
                    for x in getattr(cfg.train, "cobo_apply_to", ["kld", "cost", "nce"])
                }
            recon_sw = cobo_sw if "recon" in cobo_apply else None
            kld_sw = cobo_sw if "kld" in cobo_apply else None
            cost_sw = cobo_sw if "cost" in cobo_apply else None
            nce_sw = cobo_sw if "nce" in cobo_apply else None
            recon_loss = masked_cross_entropy(
                out.logits,
                target_ids,
                cand_masks_eff,
                tokenizer.pad_id,
                position_weights=position_weights,
                sample_weights=recon_sw,
                label_smoothing=float(getattr(cfg.train, "label_smoothing", 0.0)),
            )
            kl_loss = kl_divergence(out.mu, out.logvar, sample_weights=kld_sw)
            if cost_target_regression != cost_target:
                cost_regression_targets = _convert_cost_tensor_space(
                    batch["costs"],
                    cost_target,
                    cost_target_regression,
                    task_min_cost,
                )
            else:
                cost_regression_targets = batch["costs"]
            cost_loss = weighted_cost_loss(out.cost_pred, cost_regression_targets, batch["cost_mask"], sample_weights=cost_sw)
            latent_use_loss, latent_use_rank_loss, latent_use_top1_drop_loss = latent_use_margin_loss(
                model,
                out.logits,
                decoder_input_ids,
                decoder_var_ids,
                decoder_input_ids.eq(tokenizer.pad_id),
                out.z,
                out.memory,
                target_ids,
                cand_masks_eff,
                tokenizer.pad_id,
                position_weights,
                margin=float(getattr(cfg.train, "latent_use_margin", 1.0)),
                wrong_top1_margin=float(getattr(cfg.train, "latent_wrong_top1_margin", 0.0)),
            )
            if getattr(cfg.train, "nce_mu", False):
                nce_z = out.mu
            else:
                nce_z = out.z
            if cfg.train.order_nce:
                nce_loss = ordered_infonce_loss(
                    nce_z,
                    batch["costs"],
                    batch["cost_mask"],
                    cfg.train.tau_nce,
                    sample_weights=nce_sw,
                    pos_weight_by_percentile=bool(getattr(cfg.train, "order_nce_pos_weight_by_percentile", False)),
                    pos_weight_sigma=float(getattr(cfg.train, "order_nce_pos_weight_sigma", 0.2)),
                )
            else:
                nce_loss = soft_infonce_loss(nce_z, batch["costs"], batch["cost_mask"], cfg.train.tau_nce, sample_weights=nce_sw)
            loss = (
                float(getattr(cfg.train, "lambda_recon", 1.0)) * recon_loss
                + beta * kl_loss
                + float(cfg.train.lambda_cost) * cost_loss
                + float(cfg.train.lambda_nce) * nce_loss
                + float(getattr(cfg.train, "lambda_latent_use", 0.0)) * latent_use_loss
            )
            batch_token_correct, batch_token_total, batch_exact_count, batch_sample_total = (
                _teacher_forcing_accuracy_stats(
                    out.logits,
                    target_ids,
                    cand_masks_eff,
                    tokenizer.pad_id,
                )
            )

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
            optimizer.step()

        total_loss = total_loss + loss.detach()
        total_recon = total_recon + recon_loss.detach()
        total_kl = total_kl + kl_loss.detach()
        total_cost = total_cost + cost_loss.detach()
        total_nce = total_nce + nce_loss.detach()
        total_latent_use = total_latent_use + latent_use_loss.detach()
        total_latent_use_rank = total_latent_use_rank + latent_use_rank_loss.detach()
        total_latent_use_top1_drop = total_latent_use_top1_drop + latent_use_top1_drop_loss.detach()
        total_token_correct = total_token_correct + batch_token_correct.detach()
        total_token_count = total_token_count + batch_token_total.detach()
        total_exact_count = total_exact_count + batch_exact_count.detach()
        total_sample_count = total_sample_count + batch_sample_total.detach()
        total_batches += 1

    denom = max(total_batches, 1)
    # Single end-of-epoch sync. ``.item()`` here pulls scalars in one CUDA
    # stream flush instead of one per batch.
    token_count_i = max(int(total_token_count.item()), 1)
    sample_count_i = max(int(total_sample_count.item()), 1)
    return {
        "loss": float(total_loss.item()) / denom,
        "recon_loss": float(total_recon.item()) / denom,
        "kl_loss": float(total_kl.item()) / denom,
        "cost_loss": float(total_cost.item()) / denom,
        "nce_loss": float(total_nce.item()) / denom,
        "latent_use_loss": float(total_latent_use.item()) / denom,
        "latent_use_rank_loss": float(total_latent_use_rank.item()) / denom,
        "latent_use_top1_drop_loss": float(total_latent_use_top1_drop.item()) / denom,
        # Token / sequence accuracy collected during the train loop. With
        # singleton positions excluded by ``_teacher_forcing_accuracy_stats``
        # this matches the train-side ``evaluate_teacher_forcing`` output up
        # to dropout noise, removing the need for a second forward pass over
        # the train split.
        "token_accuracy": float(total_token_correct.item()) / token_count_i,
        "full_sequence_exact_match": float(total_exact_count.item()) / sample_count_i,
        "beta": beta,
        "early_param_weight_max": float(getattr(cfg.train, "early_param_weight_max", 1.0)),
        "early_param_weight_power": float(getattr(cfg.train, "early_param_weight_power", 1.0)),
        "lambda_latent_use": float(getattr(cfg.train, "lambda_latent_use", 0.0)),
        "latent_use_margin": float(getattr(cfg.train, "latent_use_margin", 1.0)),
        "latent_wrong_top1_margin": float(getattr(cfg.train, "latent_wrong_top1_margin", 0.0)),
    }
