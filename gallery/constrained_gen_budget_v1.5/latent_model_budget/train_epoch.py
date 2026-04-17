from __future__ import annotations

from typing import Dict

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
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_cost = 0.0
    total_nce = 0.0
    total_latent_use = 0.0
    total_latent_use_rank = 0.0
    total_latent_use_top1_drop = 0.0
    total_token_correct = 0
    total_token_count = 0
    total_exact_count = 0
    total_sample_count = 0
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
        singleton_mask = _build_singleton_position_mask(
            batch["target_ids"], candidate_masks, tokenizer.pad_id
        )
        position_weights = _build_early_param_position_weights(
            batch["seq_lens"],
            max_len=int(batch["target_ids"].shape[1]),
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
                batch["decoder_input_ids"],
                batch["decoder_var_ids"],
                pad_token_id=tokenizer.pad_id,
            )
            # CoBO-style sample weighting
            cobo_sw = None
            if getattr(cfg.train, "cobo_sample_weighting", False):
                cobo_sw = compute_cobo_sample_weights(
                    batch["costs"],
                    batch["cost_mask"],
                    quantile=float(getattr(cfg.train, "cobo_weight_quantile", 0.95)),
                    sigma=float(getattr(cfg.train, "cobo_weight_sigma", 0.1)),
                )
            recon_loss = masked_cross_entropy(
                out.logits,
                batch["target_ids"],
                candidate_masks,
                tokenizer.pad_id,
                position_weights=position_weights,
                sample_weights=cobo_sw,
                label_smoothing=float(getattr(cfg.train, "label_smoothing", 0.0)),
            )
            kl_loss = kl_divergence(out.mu, out.logvar)
            cost_loss = weighted_cost_loss(out.cost_pred, batch["costs"], batch["cost_mask"], sample_weights=cobo_sw)
            latent_use_loss, latent_use_rank_loss, latent_use_top1_drop_loss = latent_use_margin_loss(
                model,
                out.logits,
                batch["decoder_input_ids"],
                batch["decoder_var_ids"],
                batch["decoder_input_ids"].eq(tokenizer.pad_id),
                out.z,
                out.memory,
                batch["target_ids"],
                candidate_masks,
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
                nce_loss = ordered_infonce_loss(nce_z, batch["costs"], batch["cost_mask"], cfg.train.tau_nce, sample_weights=cobo_sw)
            else:
                nce_loss = soft_infonce_loss(nce_z, batch["costs"], batch["cost_mask"], cfg.train.tau_nce, sample_weights=cobo_sw)
            loss = (
                recon_loss
                + beta * kl_loss
                + float(cfg.train.lambda_cost) * cost_loss
                + float(cfg.train.lambda_nce) * nce_loss
                + float(getattr(cfg.train, "lambda_latent_use", 0.0)) * latent_use_loss
            )
            batch_token_correct, batch_token_total, batch_exact_count, batch_sample_total = (
                _teacher_forcing_accuracy_stats(
                    out.logits,
                    batch["target_ids"],
                    candidate_masks,
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

        total_loss += float(loss.item())
        total_recon += float(recon_loss.item())
        total_kl += float(kl_loss.item())
        total_cost += float(cost_loss.item())
        total_nce += float(nce_loss.item())
        total_latent_use += float(latent_use_loss.item())
        total_latent_use_rank += float(latent_use_rank_loss.item())
        total_latent_use_top1_drop += float(latent_use_top1_drop_loss.item())
        total_token_correct += int(batch_token_correct)
        total_token_count += int(batch_token_total)
        total_exact_count += int(batch_exact_count)
        total_sample_count += int(batch_sample_total)
        total_batches += 1

    denom = max(total_batches, 1)
    return {
        "loss": total_loss / denom,
        "recon_loss": total_recon / denom,
        "kl_loss": total_kl / denom,
        "cost_loss": total_cost / denom,
        "nce_loss": total_nce / denom,
        "latent_use_loss": total_latent_use / denom,
        "latent_use_rank_loss": total_latent_use_rank / denom,
        "latent_use_top1_drop_loss": total_latent_use_top1_drop / denom,
        "beta": beta,
        "early_param_weight_max": float(getattr(cfg.train, "early_param_weight_max", 1.0)),
        "early_param_weight_power": float(getattr(cfg.train, "early_param_weight_power", 1.0)),
        "lambda_latent_use": float(getattr(cfg.train, "lambda_latent_use", 0.0)),
        "latent_use_margin": float(getattr(cfg.train, "latent_use_margin", 1.0)),
        "latent_wrong_top1_margin": float(getattr(cfg.train, "latent_wrong_top1_margin", 0.0)),
    }
