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
    _build_teacher_forcing_candidate_masks,
    _compress_teacher_forcing_batch,
    _teacher_forcing_accuracy_stats,
)
from .train_losses import (
    beta_by_epoch,
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
        compressed = _compress_teacher_forcing_batch(batch, candidate_masks, tokenizer)
        position_weights = _build_early_param_position_weights(
            compressed["seq_lens"],
            max_len=int(compressed["target_ids"].shape[1]),
            max_weight=float(getattr(cfg.train, "early_param_weight_max", 1.0)),
            power=float(getattr(cfg.train, "early_param_weight_power", 1.0)),
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)
        use_amp = bool(cfg.train.use_amp and device.type == "cuda")
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(
                batch["encoder_token_ids"],
                batch["encoder_var_ids"],
                compressed["decoder_input_ids"],
                compressed["decoder_var_ids"],
                pad_token_id=tokenizer.pad_id,
            )
            recon_loss = masked_cross_entropy(
                out.logits,
                compressed["target_ids"],
                compressed["candidate_masks"],
                tokenizer.pad_id,
                position_weights=position_weights,
            )
            kl_loss = kl_divergence(out.mu, out.logvar)
            cost_loss = weighted_cost_loss(out.cost_pred, batch["costs"], batch["cost_mask"])
            latent_use_loss, latent_use_rank_loss, latent_use_top1_drop_loss = latent_use_margin_loss(
                model,
                out.logits,
                compressed["decoder_input_ids"],
                compressed["decoder_var_ids"],
                compressed["decoder_input_ids"].eq(tokenizer.pad_id),
                out.z,
                out.memory,
                compressed["target_ids"],
                compressed["candidate_masks"],
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
                nce_loss = ordered_infonce_loss(nce_z, batch["costs"], batch["cost_mask"], cfg.train.tau_nce)
            else:
                nce_loss = soft_infonce_loss(nce_z, batch["costs"], batch["cost_mask"], cfg.train.tau_nce)
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
                    compressed["target_ids"],
                    compressed["candidate_masks"],
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
