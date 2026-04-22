"""Diagnostic training: isolate the effect of *including* loss from samples
that hw_param=8 oracle would have fallback'd on.

Approach (option 3 — oracle mask diff)
--------------------------------------
- Training runs with ``max_vthread_extent = 15`` so the model sees wider
  candidate sets and no oracle fallback at gold values (no singleton fills).
- Diagnostic reference: the precomputed candidate-mask cache built under the
  default hw_param (``max_vthread_extent = 8``). File:
  ``..._v4_no_budget.pt`` (no generator suffix). That cache encodes exactly
  the singleton fallbacks that occurred under the old behavior.
- At every decoder position where the baseline (hw_param=8) mask is a
  singleton but the current (hw_param=15) training mask is not, the recon
  loss at that position is scaled by ``VIOLATION_LOSS_WEIGHT``. Set to 0.0
  to reproduce the old loss pattern; 0.1–0.3 to interpolate.

Only ``recon_loss`` is affected. ``kl / cost / nce / latent_use`` retain
their original weighting.
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


from latent_model_budget import train as train_module
from latent_model_budget import train_epoch as train_epoch_module
from latent_model_budget.adapter import GeneratorRegistry
from latent_model_budget.config import build_config, resolve_task_paths
from latent_model_budget.dataset import _candidate_mask_cache_path_for_workload
from latent_model_budget.tokenizer import ParamTokenizer
from latent_model_budget.model import LatentParamVAE
from latent_model_budget.train_eval import (
    _batch_to_device,
    _build_early_param_position_weights,
    _build_singleton_position_mask,
    _build_teacher_forcing_candidate_masks,
    _compress_teacher_forcing_batch,
    _teacher_forcing_accuracy_stats,
)
from latent_model_budget.train_losses import (
    beta_by_epoch,
    compute_cobo_sample_weights,
    kl_divergence,
    latent_use_margin_loss,
    masked_cross_entropy,
    ordered_infonce_loss,
    soft_infonce_loss,
    weighted_cost_loss,
)


# -----------------------------------------------------------------------------
# Diagnostic knobs
# -----------------------------------------------------------------------------
TASK_INDEX = 1490
VIOLATION_LOSS_WEIGHT = 0.0  # per-position multiplier for recon-loss positions
                             # where hw_param=8 baseline oracle would fallback.
                             # 0.0 mimics old hw_param=8 training; 0.1-0.3 to
                             # interpolate toward full hw_param=15 training.


_BASELINE_CACHE_BY_WORKLOAD: Dict[Tuple[str, str], Dict[str, torch.Tensor]] = {}


def _baseline_cache_config(cfg):
    """Make a cfg copy that points at the default-hw_param (hw_param=8) cache."""
    cfg_base = deepcopy(cfg)
    cfg_base.generator.hw_param = {}
    cfg_base.generator.disable_constraint = []
    return cfg_base


def _load_baseline_workload_cache(
    cfg, workload_key: str, target_kind: str
) -> Dict[str, torch.Tensor]:
    key = (str(workload_key), str(target_kind))
    cache = _BASELINE_CACHE_BY_WORKLOAD.get(key)
    if cache is not None:
        return cache
    cfg_base = _baseline_cache_config(cfg)
    path = _candidate_mask_cache_path_for_workload(cfg_base, workload_key, target_kind)
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline mask cache not found: {path}. Run training once with "
            "default hw_param to populate it."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sample_masks = payload["sample_masks"]
    cache = {
        str(sid): mask.to(dtype=torch.bool, device="cpu")
        for sid, mask in sample_masks.items()
    }
    _BASELINE_CACHE_BY_WORKLOAD[key] = cache
    print(
        f"[diag] loaded baseline mask cache ({len(cache)} samples) from {path.name}"
    )
    return cache


def _baseline_violation_scale(
    batch: Dict[str, object],
    cfg,
    tokenizer: ParamTokenizer,
    device: torch.device,
    max_len: int,
    current_candidate_masks: torch.Tensor,
    *,
    violation_weight: float = VIOLATION_LOSS_WEIGHT,
) -> Tuple[torch.Tensor, int, int]:
    """Return (scale[B, max_len], num_rows_with_downweight, num_downweighted_positions).

    Positions are marked as violations where the baseline (hw_param=8) mask
    is a singleton but the current (hw_param=15) mask is NOT a singleton.
    That is exactly the set of positions where the old oracle would have
    triggered fallback and the new oracle learns a real distribution.
    """
    sample_ids = batch["sample_ids"]
    workload_keys = batch["workload_keys"]
    target_kinds = batch["target_kinds"]
    target_ids: torch.Tensor = batch["target_ids"]
    pad_id = int(tokenizer.pad_id)
    vocab_size = len(tokenizer.id_to_token)
    bsz = int(target_ids.shape[0])

    pad_only_cpu = tokenizer.pad_only_mask(device=torch.device("cpu"))
    baseline_mask = pad_only_cpu.unsqueeze(0).unsqueeze(0).expand(bsz, max_len, vocab_size).clone()

    for i, sid in enumerate(sample_ids):
        cache = _load_baseline_workload_cache(cfg, workload_keys[i], target_kinds[i])
        m = cache.get(str(sid))
        if m is None:
            raise KeyError(
                f"sample_id={sid} not found in baseline cache "
                f"(workload={workload_keys[i][:40]}..., target_kind={target_kinds[i]})"
            )
        T = min(int(m.shape[0]), max_len)
        baseline_mask[i, :T] = m[:T]

    baseline_mask = baseline_mask.to(device=device, dtype=torch.bool, non_blocking=True)

    singleton_base = _build_singleton_position_mask(target_ids, baseline_mask, pad_id)
    singleton_cur = _build_singleton_position_mask(target_ids, current_candidate_masks, pad_id)
    downweight_mask = singleton_base & (~singleton_cur)

    scale = torch.where(
        downweight_mask,
        torch.tensor(float(violation_weight), dtype=torch.float32, device=device),
        torch.tensor(1.0, dtype=torch.float32, device=device),
    )
    rows_violating = int(downweight_mask.any(dim=-1).sum().item())
    positions_down = int(downweight_mask.sum().item())
    return scale, rows_violating, positions_down


def train_one_epoch_diag(
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
    """Copy of ``train_epoch.train_one_epoch`` with recon-loss position
    weights down-scaled from the first vthread violation (gold > 8) onward."""
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
    total_violation_rows = 0
    total_violation_positions = 0

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
            # Diagnostic not supported in compressed path (positions are re-indexed).
            recon_position_weights = position_weights
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
            position_weights = position_weights * (~singleton_mask).to(position_weights.dtype)

            diag_scale, n_rows, n_positions = _baseline_violation_scale(
                batch,
                cfg,
                tokenizer,
                device=device,
                max_len=int(target_ids.shape[1]),
                current_candidate_masks=cand_masks_eff,
                violation_weight=VIOLATION_LOSS_WEIGHT,
            )
            total_violation_rows += n_rows
            total_violation_positions += n_positions
            recon_position_weights = position_weights * diag_scale

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
                position_weights=recon_position_weights,
                sample_weights=recon_sw,
                label_smoothing=float(getattr(cfg.train, "label_smoothing", 0.0)),
            )
            kl_loss = kl_divergence(out.mu, out.logvar, sample_weights=kld_sw)
            cost_loss = weighted_cost_loss(
                out.cost_pred, batch["costs"], batch["cost_mask"], sample_weights=cost_sw
            )
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
                nce_loss = soft_infonce_loss(
                    nce_z, batch["costs"], batch["cost_mask"], cfg.train.tau_nce,
                    sample_weights=nce_sw,
                )
            loss = (
                float(getattr(cfg.train, "lambda_recon", 1.0)) * recon_loss
                + beta * kl_loss
                + float(cfg.train.lambda_cost) * cost_loss
                + float(cfg.train.lambda_nce) * nce_loss
                + float(getattr(cfg.train, "lambda_latent_use", 0.0)) * latent_use_loss
            )
            batch_token_correct, batch_token_total, batch_exact_count, batch_sample_total = (
                _teacher_forcing_accuracy_stats(
                    out.logits, target_ids, cand_masks_eff, tokenizer.pad_id,
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
    print(
        f"[diag] epoch {epoch}: downweighted rows={total_violation_rows} "
        f"downweighted positions={total_violation_positions} "
        f"(weight={VIOLATION_LOSS_WEIGHT}, baseline=hw_param default)"
    )
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
        "diag_downweighted_rows": total_violation_rows,
        "diag_downweighted_positions": total_violation_positions,
        "diag_violation_loss_weight": float(VIOLATION_LOSS_WEIGHT),
    }


def main() -> None:
    cfg = build_config()
    cfg.data.task_index = TASK_INDEX
    resolve_task_paths(cfg)
    cfg.generator.hw_param = {"max_vthread_extent": 15}
    cfg.generator.disable_constraint = []

    print(
        f"[diag] task_index={TASK_INDEX} "
        f"VIOLATION_LOSS_WEIGHT={VIOLATION_LOSS_WEIGHT} "
        f"generator.hw_param={cfg.generator.hw_param} "
        f"(baseline ref = default hw_param mask cache)"
    )

    train_epoch_module.train_one_epoch = train_one_epoch_diag
    train_module.train_one_epoch = train_one_epoch_diag

    train_module.train_main(cfg)


if __name__ == "__main__":
    main()
