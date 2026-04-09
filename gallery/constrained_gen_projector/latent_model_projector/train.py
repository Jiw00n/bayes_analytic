from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path
import time
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

from .adapter import GeneratorRegistry
from .dataset import DatasetBundle, PreparedSample, build_dataset_bundle, collate_prepared_samples
from .inference import greedy_decode_batch, greedy_decode_sample, pretty_print_reconstruction
from .model import LatentParamVAE
from .tokenizer import ParamTokenizer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def configure_runtime(cfg, device: torch.device) -> None:
    if device.type != "cuda":
        return
    allow_tf32 = bool(getattr(cfg.train, "allow_tf32", True))
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def beta_by_epoch(cfg, epoch: int) -> float:
    if cfg.beta_warmup_epochs <= 0:
        return float(cfg.beta_end)
    progress = min(max(epoch / cfg.beta_warmup_epochs, 0.0), 1.0)
    return float(cfg.beta_start + (cfg.beta_end - cfg.beta_start) * progress)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu * mu - 1.0 - logvar, dim=-1))


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    candidate_masks: torch.Tensor,
    pad_id: int,
    position_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    masked_logits = logits.masked_fill(~candidate_masks, float("-inf"))
    token_losses = F.cross_entropy(
        masked_logits.reshape(-1, masked_logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_id,
        reduction="none",
    ).view_as(targets)
    valid_mask = targets.ne(pad_id)
    if position_weights is None:
        position_weights = torch.ones_like(token_losses)
    weighted_mask = position_weights * valid_mask.to(dtype=token_losses.dtype)
    return (token_losses * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)


def weighted_cost_loss(cost_pred: torch.Tensor, costs: torch.Tensor, cost_mask: torch.Tensor) -> torch.Tensor:
    if not cost_mask.any():
        return cost_pred.new_tensor(0.0)
    return F.mse_loss(cost_pred[cost_mask], costs[cost_mask])


def _weighted_token_mean(losses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def latent_use_margin_loss(
    model: LatentParamVAE,
    true_logits: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    decoder_var_ids: torch.Tensor,
    decoder_pad_mask: torch.Tensor,
    true_latent: torch.Tensor,
    true_memory: torch.Tensor,
    targets: torch.Tensor,
    candidate_masks: torch.Tensor,
    pad_id: int,
    position_weights: torch.Tensor,
    margin: float,
    wrong_top1_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = int(true_latent.size(0))
    if batch_size > 1:
        perm = torch.roll(torch.arange(batch_size, device=true_latent.device), shifts=1)
        wrong_latent = true_latent[perm].detach()
        wrong_memory = true_memory[perm].detach()
    else:
        wrong_latent = torch.zeros_like(true_latent)
        wrong_memory = torch.zeros_like(true_memory)
    zero_latent = torch.zeros_like(true_latent)
    zero_memory = torch.zeros_like(true_memory)

    shuffled_logits = model.decode(
        decoder_input_ids,
        decoder_var_ids,
        wrong_memory,
        wrong_latent,
        decoder_pad_mask=decoder_pad_mask,
    )
    zero_logits = model.decode(
        decoder_input_ids,
        decoder_var_ids,
        zero_memory,
        zero_latent,
        decoder_pad_mask=decoder_pad_mask,
    )

    valid_mask = targets.ne(pad_id)
    safe_targets = targets.masked_fill(~valid_mask, 0)
    true_gold_logits = torch.gather(
        true_logits,
        dim=-1,
        index=safe_targets.unsqueeze(-1),
    ).squeeze(-1)
    shuffled_gold_logits = torch.gather(
        shuffled_logits,
        dim=-1,
        index=safe_targets.unsqueeze(-1),
    ).squeeze(-1)
    zero_gold_logits = torch.gather(
        zero_logits,
        dim=-1,
        index=safe_targets.unsqueeze(-1),
    ).squeeze(-1)

    gold_valid_mask = torch.gather(
        candidate_masks.to(dtype=torch.bool),
        dim=-1,
        index=safe_targets.unsqueeze(-1),
    ).squeeze(-1)
    valid_mask = valid_mask & gold_valid_mask
    weighted_mask = position_weights * valid_mask.to(dtype=true_gold_logits.dtype)
    shuffled_margin_loss = F.relu(float(margin) - (true_gold_logits - shuffled_gold_logits))
    zero_margin_loss = F.relu(float(margin) - (true_gold_logits - zero_gold_logits))
    rank_margin_loss = _weighted_token_mean(
        0.5 * (shuffled_margin_loss + zero_margin_loss),
        weighted_mask,
    )

    gold_one_hot = F.one_hot(safe_targets, num_classes=true_logits.size(-1)).to(dtype=torch.bool)
    alt_candidate_masks = candidate_masks.to(dtype=torch.bool) & (~gold_one_hot)
    alt_exists = alt_candidate_masks.any(dim=-1)
    alt_weighted_mask = weighted_mask * alt_exists.to(dtype=weighted_mask.dtype)

    shuffled_best_other_logits = shuffled_logits.masked_fill(~alt_candidate_masks, float("-inf")).amax(dim=-1)
    zero_best_other_logits = zero_logits.masked_fill(~alt_candidate_masks, float("-inf")).amax(dim=-1)
    shuffled_best_other_logits = torch.where(alt_exists, shuffled_best_other_logits, shuffled_gold_logits)
    zero_best_other_logits = torch.where(alt_exists, zero_best_other_logits, zero_gold_logits)

    # Wrong latent should demote the gold token enough that another legal token becomes top-1.
    shuffled_top1_drop_loss = F.relu(
        float(wrong_top1_margin) - (shuffled_best_other_logits - shuffled_gold_logits)
    )
    zero_top1_drop_loss = F.relu(
        float(wrong_top1_margin) - (zero_best_other_logits - zero_gold_logits)
    )
    top1_drop_loss = _weighted_token_mean(
        0.5 * (shuffled_top1_drop_loss + zero_top1_drop_loss),
        alt_weighted_mask,
    )
    total_loss = rank_margin_loss + top1_drop_loss
    return total_loss, rank_margin_loss, top1_drop_loss


def soft_infonce_loss(z: torch.Tensor, costs: torch.Tensor, cost_mask: torch.Tensor, tau: float) -> torch.Tensor:
    valid_idx = torch.nonzero(cost_mask, as_tuple=False).flatten()
    if valid_idx.numel() < 2:
        return z.new_tensor(0.0)

    z = F.normalize(z[valid_idx], dim=-1)
    y = costs[valid_idx]
    sim = z @ z.t() / max(float(tau), 1e-6)
    eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)

    with torch.no_grad():
        dist = torch.abs(y[:, None] - y[None, :])
        weights = torch.exp(-dist / dist.mean().clamp_min(1e-6))
        weights.fill_diagonal_(0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    log_probs = F.log_softmax(sim.masked_fill(eye, float("-inf")), dim=-1)
    log_probs = log_probs.masked_fill(eye, 0.0)
    loss = -(weights * log_probs).sum(dim=-1).mean()
    return loss


def ordered_infonce_loss(
    z: torch.Tensor,
    costs: torch.Tensor,
    cost_mask: torch.Tensor,
    tau: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    valid_idx = torch.nonzero(cost_mask, as_tuple=False).flatten()
    if valid_idx.numel() < 2:
        return z.new_tensor(0.0)

    z = z[valid_idx]
    cost = costs[valid_idx]
    assert z.dim() == 2
    assert cost.dim() == 1 and cost.shape[0] == z.shape[0]
    B = z.shape[0]
    device = z.device

    # Normalize embeddings -> cosine sim
    z = F.normalize(z, p=2, dim=1, eps=eps)
    sim = (z @ z.t()) / max(float(tau), eps)  # (B,B)

    # Pairwise masks by cost order
    c_i = cost[:, None]        # (B,1)
    c_j = cost[None, :]        # (1,B)
    same = torch.eye(B, device=device, dtype=torch.bool)

    pos_mask = (c_j > c_i) & (~same)   # (B,B)
    neg_mask = (c_j < c_i) & (~same)   # (B,B)

    # Automatic margin for negatives
    scale = cost.std(unbiased=False).clamp_min(eps)
    delta = (c_i - c_j).clamp_min(0.0) / scale  # (B,B)
    neg_sim = sim - delta

    neg_inf = torch.tensor(-float("inf"), device=device, dtype=sim.dtype)

    pos_logits = sim.masked_fill(~pos_mask, neg_inf)          # (B,B)
    neg_logits = neg_sim.masked_fill(~neg_mask, neg_inf)      # (B,B)

    num = torch.logsumexp(pos_logits, dim=1)                  # (B,)
    den = torch.logsumexp(torch.cat([pos_logits, neg_logits], dim=1), dim=1)  # (B,)

    # anchors that have at least one pos and one neg in batch
    has_pos = pos_mask.any(dim=1)
    has_neg = neg_mask.any(dim=1)
    valid = has_pos & has_neg

    # If no valid anchors (e.g., tiny batch), return 0 with grad
    if not valid.any():
        return z.sum() * 0.0

    loss = -(num - den)
    return loss[valid].mean()



def _build_teacher_forcing_candidate_masks(
    batch: Dict[str, object],
    registry: GeneratorRegistry,
    tokenizer: ParamTokenizer,
    device: torch.device,
    debug_invalid_step: bool = True,
) -> torch.Tensor:
    masks = batch.get("candidate_masks")
    if masks is not None:
        return masks

    target_ids: torch.Tensor = batch["target_ids"]
    ordered_names: List[List[str]] = batch["ordered_param_names"]
    ordered_values: List[List[int]] = batch["ordered_param_values"]
    task_indices: List[int] = batch["task_indices"]
    sketch_indices: List[int] = batch["sketch_indices"]
    workload_keys: List[str] = batch["workload_keys"]
    target_kinds: List[str] = batch["target_kinds"]
    sample_ids: List[str] = batch["sample_ids"]

    bsz, max_len = target_ids.shape
    vocab_size = len(tokenizer.id_to_token)
    masks = torch.zeros((bsz, max_len, vocab_size), dtype=torch.bool, device=device)

    for i in range(bsz):
        oracle = registry.build_oracle(
            task_index=task_indices[i],
            sketch_index=sketch_indices[i],
            workload_key=workload_keys[i],
            target_kind=target_kinds[i],
        )
        names = ordered_names[i]
        values = ordered_values[i]

        for t, (name, value) in enumerate(zip(names, values)):
            try:
                candidates = oracle.candidate_values(name)
                masks[i, t] = tokenizer.candidate_mask_from_values(name, candidates, device=device)

                gold_token = tokenizer.value_to_token(name, value)
                gold_id = tokenizer.token_to_id.get(gold_token, tokenizer.unk_id)
                if debug_invalid_step and not masks[i, t, gold_id]:
                    print(
                        f"[invalid-step] sample={sample_ids[i]} step={t} var={name} "
                        f"gold={value} candidates={candidates}"
                    )
                    raise ValueError("gold value is outside oracle candidates")

                oracle.assign(name, value)
            except Exception:  # pylint: disable=broad-except
                gold_token = tokenizer.value_to_token(name, value)
                gold_id = tokenizer.token_to_id.get(gold_token, tokenizer.unk_id)
                masks[i, t] = torch.zeros(vocab_size, dtype=torch.bool, device=device)
                masks[i, t, gold_id] = True
                for rem_t, rem_name, rem_value in zip(range(t + 1, len(names)), names[t + 1:], values[t + 1:]):
                    rem_gold_token = tokenizer.value_to_token(rem_name, rem_value)
                    rem_gold_id = tokenizer.token_to_id.get(rem_gold_token, tokenizer.unk_id)
                    masks[i, rem_t] = torch.zeros(vocab_size, dtype=torch.bool, device=device)
                    masks[i, rem_t, rem_gold_id] = True
                break

        for t in range(len(names), max_len):
            masks[i, t] = tokenizer.pad_only_mask(device=device)

    return masks


def _build_early_param_position_weights(
    seq_lens: torch.Tensor,
    max_len: int,
    max_weight: float,
    power: float,
    device: torch.device,
) -> torch.Tensor:
    weights = torch.ones((int(seq_lens.shape[0]), int(max_len)), dtype=torch.float32, device=device)
    if max_weight <= 1.0:
        return weights

    for sample_idx in range(int(seq_lens.shape[0])):
        seq_len = int(seq_lens[sample_idx].item())
        if seq_len <= 0:
            continue
        if seq_len == 1:
            weights[sample_idx, 0] = float(max_weight)
            continue

        pos = torch.arange(seq_len, dtype=torch.float32, device=device)
        normalized = 1.0 - pos / float(seq_len - 1)
        sample_weights = 1.0 + (float(max_weight) - 1.0) * torch.pow(normalized, float(power))
        weights[sample_idx, :seq_len] = sample_weights

    return weights


def _batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved = {}
    non_blocking = device.type == "cuda"
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=non_blocking)
        else:
            moved[key] = value
    return moved


def _prepare_loader(
    dataset,
    tokenizer: ParamTokenizer,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_prepared_samples(batch, tokenizer),
    )
    if num_workers > 0:
        kwargs["pin_memory"] = bool(pin_memory)
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = int(prefetch_factor)
    else:
        kwargs["pin_memory"] = bool(pin_memory)
    return DataLoader(**kwargs)


def save_checkpoint(
    path: str | Path,
    *,
    model: LatentParamVAE,
    optimizer,
    scheduler,
    epoch: int,
    best_exact_match: float,
    config,
    tokenizer: ParamTokenizer,
    latent_cost_ridge: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": None if scheduler is None else scheduler.state_dict(),
            "epoch": int(epoch),
            "best_exact_match": float(best_exact_match),
            "config": config.to_dict(),
            "tokenizer": tokenizer.to_state_dict(),
            "latent_cost_ridge": latent_cost_ridge,
        },
        path,
    )


def load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None) -> dict:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    return payload


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
    total_batches = 0

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc=f"train epoch {epoch}")

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
        position_weights = _build_early_param_position_weights(
            batch["seq_lens"],
            max_len=int(batch["target_ids"].shape[1]),
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
                batch["decoder_input_ids"],
                batch["decoder_var_ids"],
                pad_token_id=tokenizer.pad_id,
            )
            recon_loss = masked_cross_entropy(
                out.logits,
                batch["target_ids"],
                candidate_masks,
                tokenizer.pad_id,
                position_weights=position_weights,
            )
            kl_loss = kl_divergence(out.mu, out.logvar)
            cost_loss = weighted_cost_loss(out.cost_pred, batch["costs"], batch["cost_mask"])
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
            if cfg.train.order_nce:
                nce_loss = ordered_infonce_loss(out.z, batch["costs"], batch["cost_mask"], cfg.train.tau_nce)
            else:
                nce_loss = soft_infonce_loss(out.z, batch["costs"], batch["cost_mask"], cfg.train.tau_nce)
            loss = (
                recon_loss
                + beta * kl_loss
                + float(cfg.train.lambda_cost) * cost_loss
                + float(cfg.train.lambda_nce) * nce_loss
                + float(getattr(cfg.train, "lambda_latent_use", 0.0)) * latent_use_loss
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


@torch.no_grad()
def evaluate_autoregressive(
    model: LatentParamVAE,
    dataset,
    registry: GeneratorRegistry,
    tokenizer: ParamTokenizer,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, float]:
    model.eval()

    token_correct = 0
    token_total = 0
    exact_match = 0

    samples = list(dataset.samples)
    total_batches = (len(samples) + max(int(batch_size), 1) - 1) // max(int(batch_size), 1)
    print(f"[eval] validation batches")
    for start in tqdm(range(0, len(samples), max(int(batch_size), 1)), desc=f"eval batches", total=total_batches):
        batch_samples = samples[start:start + max(int(batch_size), 1)]
        results = greedy_decode_batch(model, batch_samples, registry, tokenizer, device)
        batch_token_correct = 0
        batch_token_total = 0
        batch_exact_match = 0
        for sample, result in zip(batch_samples, results):
            pred_values = [result.predicted_param_dict[name] for name in sample.ordered_param_names]
            gold_values = list(sample.ordered_param_values)

            for pred, gold in zip(pred_values, gold_values):
                token_correct += int(pred == gold)
                token_total += 1
                batch_token_correct += int(pred == gold)
                batch_token_total += 1

            is_exact = pred_values == gold_values
            exact_match += int(is_exact)
            batch_exact_match += int(is_exact)

        batch_tok_acc = batch_token_correct / max(batch_token_total, 1)
        batch_exact = batch_exact_match / max(len(batch_samples), 1)
        print(
            f"[eval-batch {start // max(int(batch_size), 1) + 1}/{total_batches}] "
            f"tok_acc={batch_tok_acc:.4f} "
            f"exact={batch_exact:.4f}"
        )

    n = max(len(dataset), 1)
    return {
        "token_accuracy": token_correct / max(token_total, 1),
        "full_sequence_exact_match": exact_match / n,
    }


@torch.no_grad()
def evaluate_cost_ranking(
    model: LatentParamVAE,
    dataset,
    tokenizer: ParamTokenizer,
    device: torch.device,
    batch_size: int = 64,
    latent_cost_ridge: dict | None = None,
) -> Dict[str, float]:
    model.eval()

    scored_by_source: Dict[str, List[tuple[float, float, str]]] = {"cost_head": []}
    ridge_weight = None
    ridge_bias = 0.0
    if latent_cost_ridge is not None:
        ridge_weight = latent_cost_ridge["weight"].to(device=device, dtype=torch.float32)
        ridge_bias = float(latent_cost_ridge.get("bias", 0.0))
        scored_by_source["cost_vec"] = []

    samples = list(dataset.samples)
    stride = max(int(batch_size), 1)

    for start in range(0, len(samples), stride):
        batch_samples = samples[start:start + stride]
        batch = collate_prepared_samples(batch_samples, tokenizer)
        enc_ids = batch["encoder_token_ids"].to(device, non_blocking=device.type == "cuda")
        enc_var_ids = batch["encoder_var_ids"].to(device, non_blocking=device.type == "cuda")
        enc_pad = enc_ids.eq(tokenizer.pad_id)
        _, _, z, _ = model.encode(enc_ids, enc_var_ids, enc_pad, deterministic=True)
        cost_head_pred = model.cost_head(z).squeeze(-1).detach().cpu().tolist()
        cost_vec_pred = None
        if ridge_weight is not None:
            cost_vec_pred = (z @ ridge_weight + ridge_bias).detach().cpu().tolist()

        for row_idx, sample in enumerate(batch_samples):
            if sample.cost is None or not math.isfinite(float(sample.cost)):
                continue
            actual_cost = float(sample.cost)
            sample_id = str(sample.sample_id)
            scored_by_source["cost_head"].append((float(cost_head_pred[row_idx]), actual_cost, sample_id))
            if cost_vec_pred is not None:
                scored_by_source["cost_vec"].append((float(cost_vec_pred[row_idx]), actual_cost, sample_id))

    if not any(scored_by_source.values()):
        return {}

    metrics: Dict[str, float] = {}
    for source_name, scored_items in scored_by_source.items():
        if not scored_items:
            continue
        pred_sorted = sorted(scored_items, key=lambda item: item[0], reverse=True)
        actual_sorted = sorted(scored_items, key=lambda item: item[1], reverse=True)

        actual_top1_id = actual_sorted[0][2]
        actual_top1_pred_rank = 1
        for rank, (_, _, sample_id) in enumerate(pred_sorted, start=1):
            if sample_id == actual_top1_id:
                actual_top1_pred_rank = rank
                break

        pred_top1_actual_cost = pred_sorted[0][1]
        topk = min(10, len(pred_sorted))
        pred_topk_mean_actual_cost = sum(item[1] for item in pred_sorted[:topk]) / topk
        metrics[f"{source_name}_actual_top1_pred_rank"] = float(actual_top1_pred_rank)
        metrics[f"{source_name}_pred_top1_actual_cost"] = float(pred_top1_actual_cost)
        metrics[f"{source_name}_pred_top10_mean_actual_cost"] = float(pred_topk_mean_actual_cost)

    return metrics


@torch.no_grad()
def fit_latent_cost_ridge(
    model: LatentParamVAE,
    dataset,
    tokenizer: ParamTokenizer,
    device: torch.device,
    alpha: float,
    batch_size: int = 128,
) -> dict | None:
    model.eval()

    latent_batches: List[torch.Tensor] = []
    cost_batches: List[torch.Tensor] = []
    samples = list(dataset.samples)
    stride = max(int(batch_size), 1)

    for start in range(0, len(samples), stride):
        batch_samples = samples[start:start + stride]
        batch = collate_prepared_samples(batch_samples, tokenizer)
        valid_mask = batch["cost_mask"]
        if not bool(valid_mask.any()):
            continue

        enc_ids = batch["encoder_token_ids"].to(device, non_blocking=device.type == "cuda")
        enc_var_ids = batch["encoder_var_ids"].to(device, non_blocking=device.type == "cuda")
        enc_pad = enc_ids.eq(tokenizer.pad_id)
        _, _, z, _ = model.encode(enc_ids, enc_var_ids, enc_pad, deterministic=True)

        valid_mask_device = valid_mask.to(device=device, non_blocking=device.type == "cuda")
        latent_batches.append(z[valid_mask_device].detach().cpu().to(dtype=torch.float64))
        cost_batches.append(batch["costs"][valid_mask].detach().cpu().to(dtype=torch.float64))

    if not latent_batches:
        return None

    x = torch.cat(latent_batches, dim=0)
    y = torch.cat(cost_batches, dim=0)
    num_samples, latent_dim = x.shape

    ones = torch.ones((num_samples, 1), dtype=torch.float64)
    design = torch.cat([x, ones], dim=1)
    reg = torch.eye(latent_dim + 1, dtype=torch.float64)
    reg[-1, -1] = 0.0
    reg = reg * float(alpha)

    lhs = design.T @ design + reg
    rhs = design.T @ y
    try:
        coeff = torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        coeff = torch.linalg.pinv(lhs) @ rhs

    weight = coeff[:-1].to(dtype=torch.float32).contiguous()
    bias = float(coeff[-1].item())
    pred = (design @ coeff).to(dtype=torch.float64)
    mse = float(torch.mean((pred - y) ** 2).item())

    return {
        "weight": weight,
        "bias": bias,
        "alpha": float(alpha),
        "num_samples": int(num_samples),
        "latent_dim": int(latent_dim),
        "feature_name": "deterministic_z",
        "target_name": "neg_log_cost",
        "train_mse": mse,
    }


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
    for record in list(bundle.train_records) + list(bundle.val_records) + list(bundle.test_records):
        if record.task_index is not None:
            return str(int(record.task_index))
    return "na"


def _build_wandb_project_name(config, bundle: DatasetBundle) -> str:
    task_index = _resolve_run_task_index(bundle)
    project_suffix = getattr(config.wandb, "project", None) or "single_v1"
    return f"Task{task_index}_{project_suffix}"


def _build_wandb_run_name(config, bundle: DatasetBundle) -> str:
    return (
        f"d{int(config.model.d_model)}"
        f"_z{int(config.model.latent_dim)}"
        f"_ztok{int(config.model.latent_token_count)}"
        f"_enc{int(config.model.num_encoder_layers)}"
        f"_dec{int(config.model.num_decoder_layers)}"
        # f"_cost{int(config.model.num_cost_layers)}"
    )


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
    if wandb_project:
        if wandb is None:
            print("[train] wandb project is set but wandb is not installed; skipping wandb logging")
        else:
            project_name = _build_wandb_project_name(config, bundle)
            run_name = _build_wandb_run_name(config, bundle)
            print(f"[train] initializing wandb: project={project_name} run={run_name}")
            wandb_run = wandb.init(
                project=project_name,
                name=run_name,
                config=config.to_dict(),
            )

    train_loader = _prepare_loader(
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
    scheduler = None
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.train.use_amp and device.type == "cuda"))

    checkpoint_dir = Path(config.train.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.save_json(checkpoint_dir / "config.json")
    (checkpoint_dir / "tokenizer.json").write_text(
        json.dumps(tokenizer.to_state_dict(), indent=2),
        encoding="utf-8",
    )
    print(f"[train] checkpoint dir: {checkpoint_dir}")

    start_epoch = 1
    best_exact_match = float("-inf")
    best_val_acc = float("-inf")
    best_recon_loss = float("inf")

    if config.train.resume_from:
        print(f"[train] resuming from {config.train.resume_from}")
        payload = load_checkpoint(config.train.resume_from, model, optimizer, scheduler)
        start_epoch = int(payload["epoch"]) + 1
        best_exact_match = float(payload.get("best_exact_match", best_exact_match))

    best_metrics: Dict[str, float] = {}
    latent_cost_ridge = None

    timestamp = time.strftime("%m%d%H%M")

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
        
        summary = {**train_metrics}
        print(
            f"[epoch {epoch}] "
            f"loss={summary['loss']:.4f} recon={summary['recon_loss']:.4f} "
            f"kl={summary['kl_loss']:.4f} "
        )
        if epoch == config.train.num_epochs:
            print(f"[train] evaluating validation split after epoch {epoch}")
            if bool(getattr(config.train, "cost_ridge_vec", False)):
                print("[train] fitting latent cost ridge on train split")
                latent_cost_ridge = fit_latent_cost_ridge(
                    model,
                    bundle.train_dataset,
                    tokenizer,
                    device,
                    alpha=config.train.ridge_alpha,
                    batch_size=config.eval.batch_size,
                )
                if latent_cost_ridge is not None:
                    print(
                        f"[train] latent cost ridge ready: samples={latent_cost_ridge['num_samples']} "
                        f"alpha={latent_cost_ridge['alpha']:.2e} "
                        f"train_mse={latent_cost_ridge['train_mse']:.6f}"
                    )
                    summary["train_ridge_mse"] = float(latent_cost_ridge["train_mse"])

            cost_metrics = {}
            if (
                float(config.train.lambda_cost) > 0.0
                or float(config.train.lambda_nce) > 0.0
                or latent_cost_ridge is not None
            ):
                print("[train] evaluating validation cost ranking before decode")
                cost_metrics = evaluate_cost_ranking(
                    model,
                    bundle.val_dataset,
                    tokenizer,
                    device,
                    batch_size=config.eval.batch_size,
                    latent_cost_ridge=latent_cost_ridge,
                )
                if "cost_head_actual_top1_pred_rank" in cost_metrics:
                    print(
                        f"val_cost_head_actual_top1_pred_rank : {int(cost_metrics['cost_head_actual_top1_pred_rank'])}\n"
                        f"val_cost_head_pred_top1_actual_cost : {cost_metrics['cost_head_pred_top1_actual_cost']:.6f}\n"
                        f"val_cost_head_pred_top10_mean_actual_cost : {cost_metrics['cost_head_pred_top10_mean_actual_cost']:.6f}\n"
                    )
                if "cost_vec_actual_top1_pred_rank" in cost_metrics:
                    print(
                        f"\nval_cost_vec_actual_top1_pred_rank : {int(cost_metrics['cost_vec_actual_top1_pred_rank'])}\n"
                        f"val_cost_vec_pred_top1_actual_cost : {cost_metrics['cost_vec_pred_top1_actual_cost']:.6f}\n"
                        f"val_cost_vec_pred_top10_mean_actual_cost : {cost_metrics['cost_vec_pred_top10_mean_actual_cost']:.6f}\n"
                    )

            val_metrics = evaluate_autoregressive(
                model,
                bundle.val_dataset,
                registry,
                tokenizer,
                device,
                batch_size=config.eval.batch_size,
            )

            summary = {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
            if cost_metrics:
                summary.update({f"val_{k}": v for k, v in cost_metrics.items()})
            print(
                f"val_tok_acc={summary['val_token_accuracy']:.4f} "
                f"val_exact={summary['val_full_sequence_exact_match']:.4f}"
            )


        
        if "val_full_sequence_exact_match" in summary:
            best_exact_match = max(best_exact_match, float(summary["val_full_sequence_exact_match"]))
        if "val_token_accuracy" in summary:
            best_val_acc = max(best_val_acc, float(summary["val_token_accuracy"]))
        if wandb_run is not None:
            wandb.log({"epoch": epoch, **summary}, step=epoch)
        

        # if config.train.save_every_epoch:
        #     summary = {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
        #     val_metrics = evaluate_autoregressive(
        #         model,
        #         bundle.val_dataset,
        #         registry,
        #         tokenizer,
        #         device,
        #         batch_size=config.eval.batch_size,
        #     )
        #     save_checkpoint(
        #         checkpoint_dir / "last.pt",
        #         model=model,
        #         optimizer=optimizer,
        #         scheduler=scheduler,
        #         epoch=epoch,
        #         best_exact_match=max(best_exact_match, summary["val_full_sequence_exact_match"]),
        #         config=config,
        #         tokenizer=tokenizer,
        #     )

        
        if epoch == config.train.num_epochs:
            best_recon_loss = summary["recon_loss"]
            best_metrics = dict(summary)
            checkpoint_kwargs = dict(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_exact_match=best_exact_match,
                config=config,
                tokenizer=tokenizer,
                latent_cost_ridge=latent_cost_ridge,
            )
            save_checkpoint(
                checkpoint_dir / f"last_{timestamp}_acc_{summary['val_token_accuracy']:.2f}.pt",
                **checkpoint_kwargs,
            )
            save_checkpoint(checkpoint_dir / "last.pt", **checkpoint_kwargs)
            # save_checkpoint(checkpoint_dir / "best.pt", **checkpoint_kwargs)

    final_metrics = dict(best_metrics)
    if bundle.test_dataset.samples:
        print("[train] evaluating test split")
        test_metrics = evaluate_autoregressive(
            model,
            bundle.test_dataset,
            registry,
            tokenizer,
            device,
            batch_size=config.eval.batch_size,
        )
        final_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
    print("[final]", json.dumps(final_metrics, indent=2))

    if bundle.val_dataset.samples:
        sample = bundle.val_dataset.samples[0]
        decoded = greedy_decode_sample(model, sample, registry, tokenizer, device)
        print(pretty_print_reconstruction(sample, decoded))
    elif bundle.test_dataset.samples:
        sample = bundle.test_dataset.samples[0]
        decoded = greedy_decode_sample(model, sample, registry, tokenizer, device)
        print(pretty_print_reconstruction(sample, decoded))

    if wandb_run is not None:
        wandb_run.summary.update(final_metrics)
        wandb_run.finish()

    return final_metrics
