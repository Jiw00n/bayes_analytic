from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .model import LatentParamVAE


def beta_by_epoch(cfg, epoch: int) -> float:
    if cfg.beta_warmup_epochs <= 0:
        return float(cfg.beta_end)
    progress = min(max(epoch / cfg.beta_warmup_epochs, 0.0), 1.0)
    return float(cfg.beta_start + (cfg.beta_end - cfg.beta_start) * progress)


def compute_cobo_sample_weights(
    costs: torch.Tensor,
    cost_mask: torch.Tensor,
    quantile: float = 0.95,
    sigma: float = 0.5,
) -> torch.Tensor:
    """CoBO-style CDF weighting: λ(y) = Φ((y − y_q) / σ_abs).

    Higher cost (= faster kernel) gets weight closer to 1,
    lower cost gets weight closer to 0.

    ``sigma`` is a multiplier of the cost standard deviation, so the
    transition smoothness adapts to the actual cost distribution.
    Samples without valid cost receive a neutral weight of 1.
    """
    valid_costs = costs[cost_mask]
    if valid_costs.numel() < 2:
        return torch.ones_like(costs)
    y_q = torch.quantile(valid_costs.float(), float(quantile))
    sigma_abs = valid_costs.float().std().clamp_min(1e-6) * max(float(sigma), 1e-8)
    z_scores = (costs - y_q) / sigma_abs
    weights = 0.5 * (1.0 + torch.erf(z_scores / math.sqrt(2.0)))
    weights = weights.clamp(min=0.01)
    # normalise so valid weights average to 1 (keep loss scale stable)
    valid_mean = weights[cost_mask].mean().clamp_min(1e-8)
    weights = weights / valid_mean
    # invalid-cost samples → neutral weight 1
    weights = torch.where(cost_mask, weights, torch.ones_like(weights))
    return weights.detach()


def kl_divergence(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    per_sample = 0.5 * torch.sum(torch.exp(logvar) + mu * mu - 1.0 - logvar, dim=-1)
    if sample_weights is None:
        return per_sample.mean()
    sw = sample_weights.to(dtype=per_sample.dtype, device=per_sample.device)
    return (per_sample * sw).sum() / sw.sum().clamp_min(1e-8)


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    candidate_masks: torch.Tensor,
    pad_id: int,
    position_weights: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    masked_logits = logits.masked_fill(~candidate_masks, float("-inf"))

    if label_smoothing > 0.0:
        # Custom label smoothing: distribute smoothing mass only over valid
        # candidates (not the -inf masked ones).
        #   loss = (1 - eps) * NLL(target) + eps * mean_valid(-log p_j)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        nll_target = F.nll_loss(
            log_probs.reshape(-1, log_probs.size(-1)),
            targets.reshape(-1),
            ignore_index=pad_id,
            reduction="none",
        ).view_as(targets)
        # Mean NLL over valid candidates per position
        valid_log_probs = log_probs.masked_fill(~candidate_masks, 0.0)
        num_valid = candidate_masks.float().sum(dim=-1).clamp_min(1.0)
        mean_nll_valid = -valid_log_probs.sum(dim=-1) / num_valid
        token_losses = (1.0 - label_smoothing) * nll_target + label_smoothing * mean_nll_valid
    else:
        token_losses = F.cross_entropy(
            masked_logits.reshape(-1, masked_logits.size(-1)),
            targets.reshape(-1),
            ignore_index=pad_id,
            reduction="none",
        ).view_as(targets)

    valid_mask = targets.ne(pad_id)
    if position_weights is None:
        position_weights = torch.ones_like(token_losses)
    if sample_weights is not None:
        # sample_weights: [B] → [B, 1] broadcast over sequence length
        position_weights = position_weights * sample_weights.unsqueeze(1)
    weighted_mask = position_weights * valid_mask.to(dtype=token_losses.dtype)
    return (token_losses * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)


def weighted_cost_loss(
    cost_pred: torch.Tensor,
    costs: torch.Tensor,
    cost_mask: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if not cost_mask.any():
        return cost_pred.new_tensor(0.0)
    if sample_weights is not None:
        sq_err = (cost_pred[cost_mask] - costs[cost_mask]) ** 2
        sw = sample_weights[cost_mask]
        return (sq_err * sw).sum() / sw.sum().clamp_min(1e-8)
    return F.mse_loss(cost_pred[cost_mask], costs[cost_mask])


def soft_infonce_loss(
    z: torch.Tensor,
    costs: torch.Tensor,
    cost_mask: torch.Tensor,
    tau: float,
    sample_weights: torch.Tensor | None = None,
    task_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    valid_idx = torch.nonzero(cost_mask, as_tuple=False).flatten()
    if valid_idx.numel() < 2:
        return z.new_tensor(0.0)

    sw = None
    if sample_weights is not None:
        sw = sample_weights[valid_idx]

    z = F.normalize(z[valid_idx], dim=-1)
    y = costs[valid_idx]
    sim = z @ z.t() / max(float(tau), 1e-6)
    eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)

    if task_ids is not None:
        tid = task_ids[valid_idx]
        same_task = (tid[:, None] == tid[None, :]) & (tid[:, None] >= 0)
        pair_mask = same_task & (~eye)
    else:
        pair_mask = ~eye

    with torch.no_grad():
        dist = torch.abs(y[:, None] - y[None, :])
        weights = torch.exp(-dist / dist.mean().clamp_min(1e-6))
        weights = weights * pair_mask.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    log_probs = F.log_softmax(sim.masked_fill(~pair_mask, float("-inf")), dim=-1)
    log_probs = log_probs.masked_fill(~pair_mask, 0.0)
    per_anchor = -(weights * log_probs).sum(dim=-1)
    has_pair = pair_mask.any(dim=-1)
    if not has_pair.any():
        return z.sum() * 0.0
    if sw is not None:
        sw_eff = sw * has_pair.to(sw.dtype)
        return (per_anchor * sw_eff).sum() / sw_eff.sum().clamp_min(1e-8)
    return per_anchor[has_pair].mean()


def ordered_infonce_loss(
    z: torch.Tensor,
    costs: torch.Tensor,
    cost_mask: torch.Tensor,
    tau: float,
    eps: float = 1e-12,
    sample_weights: torch.Tensor | None = None,
    pos_weight_by_percentile: bool = False,
    pos_weight_sigma: float = 0.2,
    task_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    valid_idx = torch.nonzero(cost_mask, as_tuple=False).flatten()
    if valid_idx.numel() < 2:
        return z.new_tensor(0.0)

    sw = None
    if sample_weights is not None:
        sw = sample_weights[valid_idx]

    z = z[valid_idx]
    cost = costs[valid_idx]
    z = F.normalize(z, p=2, dim=1, eps=eps)
    sim = (z @ z.t()) / max(float(tau), eps)

    c_i = cost[:, None]
    c_j = cost[None, :]
    same = torch.eye(z.shape[0], device=z.device, dtype=torch.bool)

    pos_mask = (c_j > c_i) & (~same)
    neg_mask = (c_j < c_i) & (~same)

    if task_ids is not None:
        tid = task_ids[valid_idx]
        same_task = (tid[:, None] == tid[None, :]) & (tid[:, None] >= 0)
        pos_mask = pos_mask & same_task
        neg_mask = neg_mask & same_task

    scale = cost.std(unbiased=False).clamp_min(eps)
    delta = (c_i - c_j).clamp_min(0.0) / scale
    neg_sim = sim - delta

    neg_inf = torch.tensor(-float("inf"), device=sim.device, dtype=sim.dtype)
    if pos_weight_by_percentile:
        n = cost.shape[0]
        order_idx = torch.argsort(cost)
        ranks = torch.empty_like(cost)
        ranks[order_idx] = torch.arange(n, device=cost.device, dtype=cost.dtype)
        pct = ranks / max(n - 1, 1)
        dp = (pct[None, :] - pct[:, None]).clamp_min(0.0)
        sigma = max(float(pos_weight_sigma), eps)
        pos_score = (dp / sigma).masked_fill(~pos_mask, neg_inf)
        log_pos_w = pos_score - torch.logsumexp(pos_score, dim=1, keepdim=True)
        pos_logits = (sim + log_pos_w).masked_fill(~pos_mask, neg_inf)
    else:
        pos_logits = sim.masked_fill(~pos_mask, neg_inf)
    neg_logits = neg_sim.masked_fill(~neg_mask, neg_inf)

    num = torch.logsumexp(pos_logits, dim=1)
    den = torch.logsumexp(torch.cat([pos_logits, neg_logits], dim=1), dim=1)

    valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    if not valid.any():
        return z.sum() * 0.0

    loss = -(num - den)
    if sw is not None:
        sw_valid = sw[valid]
        return (loss[valid] * sw_valid).sum() / sw_valid.sum().clamp_min(1e-8)
    return loss[valid].mean()
