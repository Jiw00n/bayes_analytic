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


def pairwise_rank_loss(
    cost_pred: torch.Tensor,
    costs: torch.Tensor,
    cost_mask: torch.Tensor,
    method: str = "ranknet",
    margin: float = 1.0,
    margin_mode: str = "fixed",
    margin_min: float | None = None,
    margin_max: float | None = None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pairwise rank loss on (pred, actual) pairs.

    A pair (i, j) contributes when actual[i] > actual[j]; the loss pushes
    pred[i] above pred[j]. ``costs`` and ``cost_pred`` must be in a space
    where "higher is better" (same ordering convention as the rest of the
    training code — see ``cost_target`` docstring).

    - ``method="ranknet"``: softplus(-(pred[i] - pred[j])) (logistic).
      ``margin`` / ``margin_mode`` are ignored.
    - ``method="hinge"``: relu(m_ij - (pred[i] - pred[j])), where
      ``m_ij`` depends on ``margin_mode``:
        * ``"fixed"``   → m_ij = margin (constant)
        * ``"pair_gap"`` → m_ij = margin * (actual[i] - actual[j])  (the
          required prediction gap scales with the true cost gap, so
          pairs that are far apart in cost must also be far apart in
          prediction)
      In ``pair_gap`` mode the per-pair margin is optionally clamped to
      ``[margin_min, margin_max]`` (each bound is ignored when None).
      In ``fixed`` mode these clamps are ignored.
    """
    if cost_mask.sum() < 2:
        return cost_pred.new_tensor(0.0)
    preds = cost_pred[cost_mask]
    actuals = costs[cost_mask]

    pred_diff = preds.unsqueeze(1) - preds.unsqueeze(0)  # [N, N]
    actual_diff = actuals.unsqueeze(1) - actuals.unsqueeze(0)
    pair_mask = actual_diff > 0
    if not bool(pair_mask.any()):
        return cost_pred.new_tensor(0.0)

    method_lower = method.lower()
    if method_lower == "ranknet":
        pair_losses = F.softplus(-pred_diff)
    elif method_lower == "hinge":
        mode_lower = margin_mode.lower()
        if mode_lower == "fixed":
            effective_margin = float(margin)
        elif mode_lower == "pair_gap":
            # ``actual_diff`` on off-diagonal entries where pair_mask is
            # False will be non-positive; pair_mask filters them out of
            # the final sum, so their negative margins never contribute.
            effective_margin = float(margin) * actual_diff
            if margin_min is not None or margin_max is not None:
                clamp_min = float(margin_min) if margin_min is not None else None
                clamp_max = float(margin_max) if margin_max is not None else None
                effective_margin = effective_margin.clamp(min=clamp_min, max=clamp_max)
        else:
            raise ValueError(
                f"Unknown cost_rank_margin_mode: {margin_mode!r} "
                "(expected 'fixed' or 'pair_gap')"
            )
        pair_losses = F.relu(effective_margin - pred_diff)
    else:
        raise ValueError(
            f"Unknown cost_rank_method: {method!r} (expected 'ranknet' or 'hinge')"
        )

    pair_weights = pair_mask.to(pair_losses.dtype)
    if sample_weights is not None:
        sw = sample_weights[cost_mask].to(pair_losses.dtype)
        pair_weights = pair_weights * sw.unsqueeze(1) * sw.unsqueeze(0)
    denom = pair_weights.sum().clamp_min(1e-8)
    return (pair_losses * pair_weights).sum() / denom


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


def soft_infonce_loss(
    z: torch.Tensor,
    costs: torch.Tensor,
    cost_mask: torch.Tensor,
    tau: float,
    sample_weights: torch.Tensor | None = None,
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

    with torch.no_grad():
        dist = torch.abs(y[:, None] - y[None, :])
        weights = torch.exp(-dist / dist.mean().clamp_min(1e-6))
        weights.fill_diagonal_(0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    log_probs = F.log_softmax(sim.masked_fill(eye, float("-inf")), dim=-1)
    log_probs = log_probs.masked_fill(eye, 0.0)
    per_anchor = -(weights * log_probs).sum(dim=-1)
    if sw is not None:
        return (per_anchor * sw).sum() / sw.sum().clamp_min(1e-8)
    return per_anchor.mean()


def ordered_infonce_loss(
    z: torch.Tensor,
    costs: torch.Tensor,
    cost_mask: torch.Tensor,
    tau: float,
    eps: float = 1e-12,
    sample_weights: torch.Tensor | None = None,
    pos_weight_by_percentile: bool = False,
    pos_weight_sigma: float = 0.2,
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
