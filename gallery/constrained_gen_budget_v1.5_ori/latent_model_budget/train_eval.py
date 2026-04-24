from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from .adapter import GeneratorRegistry
from .dataset import collate_prepared_samples
from .inference import greedy_decode_batch
from .model import LatentParamVAE
from .recon_predict_gp import fit_gp_recon_predictor
from .recon_predict_lgbm import fit_lgbm_ranker_recon_predictor
from .tokenizer import ParamTokenizer


VALID_REENCODE_PREDICTOR_NAMES = (
    "cost_head",
    "cost_vec",
    "cost_vec_weighted",
    "gp",
    "lightgbm_ranker",
)


def _build_reencode_predictor(
    *,
    name: str,
    model,
    bundle,
    tokenizer,
    device,
    latent_cost_ridges,
    config,
):
    """Construct a re-encode predictor matching the configured name.

    Training set for gp / lightgbm_ranker is the top-K + random-N training
    subset only — walk-buffer samples are NOT included here (intentionally
    different from the recon-predict GP which does include them).

    - cost_head: returns None (tune_by_latent falls back to model.cost_head).
    - cost_vec / cost_vec_weighted: wraps the corresponding ridge payload.
    - gp: fits a fresh GP on the training subset.
    - lightgbm_ranker: fits an LGBMRanker on the same selection as GP.
    """
    import sys as _sys
    from pathlib import Path as _Path

    _here = _Path(__file__).resolve().parent.parent
    if str(_here) not in _sys.path:
        _sys.path.insert(0, str(_here))
    from tune_by_latent import RidgeReEncodePredictor  # lazy import to avoid cycles

    name = (name or "cost_head").lower()
    if name not in VALID_REENCODE_PREDICTOR_NAMES:
        print(
            f"[reencode] unknown re_encode_predictor={name!r}; "
            f"falling back to cost_head"
        )
        name = "cost_head"

    if name == "cost_head":
        return None
    if name in ("cost_vec", "cost_vec_weighted"):
        want_weighted = name == "cost_vec_weighted"
        payload = next(
            (
                p for p in (latent_cost_ridges or [])
                if bool(p.get("weighted", False)) == want_weighted
            ),
            None,
        )
        if payload is None or "weight" not in payload:
            print(
                f"[reencode] {name} ridge not fitted yet; "
                f"falling back to cost_head"
            )
            return None
        weight = payload["weight"].detach().to(dtype=torch.float32, device=device)
        bias = float(payload.get("bias", 0.0))
        fit_target = str(payload.get("target_name", "neg_log"))
        output_target = str(payload.get("cost_target", fit_target))
        tmc = payload.get("task_min_cost")
        return RidgeReEncodePredictor(
            weight=weight,
            bias=bias,
            name=name,
            fit_target=fit_target,
            output_target=output_target,
            task_min_cost=float(tmc) if tmc is not None else None,
        )
    if name == "gp":
        return fit_gp_recon_predictor(
            model=model,
            dataset=bundle.train_dataset,
            tokenizer=tokenizer,
            device=device,
            top_k=int(getattr(config.train, "latent_walk_predict_gp_top_k", 800)),
            random_n=int(getattr(config.train, "latent_walk_predict_gp_random_n", 200)),
            batch_size=config.eval.batch_size,
            seed=int(getattr(config.data, "seed", 0)),
            walk_buffer=None,
        )
    if name == "lightgbm_ranker":
        return fit_lgbm_ranker_recon_predictor(
            model=model,
            dataset=bundle.train_dataset,
            tokenizer=tokenizer,
            device=device,
            top_k=int(getattr(config.train, "latent_walk_predict_gp_top_k", 800)),
            random_n=int(getattr(config.train, "latent_walk_predict_gp_random_n", 200)),
            batch_size=config.eval.batch_size,
            seed=int(getattr(config.data, "seed", 0)),
            walk_buffer=None,
        )
    return None


def _build_singleton_position_mask(
    targets: torch.Tensor,
    candidate_masks: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    valid_mask = targets.ne(int(pad_id))
    singleton_mask = candidate_masks.to(dtype=torch.bool).sum(dim=-1).eq(1)
    return singleton_mask & valid_mask


def _compress_teacher_forcing_batch(
    batch: Dict[str, object],
    candidate_masks: torch.Tensor,
    tokenizer: ParamTokenizer,
) -> Dict[str, torch.Tensor]:
    """Legacy compression: drop PAD and singleton positions per row, then
    re-pad each row up to the batch's longest kept length with PAD.

    Selectable via ``TrainConfig.use_compressed_teacher_forcing`` for parity
    with pre-1964b7a behaviour. The default training path keeps singletons in
    place and zeros their loss instead.
    """
    target_ids: torch.Tensor = batch["target_ids"]
    decoder_input_ids: torch.Tensor = batch["decoder_input_ids"]
    decoder_var_ids: torch.Tensor = batch["decoder_var_ids"]

    keep_mask = target_ids.ne(tokenizer.pad_id) & (
        ~_build_singleton_position_mask(target_ids, candidate_masks, tokenizer.pad_id)
    )
    seq_lens = keep_mask.sum(dim=-1)
    batch_size = int(target_ids.shape[0])
    vocab_size = int(candidate_masks.shape[-1])
    max_len = max(int(seq_lens.max().item()), 1)

    new_decoder_input_ids = torch.full(
        (batch_size, max_len),
        tokenizer.pad_id,
        dtype=decoder_input_ids.dtype,
        device=decoder_input_ids.device,
    )
    new_decoder_var_ids = torch.full(
        (batch_size, max_len),
        tokenizer.var_pad_id,
        dtype=decoder_var_ids.dtype,
        device=decoder_var_ids.device,
    )
    new_target_ids = torch.full(
        (batch_size, max_len),
        tokenizer.pad_id,
        dtype=target_ids.dtype,
        device=target_ids.device,
    )
    new_candidate_masks = torch.zeros(
        (batch_size, max_len, vocab_size),
        dtype=torch.bool,
        device=candidate_masks.device,
    )
    new_candidate_masks[:, :, tokenizer.pad_id] = True

    for row_idx in range(batch_size):
        keep_indices = torch.nonzero(keep_mask[row_idx], as_tuple=False).flatten()
        if int(keep_indices.numel()) <= 0:
            continue
        length = int(keep_indices.numel())
        new_decoder_input_ids[row_idx, :length] = decoder_input_ids[row_idx, keep_indices]
        new_decoder_var_ids[row_idx, :length] = decoder_var_ids[row_idx, keep_indices]
        new_target_ids[row_idx, :length] = target_ids[row_idx, keep_indices]
        new_candidate_masks[row_idx, :length] = candidate_masks[row_idx, keep_indices]

    return {
        "decoder_input_ids": new_decoder_input_ids,
        "decoder_var_ids": new_decoder_var_ids,
        "target_ids": new_target_ids,
        "candidate_masks": new_candidate_masks,
        "seq_lens": seq_lens.to(dtype=torch.long),
    }


def _teacher_forcing_accuracy_stats(
    logits: torch.Tensor,
    targets: torch.Tensor,
    candidate_masks: torch.Tensor,
    pad_id: int,
) -> tuple[int, int, int, int]:
    masked_logits = logits.masked_fill(~candidate_masks, float("-inf"))
    pred_ids = torch.argmax(masked_logits, dim=-1)
    singleton_mask = _build_singleton_position_mask(targets, candidate_masks, pad_id)
    valid_mask = targets.ne(int(pad_id)) & (~singleton_mask)
    token_correct = int((pred_ids.eq(targets) & valid_mask).sum().item())
    token_total = int(valid_mask.sum().item())
    if targets.dim() != 2:
        raise ValueError("teacher forcing accuracy expects [batch, seq] targets")
    sample_valid = valid_mask.any(dim=-1)
    sample_all_correct = pred_ids.eq(targets) | (~valid_mask)
    exact_count = int((sample_all_correct.all(dim=-1) & sample_valid).sum().item())
    sample_total = int(sample_valid.sum().item())
    return token_correct, token_total, exact_count, sample_total


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
                for rem_t, rem_name, rem_value in zip(
                    range(t + 1, len(names)),
                    names[t + 1:],
                    values[t + 1:],
                ):
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


def _resolve_ridge_alphas(cfg) -> List[float]:
    raw = getattr(cfg.train, "ridge_alpha", 0.1)
    if isinstance(raw, (list, tuple)):
        candidates = list(raw)
    else:
        candidates = [raw]

    resolved: List[float] = []
    seen: set[float] = set()
    for value in candidates:
        alpha = float(value)
        if not math.isfinite(alpha):
            raise ValueError(f"ridge_alpha must be finite, got {value}")
        if alpha < 0.0:
            raise ValueError(f"ridge_alpha must be non-negative, got {value}")
        if alpha in seen:
            continue
        seen.add(alpha)
        resolved.append(alpha)
    if not resolved:
        raise ValueError("ridge_alpha must contain at least one value")
    return resolved


def _alpha_metric_suffix(alpha: float) -> str:
    text = f"{float(alpha):.12g}"
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def _spearman_from_pairs(scored_items: Sequence[tuple[float, float, str]]) -> float:
    if len(scored_items) < 2:
        return float("nan")
    try:
        from scipy.stats import spearmanr
    except Exception:  # pragma: no cover
        return float("nan")
    preds = [item[0] for item in scored_items]
    actuals = [item[1] for item in scored_items]
    result = spearmanr(preds, actuals)
    rho = getattr(result, "correlation", None)
    if rho is None:
        rho = result[0]
    rho = float(rho)
    return rho if math.isfinite(rho) else float("nan")


def _build_named_latent_cost_ridges(latent_cost_ridges: Sequence[dict] | None) -> Dict[str, dict]:
    if not latent_cost_ridges:
        return {}

    unweighted = [p for p in latent_cost_ridges if not bool(p.get("weighted", False))]
    weighted = [p for p in latent_cost_ridges if bool(p.get("weighted", False))]

    named: Dict[str, dict] = {}
    if unweighted:
        named["cost_vec"] = unweighted[0]
        if len(unweighted) > 1:
            for payload in unweighted:
                alpha = float(payload["alpha"])
                named[f"cost_vec_alpha_{_alpha_metric_suffix(alpha)}"] = payload
    if weighted:
        named["cost_vec_weighted"] = weighted[0]
        if len(weighted) > 1:
            for payload in weighted:
                alpha = float(payload["alpha"])
                named[f"cost_vec_weighted_alpha_{_alpha_metric_suffix(alpha)}"] = payload
    return named


@torch.no_grad()
def evaluate_teacher_forcing(
    model: LatentParamVAE,
    dataset,
    registry: GeneratorRegistry,
    tokenizer: ParamTokenizer,
    device: torch.device,
    batch_size: int = 64,
    use_compressed: bool = False,
) -> Dict[str, float]:
    model.eval()

    token_correct = 0
    token_total = 0
    exact_count = 0
    sample_total = 0
    samples = list(dataset.samples)
    stride = max(int(batch_size), 1)

    for start in range(0, len(samples), stride):
        batch_samples = samples[start:start + stride]
        batch = collate_prepared_samples(batch_samples, tokenizer)
        batch = _batch_to_device(batch, device)
        candidate_masks = _build_teacher_forcing_candidate_masks(
            batch,
            registry,
            tokenizer,
            device=device,
            debug_invalid_step=False,
        )
        if use_compressed:
            compressed = _compress_teacher_forcing_batch(batch, candidate_masks, tokenizer)
            decoder_input_ids = compressed["decoder_input_ids"]
            decoder_var_ids = compressed["decoder_var_ids"]
            target_ids = compressed["target_ids"]
            cand_masks_eval = compressed["candidate_masks"]
        else:
            decoder_input_ids = batch["decoder_input_ids"]
            decoder_var_ids = batch["decoder_var_ids"]
            target_ids = batch["target_ids"]
            cand_masks_eval = candidate_masks
        out = model(
            batch["encoder_token_ids"],
            batch["encoder_var_ids"],
            decoder_input_ids,
            decoder_var_ids,
            pad_token_id=tokenizer.pad_id,
        )
        batch_token_correct, batch_token_total, batch_exact_count, batch_sample_total = (
            _teacher_forcing_accuracy_stats(
                out.logits,
                target_ids,
                cand_masks_eval,
                tokenizer.pad_id,
            )
        )
        token_correct += int(batch_token_correct)
        token_total += int(batch_token_total)
        exact_count += int(batch_exact_count)
        sample_total += int(batch_sample_total)

    return {
        "token_accuracy": token_correct / max(token_total, 1),
        "full_sequence_exact_match": exact_count / max(sample_total, 1),
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
    stride = max(int(batch_size), 1)
    total_batches = (len(samples) + stride - 1) // stride
    print("[eval] validation batches")
    for start in tqdm(range(0, len(samples), stride), desc="eval batches", total=total_batches):
        batch_samples = samples[start:start + stride]
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
            f"[eval-batch {start // stride + 1}/{total_batches}] "
            f"tok_acc={batch_tok_acc:.4f} exact={batch_exact:.4f}"
        )

    return {
        "token_accuracy": token_correct / max(token_total, 1),
        "full_sequence_exact_match": exact_match / max(len(dataset), 1),
    }


@torch.no_grad()
def evaluate_cost_ranking(
    model: LatentParamVAE,
    dataset,
    tokenizer: ParamTokenizer,
    device: torch.device,
    batch_size: int = 64,
    latent_cost_ridge: dict | None = None,
    latent_cost_ridges: Dict[str, dict] | None = None,
) -> Dict[str, float]:
    model.eval()

    scored_by_source: Dict[str, List[tuple[float, float, str]]] = {"cost_head": []}
    vector_payloads: Dict[str, dict] = {}
    if latent_cost_ridges:
        vector_payloads.update(latent_cost_ridges)
    elif latent_cost_ridge is not None:
        vector_payloads["cost_vec"] = latent_cost_ridge

    vector_weights: Dict[str, tuple[torch.Tensor, float]] = {}
    for source_name, payload in vector_payloads.items():
        vector_weights[source_name] = (
            payload["weight"].to(device=device, dtype=torch.float32),
            float(payload.get("bias", 0.0)),
        )
        scored_by_source[source_name] = []

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
        vector_preds = {
            source_name: (z @ weight + bias).detach().cpu().tolist()
            for source_name, (weight, bias) in vector_weights.items()
        }

        for row_idx, sample in enumerate(batch_samples):
            if sample.cost is None or not math.isfinite(float(sample.cost)):
                continue
            actual_cost = float(sample.cost)
            sample_id = str(sample.sample_id)
            scored_by_source["cost_head"].append((float(cost_head_pred[row_idx]), actual_cost, sample_id))
            for source_name, pred_values in vector_preds.items():
                scored_by_source[source_name].append((float(pred_values[row_idx]), actual_cost, sample_id))

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
        metrics[f"{source_name}_actual_top1_pred_rank"] = float(actual_top1_pred_rank)
        metrics[f"{source_name}_pred_top1_actual_cost"] = float(pred_top1_actual_cost)
        for k in (10, 20):
            topk = min(k, len(pred_sorted))
            pred_topk_mean_actual_cost = sum(item[1] for item in pred_sorted[:topk]) / topk
            metrics[f"{source_name}_pred_top{k}_mean_actual_cost"] = float(pred_topk_mean_actual_cost)

        metrics[f"{source_name}_spearman"] = _spearman_from_pairs(scored_items)
        top5_n = max(2, int(math.ceil(len(actual_sorted) * 0.05)))
        top5_n = min(top5_n, len(actual_sorted))
        metrics[f"{source_name}_spearman_top5pct"] = _spearman_from_pairs(
            actual_sorted[:top5_n]
        )

    return metrics


@torch.no_grad()
def fit_latent_cost_ridge(
    model: LatentParamVAE,
    dataset,
    tokenizer: ParamTokenizer,
    device: torch.device,
    alpha: float,
    batch_size: int = 128,
    sample_weight_quantile: float | None = None,
    sample_weight_sigma: float | None = None,
    cost_target: str = "neg_log",
    cost_target_regression: str | None = None,
    task_min_cost: float | None = None,
) -> dict | None:
    model.eval()

    # Ridge is fit in ``cost_target_regression`` space (defaults to
    # ``cost_target``). ``batch["costs"]`` is in ``cost_target`` space, so we
    # convert it once per batch before fitting.
    fit_target = cost_target_regression or cost_target
    if fit_target != cost_target:
        from .train_epoch import _convert_cost_tensor_space
    else:
        _convert_cost_tensor_space = None  # type: ignore[assignment]

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
        costs_for_fit = batch["costs"][valid_mask]
        if _convert_cost_tensor_space is not None:
            costs_for_fit = _convert_cost_tensor_space(
                costs_for_fit, cost_target, fit_target, task_min_cost
            )
        cost_batches.append(costs_for_fit.detach().cpu().to(dtype=torch.float64))

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

    weighted = sample_weight_quantile is not None and sample_weight_sigma is not None
    if weighted:
        from .train_losses import compute_cobo_sample_weights

        y32 = y.to(dtype=torch.float32)
        mask_all = torch.ones_like(y32, dtype=torch.bool)
        w = compute_cobo_sample_weights(
            y32,
            mask_all,
            quantile=float(sample_weight_quantile),
            sigma=float(sample_weight_sigma),
        ).to(dtype=torch.float64)
        sqrt_w = torch.sqrt(w.clamp_min(1e-8))
        design_w = sqrt_w.unsqueeze(1) * design
        y_w = sqrt_w * y
        lhs = design_w.T @ design_w + reg
        rhs = design_w.T @ y_w
    else:
        lhs = design.T @ design + reg
        rhs = design.T @ y
    try:
        coeff = torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        coeff = torch.linalg.pinv(lhs) @ rhs

    pred = (design @ coeff).to(dtype=torch.float64)
    payload = {
        "weight": coeff[:-1].to(dtype=torch.float32).contiguous(),
        "bias": float(coeff[-1].item()),
        "alpha": float(alpha),
        "num_samples": int(num_samples),
        "latent_dim": int(latent_dim),
        "feature_name": "deterministic_z",
        "target_name": fit_target,
        # Output-space metadata: ``z @ weight + bias`` is in ``fit_target``
        # space. Consumers of ridge predictions (predict_score,
        # RidgeReEncodePredictor.score) convert back to ``cost_target`` space
        # using ``task_min_cost`` so walk-side comparisons remain in
        # ``cost_target`` space regardless of ``cost_target_regression``.
        "cost_target": cost_target,
        "task_min_cost": task_min_cost,
        "train_mse": float(torch.mean((pred - y) ** 2).item()),
        "weighted": bool(weighted),
    }
    if weighted:
        payload["weight_quantile"] = float(sample_weight_quantile)
        payload["weight_sigma"] = float(sample_weight_sigma)
    return payload


@torch.no_grad()
def fit_latent_cost_ridges(
    model: LatentParamVAE,
    dataset,
    tokenizer: ParamTokenizer,
    device: torch.device,
    alphas: Sequence[float],
    batch_size: int = 128,
    sample_weight_quantile: float | None = None,
    sample_weight_sigma: float | None = None,
    cost_target: str = "neg_log",
    cost_target_regression: str | None = None,
    task_min_cost: float | None = None,
) -> List[dict]:
    fitted: List[dict] = []
    for alpha in alphas:
        payload = fit_latent_cost_ridge(
            model,
            dataset,
            tokenizer,
            device,
            alpha=float(alpha),
            batch_size=batch_size,
            sample_weight_quantile=sample_weight_quantile,
            sample_weight_sigma=sample_weight_sigma,
            cost_target=cost_target,
            cost_target_regression=cost_target_regression,
            task_min_cost=task_min_cost,
        )
        if payload is not None:
            fitted.append(payload)
    return fitted
