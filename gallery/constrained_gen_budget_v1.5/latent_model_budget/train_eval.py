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
from .tokenizer import ParamTokenizer


def _build_singleton_position_mask(
    targets: torch.Tensor,
    candidate_masks: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    valid_mask = targets.ne(int(pad_id))
    singleton_mask = candidate_masks.to(dtype=torch.bool).sum(dim=-1).eq(1)
    return singleton_mask & valid_mask


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


def _build_named_latent_cost_ridges(latent_cost_ridges: Sequence[dict] | None) -> Dict[str, dict]:
    if not latent_cost_ridges:
        return {}

    named: Dict[str, dict] = {"cost_vec": latent_cost_ridges[0]}
    if len(latent_cost_ridges) == 1:
        return named

    for payload in latent_cost_ridges:
        alpha = float(payload["alpha"])
        named[f"cost_vec_alpha_{_alpha_metric_suffix(alpha)}"] = payload
    return named


@torch.no_grad()
def evaluate_teacher_forcing(
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
        out = model(
            batch["encoder_token_ids"],
            batch["encoder_var_ids"],
            batch["decoder_input_ids"],
            batch["decoder_var_ids"],
            pad_token_id=tokenizer.pad_id,
        )
        batch_token_correct, batch_token_total, batch_exact_count, batch_sample_total = (
            _teacher_forcing_accuracy_stats(
                out.logits,
                batch["target_ids"],
                candidate_masks,
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
        mu, _, z, _ = model.encode(enc_ids, enc_var_ids, enc_pad, deterministic=True)
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

    pred = (design @ coeff).to(dtype=torch.float64)
    return {
        "weight": coeff[:-1].to(dtype=torch.float32).contiguous(),
        "bias": float(coeff[-1].item()),
        "alpha": float(alpha),
        "num_samples": int(num_samples),
        "latent_dim": int(latent_dim),
        "feature_name": "deterministic_z",
        "target_name": "neg_log_cost",
        "train_mse": float(torch.mean((pred - y) ** 2).item()),
    }


@torch.no_grad()
def fit_latent_cost_ridges(
    model: LatentParamVAE,
    dataset,
    tokenizer: ParamTokenizer,
    device: torch.device,
    alphas: Sequence[float],
    batch_size: int = 128,
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
        )
        if payload is not None:
            fitted.append(payload)
    return fitted
