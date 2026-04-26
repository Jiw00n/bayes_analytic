from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader

from .adapter import GeneratorRegistry
from .dataset import LatentParamDataset, collate_prepared_samples
from .model import LatentParamVAE
from .recon_predict_gp import fit_gp_recon_predictor
from .recon_predict_lgbm import fit_lgbm_ranker_recon_predictor
from .tokenizer import ParamTokenizer


def _build_eval_loader(
    dataset,
    tokenizer: ParamTokenizer,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """Background-worker DataLoader for read-only eval / ridge passes. Lets
    the next batch's collation overlap with the current GPU step instead of
    serializing on the main thread (the source of the 10–40% GPU utilization
    during eval/ridge)."""
    nw = max(0, int(num_workers))
    kwargs = dict(
        dataset=dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=nw,
        collate_fn=lambda b: collate_prepared_samples(b, tokenizer),
        pin_memory=bool(pin_memory),
    )
    if nw > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(**kwargs)


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

    - cost_head: returns None (latent_walk falls back to model.cost_head).
    - cost_vec / cost_vec_weighted: wraps the corresponding ridge payload.
    - gp: fits a fresh GP on the training subset.
    - lightgbm_ranker: fits an LGBMRanker on the same selection as GP.
    """
    from .latent_walk import RidgeReEncodePredictor

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
            top_k=int(getattr(config.latent_walk, "predict_gp_top_k", 800)),
            random_n=int(getattr(config.latent_walk, "predict_gp_random_n", 200)),
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
            top_k=int(getattr(config.latent_walk, "predict_gp_top_k", 800)),
            random_n=int(getattr(config.latent_walk, "predict_gp_random_n", 200)),
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


def _teacher_forcing_accuracy_stats(
    logits: torch.Tensor,
    targets: torch.Tensor,
    candidate_masks: torch.Tensor,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns 0-d device tensors so the caller can decide when to incur the
    GPU→CPU sync. The previous version's per-call ``.item()`` quartet stalled
    every training batch four extra times."""
    if targets.dim() != 2:
        raise ValueError("teacher forcing accuracy expects [batch, seq] targets")
    masked_logits = logits.masked_fill(~candidate_masks, float("-inf"))
    pred_ids = torch.argmax(masked_logits, dim=-1)
    singleton_mask = _build_singleton_position_mask(targets, candidate_masks, pad_id)
    valid_mask = targets.ne(int(pad_id)) & (~singleton_mask)
    token_correct = (pred_ids.eq(targets) & valid_mask).sum()
    token_total = valid_mask.sum()
    sample_valid = valid_mask.any(dim=-1)
    sample_all_correct = pred_ids.eq(targets) | (~valid_mask)
    exact_count = (sample_all_correct.all(dim=-1) & sample_valid).sum()
    sample_total = sample_valid.sum()
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
def evaluate_teacher_forcing_with_encoded(
    model: LatentParamVAE,
    dataset,
    registry: GeneratorRegistry,
    tokenizer: ParamTokenizer,
    device: torch.device,
    *,
    encoded: Dict[str, object],
    batch_size: int = 64,
    num_workers: int = 0,
    loader: DataLoader | None = None,
) -> Dict[str, float]:
    """Decoder-only teacher-forcing eval that reuses ``z`` already produced
    by :func:`encode_dataset`. Skips the redundant encoder pass over the
    same dataset (~half the work of :func:`evaluate_teacher_forcing`).

    Note: ``encode_dataset`` runs the encoder with ``deterministic=True``
    (z = mu), so the metric here is computed on deterministic z's rather
    than the sampled z's used by the full ``evaluate_teacher_forcing``.
    """
    model.eval()

    z_all: torch.Tensor = encoded["z"]
    if z_all.numel() == 0:
        return {"token_accuracy": 0.0, "full_sequence_exact_match": 0.0}

    sample_id_to_row: Dict[str, int] = {
        str(sid): i for i, sid in enumerate(encoded["sample_ids"])
    }

    if loader is None:
        loader = _build_eval_loader(
            dataset,
            tokenizer,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )

    token_correct = torch.zeros((), device=device, dtype=torch.long)
    token_total = torch.zeros((), device=device, dtype=torch.long)
    exact_count = torch.zeros((), device=device, dtype=torch.long)
    sample_total = torch.zeros((), device=device, dtype=torch.long)

    latent_token_count = int(model.cfg.latent_token_count)
    embed_dim = int(model.cfg.embed_dim)

    for batch in loader:
        rows = [sample_id_to_row[str(sid)] for sid in batch["sample_ids"]]
        z = z_all[rows].to(
            device=device,
            dtype=torch.float32,
            non_blocking=device.type == "cuda",
        )
        memory = model.latent_to_memory(z).view(z.size(0), latent_token_count, embed_dim)

        batch = _batch_to_device(batch, device)
        candidate_masks = _build_teacher_forcing_candidate_masks(
            batch, registry, tokenizer, device=device, debug_invalid_step=False,
        )
        decoder_input_ids = batch["decoder_input_ids"]
        decoder_var_ids = batch["decoder_var_ids"]
        target_ids = batch["target_ids"]
        cand_masks_eval = candidate_masks

        dec_pad = decoder_input_ids.eq(tokenizer.pad_id)
        logits = model.decode(decoder_input_ids, decoder_var_ids, memory, z, dec_pad)

        b_correct, b_total, b_exact, b_sample_total = _teacher_forcing_accuracy_stats(
            logits, target_ids, cand_masks_eval, tokenizer.pad_id,
        )
        token_correct = token_correct + b_correct.detach()
        token_total = token_total + b_total.detach()
        exact_count = exact_count + b_exact.detach()
        sample_total = sample_total + b_sample_total.detach()

    return {
        "token_accuracy": float(token_correct.item()) / max(int(token_total.item()), 1),
        "full_sequence_exact_match": float(exact_count.item()) / max(int(sample_total.item()), 1),
    }


@torch.no_grad()
def evaluate_teacher_forcing(
    model: LatentParamVAE,
    dataset,
    registry: GeneratorRegistry,
    tokenizer: ParamTokenizer,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 0,
    loader: DataLoader | None = None,
) -> Dict[str, float]:
    model.eval()

    token_correct = torch.zeros((), device=device, dtype=torch.long)
    token_total = torch.zeros((), device=device, dtype=torch.long)
    exact_count = torch.zeros((), device=device, dtype=torch.long)
    sample_total = torch.zeros((), device=device, dtype=torch.long)

    if loader is None:
        loader = _build_eval_loader(
            dataset,
            tokenizer,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )

    for batch in loader:
        batch = _batch_to_device(batch, device)
        candidate_masks = _build_teacher_forcing_candidate_masks(
            batch,
            registry,
            tokenizer,
            device=device,
            debug_invalid_step=False,
        )
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
        token_correct = token_correct + batch_token_correct.detach()
        token_total = token_total + batch_token_total.detach()
        exact_count = exact_count + batch_exact_count.detach()
        sample_total = sample_total + batch_sample_total.detach()

    token_correct_i = int(token_correct.item())
    token_total_i = int(token_total.item())
    exact_count_i = int(exact_count.item())
    sample_total_i = int(sample_total.item())
    return {
        "token_accuracy": token_correct_i / max(token_total_i, 1),
        "full_sequence_exact_match": exact_count_i / max(sample_total_i, 1),
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
    num_workers: int = 0,
    loader: DataLoader | None = None,
    encoded: Dict[str, object] | None = None,
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

    if encoded is None:
        encoded = encode_dataset(
            model,
            dataset,
            tokenizer,
            device,
            batch_size=batch_size,
            num_workers=num_workers,
            loader=loader,
        )
    z_all = encoded["z"]
    cost_all = encoded["cost"]
    mask_all = encoded["cost_mask"]
    sample_ids_all = list(encoded["sample_ids"])
    if z_all.numel() == 0:
        return {}

    # cost_head + ridge predictions for all samples in one batched call. Way
    # cheaper than per-batch CPU scatter since we only run cost_head and a
    # matmul over z (already on CPU); push to device only for cost_head.
    z_dev = z_all.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
    cost_head_pred_all = model.cost_head(z_dev).squeeze(-1).detach().cpu().tolist()
    vector_preds_all: Dict[str, list] = {}
    for source_name, (weight, bias) in vector_weights.items():
        weight_cpu = weight.detach().to(device="cpu", dtype=torch.float32)
        preds = (z_all @ weight_cpu + float(bias)).tolist()
        vector_preds_all[source_name] = preds

    cost_list = cost_all.tolist()
    mask_list = mask_all.tolist()
    for row_idx in range(len(sample_ids_all)):
        if not bool(mask_list[row_idx]):
            continue
        actual_cost = float(cost_list[row_idx])
        if not math.isfinite(actual_cost):
            continue
        sample_id = sample_ids_all[row_idx]
        scored_by_source["cost_head"].append(
            (float(cost_head_pred_all[row_idx]), actual_cost, sample_id)
        )
        for source_name, preds in vector_preds_all.items():
            scored_by_source[source_name].append(
                (float(preds[row_idx]), actual_cost, sample_id)
            )

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
def encode_dataset(
    model: LatentParamVAE,
    dataset,
    tokenizer: ParamTokenizer,
    device: torch.device,
    *,
    batch_size: int = 128,
    num_workers: int = 0,
    loader: DataLoader | None = None,
) -> Dict[str, object]:
    """Single forward pass over the encoder. Returns CPU tensors keyed by
    ``z`` / ``cost`` / ``cost_mask`` plus a ``sample_ids`` list. ``cost`` is
    in the loader's ``cost_target`` space (caller converts as needed).

    Used by both ridge fit and cost-ranking evaluation so a per-epoch encode
    happens exactly once even when both consumers (and val + train splits)
    need the same z's.
    """
    model.eval()
    if loader is None:
        loader = _build_eval_loader(
            dataset,
            tokenizer,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
    z_chunks: List[torch.Tensor] = []
    cost_chunks: List[torch.Tensor] = []
    mask_chunks: List[torch.Tensor] = []
    sample_ids: List[str] = []
    for batch in loader:
        enc_ids = batch["encoder_token_ids"].to(device, non_blocking=device.type == "cuda")
        enc_var_ids = batch["encoder_var_ids"].to(device, non_blocking=device.type == "cuda")
        enc_pad = enc_ids.eq(tokenizer.pad_id)
        _, _, z, _ = model.encode(enc_ids, enc_var_ids, enc_pad, deterministic=True)
        z_chunks.append(z.detach().to(device="cpu", dtype=torch.float32))
        cost_chunks.append(batch["costs"].detach().to(dtype=torch.float32))
        mask_chunks.append(batch["cost_mask"].detach().to(dtype=torch.bool))
        sample_ids.extend(str(x) for x in batch["sample_ids"])
    if not z_chunks:
        return {
            "z": torch.zeros((0, 0), dtype=torch.float32),
            "cost": torch.zeros((0,), dtype=torch.float32),
            "cost_mask": torch.zeros((0,), dtype=torch.bool),
            "sample_ids": [],
        }
    return {
        "z": torch.cat(z_chunks, dim=0),
        "cost": torch.cat(cost_chunks, dim=0),
        "cost_mask": torch.cat(mask_chunks, dim=0),
        "sample_ids": sample_ids,
    }


def _concat_encoded(parts: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Stack the per-dataset outputs of :func:`encode_dataset` into one."""
    parts = [p for p in parts if p and len(p["sample_ids"]) > 0]
    if not parts:
        return {
            "z": torch.zeros((0, 0), dtype=torch.float32),
            "cost": torch.zeros((0,), dtype=torch.float32),
            "cost_mask": torch.zeros((0,), dtype=torch.bool),
            "sample_ids": [],
        }
    return {
        "z": torch.cat([p["z"] for p in parts], dim=0),
        "cost": torch.cat([p["cost"] for p in parts], dim=0),
        "cost_mask": torch.cat([p["cost_mask"] for p in parts], dim=0),
        "sample_ids": [sid for p in parts for sid in p["sample_ids"]],
    }


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
    num_workers: int = 0,
    loader: DataLoader | None = None,
    encoded: Dict[str, object] | None = None,
) -> dict | None:
    model.eval()

    # Ridge is fit in ``cost_target_regression`` space (defaults to
    # ``cost_target``). ``batch["costs"]`` is in ``cost_target`` space, so we
    # convert it once before fitting.
    fit_target = cost_target_regression or cost_target
    if fit_target != cost_target:
        from .train_epoch import _convert_cost_tensor_space
    else:
        _convert_cost_tensor_space = None  # type: ignore[assignment]

    if encoded is None:
        encoded = encode_dataset(
            model,
            dataset,
            tokenizer,
            device,
            batch_size=batch_size,
            num_workers=num_workers,
            loader=loader,
        )
    z_all = encoded["z"]
    cost_all = encoded["cost"]
    mask_all = encoded["cost_mask"]
    if int(mask_all.sum().item()) == 0:
        return None
    z_valid = z_all[mask_all]
    cost_valid = cost_all[mask_all]
    if _convert_cost_tensor_space is not None:
        cost_valid = _convert_cost_tensor_space(
            cost_valid, cost_target, fit_target, task_min_cost
        )
    x = z_valid.to(dtype=torch.float64)
    y = cost_valid.to(dtype=torch.float64)
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
    loader: DataLoader | None = None,
    encoded: Dict[str, object] | None = None,
) -> List[dict]:
    if encoded is None:
        # Encode once even when fitting multiple alphas — the previous code
        # ran a fresh forward per alpha, paying ``alphas × N_samples`` of
        # encoder cost.
        encoded = encode_dataset(
            model,
            dataset,
            tokenizer,
            device,
            batch_size=batch_size,
            loader=loader,
        )
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
            loader=loader,
            encoded=encoded,
        )
        if payload is not None:
            fitted.append(payload)
    return fitted


# ---------------------------------------------------------------------------
# Per-epoch eval orchestration (formerly in train.py)
# ---------------------------------------------------------------------------

def _fit_epoch_ridges(
    model, bundle, tokenizer, config, device,
    *,
    ridge_dataset=None,
    ridge_loader=None,
    encoded=None,
):
    if not bool(getattr(config.train, "cost_ridge_vec", False)):
        return [], None, {}

    if ridge_dataset is None:
        include_val = bool(getattr(config.train, "cost_ridge_include_val", False))
        if include_val and len(bundle.val_dataset.samples) > 0:
            ridge_dataset = LatentParamDataset(
                list(bundle.train_dataset.samples) + list(bundle.val_dataset.samples)
            )
        else:
            ridge_dataset = bundle.train_dataset

    ridge_alphas = _resolve_ridge_alphas(config)
    _ridge_cost_target = str(getattr(config.data, "cost_target", "neg_log"))
    _ridge_cost_target_regression = getattr(config.data, "cost_target_regression", None)
    _ridge_mins = list(bundle.task_min_costs.values())
    _ridge_task_min_cost = float(_ridge_mins[0]) if _ridge_mins else None
    _ridge_fit_target = _ridge_cost_target_regression or _ridge_cost_target
    print(
        f"[ridge] fit_target={_ridge_fit_target!r} output_target={_ridge_cost_target!r} "
        f"task_min_cost={_ridge_task_min_cost!r}"
    )
    latent_cost_ridges = fit_latent_cost_ridges(
        model,
        ridge_dataset,
        tokenizer,
        device,
        alphas=ridge_alphas,
        batch_size=config.eval.batch_size,
        cost_target=_ridge_cost_target,
        cost_target_regression=_ridge_cost_target_regression,
        task_min_cost=_ridge_task_min_cost,
        loader=ridge_loader,
        encoded=encoded,
    )
    ridge_metrics: Dict[str, float] = {}
    for ridge_payload in latent_cost_ridges:
        alpha = float(ridge_payload["alpha"])
        alpha_suffix = _alpha_metric_suffix(alpha)
        if alpha == float(ridge_alphas[0]):
            ridge_metrics["train_ridge_mse"] = float(ridge_payload["train_mse"])
        ridge_metrics[f"train_ridge_alpha_{alpha_suffix}_mse"] = float(ridge_payload["train_mse"])

    if bool(getattr(config.train, "cost_ridge_weighted", False)):
        weighted_ridges = fit_latent_cost_ridges(
            model,
            ridge_dataset,
            tokenizer,
            device,
            alphas=ridge_alphas,
            batch_size=config.eval.batch_size,
            sample_weight_quantile=float(getattr(config.train, "weight_quantile", 0.85)),
            sample_weight_sigma=float(getattr(config.train, "weight_sigma", 0.25)),
            cost_target=_ridge_cost_target,
            cost_target_regression=_ridge_cost_target_regression,
            task_min_cost=_ridge_task_min_cost,
            loader=ridge_loader,
            encoded=encoded,
        )
        for ridge_payload in weighted_ridges:
            alpha = float(ridge_payload["alpha"])
            alpha_suffix = _alpha_metric_suffix(alpha)
            if alpha == float(ridge_alphas[0]):
                ridge_metrics["train_ridge_weighted_mse"] = float(ridge_payload["train_mse"])
            ridge_metrics[f"train_ridge_weighted_alpha_{alpha_suffix}_mse"] = float(
                ridge_payload["train_mse"]
            )
        latent_cost_ridges = list(latent_cost_ridges) + list(weighted_ridges)

    return latent_cost_ridges, ridge_metrics


def _evaluate_validation_epoch(
    model, bundle, registry, tokenizer, config, device, epoch, latent_cost_ridges,
    *,
    val_loader=None,
    encoded_val=None,
):
    summary: Dict[str, float] = {}
    if not bundle.val_dataset.samples:
        return summary

    val_tf_metrics = evaluate_teacher_forcing(
        model,
        bundle.val_dataset,
        registry,
        tokenizer,
        device,
        batch_size=config.eval.batch_size,
        loader=val_loader,
    )
    summary.update({f"val_{k}": float(v) for k, v in val_tf_metrics.items()})
    print(
        f"[epoch {epoch}] val_tok_acc={summary['val_token_accuracy']:.4f} "
        f"val_exact={summary['val_full_sequence_exact_match']:.4f}"
    )

    cost_metrics = evaluate_cost_ranking(
        model,
        bundle.val_dataset,
        tokenizer,
        device,
        batch_size=config.eval.batch_size,
        latent_cost_ridges=_build_named_latent_cost_ridges(latent_cost_ridges),
        loader=val_loader,
        encoded=encoded_val,
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
    if "cost_vec_weighted_actual_top1_pred_rank" in cost_metrics:
        print(
            f"val_cost_vec_weighted_actual_top1_pred_rank : {int(cost_metrics['cost_vec_weighted_actual_top1_pred_rank'])}\n"
            f"val_cost_vec_weighted_pred_top1_actual_cost : {cost_metrics['cost_vec_weighted_pred_top1_actual_cost']:.6f}\n"
            f"val_cost_vec_weighted_pred_top10_mean_actual_cost : {cost_metrics['cost_vec_weighted_pred_top10_mean_actual_cost']:.6f}\n"
        )
    for key, value in sorted(cost_metrics.items()):
        if key.startswith("cost_vec_alpha_") and key.endswith("_actual_top1_pred_rank"):
            prefix = key[: -len("_actual_top1_pred_rank")]
            top1_cost_key = f"{prefix}_pred_top1_actual_cost"
            top10_key = f"{prefix}_pred_top10_mean_actual_cost"
            top20_key = f"{prefix}_pred_top20_mean_actual_cost"
            print(
                f"{'val_' + key} : {int(value)}\n"
                f"{'val_' + top1_cost_key} : {cost_metrics[top1_cost_key]:.6f}\n"
                f"{'val_' + top10_key} : {cost_metrics[top10_key]:.6f}\n"
                f"{'val_' + top20_key} : {cost_metrics[top20_key]:.6f}\n"
            )

    return summary
