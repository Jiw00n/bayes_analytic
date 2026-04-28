"""Cost-prediction analysis for a trained LatentParamVAE checkpoint.

Mirrors :mod:`analyze_reconstruction_with_ar` but evaluates the **cost** side:
loads the checkpoint, encodes each split into ``z`` via the encoder, then
scores every record with both ``cost_head`` and a ``cost_vec`` ridge fit on
the training latents. Reports per-task and overall metrics on three slices
(all / actual top-5% / actual top-10%) for train / val / train+val.

This mirrors the metrics produced by ``train_eval.evaluate_cost_ranking``
(spearman, actual_top1_pred_rank, pred_top1_actual_cost, ...) and adds
top-K recall and pairwise accuracy that the user asked for.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

try:
    from latent_model_budget.dataset import LatentParamDataset
    from latent_model_budget.train_eval import (
        _resolve_ridge_alphas,
        encode_dataset,
        fit_latent_cost_ridges,
    )
    from analyze_reconstruction_with_ar import _load_model_and_bundle
except ImportError:  # pragma: no cover
    import sys

    _HERE = Path(__file__).resolve().parent
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))

    from latent_model_budget.dataset import LatentParamDataset
    from latent_model_budget.train_eval import (
        _resolve_ridge_alphas,
        encode_dataset,
        fit_latent_cost_ridges,
    )
    from analyze_reconstruction_with_ar import _load_model_and_bundle


TaskKey = Tuple[Optional[int], Optional[str], Optional[str]]


def _task_key(sample) -> TaskKey:
    return (
        None if sample.task_index is None else int(sample.task_index),
        None if sample.workload_key is None else str(sample.workload_key),
        None if sample.target_kind is None else str(sample.target_kind),
    )


def _task_key_sort(key: TaskKey):
    ti, wk, tk = key
    return (ti is None, ti if ti is not None else -1, wk or "", tk or "")


def _spearman(preds: Sequence[float], actuals: Sequence[float]) -> float:
    if len(preds) < 2:
        return float("nan")
    try:
        from scipy.stats import spearmanr
    except Exception:  # pragma: no cover
        return float("nan")
    result = spearmanr(list(preds), list(actuals))
    rho = getattr(result, "correlation", None)
    if rho is None:
        rho = result[0]
    rho = float(rho)
    return rho if math.isfinite(rho) else float("nan")


def _pairwise_accuracy(preds: Sequence[float], actuals: Sequence[float]) -> float:
    """Fraction of unordered pairs (i, j) with the same ordering under preds
    and actuals (ties on either side are excluded). Equivalent to
    (kendalltau + 1) / 2 when there are no ties; computed via scipy when
    available, falling back to an O(N^2) double-loop otherwise.
    """
    n = len(preds)
    if n < 2:
        return float("nan")
    try:
        from scipy.stats import kendalltau

        result = kendalltau(list(preds), list(actuals))
        tau = getattr(result, "correlation", None)
        if tau is None:
            tau = result[0]
        tau = float(tau)
        if math.isfinite(tau):
            return 0.5 * (tau + 1.0)
    except Exception:  # pragma: no cover
        pass

    concordant = 0
    discordant = 0
    for i in range(n):
        pi, ai = preds[i], actuals[i]
        for j in range(i + 1, n):
            pj, aj = preds[j], actuals[j]
            dp = pi - pj
            da = ai - aj
            if dp == 0 or da == 0:
                continue
            if (dp > 0) == (da > 0):
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return float("nan")
    return concordant / total


def _topk_recall(
    preds: Sequence[float],
    actuals: Sequence[float],
    *,
    k_pct: float,
    higher_is_better: bool,
) -> Optional[float]:
    """``|pred_topK ∩ actual_topK| / |actual_topK|``, where ``topK`` is the top
    ``ceil(k_pct * N)`` rows. Returns ``None`` when fewer than two items would
    be selected.
    """
    n = len(preds)
    if n < 2:
        return None
    k = max(1, int(math.ceil(n * float(k_pct))))
    if k < 1:
        return None

    indices = list(range(n))
    pred_top = set(
        sorted(indices, key=lambda i: preds[i], reverse=higher_is_better)[:k]
    )
    actual_top = set(
        sorted(indices, key=lambda i: actuals[i], reverse=higher_is_better)[:k]
    )
    if not actual_top:
        return None
    return len(pred_top & actual_top) / float(len(actual_top))


def _slice_actual_top(
    preds: Sequence[float],
    actuals: Sequence[float],
    *,
    fraction: float,
    higher_is_better: bool,
) -> Tuple[List[float], List[float]]:
    n = len(preds)
    if n == 0:
        return [], []
    k = max(2, int(math.ceil(n * float(fraction))))
    k = min(k, n)
    order = sorted(range(n), key=lambda i: actuals[i], reverse=higher_is_better)[:k]
    return [preds[i] for i in order], [actuals[i] for i in order]


def _compute_slice_metrics(
    preds: Sequence[float],
    actuals: Sequence[float],
    sample_ids: Sequence[str],
    *,
    higher_is_better: bool,
) -> Dict[str, Any]:
    """All metrics on a single (preds, actuals) slice.

    ``higher_is_better`` controls the meaning of "best": for ``norm_throughput``
    higher cost is better (more throughput); for ``neg_log`` and
    ``log_norm_throughput`` higher is also better. Caller resolves it from the
    bundle's ``cost_target``.
    """
    n = len(preds)
    if n == 0:
        return {"num_samples": 0}
    if n == 1:
        return {
            "num_samples": 1,
            "spearman": float("nan"),
            "pairwise_accuracy": float("nan"),
            "recall_at_5pct": None,
            "recall_at_10pct": None,
            "actual_top1_pred_rank": 1,
            "pred_top1_actual_cost": float(actuals[0]),
        }

    pred_sorted_idx = sorted(range(n), key=lambda i: preds[i], reverse=higher_is_better)
    actual_sorted_idx = sorted(range(n), key=lambda i: actuals[i], reverse=higher_is_better)

    actual_top1 = actual_sorted_idx[0]
    actual_top1_pred_rank = 1
    for rank, idx in enumerate(pred_sorted_idx, start=1):
        if idx == actual_top1:
            actual_top1_pred_rank = rank
            break

    pred_top1_actual_cost = float(actuals[pred_sorted_idx[0]])
    metrics: Dict[str, Any] = {
        "num_samples": int(n),
        "spearman": _spearman(preds, actuals),
        "pairwise_accuracy": _pairwise_accuracy(preds, actuals),
        "recall_at_5pct": _topk_recall(
            preds, actuals, k_pct=0.05, higher_is_better=higher_is_better
        ),
        "recall_at_10pct": _topk_recall(
            preds, actuals, k_pct=0.10, higher_is_better=higher_is_better
        ),
        "actual_top1_pred_rank": int(actual_top1_pred_rank),
        "pred_top1_actual_cost": pred_top1_actual_cost,
    }
    for k in (10, 20):
        topk = min(k, n)
        mean_actual = sum(actuals[i] for i in pred_sorted_idx[:topk]) / topk
        metrics[f"pred_top{k}_mean_actual_cost"] = float(mean_actual)
    return metrics


def _evaluate_source(
    preds_full: Sequence[float],
    actuals_full: Sequence[float],
    sample_ids: Sequence[str],
    *,
    higher_is_better: bool,
) -> Dict[str, Any]:
    """Run the metric battery on ``all`` / ``top5pct`` / ``top10pct`` slices."""
    overall = _compute_slice_metrics(
        preds_full,
        actuals_full,
        sample_ids,
        higher_is_better=higher_is_better,
    )

    p5, a5 = _slice_actual_top(
        preds_full, actuals_full, fraction=0.05, higher_is_better=higher_is_better
    )
    top5 = _compute_slice_metrics(
        p5,
        a5,
        sample_ids[: len(p5)],  # ids unused beyond size
        higher_is_better=higher_is_better,
    )

    p10, a10 = _slice_actual_top(
        preds_full, actuals_full, fraction=0.10, higher_is_better=higher_is_better
    )
    top10 = _compute_slice_metrics(
        p10,
        a10,
        sample_ids[: len(p10)],
        higher_is_better=higher_is_better,
    )

    return {"all": overall, "top5pct": top5, "top10pct": top10}


def _predictions_for_split(
    encoded: Dict[str, object],
    *,
    model,
    device: torch.device,
    ridge_payloads: Sequence[dict],
) -> Dict[str, Dict[str, Any]]:
    """For one encoded split, return a mapping ``source_name -> {sample_ids, preds, actuals, task_keys}``.

    Sources are ``cost_head`` plus one entry per fitted ridge payload (named
    ``cost_vec`` / ``cost_vec_weighted`` / ``cost_vec_alpha_<suffix>``).
    Samples whose actual cost is missing or non-finite are filtered out so
    each source iterates the same valid index set.
    """
    z_all: torch.Tensor = encoded["z"]
    cost_all: torch.Tensor = encoded["cost"]
    mask_all: torch.Tensor = encoded["cost_mask"]
    sample_ids_all: List[str] = list(encoded["sample_ids"])
    n = z_all.shape[0]
    if n == 0:
        return {}

    cost_list = cost_all.tolist()
    mask_list = mask_all.tolist()
    valid_idx = [
        i for i in range(n)
        if bool(mask_list[i]) and math.isfinite(float(cost_list[i]))
    ]
    if not valid_idx:
        return {}

    z_dev = z_all.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
    cost_head_full = model.cost_head(z_dev).squeeze(-1).detach().cpu().tolist()

    sources: Dict[str, Dict[str, Any]] = {}
    sources["cost_head"] = {
        "preds": [float(cost_head_full[i]) for i in valid_idx],
    }

    z_cpu = z_all.to(dtype=torch.float32)
    for payload in ridge_payloads or []:
        weight = payload["weight"].detach().to(device="cpu", dtype=torch.float32)
        bias = float(payload.get("bias", 0.0))
        weighted = bool(payload.get("weighted", False))
        alpha = float(payload.get("alpha", 0.0))
        # Match the naming convention used by _build_named_latent_cost_ridges
        # so downstream consumers see "cost_vec" / "cost_vec_weighted" rather
        # than alpha-suffixed names when only one alpha is present.
        base = "cost_vec_weighted" if weighted else "cost_vec"
        name = base
        if name in sources:
            text = f"{alpha:.12g}".replace("-", "m").replace("+", "").replace(".", "p")
            name = f"{base}_alpha_{text}"
        preds_full = (z_cpu @ weight + bias).tolist()
        sources[name] = {
            "preds": [float(preds_full[i]) for i in valid_idx],
        }

    actuals = [float(cost_list[i]) for i in valid_idx]
    sample_ids = [sample_ids_all[i] for i in valid_idx]
    for name, payload in sources.items():
        payload["actuals"] = actuals
        payload["sample_ids"] = sample_ids
        payload["valid_idx"] = valid_idx
    return sources


def _build_task_index_map(
    samples,
    sample_ids: Sequence[str],
) -> Dict[str, TaskKey]:
    by_id: Dict[str, TaskKey] = {}
    wanted = set(sample_ids)
    for s in samples:
        sid = str(s.sample_id)
        if sid in wanted and sid not in by_id:
            by_id[sid] = _task_key(s)
    return by_id


def _aggregate_split(
    sources: Dict[str, Dict[str, Any]],
    task_by_sid: Dict[str, TaskKey],
    *,
    higher_is_better: bool,
) -> Dict[str, Any]:
    """Build the report body for one split (e.g. ``val``)."""
    if not sources:
        return {"num_samples": 0, "per_source": {}, "per_task": []}

    any_source = next(iter(sources.values()))
    sample_ids = list(any_source["sample_ids"])
    n = len(sample_ids)

    per_source_overall: Dict[str, Any] = {}
    for name, payload in sources.items():
        per_source_overall[name] = _evaluate_source(
            payload["preds"],
            payload["actuals"],
            payload["sample_ids"],
            higher_is_better=higher_is_better,
        )

    # Per-task aggregation: bucket sample indices by task key, then evaluate
    # each task with each source.
    task_buckets: Dict[TaskKey, List[int]] = defaultdict(list)
    for i, sid in enumerate(sample_ids):
        tkey = task_by_sid.get(sid)
        if tkey is None:
            continue
        task_buckets[tkey].append(i)

    per_task_report: List[Dict[str, Any]] = []
    for tkey in sorted(task_buckets.keys(), key=_task_key_sort):
        idxs = task_buckets[tkey]
        ti, wk, tk = tkey
        entry: Dict[str, Any] = {
            "task_index": ti,
            "workload_key": wk,
            "target_kind": tk,
            "num_samples": len(idxs),
            "per_source": {},
        }
        for name, payload in sources.items():
            preds = [payload["preds"][i] for i in idxs]
            actuals = [payload["actuals"][i] for i in idxs]
            sids = [payload["sample_ids"][i] for i in idxs]
            entry["per_source"][name] = _evaluate_source(
                preds, actuals, sids, higher_is_better=higher_is_better
            )
        per_task_report.append(entry)

    return {
        "num_samples": int(n),
        "per_source": per_source_overall,
        "per_task": per_task_report,
    }


def _resolve_higher_is_better(cost_target: str) -> bool:
    """In all three loader-side cost_target spaces (``norm_throughput``,
    ``neg_log``, ``log_norm_throughput``), higher numbers correspond to
    faster / better schedules."""
    return True


def _format_split_summary(name: str, body: Dict[str, Any]) -> str:
    n = body.get("num_samples", 0)
    per = body.get("per_source", {})
    if not per:
        return f"[{name}] num_samples={n} (no valid predictions)"
    parts = [f"[{name}] num_samples={n}"]
    for source, metrics in per.items():
        all_m = metrics.get("all", {})
        spr = all_m.get("spearman", float("nan"))
        pacc = all_m.get("pairwise_accuracy", float("nan"))
        r5 = all_m.get("recall_at_5pct")
        r10 = all_m.get("recall_at_10pct")
        parts.append(
            f"  {source}: spearman={spr:.4f} pairwise={pacc:.4f} "
            f"recall@5%={'nan' if r5 is None else f'{r5:.4f}'} "
            f"recall@10%={'nan' if r10 is None else f'{r10:.4f}'}"
        )
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--json-path",
        type=str,
        action="append",
        default=None,
        help="Override checkpoint json_paths (repeatable).",
    )
    parser.add_argument("--network-info-folder", type=str, default=None)
    parser.add_argument(
        "--ridge-fit-on",
        type=str,
        default="train",
        choices=["train", "train_val", "all"],
        help=(
            "Which split(s) to fit the cost_vec ridge on. ``train`` is the "
            "honest setting for val metrics; the trainer's default is "
            "``train_val`` (cost_ridge_include_val=True)."
        ),
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help=(
            "Also evaluate the test split (and include it in ``combined`` and "
            "``ridge_fit_on=all``). Off by default since most checkpoints have "
            "no held-out test set."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output JSON path. Defaults to "
            "``<checkpoint_dir>/cost_ranking_<checkpoint_stem>.json``."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload, cfg, registry, bundle, tokenizer, model, device = _load_model_and_bundle(
        args.checkpoint,
        device=args.device,
        json_paths=args.json_path,
        network_info_folder=args.network_info_folder,
    )

    cost_target = str(getattr(cfg.data, "cost_target", "neg_log"))
    cost_target_regression = getattr(cfg.data, "cost_target_regression", None)
    higher_is_better = _resolve_higher_is_better(cost_target)
    mins = list(bundle.task_min_costs.values())
    task_min_cost = float(mins[0]) if mins else None
    print(
        f"[cost-analyze] cost_target={cost_target!r} "
        f"cost_target_regression={cost_target_regression!r} "
        f"task_min_cost={task_min_cost!r} higher_is_better={higher_is_better}"
    )

    encoded_train = encode_dataset(
        model, bundle.train_dataset, tokenizer, device, batch_size=args.batch_size
    )
    encoded_val = encode_dataset(
        model, bundle.val_dataset, tokenizer, device, batch_size=args.batch_size
    )
    encoded_test = (
        encode_dataset(
            model, bundle.test_dataset, tokenizer, device, batch_size=args.batch_size
        )
        if args.include_test
        else None
    )
    print(
        f"[cost-analyze] encoded "
        f"train={encoded_train['z'].shape[0]} "
        f"val={encoded_val['z'].shape[0]} "
        f"test={'-' if encoded_test is None else encoded_test['z'].shape[0]}"
    )

    # Ridge fit dataset selection.
    if args.ridge_fit_on == "train":
        ridge_dataset = bundle.train_dataset
        ridge_encoded = encoded_train
    elif args.ridge_fit_on == "train_val":
        ridge_dataset = LatentParamDataset(
            list(bundle.train_dataset.samples) + list(bundle.val_dataset.samples)
        )
        from latent_model_budget.train_eval import _concat_encoded
        ridge_encoded = _concat_encoded([encoded_train, encoded_val])
    else:  # "all"
        all_samples = (
            list(bundle.train_dataset.samples)
            + list(bundle.val_dataset.samples)
            + (list(bundle.test_dataset.samples) if args.include_test else [])
        )
        ridge_dataset = LatentParamDataset(all_samples)
        from latent_model_budget.train_eval import _concat_encoded
        parts = [encoded_train, encoded_val]
        if encoded_test is not None:
            parts.append(encoded_test)
        ridge_encoded = _concat_encoded(parts)

    alphas = _resolve_ridge_alphas(cfg)
    ridge_payloads = fit_latent_cost_ridges(
        model,
        ridge_dataset,
        tokenizer,
        device,
        alphas=alphas,
        batch_size=args.batch_size,
        cost_target=cost_target,
        cost_target_regression=cost_target_regression,
        task_min_cost=task_min_cost,
        encoded=ridge_encoded,
    )
    ridge_summary = [
        {
            "alpha": float(p["alpha"]),
            "weighted": bool(p.get("weighted", False)),
            "num_samples": int(p["num_samples"]),
            "train_mse": float(p["train_mse"]),
            "fit_target": str(p.get("target_name", "")),
            "output_target": str(p.get("cost_target", "")),
        }
        for p in ridge_payloads
    ]
    print(f"[cost-analyze] fit {len(ridge_payloads)} ridge(s) on {args.ridge_fit_on!r}: {ridge_summary}")

    # Build per-split source predictions.
    sources_train = _predictions_for_split(
        encoded_train, model=model, device=device, ridge_payloads=ridge_payloads
    )
    sources_val = _predictions_for_split(
        encoded_val, model=model, device=device, ridge_payloads=ridge_payloads
    )
    if encoded_test is not None:
        sources_test = _predictions_for_split(
            encoded_test, model=model, device=device, ridge_payloads=ridge_payloads
        )
    else:
        sources_test = {}

    # Combined: just concatenate the per-source preds/actuals/sample_ids.
    def _concat_sources(*all_sources: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        combined: Dict[str, Dict[str, Any]] = {}
        names: List[str] = []
        for src in all_sources:
            for name in src.keys():
                if name not in combined:
                    combined[name] = {"preds": [], "actuals": [], "sample_ids": []}
                    names.append(name)
        for src in all_sources:
            for name in names:
                if name not in src:
                    continue
                combined[name]["preds"].extend(src[name]["preds"])
                combined[name]["actuals"].extend(src[name]["actuals"])
                combined[name]["sample_ids"].extend(src[name]["sample_ids"])
        return combined

    sources_combined = _concat_sources(sources_train, sources_val, sources_test)

    # task_index lookup by sample_id.
    task_by_sid: Dict[str, TaskKey] = {}
    for ds in (bundle.train_dataset, bundle.val_dataset, bundle.test_dataset):
        for s in ds.samples:
            sid = str(s.sample_id)
            if sid not in task_by_sid:
                task_by_sid[sid] = _task_key(s)

    splits_report: Dict[str, Any] = {}
    splits_report["train"] = _aggregate_split(
        sources_train, task_by_sid, higher_is_better=higher_is_better
    )
    splits_report["val"] = _aggregate_split(
        sources_val, task_by_sid, higher_is_better=higher_is_better
    )
    if args.include_test:
        splits_report["test"] = _aggregate_split(
            sources_test, task_by_sid, higher_is_better=higher_is_better
        )
    splits_report["combined"] = _aggregate_split(
        sources_combined, task_by_sid, higher_is_better=higher_is_better
    )

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "json_paths": list(getattr(cfg.data, "json_paths", []) or []),
        "cost_target": cost_target,
        "cost_target_regression": cost_target_regression,
        "task_min_cost": task_min_cost,
        "higher_is_better": higher_is_better,
        "ridge_fit_on": args.ridge_fit_on,
        "ridge_alphas": [float(a) for a in alphas],
        "ridge_payloads": ridge_summary,
        "splits": splits_report,
    }

    output_path = (
        Path(args.output)
        if args.output
        else Path(args.checkpoint).resolve().parent
        / f"cost_ranking_{Path(args.checkpoint).stem}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[cost-analyze] wrote {output_path}")

    for name in ("train", "val", "test", "combined"):
        if name not in splits_report:
            continue
        print(_format_split_summary(name, splits_report[name]))


if __name__ == "__main__":
    main()
