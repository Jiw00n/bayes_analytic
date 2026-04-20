"""Evaluate GP ranking quality on a measure-records JSON.

Minimal path:
  1. Load bundle (checkpoint + tokenizer + registry).
  2. Load training records from config.data.json_paths; resolve param order once
     per unique (workload_key, target_kind, sketch_index, param_signature) group
     so we don't build a generator per record.
  3. Tokenize -> encode (mu) on training set, select top_k + random_n, fit GP.
  4. Tokenize -> encode target records, predict, report correlations.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

sys.path.insert(0, str(Path(__file__).resolve().parent))

from latent_model_budget.adapter import (
    GeneratorRegistry,
    JsonSampleRecord,
    load_json_samples,
    _augment_record_budget_params,
)
from latent_model_budget.config import ExperimentConfig
from latent_model_budget.dataset import (
    _build_prepared_sample,
    budget_enabled,
    filter_param_order,
    _expand_json_paths,
)
from latent_model_budget.recon_predict_gp import _encode_samples, _select_indices
from tune_by_latent import load_bundle


def _build_samples_fast(records, registry, tokenizer, include_budget):
    """Build PreparedSamples without per-record generator construction.

    Resolves the param order once per unique group via `get_generator_from_record`
    on a representative record, then tokenizes all records in the group directly.
    """
    prepared, skipped = [], 0

    group_cache: dict = {}

    def _group_key(rec: JsonSampleRecord):
        return (
            rec.workload_key,
            rec.target_kind,
            rec.task_index,
            rec.sketch_index,
            tuple(rec.param_signature or ()),
        )

    for rec in records:
        if rec.cost is None:
            skipped += 1
            continue
        key = _group_key(rec)
        if key in group_cache:
            meta = group_cache[key]
            if meta is None:
                skipped += 1
                continue
        else:
            try:
                gen = registry.get_generator_from_record(rec)
            except Exception:
                group_cache[key] = None
                skipped += 1
                continue
            full_order = list(gen.get_full_var_order_entries()["param_order"])
            order = filter_param_order(full_order, include_budget=include_budget)
            budget_specs = [
                (str(spec["name"]), tuple(str(n) for n in spec.get("factor_names", ())))
                for spec in getattr(gen, "_budget_specs", ())
            ]
            meta = (order, budget_specs)
            group_cache[key] = meta

        order, budget_specs = meta
        params = rec.params
        for budget_name, factor_names in budget_specs:
            if budget_name in params:
                params[budget_name] = int(params[budget_name])
                continue
            if not factor_names or any(n not in params for n in factor_names):
                continue
            v = 1
            for fn in factor_names:
                v *= int(params[fn])
            params[budget_name] = int(v)

        missing = [n for n in order if n not in params]
        if missing:
            skipped += 1
            continue
        ordered_values = [int(params[n]) for n in order]
        try:
            sample = _build_prepared_sample(rec, order, ordered_values, tokenizer)
        except Exception:
            skipped += 1
            continue
        prepared.append(sample)
    return prepared, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--records", required=True, help="measure-records JSON to rank")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--random-n", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = ExperimentConfig.load_json(args.config)
    include_budget = budget_enabled(cfg)

    t0 = time.time()
    bundle = load_bundle(args.checkpoint, device=args.device)
    print(f"[eval] bundle loaded in {time.time()-t0:.1f}s device={bundle.device}")

    t0 = time.time()
    train_paths = _expand_json_paths(cfg.data.json_paths)
    train_records = []
    for p in train_paths:
        train_records.extend(load_json_samples(p))
    print(f"[eval] loaded {len(train_records)} train records from {len(train_paths)} file(s) in {time.time()-t0:.1f}s")

    t0 = time.time()
    train_samples, train_skipped = _build_samples_fast(
        train_records, bundle.registry, bundle.tokenizer, include_budget
    )
    print(f"[eval] built {len(train_samples)} train samples (skipped={train_skipped}) in {time.time()-t0:.1f}s")

    t0 = time.time()
    mu_train, costs_train = _encode_samples(
        bundle.model, train_samples, bundle.tokenizer, bundle.device, args.batch_size
    )
    print(f"[eval] encoded train mu shape={mu_train.shape} in {time.time()-t0:.1f}s")

    top_k = args.top_k if args.top_k is not None else int(cfg.train.latent_walk_predict_gp_top_k)
    random_n = args.random_n if args.random_n is not None else int(cfg.train.latent_walk_predict_gp_random_n)
    indices, selection = _select_indices(costs_train, top_k=top_k, random_n=random_n, seed=args.seed)
    x_train = mu_train[indices]
    y_train = costs_train[indices]

    t0 = time.time()
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1.0))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=int(args.seed),
    )
    gp.fit(x_train, y_train)
    print(f"[eval] GP fit selection={selection} n={len(y_train)} dim={x_train.shape[1]} in {time.time()-t0:.1f}s")

    t0 = time.time()
    tgt_records = load_json_samples(args.records)
    print(f"[eval] loaded {len(tgt_records)} target records from {args.records} in {time.time()-t0:.1f}s")

    t0 = time.time()
    tgt_samples, tgt_skipped = _build_samples_fast(
        tgt_records, bundle.registry, bundle.tokenizer, include_budget
    )
    print(f"[eval] built {len(tgt_samples)} target samples (skipped={tgt_skipped}) in {time.time()-t0:.1f}s")
    if not tgt_samples:
        return

    mu_tgt, costs_true = _encode_samples(
        bundle.model, tgt_samples, bundle.tokenizer, bundle.device, args.batch_size
    )
    pred = gp.predict(mu_tgt)
    print(f"[eval] predicted {len(pred)} costs")

    from scipy.stats import kendalltau, spearmanr, pearsonr

    spearman = spearmanr(costs_true, pred).correlation
    kendall = kendalltau(costs_true, pred).correlation
    pearson = pearsonr(costs_true, pred)[0]
    mse = float(np.mean((costs_true - pred) ** 2))
    print("=" * 60)
    print(f"n={len(costs_true)}")
    print(f"Spearman rho = {spearman:+.4f}")
    print(f"Kendall  tau = {kendall:+.4f}")
    print(f"Pearson  r   = {pearson:+.4f}")
    print(f"MSE          = {mse:.6f}")
    print("=" * 60)

    order_true = np.argsort(-costs_true)
    order_pred = np.argsort(-pred)
    for k in (1, 5, 10, 20, 50):
        if k > len(costs_true):
            continue
        top_true = set(int(i) for i in order_true[:k])
        top_pred = set(int(i) for i in order_pred[:k])
        inter = len(top_true & top_pred)
        print(f"top-{k} overlap: {inter}/{k}")


if __name__ == "__main__":
    main()
