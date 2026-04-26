"""Periodic LightGBM Ranker fit on (mu, cost) for recon-predict scoring.

Same role as ``recon_predict_gp.GPReconPredictor`` but uses an LGBMRanker
instead of a GP. It is a separate inference-only predictor that swaps in for
``LatentParamVAE.cost_head`` only inside the latent walk's recon-predict /
re-encode path — it never touches the VAE's parameters.

Ranking target: ``cost`` (negative-log mean_cost, higher = better performance).
The ranker is trained on a single group (all selected training samples plus
walk-buffer samples).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

import lightgbm as lgb

from .dataset import PreparedSample
from .model import LatentParamVAE
from .recon_predict_gp import (
    WalkSampleBuffer,
    _collect_mu_and_cost,
    _encode_samples,
    _select_indices,
)
from .tokenizer import ParamTokenizer


@dataclass
class LightGBMRankerReconPredictor:
    """Inference-only LightGBM ranker mapping deterministic z (=mu) to a score.

    The ranker's output is not calibrated to absolute cost values — it is a
    monotone score where higher means "ranked better" within the training set.
    For display/ranking purposes in the latent walk this is still useful and
    directly comparable across candidates from the same fit.
    """

    booster: "lgb.Booster"
    num_samples: int
    train_ndcg: float
    selection: str

    def predict(self, z: torch.Tensor) -> float:
        z_np = z.detach().to(dtype=torch.float32).cpu().numpy().reshape(1, -1)
        return float(self.booster.predict(z_np)[0])


def fit_lgbm_ranker_recon_predictor(
    *,
    model: LatentParamVAE,
    dataset,
    tokenizer: ParamTokenizer,
    device: torch.device,
    top_k: int,
    random_n: int,
    batch_size: int = 128,
    seed: int = 0,
    walk_buffer: Optional[WalkSampleBuffer] = None,
) -> Optional[LightGBMRankerReconPredictor]:
    """Fit an LGBMRanker on (deterministic z, cost) for the selected subset.

    Mirrors ``fit_gp_recon_predictor``: picks top-K by cost and random-N
    disjoint samples, encodes them to mu, optionally appends walk-buffer
    samples, and trains an LGBMRanker with a single group.
    """
    if lgb is None:
        print(
            f"[lgbm-recon] lightgbm not installed ({_LGB_IMPORT_ERROR}); "
            f"skipping ranker fit"
        )
        return None

    was_training = model.training
    model.eval()
    try:
        mu, costs = _collect_mu_and_cost(model, dataset, tokenizer, device, batch_size)
        indices, selection = _select_indices(
            costs, top_k=top_k, random_n=random_n, seed=seed
        )
        if mu.shape[0] == 0 or indices.shape[0] == 0:
            x_train = np.empty(
                (0, mu.shape[1] if mu.ndim == 2 else 0), dtype=np.float32
            )
            y_train = np.empty((0,), dtype=np.float32)
        else:
            x_train = mu[indices]
            y_train = costs[indices]

        walk_added = 0
        if walk_buffer is not None and len(walk_buffer) > 0:
            wb_mu, wb_costs = _encode_samples(
                model, walk_buffer.samples(), tokenizer, device, batch_size
            )
            if wb_mu.shape[0] > 0:
                if x_train.shape[0] == 0:
                    x_train = wb_mu
                    y_train = wb_costs
                else:
                    x_train = np.concatenate([x_train, wb_mu], axis=0)
                    y_train = np.concatenate([y_train, wb_costs], axis=0)
                walk_added = int(wb_mu.shape[0])
    finally:
        if was_training:
            model.train()

    if x_train.shape[0] == 0:
        print("[lgbm-recon] no samples available (train+walk); skipping LGBM fit")
        return None

    if walk_added > 0:
        selection = f"{selection}+walk{walk_added}"

    # LGBMRanker expects integer-like gains as labels. Map continuous costs
    # to rank-based integer labels: higher cost → higher label. We use the
    # argsort rank so labels are 0..N-1 and monotone in cost.
    order = np.argsort(np.argsort(y_train))  # 0..N-1, higher cost → higher rank
    labels = order.astype(np.int32)

    group = np.array([int(x_train.shape[0])], dtype=np.int32)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": max(1, int(x_train.shape[0] // 20)),
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "verbose": -1,
        "seed": int(seed),
        "label_gain": list(range(int(labels.max()) + 1)),
    }
    train_set = lgb.Dataset(x_train, label=labels, group=group)
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=200,
        valid_sets=[train_set],
        valid_names=["train"],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    train_ndcg = float(booster.best_score.get("train", {}).get("ndcg@1", float("nan")))

    print(
        f"[lgbm-recon] ranker fitted: selection={selection} "
        f"n={int(x_train.shape[0])} dim={int(x_train.shape[1])} "
        f"train_ndcg@1={train_ndcg:.6f}"
    )
    return LightGBMRankerReconPredictor(
        booster=booster,
        num_samples=int(x_train.shape[0]),
        train_ndcg=train_ndcg,
        selection=selection,
    )
