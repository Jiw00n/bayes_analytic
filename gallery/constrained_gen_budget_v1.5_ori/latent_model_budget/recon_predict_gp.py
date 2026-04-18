"""Periodic GP fit on (mu, cost) for recon-predict scoring.

The GP is a *separate* inference-only predictor that swaps in for
``LatentParamVAE.cost_head`` only inside the latent walk's recon-predict path.
It never touches the VAE's parameters and is rebuilt from scratch each cycle so
the VAE's training is completely unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .dataset import collate_prepared_samples
from .model import LatentParamVAE
from .tokenizer import ParamTokenizer


@dataclass
class GPReconPredictor:
    """Inference-only GP that maps deterministic z (=mu) to predicted cost."""

    gp: GaussianProcessRegressor
    num_samples: int
    train_mse: float
    selection: str

    def predict(self, z: torch.Tensor) -> float:
        z_np = z.detach().to(dtype=torch.float32).cpu().numpy().reshape(1, -1)
        return float(self.gp.predict(z_np)[0])


def _select_indices(
    costs: np.ndarray,
    *,
    top_k: int,
    random_n: int,
    seed: int,
) -> tuple[np.ndarray, str]:
    """Return indices for top-K (by cost desc) plus disjoint random-N samples."""
    n = int(costs.shape[0])
    if n == 0:
        return np.empty((0,), dtype=np.int64), "empty"

    top_k = max(0, int(top_k))
    random_n = max(0, int(random_n))

    order = np.argsort(-costs)  # descending: highest neg_log_cost = best perf
    top_idx = order[: min(top_k, n)]

    picked = set(int(x) for x in top_idx.tolist())
    rng = np.random.default_rng(int(seed))
    remaining = np.array(
        [i for i in range(n) if i not in picked],
        dtype=np.int64,
    )
    take = min(random_n, remaining.shape[0])
    if take > 0:
        rand_idx = rng.choice(remaining, size=take, replace=False)
    else:
        rand_idx = np.empty((0,), dtype=np.int64)

    selection = f"top{top_idx.shape[0]}+rand{rand_idx.shape[0]}"
    return np.concatenate([top_idx, rand_idx], axis=0), selection


@torch.no_grad()
def _collect_mu_and_cost(
    model: LatentParamVAE,
    dataset,
    tokenizer: ParamTokenizer,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    samples = list(dataset.samples)
    stride = max(int(batch_size), 1)
    mu_chunks: List[np.ndarray] = []
    cost_chunks: List[np.ndarray] = []
    for start in range(0, len(samples), stride):
        chunk = samples[start : start + stride]
        batch = collate_prepared_samples(chunk, tokenizer)
        valid_mask = batch["cost_mask"]
        if not bool(valid_mask.any()):
            continue
        enc_ids = batch["encoder_token_ids"].to(device, non_blocking=device.type == "cuda")
        enc_var_ids = batch["encoder_var_ids"].to(device, non_blocking=device.type == "cuda")
        enc_pad = enc_ids.eq(tokenizer.pad_id)
        mu, _, _, _ = model.encode(enc_ids, enc_var_ids, enc_pad, deterministic=True)
        valid_mask_device = valid_mask.to(device=device, non_blocking=device.type == "cuda")
        mu_chunks.append(
            mu[valid_mask_device].detach().cpu().to(dtype=torch.float32).numpy()
        )
        cost_chunks.append(
            batch["costs"][valid_mask].detach().cpu().to(dtype=torch.float32).numpy()
        )
    if not mu_chunks:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.concatenate(mu_chunks, axis=0), np.concatenate(cost_chunks, axis=0)


def fit_gp_recon_predictor(
    *,
    model: LatentParamVAE,
    dataset,
    tokenizer: ParamTokenizer,
    device: torch.device,
    top_k: int,
    random_n: int,
    batch_size: int = 128,
    seed: int = 0,
) -> Optional[GPReconPredictor]:
    """Fit a GP on (deterministic z, cost) for the selected training subset.

    The resulting predictor is intended to be passed to the latent walk and
    used in place of ``cost_head`` for the recon-predict step. The VAE itself
    stays in eval mode and its parameters are not touched.
    """
    was_training = model.training
    model.eval()
    try:
        mu, costs = _collect_mu_and_cost(model, dataset, tokenizer, device, batch_size)
    finally:
        if was_training:
            model.train()

    if mu.shape[0] == 0:
        print("[gp-recon] no valid (mu, cost) samples available; skipping GP fit")
        return None

    indices, selection = _select_indices(
        costs, top_k=top_k, random_n=random_n, seed=seed
    )
    if indices.shape[0] == 0:
        print("[gp-recon] selection produced 0 samples; skipping GP fit")
        return None

    x_train = mu[indices]
    y_train = costs[indices]

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1.0))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=int(seed),
    )
    gp.fit(x_train, y_train)
    pred = gp.predict(x_train)
    mse = float(np.mean((pred - y_train) ** 2))
    print(
        f"[gp-recon] GP fitted: selection={selection} "
        f"n={int(x_train.shape[0])} dim={int(x_train.shape[1])} "
        f"train_mse={mse:.6f}"
    )
    return GPReconPredictor(
        gp=gp,
        num_samples=int(x_train.shape[0]),
        train_mse=mse,
        selection=selection,
    )
