"""Periodic GP fit on (mu, cost) for recon-predict scoring.

The GP is a *separate* inference-only predictor that swaps in for
``LatentParamVAE.cost_head`` only inside the latent walk's recon-predict path.
It never touches the VAE's parameters and is rebuilt from scratch each cycle so
the VAE's training is completely unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .dataset import collate_prepared_samples, PreparedSample
from .model import LatentParamVAE
from .tokenizer import ParamTokenizer


SymMapKey = Tuple[Tuple[str, int], ...]
# Task-aware cache key: (workload_key, sym_map). Including workload_key in the
# key prevents cross-task contamination of the measurement cache — distinct
# tasks frequently produce overlapping sym_maps (e.g. small integer tile sizes)
# and without the prefix a small workload's fast cost would leak into a larger
# workload's walk via the sym_map-only lookup.
TaskSymMapKey = Tuple[str, SymMapKey]


def make_sym_map_key(sym_map: Dict[str, int]) -> SymMapKey:
    return tuple(sorted((str(k), int(v)) for k, v in sym_map.items()))


def make_task_sym_map_key(
    workload_key: Optional[str], sym_map: Dict[str, int]
) -> TaskSymMapKey:
    """Task-aware cache key. ``workload_key`` is normalized to ``""`` when
    missing so the tuple shape is stable and comparable."""
    return (str(workload_key or ""), make_sym_map_key(sym_map))


class WalkSampleBuffer:
    """Dedup buffer of (task_sym_map -> PreparedSample) collected from latent
    walks.

    Each entry corresponds to a measured walk candidate whose cost is the
    measured ``mean_cost`` (already in negative-log scale, matching dataset
    costs). Used to augment the GP training set on subsequent fits.
    """

    def __init__(self) -> None:
        self._samples: Dict[TaskSymMapKey, PreparedSample] = {}

    def __len__(self) -> int:
        return len(self._samples)

    def __contains__(self, key: TaskSymMapKey) -> bool:
        return key in self._samples

    def add(self, key: TaskSymMapKey, sample: PreparedSample) -> None:
        # latest walk wins for the same (workload_key, sym_map)
        self._samples[key] = sample

    def samples(self) -> List[PreparedSample]:
        return list(self._samples.values())

    def items(self):
        return self._samples.items()


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

    def predict_with_std(self, z: torch.Tensor) -> Tuple[float, float]:
        """Posterior mean and standard deviation at z."""
        z_np = z.detach().to(dtype=torch.float32).cpu().numpy().reshape(1, -1)
        mean, std = self.gp.predict(z_np, return_std=True)
        return float(mean[0]), float(std[0])


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
def _encode_samples(
    model: LatentParamVAE,
    samples: Sequence[PreparedSample],
    tokenizer: ParamTokenizer,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    samples = list(samples)
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


@torch.no_grad()
def _collect_mu_and_cost(
    model: LatentParamVAE,
    dataset,
    tokenizer: ParamTokenizer,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    return _encode_samples(model, list(dataset.samples), tokenizer, device, batch_size)


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
    walk_buffer: Optional[WalkSampleBuffer] = None,
) -> Optional[GPReconPredictor]:
    """Fit a GP on (deterministic z, cost) for the selected training subset.

    The resulting predictor is intended to be passed to the latent walk and
    used in place of ``cost_head`` for the recon-predict step. The VAE itself
    stays in eval mode and its parameters are not touched.

    When ``walk_buffer`` is supplied, its samples (one per unique sym_map seen
    across prior walks, with measured mean_cost) are encoded with the current
    encoder and concatenated to the training subset.
    """
    was_training = model.training
    model.eval()
    try:
        mu, costs = _collect_mu_and_cost(model, dataset, tokenizer, device, batch_size)
        indices, selection = _select_indices(
            costs, top_k=top_k, random_n=random_n, seed=seed
        )
        if mu.shape[0] == 0 or indices.shape[0] == 0:
            x_train = np.empty((0, mu.shape[1] if mu.ndim == 2 else 0), dtype=np.float32)
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
        print("[gp-recon] no samples available (train+walk); skipping GP fit")
        return None

    if walk_added > 0:
        selection = f"{selection}+walk{walk_added}"

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
    pred, pred_std = gp.predict(x_train, return_std=True)
    mse = float(np.mean((pred - y_train) ** 2))
    train_std_mean = float(np.mean(pred_std))
    train_std_max = float(np.max(pred_std))
    print(
        f"[gp-recon] GP fitted: selection={selection} "
        f"n={int(x_train.shape[0])} dim={int(x_train.shape[1])} "
        f"train_mse={mse:.6f} train_std_mean={train_std_mean:.6f} "
        f"train_std_max={train_std_max:.6f}"
    )
    return GPReconPredictor(
        gp=gp,
        num_samples=int(x_train.shape[0]),
        train_mse=mse,
        selection=selection,
    )
