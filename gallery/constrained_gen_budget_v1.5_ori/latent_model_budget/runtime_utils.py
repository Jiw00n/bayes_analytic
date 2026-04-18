from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import collate_prepared_samples
from .model import LatentParamVAE
from .tokenizer import ParamTokenizer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def configure_runtime(cfg, device: torch.device) -> None:
    if device.type != "cuda":
        return
    allow_tf32 = bool(getattr(cfg.train, "allow_tf32", True))
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def prepare_loader(
    dataset,
    tokenizer: ParamTokenizer,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_prepared_samples(batch, tokenizer),
    )
    if num_workers > 0:
        kwargs["pin_memory"] = bool(pin_memory)
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = int(prefetch_factor)
    else:
        kwargs["pin_memory"] = bool(pin_memory)
    return DataLoader(**kwargs)


def save_checkpoint(
    path: str | Path,
    *,
    model: LatentParamVAE,
    optimizer,
    scheduler,
    epoch: int,
    best_exact_match: float,
    config,
    tokenizer: ParamTokenizer,
    latent_cost_ridge: dict | None = None,
    latent_cost_ridges: list[dict] | None = None,
    timestamp,
    **extra_state,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "epoch": int(epoch),
        "best_exact_match": float(best_exact_match),
        "config": config.to_dict(),
        "tokenizer": tokenizer.to_state_dict(),
        "tokenizer_state": tokenizer.to_state_dict(),
        "latent_cost_ridge": latent_cost_ridge,
        "latent_cost_ridges": list(latent_cost_ridges) if latent_cost_ridges is not None else None,
        "timestamp": timestamp,
    }
    payload.update(extra_state)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None) -> dict:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    return payload


def save_training_artifacts(checkpoint_dir: str | Path, config, tokenizer: ParamTokenizer) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.save_json(checkpoint_dir / "config.json")
    (checkpoint_dir / "tokenizer.json").write_text(
        json.dumps(tokenizer.to_state_dict(), indent=2),
        encoding="utf-8",
    )
    return checkpoint_dir
