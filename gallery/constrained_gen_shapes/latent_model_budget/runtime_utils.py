from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .dataset import _generator_cache_suffix, collate_prepared_samples
from .model import LatentParamVAE
from .tokenizer import ParamTokenizer

if TYPE_CHECKING:
    from .dataset import DatasetBundle


class TaskBalancedBatchSampler(Sampler[List[int]]):
    """Yield batches whose composition is constrained to ``tasks_per_batch``
    distinct task ids, each contributing ``batch_size // tasks_per_batch``
    samples. Tasks are sampled uniformly without replacement per batch; each
    task's index pool is reshuffled when exhausted so the sampler tolerates
    unequal task sizes. Samples with ``task_index is None`` are bucketed under
    the ``-1`` sentinel and treated as their own task.
    """

    def __init__(
        self,
        task_indices: Sequence[Optional[int]],
        batch_size: int,
        tasks_per_batch: int,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if tasks_per_batch <= 0:
            raise ValueError(f"tasks_per_batch must be positive, got {tasks_per_batch}")
        if batch_size % tasks_per_batch != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by "
                f"tasks_per_batch ({tasks_per_batch})"
            )

        groups: dict[int, List[int]] = defaultdict(list)
        for idx, tid in enumerate(task_indices):
            key = -1 if tid is None else int(tid)
            groups[key].append(idx)

        self._groups: dict[int, List[int]] = {k: list(v) for k, v in groups.items()}
        self._task_ids: List[int] = sorted(self._groups.keys())
        if tasks_per_batch > len(self._task_ids):
            raise ValueError(
                f"tasks_per_batch ({tasks_per_batch}) exceeds number of "
                f"distinct tasks in dataset ({len(self._task_ids)})"
            )

        self._batch_size = int(batch_size)
        self._tasks_per_batch = int(tasks_per_batch)
        self._samples_per_task = self._batch_size // self._tasks_per_batch

        for tid, pool in self._groups.items():
            if len(pool) < self._samples_per_task:
                raise ValueError(
                    f"task {tid} has {len(pool)} samples, fewer than "
                    f"samples_per_task ({self._samples_per_task}); reduce "
                    f"batch_size or tasks_per_batch"
                )

        self._shuffle = bool(shuffle)
        self._seed = seed
        self._epoch = 0
        total = sum(len(v) for v in self._groups.values())
        self._length = total // self._batch_size

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[List[int]]:
        if self._seed is None:
            rng = random.Random()
        else:
            rng = random.Random(int(self._seed) + self._epoch)

        pools: dict[int, List[int]] = {
            tid: list(idxs) for tid, idxs in self._groups.items()
        }
        cursors: dict[int, int] = {tid: 0 for tid in self._task_ids}
        if self._shuffle:
            for pool in pools.values():
                rng.shuffle(pool)

        for _ in range(self._length):
            if self._shuffle:
                chosen_tasks = rng.sample(self._task_ids, self._tasks_per_batch)
            else:
                chosen_tasks = self._task_ids[: self._tasks_per_batch]
            batch: List[int] = []
            for tid in chosen_tasks:
                pool = pools[tid]
                cur = cursors[tid]
                if cur + self._samples_per_task > len(pool):
                    if self._shuffle:
                        rng.shuffle(pool)
                    cur = 0
                batch.extend(pool[cur : cur + self._samples_per_task])
                cursors[tid] = cur + self._samples_per_task
            yield batch

        self._epoch += 1


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
    batch_sampler: Optional[Sampler] = None,
):
    kwargs = dict(
        dataset=dataset,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_prepared_samples(batch, tokenizer),
    )
    if batch_sampler is not None:
        kwargs["batch_sampler"] = batch_sampler
    else:
        kwargs["batch_size"] = batch_size
        kwargs["shuffle"] = shuffle
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


# ---------------------------------------------------------------------------
# Naming / path helpers (formerly in train.py)
# ---------------------------------------------------------------------------

def _wandb_section(key: str) -> str:
    """Group metric keys into wandb sections (train/, val/, walk/, ...)."""
    if "/" in key:
        return key
    if key == "epoch":
        return key
    section_prefixes = (
        ("train_", "train/"),
        ("val_", "val/"),
        ("eval_val_", "eval_val/"),
        ("eval_test_", "eval_test/"),
        ("test_", "test/"),
        ("final_", "final/"),
    )
    for old, new in section_prefixes:
        if key.startswith(old):
            return new + key[len(old):]
    return f"train/{key}"


def _remap_for_wandb(metrics: Dict) -> Dict:
    return {_wandb_section(k): v for k, v in metrics.items()}


def _family_from_json_path(json_path: str | Path) -> str:
    """Extract the dataset family name from a measure-record JSON path.

    Layout: ``.../measure_tenset_filtered_family/{family}/{target}/{N}_*.json``
    so the family is ``parents[1].name``. Returns "na" if the layout is too
    shallow.
    """
    p = Path(json_path)
    if len(p.parents) >= 2:
        return p.parents[1].name
    return "na"


def _resolve_run_family(bundle: "DatasetBundle") -> str:
    """Family identifier for run-level naming (wandb / checkpoints).
    Per-task / per-workload artifacts (e.g. the candidate_mask_cache) and the
    measurement lookup file intentionally keep task-specific names and do
    NOT use this helper.
    """
    all_records = (
        list(bundle.train_records) + list(bundle.val_records) + list(bundle.test_records)
    )
    for record in all_records:
        json_path = getattr(record, "json_path", None)
        if not json_path:
            continue
        family = _family_from_json_path(json_path)
        if family != "na":
            return family
    return "na"


def _resolve_run_family_from_config(config) -> str:
    for p in getattr(config.data, "json_paths", []) or []:
        family = _family_from_json_path(p)
        if family != "na":
            return family
    return "na"


def _build_wandb_project_name(config, bundle: Optional["DatasetBundle"] = None) -> str:
    if bundle is not None:
        family = _resolve_run_family(bundle)
    else:
        family = _resolve_run_family_from_config(config)
    project_suffix = getattr(config.wandb, "project", None) or "single_v1"
    return f"{family}_{project_suffix}"


def _build_wandb_run_name(config, bundle: Optional["DatasetBundle"] = None) -> str:
    name = ""

    if config.model.num_encoder_layers != 4:
        name += f"_enc{config.model.num_encoder_layers}"
    if config.model.num_decoder_layers != 4:
        name += f"_dec{config.model.num_decoder_layers}"
    if config.model.nhead != 4:
        name += f"_head{config.model.nhead}"

    if config.model.latent_dim != 64:
        name += f"_zdim{config.model.latent_dim}"
    if config.model.dim_feedforward != 384:
        name += f"_fdim{config.model.dim_feedforward}"
    if config.model.cost_hidden_dim != 128:
        name += f"_cdim{config.model.cost_hidden_dim}"
    if config.model.latent_token_count != 4:
        name += f"_ztok{config.model.latent_token_count}"
    if config.model.pos_embedding_length != 2048:
        name += f"_pemb{config.model.pos_embedding_length}"

    if config.train.num_epochs != 100:
        name += f"_ep{config.train.num_epochs}"
    name += (
        f"_bs{config.train.batch_size}"
        f"_lr{config.train.learning_rate}"
        f"_wd{config.train.weight_decay}"
        f"_nce{config.train.lambda_nce}"
        f"_tau{config.train.tau_nce}"
        f"_kl{config.train.beta_end}"
        f"_bw{config.train.beta_warmup_epochs}"
    )

    if config.train.lambda_recon != 1.0:
        name += f"_lamr{config.train.lambda_recon}"
    if config.train.lambda_cost != 0.01:
        name += f"_lamc{config.train.lambda_cost}"
    if config.data.cost_target_regression:
        if config.data.cost_target_regression == "log_norm_throughput":
            name += "_reglognorm"
    if bool(getattr(config.train, "order_nce", False)):
        name += "_softnce"
    if bool(getattr(config.train, "nce_mu", False)):
        name += "_nce_mu"
    if bool(getattr(config.model, "adaln", False)):
        name += "_adaln"
    if bool(getattr(config.train, "cobo_sample_weighting", False)):
        cobo_tag = getattr(config.train, "cobo_apply_to", [])
        name += (
            f"_cobo{float(config.train.weight_quantile):.1f}"
            f"_{float(config.train.weight_sigma):.1f}"
            f"_{cobo_tag}"
        )
    if bool(getattr(config.train, "cost_ridge_weighted", False)):
        name += "_wridge"
        if not bool(getattr(config.train, "cobo_sample_weighting", False)):
            name += (
                f"{float(config.train.weight_quantile):.1f}"
                f"_{float(config.train.weight_sigma):.1f}"
            )
    ls = float(getattr(config.train, "label_smoothing", 0.0))
    if ls > 0.0:
        name += f"_ls{ls}"
    if bool(getattr(config.train, "order_nce_pos_weight_by_percentile", False)):
        name += f"_pos{float(config.train.order_nce_pos_weight_sigma):.1f}"
    if getattr(config.sampling, "strategy", "greedy") != "greedy":
        name += f"_{config.sampling.strategy}"
        if config.sampling.strategy == "sampling":
            name += (
                f"_t{config.sampling.temperature}"
                f"_k{config.sampling.top_k}"
            )
            if config.sampling.top_p != 1.0:
                name += f"_p{config.sampling.top_p}"
            name += f"_sseed{config.sampling.seed}"
    name += f"_dseed{config.data.seed}"
    name += f"_mseed{config.model.seed}"

    name += _generator_cache_suffix(config)
    return name


def _resolve_pt_dir(config) -> Path:
    """Mirror of the inline ``pt_dir`` derivation that runs once early so we
    can peek at ``last.pt`` (for resume) before ``wandb.init``. Uses the
    config-only family resolver — equivalent to the bundle-based one because
    bundle records originate from ``config.data.json_paths``.
    """
    base = Path(config.train.checkpoint_dir).expanduser().resolve()
    family = _resolve_run_family_from_config(config)
    if family and family != "na" and base.name != family:
        base = base / family
    base = base / "checkpoints"
    project = getattr(config.wandb, "project", None)
    if project:
        base = base / str(project)
    name = _build_wandb_run_name(config)
    if name:
        base = base / str(name)
    return base


def _config_for_resume_compare(payload: dict) -> dict:
    """Strip resume-related infra fields before comparing two config dumps.
    These are control flags that legitimately differ between original run
    and resume; everything else must match exactly.
    """
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    train = dict(out.get("train") or {})
    train.pop("resume", None)
    train.pop("resume_from", None)
    out["train"] = train
    return out
