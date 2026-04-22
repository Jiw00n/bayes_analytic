"""Smoke test: verify baseline-mask diff catches the expected number of rows."""
from __future__ import annotations
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from latent_model_budget import train as train_module
from latent_model_budget.config import build_config, resolve_task_paths
from latent_model_budget.dataset import collate_prepared_samples
from latent_model_budget.train_eval import (
    _batch_to_device,
    _build_teacher_forcing_candidate_masks,
)

import diag_vthread_violation as diag


def main() -> None:
    cfg = build_config()
    cfg.data.task_index = 1490
    resolve_task_paths(cfg)
    cfg.generator.hw_param = {"max_vthread_extent": 15}
    cfg.generator.disable_constraint = []

    registry = train_module.GeneratorRegistry(
        cfg.data.network_info_folder,
        hw_param=cfg.generator.hw_param,
        disable_constraint=cfg.generator.disable_constraint,
    )
    bundle = train_module.build_dataset_bundle(cfg, registry)
    tokenizer = bundle.tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DataLoader(
        bundle.train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda items: collate_prepared_samples(items, tokenizer),
    )

    total_rows = 0
    total_positions = 0
    seen = 0
    for batch in loader:
        batch = _batch_to_device(batch, device)
        cand_masks = _build_teacher_forcing_candidate_masks(
            batch, registry, tokenizer, device=device, debug_invalid_step=False,
        )
        scale, n_rows, n_pos = diag._baseline_violation_scale(
            batch, cfg, tokenizer,
            device=device,
            max_len=int(batch["target_ids"].shape[1]),
            current_candidate_masks=cand_masks,
            violation_weight=0.0,
        )
        total_rows += n_rows
        total_positions += n_pos
        seen += int(batch["target_ids"].shape[0])
        print(
            f"[smoke] batch samples={seen} rows_downweighted={total_rows} "
            f"positions_downweighted={total_positions}"
        )

    print()
    print(f"[smoke] TOTAL train samples scanned = {seen}")
    print(f"[smoke] TOTAL rows downweighted   = {total_rows}")
    print(f"[smoke] TOTAL positions downweighted = {total_positions}")


if __name__ == "__main__":
    main()
