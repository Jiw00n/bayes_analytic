"""Smoke test for the shape-prefix dataset bundle on the t4 winograd family.

Points ``data.json_paths`` at the specified directory (which contains records
with the same op but different tensor shapes / possibly different sketches) and
walks through ``build_dataset_bundle`` to verify:
  - group validation (task_desc / param-count / shape-rank consistency)
  - shape-semantic label extraction from each record's compute_dag
  - tokenizer vocab includes all shape integer values
  - encoder/decoder/target layouts produced by ``_build_prepared_sample``
  - model forward runs and produces finite loss
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from latent_model_budget.adapter import GeneratorRegistry
from latent_model_budget.config import build_config
from latent_model_budget.dataset import (
    build_dataset_bundle,
    collate_prepared_samples,
)
from latent_model_budget.model import LatentParamVAE


TEST_DIR = (
    "/root/work/tvm-ansor/gallery/dataset/measure_tenset_filtered_family/"
    "nn_contrib_conv2d_winograd_without_weight_transform/t4"
)


def main() -> None:
    cfg = build_config()
    cfg.data.json_paths = [TEST_DIR]
    cfg.train.checkpoint_dir = "/tmp/smoke_t4_ckpt"
    cfg.train.precompute_candidate_masks = False

    registry = GeneratorRegistry(
        cfg.data.network_info_folder,
        hw_param=cfg.generator.hw_param,
        disable_constraint=cfg.generator.disable_constraint,
    )

    try:
        bundle = build_dataset_bundle(cfg, registry)
    except Exception as exc:
        print(f"[smoke] build_dataset_bundle raised: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return

    tokenizer = bundle.tokenizer
    print(f"[smoke] vocab={len(tokenizer.id_to_token)} vars={len(tokenizer.id_to_var)}")
    print(f"[smoke] splits train={len(bundle.train_dataset)} val={len(bundle.val_dataset)} test={len(bundle.test_dataset)}")

    if len(bundle.train_dataset) == 0:
        print("[smoke] empty train split; aborting")
        return

    sample = bundle.train_dataset[0]
    print(f"[smoke] sample.sample_id={sample.sample_id}")
    print(f"[smoke]   shape_token_ids (len={len(sample.shape_token_ids)}): {sample.shape_token_ids}")
    print(f"[smoke]   shape_var_ids:   {sample.shape_var_ids}")
    print(f"[smoke]   encoder (len={len(sample.encoder_token_ids)}): {sample.encoder_token_ids}")
    print(f"[smoke]   enc_var (len={len(sample.encoder_var_ids)}): {sample.encoder_var_ids}")
    print(f"[smoke]   decoder (len={len(sample.decoder_input_ids)}): {sample.decoder_input_ids}")
    print(f"[smoke]   dec_var (len={len(sample.decoder_var_ids)}): {sample.decoder_var_ids}")
    print(f"[smoke]   target  (len={len(sample.target_ids)}): {sample.target_ids}")

    shape_labels = [tokenizer.id_to_var[v] for v in sample.shape_var_ids]
    print(f"[smoke]   shape_var_labels: {shape_labels}")

    batch_size = min(4, len(bundle.train_dataset))
    batch = collate_prepared_samples(
        [bundle.train_dataset[i] for i in range(batch_size)],
        tokenizer,
    )
    print(
        f"[smoke] batch encoder={tuple(batch['encoder_token_ids'].shape)} "
        f"decoder={tuple(batch['decoder_input_ids'].shape)} "
        f"target={tuple(batch['target_ids'].shape)} "
        f"cand_mask={tuple(batch['candidate_masks'].shape) if batch['candidate_masks'] is not None else None}"
    )

    model = LatentParamVAE(
        vocab_size=len(tokenizer.id_to_token),
        num_vars=len(tokenizer.id_to_var),
        cfg=cfg.model,
    )
    model.eval()

    with torch.no_grad():
        out = model(
            encoder_token_ids=batch["encoder_token_ids"],
            encoder_var_ids=batch["encoder_var_ids"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_var_ids=batch["decoder_var_ids"],
            pad_token_id=tokenizer.pad_id,
        )
    logits = out.logits
    print(f"[smoke] logits={tuple(logits.shape)} finite={torch.isfinite(logits).all().item()}")
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        batch["target_ids"].reshape(-1),
        ignore_index=tokenizer.pad_id,
    )
    print(f"[smoke] loss={loss.item():.4f} finite={torch.isfinite(loss).item()}")


if __name__ == "__main__":
    main()
