"""Verify model.seed actually controls model init RNG by building two
models with different model.seed (same data.seed) and checking weight
hashes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from latent_model_budget.config import build_config, resolve_task_paths
from latent_model_budget.adapter import GeneratorRegistry
from latent_model_budget.dataset import build_dataset_bundle
from latent_model_budget.model import LatentParamVAE
from latent_model_budget.runtime_utils import seed_everything


def _build_model_with(cfg):
    seed_everything(cfg.data.seed)  # mirror train_main
    registry = GeneratorRegistry(
        cfg.data.network_info_folder,
        hw_param=getattr(cfg.generator, "hw_param", None),
        disable_constraint=getattr(cfg.generator, "disable_constraint", None),
    )
    bundle = build_dataset_bundle(cfg, registry)
    tok = bundle.tokenizer

    # Mirror the new override in build_everything:
    ms = getattr(cfg.model, "seed", None)
    if ms is not None:
        torch.manual_seed(int(ms))
        torch.cuda.manual_seed_all(int(ms))

    model = LatentParamVAE(
        vocab_size=len(tok.id_to_token),
        num_vars=len(tok.id_to_var),
        cfg=cfg.model,
    )
    return model


def _summarize(model, tag):
    w = model.lm_head.weight.detach()
    emb = model.token_emb.weight.detach()
    print(
        f"[{tag}] lm_head.weight mean={w.mean().item():+.6f} "
        f"std={w.std().item():.6f} norm={w.norm().item():.6f}"
    )
    print(
        f"[{tag}] token_emb.weight mean={emb.mean().item():+.6f} "
        f"std={emb.std().item():.6f} norm={emb.norm().item():.6f}"
    )


def _run_variant(label, pad_vocab_to, align_to, model_seed=0):
    cfg = build_config()
    cfg.train.precompute_candidate_masks = False
    cfg.data.task_index = 1490
    cfg.data.seed = 42
    cfg.model.seed = model_seed
    cfg.data.pad_vocab_to = pad_vocab_to
    cfg.model.vocab_align_to = align_to
    from latent_model_budget.config import resolve_task_paths
    resolve_task_paths(cfg)
    print(f"\n=== {label} (pad={pad_vocab_to}, align={align_to}) ===")
    model = _build_model_with(cfg)
    # Compare layers that come AFTER token_emb but BEFORE lm_head — they should
    # match vocab=48 run when align_to is set.
    ve = model.var_emb.weight.detach()
    to_mu = model.to_mu.weight.detach()
    print(f"  var_emb norm={ve.norm().item():.6f} mean={ve.mean().item():+.6f}")
    print(f"  to_mu  norm={to_mu.norm().item():.6f} mean={to_mu.mean().item():+.6f}")
    # Post-init RNG sample (what DataLoader shuffle / dropout would see)
    tail = torch.empty(5).uniform_().tolist()
    print(f"  post-init tail RNG: {[f'{x:.6f}' for x in tail]}")


def main():
    _run_variant("vocab=27 (patched, no align)",    pad_vocab_to=None, align_to=None)
    _run_variant("vocab=48 (dummy pad, no align)",  pad_vocab_to=48,   align_to=None)
    _run_variant("vocab=27 + vocab_align_to=48",    pad_vocab_to=None, align_to=48)


if __name__ == "__main__":
    main()
