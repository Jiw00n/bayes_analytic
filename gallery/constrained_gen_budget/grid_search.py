from __future__ import annotations

import itertools
import json
from copy import deepcopy
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

if __package__ in (None, ""):
    sys.path.insert(0, str(HERE.parent))
    from latent_model_budget.config import build_config
    from latent_model_budget.model import LatentParamVAE
    import latent_model_budget.train as train_module
else:
    from .latent_model_budget.config import build_config
    from .latent_model_budget.model import LatentParamVAE
    from .latent_model_budget import train as train_module

TASK_INDEX = 1490


SEARCH_SPACE = {
    "train.learning_rate": [7e-4],
    "train.lambda_nce": [0.2],
    "train.tau_nce": [0.2],
    "train.beta_end": [0.002],
    "train.beta_warmup_epochs": [20],
    "train.order_nce": [False],
    "train.nce_mu": [False],
    "model.adaln": [True],
}

BEST_METRIC = "val_full_sequence_exact_match"
BEST_MODE = "max"
EVALUATE_AUTOREGRESSIVE_EACH_EPOCH = 0


_SHARED_DATASET_ARTIFACTS: dict[str, dict] = {}


def _dataset_cache_payload(cfg) -> dict:
    cfg_payload = cfg.to_dict() if hasattr(cfg, "to_dict") else {}
    return {
        "data": cfg_payload.get("data", {}),
        "precompute_candidate_masks": bool(getattr(cfg.train, "precompute_candidate_masks", False)),
    }


def _dataset_cache_key(cfg) -> str:
    payload = _dataset_cache_payload(cfg)
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _dataset_cache_tag(cfg) -> str:
    return hashlib.sha256(_dataset_cache_key(cfg).encode("utf-8")).hexdigest()[:12]


def _build_shared_dataset_artifacts(cfg) -> dict:
    cache_key = _dataset_cache_key(cfg)
    cached = _SHARED_DATASET_ARTIFACTS.get(cache_key)
    if cached is not None:
        print(
            f"[grid] reusing shared dataset artifacts "
            f"(cache_key={cached['cache_tag']})"
        )
        return cached

    cache_tag = _dataset_cache_tag(cfg)
    print(f"[grid] building shared dataset artifacts (cache_key={cache_tag})")
    registry = train_module.GeneratorRegistry(cfg.data.network_info_folder)
    bundle = train_module.build_dataset_bundle(cfg, registry)
    tokenizer = bundle.tokenizer
    cached = {
        "cache_tag": cache_tag,
        "registry": registry,
        "bundle": bundle,
        "tokenizer": tokenizer,
    }
    _SHARED_DATASET_ARTIFACTS[cache_key] = cached
    return cached


def _grid_build_everything(cfg):
    shared = _build_shared_dataset_artifacts(cfg)
    tokenizer = shared["tokenizer"]
    print(
        f"[grid] constructing model from shared dataset artifacts "
        f"(cache_key={shared['cache_tag']})"
    )
    model = LatentParamVAE(
        vocab_size=len(tokenizer.id_to_token),
        num_vars=len(tokenizer.id_to_var),
        cfg=cfg.model,
    )
    return shared["registry"], shared["bundle"], tokenizer, model


train_module.build_everything = _grid_build_everything
train_main = train_module.train_main


def set_nested_attr(obj, dotted_key: str, value):
    parts = dotted_key.split(".")
    cur = obj
    for part in parts[:-1]:
        cur = getattr(cur, part)
    setattr(cur, parts[-1], value)


def run_one(exp_idx: int, params: dict) -> None:
    
    cfg = build_config()
    from glob import glob
    json_paths = glob("/root/work/tvm-ansor/gallery/constrained_gen/data/measured_*/*.json")
    for p in json_paths:
        if f"/{TASK_INDEX}_" in p:
            cfg.data.json_paths = [p]
            cfg.train.checkpoint_dir = str(Path(cfg.train.checkpoint_dir) / f"{TASK_INDEX}" / "grid_search")
    cfg.train.num_epochs = 100
    cfg.train.early_stop_min_delta = 1e-4
    cfg.train.best_metric_name = BEST_METRIC
    cfg.train.best_metric_mode = BEST_MODE
    cfg.train.evaluate_autoregressive_each_epoch = EVALUATE_AUTOREGRESSIVE_EACH_EPOCH

    for k, v in params.items():
        set_nested_attr(cfg, k, v)

    try:
        train_main(cfg)
    except Exception as e:
        print(f"[grid] run failed: {type(e).__name__}: {e}")


def main():
    keys = list(SEARCH_SPACE.keys())
    values = [SEARCH_SPACE[k] for k in keys]
    combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    for idx, params in enumerate(combos, start=1):
        print(f"\n===== [{idx}/{len(combos)}] {params} =====")
        run_one(idx, deepcopy(params))


if __name__ == "__main__":
    main()
