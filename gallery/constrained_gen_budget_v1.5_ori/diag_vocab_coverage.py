"""Diagnose vocab coverage holes: for each record across train/val/test,
identify gold (name, value) pairs whose token is UNK in the built tokenizer.
Breaks down by split and by dynamic (vectorize-followed) vs static SP.
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from latent_model_budget.config import build_config, resolve_task_paths
from latent_model_budget.adapter import GeneratorRegistry
from latent_model_budget.dataset import (
    build_dataset_bundle,
    _get_generator_for_record,
)


def main():
    task_index = int(sys.argv[1]) if len(sys.argv) > 1 else 1490
    cfg = build_config()
    cfg.train.precompute_candidate_masks = False
    cfg.data.task_index = task_index
    resolve_task_paths(cfg)
    print(f"[diag] task_index={task_index} json_paths={len(cfg.data.json_paths)}")

    registry = GeneratorRegistry(
        cfg.data.network_info_folder,
        hw_param=getattr(cfg.generator, "hw_param", None),
        disable_constraint=getattr(cfg.generator, "disable_constraint", None),
    )
    bundle = build_dataset_bundle(cfg, registry)
    tok = bundle.tokenizer
    print(f"[diag] vocab={len(tok.id_to_token)} vars={len(tok.id_to_var)}")

    split_records = {
        "train": list(bundle.train_records),
        "val": list(bundle.val_records),
        "test": list(bundle.test_records),
    }

    # Figure out dynamic SP names per (workload, task, sketch) via generator cache.
    dyn_name_cache = {}

    def _dynamic_names_for(record):
        key = (record.workload_key, record.target_kind, record.task_index, record.sketch_index)
        if key in dyn_name_cache:
            return dyn_name_cache[key]
        gen = _get_generator_for_record(record, registry)
        vec_step_indices = getattr(gen, "_vectorize_split_step_indices", set())
        innermost = getattr(gen, "_innermost_names", set())
        dyn_names = set()
        for name in gen._all_sp_names:
            try:
                step_idx = int(name.split("_")[1])
            except (ValueError, IndexError):
                continue
            if step_idx in vec_step_indices:
                dyn_names.add(name)
        dyn_name_cache[key] = (dyn_names, innermost)
        return dyn_name_cache[key]

    report = {}
    for split, records in split_records.items():
        total_positions = 0
        unk_positions = 0
        unk_by_kind = Counter()
        unk_examples = defaultdict(list)  # (kind, name) -> [values]
        missing_values_by_name = defaultdict(set)

        for record in records:
            dyn_names, innermost = _dynamic_names_for(record)
            order = list(record.params.keys())
            # Use the same ordering the bundle actually prepared (record.params may
            # contain budget entries etc.); stick to SP and UR only.
            for name, value in record.params.items():
                if not (name.startswith("sp_") or name.startswith("ur_")):
                    continue
                total_positions += 1
                token = tok.value_to_token(name, int(value))
                tok_id = tok.token_to_id.get(token, tok.unk_id)
                if tok_id == tok.unk_id:
                    unk_positions += 1
                    if name.startswith("ur_"):
                        kind = "ur"
                    elif name in dyn_names:
                        kind = "sp_dynamic"
                    elif name in innermost:
                        kind = "sp_innermost"
                    else:
                        kind = "sp_static"
                    unk_by_kind[kind] += 1
                    unk_examples[(kind, name)].append(int(value))
                    missing_values_by_name[name].add(int(value))

        report[split] = {
            "n_records": len(records),
            "total_positions": total_positions,
            "unk_positions": unk_positions,
            "unk_rate": (unk_positions / total_positions) if total_positions else 0.0,
            "unk_by_kind": dict(unk_by_kind),
            "missing_values_by_name": {k: sorted(v) for k, v in missing_values_by_name.items()},
        }

    for split, r in report.items():
        print(f"\n=== split={split} ({r['n_records']} records) ===")
        print(f"  total gold positions : {r['total_positions']}")
        print(f"  UNK positions        : {r['unk_positions']} "
              f"({100 * r['unk_rate']:.2f}%)")
        print(f"  UNK breakdown        : {r['unk_by_kind']}")
        for name, values in sorted(r["missing_values_by_name"].items()):
            print(f"    {name}: missing gold values = {values[:20]}"
                  f"{' ...' if len(values) > 20 else ''} (unique={len(values)})")


if __name__ == "__main__":
    main()
