"""Check whether oracle candidate values (per record, per position) are
all mappable to vocab tokens. A candidate value that isn't in vocab is
silently dropped from the candidate_mask -> model can never emit it,
narrowing the reachable search space beyond the tokenizer's apparent vocab.
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
    print(f"[diag] task_index={task_index}")

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
    }

    # Resolve dynamic / innermost names per unique generator
    gen_info_cache = {}

    def _gen_info(record):
        key = (record.workload_key, record.target_kind, record.task_index, record.sketch_index)
        if key in gen_info_cache:
            return gen_info_cache[key]
        gen = _get_generator_for_record(record, registry)
        vec_step_indices = getattr(gen, "_vectorize_split_step_indices", set())
        innermost = getattr(gen, "_innermost_names", set())
        dyn = set()
        for name in gen._all_sp_names:
            try:
                step_idx = int(name.split("_")[1])
            except (ValueError, IndexError):
                continue
            if step_idx in vec_step_indices:
                dyn.add(name)
        gen_info_cache[key] = (gen, dyn, innermost)
        return gen_info_cache[key]

    # Build an oracle per record and iterate positions; compute candidate values
    # and count how many are missing from vocab.
    for split, records in split_records.items():
        total_cand = 0
        missed_cand = 0
        missed_kind = Counter()
        missed_values_by_name = defaultdict(lambda: Counter())
        pos_with_any_miss = 0
        pos_total = 0
        # Sample a subset to keep runtime manageable
        sample_records = records[: min(len(records), 200)]

        for record in sample_records:
            gen, dyn_names, innermost = _gen_info(record)
            oracle = registry.build_oracle_from_record(record)
            # Walk the gold path to preserve dependency constraints
            order = [n for n in record.params.keys() if n.startswith("sp_") or n.startswith("ur_")]
            # Use the prepared order from the bundle (oracle expects the canonical order)
            # Retrieve canonical order from generator
            order = list(gen._all_sp_names) + list(getattr(gen, "_ur_names", []))
            # Filter to what the record actually has
            order = [n for n in order if n in record.params]

            for name in order:
                pos_total += 1
                try:
                    cands = list(oracle.candidate_values(name))
                except Exception:
                    # Skip position if oracle can't enumerate
                    oracle.assign(name, int(record.params[name]))
                    continue
                missed_here = 0
                for v in cands:
                    total_cand += 1
                    token = tok.value_to_token(name, int(v))
                    if tok.token_to_id.get(token, tok.unk_id) == tok.unk_id:
                        missed_cand += 1
                        missed_here += 1
                        if name.startswith("ur_"):
                            kind = "ur"
                        elif name in dyn_names:
                            kind = "sp_dynamic"
                        elif name in innermost:
                            kind = "sp_innermost"
                        else:
                            kind = "sp_static"
                        missed_kind[kind] += 1
                        missed_values_by_name[name][int(v)] += 1
                if missed_here > 0:
                    pos_with_any_miss += 1
                oracle.assign(name, int(record.params[name]))

        print(f"\n=== split={split} (sampled {len(sample_records)}/{len(records)}) ===")
        print(f"  total candidate checks  : {total_cand}")
        print(f"  candidate UNK count     : {missed_cand} "
              f"({100 * missed_cand / max(1,total_cand):.2f}%)")
        print(f"  positions with any miss : {pos_with_any_miss}/{pos_total} "
              f"({100 * pos_with_any_miss / max(1,pos_total):.2f}%)")
        print(f"  miss breakdown by kind  : {dict(missed_kind)}")
        # Top missing values per name
        for name, ctr in sorted(missed_values_by_name.items()):
            top = ctr.most_common(10)
            print(f"    {name}: top-missing = {top}")


if __name__ == "__main__":
    main()
