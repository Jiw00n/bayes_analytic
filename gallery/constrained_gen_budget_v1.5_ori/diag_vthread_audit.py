"""Audit why `violating rows` count looks too small.

For task 1490, load the training dataset with `max_vthread_extent=15`, walk
every training sample, resolve its generator, and report:
- how many samples have a non-empty `_vthread_clamped_sp_names`
- the distribution of gold values at those clamped positions
- how many rows have any clamped gold > 8  (this should match the diag counter)
- sample-level mismatches (clamped set empty when it shouldn't be, etc.)
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from latent_model_budget.config import build_config, resolve_task_paths
from latent_model_budget import train as train_module


TASK_INDEX = 1490
THRESHOLD = 8


def main() -> None:
    cfg = build_config()
    cfg.data.task_index = TASK_INDEX
    resolve_task_paths(cfg)
    cfg.generator.hw_param = {"max_vthread_extent": 15}
    cfg.generator.disable_constraint = []

    registry = train_module.GeneratorRegistry(
        cfg.data.network_info_folder,
        hw_param=cfg.generator.hw_param,
        disable_constraint=cfg.generator.disable_constraint,
    )
    bundle = train_module.build_dataset_bundle(cfg, registry)

    train_ds = bundle.train_dataset
    print(f"[audit] train samples = {len(train_ds)}")

    rows_total = 0
    rows_with_clamped = 0
    rows_clamped_empty = 0
    rows_any_clamped_gt_thresh = 0
    rows_product_gt_thresh = 0
    product_hist: Counter = Counter()
    value_hist_at_clamped: Counter = Counter()
    clamped_hits: Counter = Counter()  # which clamped name matched
    sample_signatures: Counter = Counter()

    first_examples_printed = 0

    for idx in range(len(train_ds)):
        item = train_ds[idx]
        names = list(item.ordered_param_names)
        values = list(item.ordered_param_values)
        workload_key = item.workload_key
        target_kind = item.target_kind
        sketch_index = int(item.sketch_index)

        gen = registry.get_generator(
            workload_key=workload_key,
            target_kind=target_kind,
            sketch_index=sketch_index,
        )
        clamped = set(getattr(gen, "_vthread_clamped_sp_names", set()))
        sample_signatures[(str(workload_key), str(target_kind), sketch_index, tuple(sorted(clamped)))] += 1

        rows_total += 1
        if clamped:
            rows_with_clamped += 1
        else:
            rows_clamped_empty += 1

        any_violation = False
        product = 1
        for name, value in zip(names, values):
            if name in clamped:
                value_hist_at_clamped[int(value)] += 1
                clamped_hits[name] += 1
                product *= int(value)
                if int(value) > THRESHOLD:
                    any_violation = True
        if any_violation:
            rows_any_clamped_gt_thresh += 1
        if product > THRESHOLD:
            rows_product_gt_thresh += 1
        product_hist[int(product)] += 1

        if first_examples_printed < 3:
            print(
                f"[audit] sample {idx}: sketch={sketch_index} "
                f"clamped={sorted(clamped)} "
                f"names[:12]={names[:12]} values[:12]={values[:12]}"
            )
            first_examples_printed += 1

    print()
    print(f"[audit] rows total                    = {rows_total}")
    print(f"[audit] rows with non-empty clamped   = {rows_with_clamped}")
    print(f"[audit] rows with empty clamped       = {rows_clamped_empty}")
    print(f"[audit] rows any clamped value > {THRESHOLD}    = {rows_any_clamped_gt_thresh}")
    print(f"[audit] rows clamped PRODUCT > {THRESHOLD}       = {rows_product_gt_thresh}")
    print()
    print("[audit] clamped-value product distribution (top 30):")
    for p, c in sorted(product_hist.items()):
        marker = "  <-- violating" if p > THRESHOLD else ""
        print(f"        product={p:>6}  count={c}{marker}")
    print()
    print("[audit] value distribution at clamped positions (top 20):")
    for v, c in sorted(value_hist_at_clamped.items()):
        print(f"        value={v:>4}  count={c}")
    print()
    print("[audit] clamped name hit counts:")
    for name, c in clamped_hits.most_common():
        print(f"        {name}: {c}")
    print()
    print("[audit] distinct (workload, target, sketch, clamped) signatures:")
    for sig, c in sample_signatures.most_common():
        print(f"        n={c}  sketch={sig[2]}  clamped={list(sig[3])}")


if __name__ == "__main__":
    main()
