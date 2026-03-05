#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verification/benchmark script for constraint_formula.py."""

import argparse
import json
import random
import statistics
import time
from collections import defaultdict

from constraint_formula import (
    DEFAULT_HW,
    build_system,
    build_task_map,
    evaluate_record,
    get_storage_rewrite_merge_report,
    lower_with_gpu_passes,
    randomize_record_params,
    record_to_task_state,
    verify_gpu_module,
)


def _load_records_grouped(log_path):
    groups = defaultdict(list)
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            wk = rec["i"][0][0]
            groups[wk].append(rec)
    return groups


def _verify_record_with_lowering(task_map, rec, hw):
    try:
        task, state = record_to_task_state(rec, task_map)
        mod = lower_with_gpu_passes(task, state)
        return verify_gpu_module(mod, hw)
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        type=str,
        default="/root/work/tvm-ansor/gallery/logs_json/resnet_18/resnet_18-B1.json",
    )
    parser.add_argument(
        "--tasks-pkl",
        type=str,
        default="/root/work/tvm-ansor/gallery/ansor_tasks_pkl/resnet_18-(1,224,224,3).pkl",
    )
    parser.add_argument("--max-tasks", type=int, default=24)
    parser.add_argument("--random-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    hw = dict(DEFAULT_HW)

    print("=" * 80)
    print("Step-simulator constraint formula verification")
    print("=" * 80)
    print("log        :", args.log)
    print("tasks_pkl  :", args.tasks_pkl)
    print("max_tasks  :", args.max_tasks)
    print("trials/task:", args.random_trials)

    task_map = build_task_map(args.tasks_pkl)
    groups = _load_records_grouped(args.log)

    all_wks = sorted(groups.keys())[: args.max_tasks]
    if not all_wks:
        print("No records found.")
        return 1

    base_mismatch = 0
    rand_mismatch = 0
    n_rand_total = 0

    filter_times = []
    lower_times = []

    for idx, wk in enumerate(all_wks):
        base_rec = groups[wk][0]

        task, base_state = record_to_task_state(base_rec, task_map)
        merge_report = get_storage_rewrite_merge_report(task, base_state)
        system = build_system(base_rec, task, hw=hw, merge_report=merge_report)

        pred_base = evaluate_record(system, base_rec)["valid"]
        act_base = _verify_record_with_lowering(task_map, base_rec, hw)
        if pred_base != act_base:
            base_mismatch += 1

        task_rand_mismatch = 0
        task_rand_total = 0

        for _ in range(args.random_trials):
            rec_mut = randomize_record_params(base_rec, rng)

            t0 = time.perf_counter()
            pred = evaluate_record(system, rec_mut)["valid"]
            t1 = time.perf_counter()
            act = _verify_record_with_lowering(task_map, rec_mut, hw)
            t2 = time.perf_counter()

            filter_times.append(t1 - t0)
            lower_times.append(t2 - t1)

            if pred != act:
                task_rand_mismatch += 1
                rand_mismatch += 1
            task_rand_total += 1
            n_rand_total += 1

        print(
            "Task %2d: base(%s/%s) random mismatch %d/%d"
            % (idx, pred_base, act_base, task_rand_mismatch, task_rand_total)
        )

    print("-" * 80)
    print("Base mismatch      : %d/%d" % (base_mismatch, len(all_wks)))
    print("Random mismatch    : %d/%d" % (rand_mismatch, n_rand_total))

    if filter_times:
        fs = statistics.mean(filter_times)
        ls = statistics.mean(lower_times)
        print("Avg formula check  : %.6f s" % fs)
        print("Avg lower+verify   : %.6f s" % ls)
        if fs > 0:
            print("Speedup            : %.2fx" % (ls / fs))

    return 0 if (base_mismatch == 0 and rand_mismatch == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
