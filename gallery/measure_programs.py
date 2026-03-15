"""Measure generated programs with per-sketch sampling.

Default behavior:
1. Read generated task files from ``to_measure_gen_programs``.
2. Group records in each task file by sketch fingerprint.
3. Randomly sample up to 32 records per sketch.
4. Measure the sampled states and save logs under ``measured_gen_programs``.
"""

import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict

from tvm import auto_scheduler

from modules.task_paths import (
    TO_MEASURE_GEN_PROGRAM_FOLDER,
    load_and_register_tasks,
    get_measure_record_filename,
    get_to_measure_gen_filename,
)
from modules.legacy_record_sketch_io import sketch_fingerprint_hash, state_sketch_fingerprint


def make_measurer(run_timeout, repeat, number, enable_cpu_cache_flush, verbose, log_filename):
    builder = auto_scheduler.measure.LocalBuilder()
    runner = auto_scheduler.measure.LocalRunner(
        timeout=run_timeout,
        repeat=repeat,
        number=number,
        enable_cpu_cache_flush=enable_cpu_cache_flush,
    )
    return auto_scheduler.measure.ProgramMeasurer(
        builder,
        runner,
        [auto_scheduler.RecordToFile(log_filename)],
        verbose=verbose,
    )


def _task_repeat(task):
    if task.compute_dag.flop_ct >= 2416443392.0:
        return 4
    if task.compute_dag.flop_ct >= 834928640.0:
        return 6
    if task.compute_dag.flop_ct <= 2097152.0:
        return 10
    return 8


def _load_source_inputs(source_path):
    if not os.path.exists(source_path):
        return []
    inputs, _ = auto_scheduler.RecordReader(source_path).read_lines()
    return list(inputs)


def _sample_grouped_inputs(raw_inputs, sample_per_sketch, seed):
    grouped = defaultdict(list)
    for inp in raw_inputs:
        grouped[state_sketch_fingerprint(inp.state)].append(inp)

    rng = random.Random(seed)
    sampled = []
    sample_meta = []
    for fp, records in grouped.items():
        if len(records) <= sample_per_sketch:
            chosen = list(records)
        else:
            indices = sorted(rng.sample(range(len(records)), sample_per_sketch))
            chosen = [records[idx] for idx in indices]
        sampled.extend(chosen)
        sample_meta.append(
            {
                "sketch_fp_hash": sketch_fingerprint_hash(fp),
                "group_size": len(records),
                "sampled_size": len(chosen),
            }
        )
    return sampled, sample_meta


def measure_task_file(task_idx, task, args):
    source_path = get_to_measure_gen_filename(task, output_dir=args.source_dir)
    raw_inputs = _load_source_inputs(source_path)
    if not raw_inputs:
        return {
            "task_idx": task_idx,
            "task_desc": task.desc,
            "workload_key": task.workload_key,
            "source_path": source_path,
            "status": "missing_or_empty",
            "selected_inputs": 0,
            "measured": 0,
            "measure_errors": 0,
            "sample_meta": [],
        }

    sampled_inputs, sample_meta = _sample_grouped_inputs(
        raw_inputs,
        sample_per_sketch=args.sample_per_sketch,
        seed=args.seed + task_idx,
    )
    measure_inputs = [auto_scheduler.MeasureInput(task, inp.state) for inp in sampled_inputs]

    log_filename = get_measure_record_filename(task, task.target)
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    measurer = make_measurer(
        run_timeout=args.run_timeout,
        repeat=_task_repeat(task),
        number=args.number,
        enable_cpu_cache_flush=args.enable_cpu_cache_flush,
        verbose=args.verbose,
        log_filename=log_filename,
    )
    empty_policy = auto_scheduler.search_policy.EmptyPolicy(task)

    started_at = time.time()
    counter = Counter()
    for i in range(0, len(measure_inputs), args.batch_size):
        batch = measure_inputs[i:i + args.batch_size]
        print(
            f"===== task: {task_idx}/{args.total_tasks}\t"
            f"sampled: {i + len(batch)}/{len(measure_inputs)} ====="
        )
        res_batch = measurer.measure(task, empty_policy, batch)
        counter["measured"] += len(res_batch)
        for res in res_batch:
            if res.error_no != auto_scheduler.measure.MeasureErrorNo.NO_ERROR:
                counter["measure_errors"] += 1

    return {
        "task_idx": task_idx,
        "task_desc": task.desc,
        "workload_key": task.workload_key,
        "source_path": source_path,
        "log_filename": log_filename,
        "status": "ok",
        "selected_inputs": len(measure_inputs),
        "measured": counter["measured"],
        "measure_errors": counter["measure_errors"],
        "sample_meta": sample_meta,
        "elapsed_sec": time.time() - started_at,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=TO_MEASURE_GEN_PROGRAM_FOLDER)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-per-sketch", type=int, default=32)
    parser.add_argument("--start-task", type=int, default=0)
    parser.add_argument("--end-task", type=int, default=None)
    parser.add_argument("--task-step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-timeout", type=int, default=5)
    parser.add_argument("--number", type=int, default=1)
    parser.add_argument("--enable-cpu-cache-flush", action="store_true")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument(
        "--summary-json",
        default="/root/work/tvm-ansor/gallery/dataset/measured_gen_programs/measurement_summary.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tasks = load_and_register_tasks()
    end_task = len(tasks) if args.end_task is None else min(args.end_task, len(tasks))
    task_indices = list(range(args.start_task, end_task, args.task_step))
    args.total_tasks = len(tasks)

    overall = Counter()
    reports = []
    for task_idx in task_indices:
        task = tasks[task_idx]
        with open("progress.txt", "a") as fout:
            fout.write(f"Begin measure {task_idx}/{len(tasks)}: {time.time():.2f}\n")
        report = measure_task_file(task_idx, task, args)
        reports.append(report)
        overall["tasks_seen"] += 1
        overall[f"status::{report['status']}"] += 1
        overall["selected_inputs"] += report["selected_inputs"]
        overall["measured"] += report["measured"]
        overall["measure_errors"] += report["measure_errors"]
        with open("progress.txt", "a") as fout:
            fout.write(f"End measure {task_idx}/{len(tasks)}: {time.time():.2f}\n")

    out_dir = os.path.dirname(args.summary_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.summary_json, "w") as f:
        json.dump(
            {
                "source_dir": args.source_dir,
                "task_count": len(task_indices),
                "sample_per_sketch": args.sample_per_sketch,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "counts": dict(overall),
                "reports": reports,
            },
            f,
            indent=2,
        )

    print(
        "measurement_done "
        f"tasks={overall['tasks_seen']} "
        f"selected_inputs={overall['selected_inputs']} "
        f"measured={overall['measured']} "
        f"errors={overall['measure_errors']}"
    )


if __name__ == "__main__":
    main()
