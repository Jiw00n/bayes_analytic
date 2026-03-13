"""Validate generated params from representative projected GPU sketches."""

import argparse
import json
import random
import time
from collections import Counter

from modules.common import TO_MEASURE_PROGRAM_FOLDER
from modules.projected_gpu_validation import (
    collect_gpu_projection_diagnostics,
    ensure_parent_dir,
    load_sketch_lines,
    load_sketch_record,
    load_tasks_by_workload,
    build_schedule_generator,
)
from modules.tvm_verify import lower_with_gpu_passes, params_to_state, verify_gpu_module


def _load_selection(selection_path):
    with open(selection_path) as f:
        payload = json.load(f)

    categories_by_index = {}
    for category in ("vectorize", "shared_memory", "max_vthread"):
        for entry in payload.get(category, []):
            idx = int(entry["index"])
            categories_by_index.setdefault(idx, set()).add(category)

    return payload, {
        idx: {
            "categories": sorted(categories),
        }
        for idx, categories in categories_by_index.items()
    }


def _invalid_record_base(sketch_index, attempt, seed, task, categories, params):
    return {
        "sketch_index": sketch_index,
        "attempt": attempt,
        "seed": seed,
        "task_desc": task.desc,
        "workload_key": task.workload_key,
        "categories": categories,
        "params": params,
    }


def _finalize_report(report):
    if report["randomize_success"]:
        report["concrete_invalid_rate"] = report["concrete_invalid"] / report["randomize_success"]
    else:
        report["concrete_invalid_rate"] = None
    return report


def run(args):
    started = time.time()
    sketch_lines = load_sketch_lines(args.sketches_path)
    _, tasks_by_wkey = load_tasks_by_workload(args.network_info_dir)
    attempts_per_sketch = args.attempts_per_sketch

    if args.all_sketches:
        selected = {
            idx: {"categories": ["all_sketches"]}
            for idx, line in enumerate(sketch_lines)
            if line
        }
        selection_payload = {"selected_counts": {"all_sketches": len(selected)}}
    else:
        selection_payload, selected = _load_selection(args.selection_path)

    selected_indices = sorted(selected.keys())
    if args.start:
        selected_indices = selected_indices[args.start:]
    if args.limit is not None:
        selected_indices = selected_indices[:args.limit]

    invalid_records = []
    reports = []
    overall = Counter()

    for ordinal, sketch_index in enumerate(selected_indices, start=1):
        if sketch_index >= len(sketch_lines):
            raise IndexError(
                f"Selected sketch index {sketch_index} is out of range for {args.sketches_path}"
            )

        line = sketch_lines[sketch_index]
        if not line:
            raise ValueError(f"Selected sketch index {sketch_index} points to an empty line")
        sketch_started = time.time()
        task, base_inp, base_res, state = load_sketch_record(line, tasks_by_wkey)
        gen = build_schedule_generator(task, state, base_inp=base_inp, base_res=base_res)
        categories = selected[sketch_index]["categories"]
        seed = args.seed_base + sketch_index
        rng = random.Random(seed)

        report = {
            "sketch_index": sketch_index,
            "task_desc": task.desc,
            "workload_key": task.workload_key,
            "categories": categories,
            "seed": seed,
            "attempts": attempts_per_sketch,
            "randomize_success": 0,
            "randomize_fail": 0,
            "concrete_pass": 0,
            "concrete_invalid": 0,
            "invalid_root_causes": Counter(),
            "randomize_fail_examples": [],
            "invalid_examples": [],
        }

        for attempt in range(1, attempts_per_sketch + 1):
            try:
                params = gen.randomize_params(rng=rng, max_retries=args.max_retries)
            except Exception as err:  # pylint: disable=broad-except
                report["randomize_fail"] += 1
                if len(report["randomize_fail_examples"]) < args.example_limit:
                    report["randomize_fail_examples"].append(
                        {
                            "attempt": attempt,
                            "error": f"{type(err).__name__}: {err}",
                        }
                    )
                continue

            report["randomize_success"] += 1
            error = None
            concrete_ok = False
            concrete_result = gen.get_concrete_final_result(params)
            if concrete_result is not None:
                concrete_ok = bool(concrete_result.get("ok"))
                error = concrete_result.get("error")
            else:
                try:
                    new_state = params_to_state(task, base_inp, base_res, params)
                    mod = lower_with_gpu_passes(task, new_state)
                    concrete_ok = verify_gpu_module(mod)
                    if not concrete_ok:
                        error = "verify_gpu_module returned False"
                except Exception as err:  # pylint: disable=broad-except
                    error = f"{type(err).__name__}: {err}"

            if concrete_ok:
                report["concrete_pass"] += 1
                continue

            report["concrete_invalid"] += 1
            diagnostics = collect_gpu_projection_diagnostics(gen, params)
            report["invalid_root_causes"][diagnostics["root_cause"]] += 1
            invalid_record = _invalid_record_base(
                sketch_index, attempt, seed, task, categories, params
            )
            invalid_record.update(
                {
                    "error": error,
                    "root_cause": diagnostics["root_cause"],
                    "exact_violations": diagnostics["exact_violations"],
                    "projected_violations": diagnostics["projected_violations"],
                    "constraint_snapshots": diagnostics["snapshots"],
                }
            )
            invalid_records.append(invalid_record)

            if len(report["invalid_examples"]) < args.example_limit:
                report["invalid_examples"].append(invalid_record)

        report["invalid_root_causes"] = dict(report["invalid_root_causes"])
        report["elapsed_sec"] = time.time() - sketch_started
        if report["randomize_success"] == 0:
            report["status"] = "no_randomize_success"
        else:
            invalid_rate = report["concrete_invalid"] / report["randomize_success"]
            report["status"] = "ok" if invalid_rate <= args.acceptance_threshold else "invalid_rate_exceeded"
        _finalize_report(report)
        reports.append(report)

        overall["selected_sketches"] += 1
        overall["attempts"] += report["attempts"]
        overall["randomize_success"] += report["randomize_success"]
        overall["randomize_fail"] += report["randomize_fail"]
        overall["concrete_pass"] += report["concrete_pass"]
        overall["concrete_invalid"] += report["concrete_invalid"]
        overall[f"status::{report['status']}"] += 1
        for root_cause, count in report["invalid_root_causes"].items():
            overall[f"root_cause::{root_cause}"] += count

        if ordinal % args.print_every == 0 or report["status"] != "ok":
            print(
                f"[{ordinal}/{len(selected_indices)}] sketch={sketch_index} status={report['status']} "
                f"success={report['randomize_success']} invalid={report['concrete_invalid']} "
                f"rand_fail={report['randomize_fail']}",
                flush=True,
            )

    aggregate = {
        "selection_path": args.selection_path,
        "all_sketches": args.all_sketches,
        "start": args.start,
        "limit": args.limit,
        "sketches_path": args.sketches_path,
        "network_info_dir": args.network_info_dir,
        "seed_base": args.seed_base,
        "attempts_per_sketch": attempts_per_sketch,
        "max_retries": args.max_retries,
        "acceptance_threshold": args.acceptance_threshold,
        "selected_unique_sketches": len(selected_indices),
        "elapsed_sec": time.time() - started,
        "counts": dict(overall),
        "reports": reports,
        "selection_meta": {
            "selected_counts": selection_payload.get("selected_counts", {}),
        },
    }
    if overall["randomize_success"]:
        aggregate["concrete_invalid_rate"] = overall["concrete_invalid"] / overall["randomize_success"]
    else:
        aggregate["concrete_invalid_rate"] = None

    ensure_parent_dir(args.summary_path)
    with open(args.summary_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    if args.invalid_path:
        ensure_parent_dir(args.invalid_path)
        with open(args.invalid_path, "w") as f:
            for record in invalid_records:
                f.write(json.dumps(record) + "\n")

    print(
        "generation_validation_done "
        f"sketches={len(selected_indices)} "
        f"randomize_success={overall['randomize_success']} "
        f"concrete_invalid={overall['concrete_invalid']} "
        f"invalid_rate={aggregate['concrete_invalid_rate']}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-path", default=None)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--invalid-path", default=None)
    parser.add_argument("--all-sketches", action="store_true")
    parser.add_argument("--network-info-dir", default="/root/work/tvm-ansor/gallery/dataset/network_info")
    parser.add_argument(
        "--sketches-path",
        default=f"{TO_MEASURE_PROGRAM_FOLDER}/all_sketches.json",
    )
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--attempts-per-sketch", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=64)
    parser.add_argument("--acceptance-threshold", type=float, default=0.05)
    parser.add_argument("--example-limit", type=int, default=8)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if not args.all_sketches and not args.selection_path:
        parser.error("--selection-path is required unless --all-sketches is set")
    if args.attempts_per_sketch is None:
        args.attempts_per_sketch = 3 if args.all_sketches else 200
    return args


if __name__ == "__main__":
    run(parse_args())
