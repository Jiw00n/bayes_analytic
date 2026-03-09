"""Generate verified schedules from unique sketches.

The pipeline is:
1. Load tasks from ``all_tasks.pkl`` and register workloads.
2. Load unique sketch records from ``to_measure_programs/all_sketches.json``.
3. Build a symbolic state per sketch.
4. Randomize schedule params from the active symbolic constraints.
5. Verify each generated state with ``verify_gpu_module``.
6. Save passing records into task-level JSON files under
   ``gallery/dataset/to_measure_gen_programs``.
7. Write per-sketch summaries and failure reports for later inspection.
"""

import argparse
import json
import os
import random
import time
from collections import Counter

from tvm import auto_scheduler
from tvm.auto_scheduler.measure_record import load_record_from_string, save_records

from modules.common import (
    TO_MEASURE_GEN_PROGRAM_FOLDER,
    TO_MEASURE_PROGRAM_FOLDER,
    load_and_register_tasks,
    get_to_measure_gen_filename,
)
from modules.param_manager import build_symbolic_state
from modules.record_loader import (
    sketch_fingerprint_hash,
    sketch_fingerprint_repr,
    state_sketch_fingerprint,
    state_step_signature,
)
from modules.schedule_generator import ScheduleGenerator
from modules.tvm_verify import lower_with_gpu_passes, params_to_state, verify_gpu_module


DEFAULT_SKETCH_PATH = f"{TO_MEASURE_PROGRAM_FOLDER}/all_sketches.json"
DEFAULT_SUMMARY_PATH = f"{TO_MEASURE_GEN_PROGRAM_FOLDER}/generation_sketch_summary.jsonl"
DEFAULT_FAILURE_PATH = f"{TO_MEASURE_GEN_PROGRAM_FOLDER}/generation_failures.jsonl"
DEFAULT_OVERVIEW_PATH = f"{TO_MEASURE_GEN_PROGRAM_FOLDER}/generation_overview.json"


def _iter_sketch_lines(sketch_path, start_sketch=0, end_sketch=None):
    with open(sketch_path) as f:
        for idx, line in enumerate(f):
            if idx < start_sketch:
                continue
            if end_sketch is not None and idx >= end_sketch:
                break
            line = line.strip()
            if line:
                yield idx, line


def _append_jsonl(path, payload):
    with open(path, "a") as f:
        f.write(json.dumps(payload) + "\n")


def _flush_records(output_path, inputs_buf, results_buf):
    if not inputs_buf:
        return 0
    save_records(output_path, inputs_buf, results_buf)
    count = len(inputs_buf)
    inputs_buf.clear()
    results_buf.clear()
    return count


def _load_sketch_record(line, tasks_by_wkey):
    base_inp, base_res = load_record_from_string(line)
    recovered = auto_scheduler.measure.recover_measure_input(base_inp)
    task = tasks_by_wkey[recovered.task.workload_key]
    base_inp = auto_scheduler.MeasureInput(task, recovered.state)
    return task, base_inp, base_res


def _trim_examples(examples, limit):
    if len(examples) > limit:
        del examples[limit:]


def generate_for_sketch(
    sketch_idx,
    line,
    tasks_by_wkey,
    output_dir,
    schedules_per_sketch,
    flush_every,
    max_retries,
    max_attempt_factor,
    seed_base,
    example_limit,
):
    started_at = time.time()
    task = None
    base_inp = None
    base_res = None
    output_path = None
    inputs_buf = []
    results_buf = []
    failure_examples = []
    counter = Counter()

    try:
        task, base_inp, base_res = _load_sketch_record(line, tasks_by_wkey)
        output_path = get_to_measure_gen_filename(task, output_dir=output_dir)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        sketch_fp = state_sketch_fingerprint(base_inp.state)
        sketch_fp_hash = sketch_fingerprint_hash(sketch_fp)
        step_signature = state_step_signature(base_inp.state)
        sym_state = build_symbolic_state(task.compute_dag, base_inp.state)
        gen = ScheduleGenerator(sym_state)

        rng = random.Random(seed_base + sketch_idx)
        max_attempts = schedules_per_sketch * max_attempt_factor

        while counter["verified_ok"] < schedules_per_sketch and counter["attempted"] < max_attempts:
            counter["attempted"] += 1
            params = None
            try:
                params = gen.randomize_params(rng=rng, max_retries=max_retries)
            except Exception as err:  # pylint: disable=broad-except
                counter["randomize_fail"] += 1
                failure_examples.append(
                    {
                        "kind": "randomize_fail",
                        "attempt": counter["attempted"],
                        "error": f"{type(err).__name__}: {err}",
                    }
                )
                _trim_examples(failure_examples, example_limit)
                continue

            try:
                new_state = params_to_state(task, base_inp, base_res, params)
                mod = lower_with_gpu_passes(task, new_state)
                if not verify_gpu_module(mod):
                    counter["verify_fail"] += 1
                    failure_examples.append(
                        {
                            "kind": "verify_fail",
                            "attempt": counter["attempted"],
                            "error": "verify_gpu_module returned False",
                            "params": params,
                        }
                    )
                    _trim_examples(failure_examples, example_limit)
                    continue
                inputs_buf.append(auto_scheduler.MeasureInput(task, new_state))
                results_buf.append(base_res)
                counter["verified_ok"] += 1
                if len(inputs_buf) >= flush_every:
                    counter["saved_records"] += _flush_records(output_path, inputs_buf, results_buf)
            except Exception as err:  # pylint: disable=broad-except
                counter["verify_fail"] += 1
                failure_examples.append(
                    {
                        "kind": "lower_or_verify_exception",
                        "attempt": counter["attempted"],
                        "error": f"{type(err).__name__}: {err}",
                        "params": params,
                    }
                )
                _trim_examples(failure_examples, example_limit)

        counter["saved_records"] += _flush_records(output_path, inputs_buf, results_buf)
        status = "ok" if counter["verified_ok"] >= schedules_per_sketch else "shortfall"
        summary = {
            "status": status,
            "sketch_index": sketch_idx,
            "task_desc": task.desc,
            "workload_key": task.workload_key,
            "target": str(task.target),
            "output_path": output_path,
            "sketch_fp_hash": sketch_fp_hash,
            "sketch_fp_repr": sketch_fingerprint_repr(sketch_fp),
            "step_signature": step_signature,
            "requested_records": schedules_per_sketch,
            "verified_ok": counter["verified_ok"],
            "saved_records": counter["saved_records"],
            "attempted": counter["attempted"],
            "randomize_fail": counter["randomize_fail"],
            "verify_fail": counter["verify_fail"],
            "elapsed_sec": time.time() - started_at,
            "failure_examples": failure_examples,
        }
        return summary
    except Exception as err:  # pylint: disable=broad-except
        return {
            "status": "sketch_error",
            "sketch_index": sketch_idx,
            "task_desc": task.desc if task is not None else "",
            "workload_key": task.workload_key if task is not None else None,
            "target": str(task.target) if task is not None else None,
            "output_path": output_path,
            "requested_records": schedules_per_sketch,
            "verified_ok": counter["verified_ok"],
            "saved_records": counter["saved_records"],
            "attempted": counter["attempted"],
            "randomize_fail": counter["randomize_fail"],
            "verify_fail": counter["verify_fail"],
            "elapsed_sec": time.time() - started_at,
            "failure_examples": [
                {
                    "kind": "sketch_error",
                    "error": f"{type(err).__name__}: {err}",
                }
            ],
        }


def run_generation(args):
    os.makedirs(args.output_dir, exist_ok=True)
    tasks = load_and_register_tasks(args.network_info_dir)
    tasks_by_wkey = {task.workload_key: task for task in tasks}

    overall = Counter()
    started_at = time.time()
    selected_lines = 0
    unique_outputs = set()

    for sketch_idx, line in _iter_sketch_lines(
        args.sketches_path,
        start_sketch=args.start_sketch,
        end_sketch=args.end_sketch,
    ):
        selected_lines += 1
        summary = generate_for_sketch(
            sketch_idx=sketch_idx,
            line=line,
            tasks_by_wkey=tasks_by_wkey,
            output_dir=args.output_dir,
            schedules_per_sketch=args.schedules_per_sketch,
            flush_every=args.flush_every,
            max_retries=args.max_retries,
            max_attempt_factor=args.max_attempt_factor,
            seed_base=args.seed,
            example_limit=args.failure_example_limit,
        )
        _append_jsonl(args.summary_path, summary)

        overall["sketches_seen"] += 1
        overall["requested_records"] += summary["requested_records"]
        overall["verified_ok"] += summary["verified_ok"]
        overall["saved_records"] += summary["saved_records"]
        overall["attempted"] += summary["attempted"]
        overall["randomize_fail"] += summary["randomize_fail"]
        overall["verify_fail"] += summary["verify_fail"]
        overall[f"status::{summary['status']}"] += 1
        if summary.get("output_path"):
            unique_outputs.add(summary["output_path"])

        if summary["status"] != "ok" or summary["randomize_fail"] or summary["verify_fail"]:
            _append_jsonl(args.failure_path, summary)

        if overall["sketches_seen"] % args.print_every == 0 or summary["status"] != "ok":
            print(
                f"[{overall['sketches_seen']}] sketch={sketch_idx} status={summary['status']} "
                f"saved={summary['saved_records']}/{summary['requested_records']} "
                f"attempted={summary['attempted']} "
                f"rand_fail={summary['randomize_fail']} verify_fail={summary['verify_fail']}"
            )

    overview = {
        "network_info_dir": args.network_info_dir,
        "sketches_path": args.sketches_path,
        "output_dir": args.output_dir,
        "summary_path": args.summary_path,
        "failure_path": args.failure_path,
        "schedules_per_sketch": args.schedules_per_sketch,
        "max_retries": args.max_retries,
        "max_attempt_factor": args.max_attempt_factor,
        "seed": args.seed,
        "start_sketch": args.start_sketch,
        "end_sketch": args.end_sketch,
        "selected_sketches": selected_lines,
        "unique_output_files": len(unique_outputs),
        "elapsed_sec": time.time() - started_at,
        "counts": dict(overall),
    }
    with open(args.overview_path, "w") as f:
        json.dump(overview, f, indent=2)

    print(
        "generation_done "
        f"sketches={overview['selected_sketches']} "
        f"saved={overall['saved_records']} "
        f"verify_fail={overall['verify_fail']} "
        f"randomize_fail={overall['randomize_fail']} "
        f"elapsed_sec={overview['elapsed_sec']:.2f}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-info-dir", default="/root/work/tvm-ansor/gallery/dataset/network_info")
    parser.add_argument("--sketches-path", default=DEFAULT_SKETCH_PATH)
    parser.add_argument("--output-dir", default=TO_MEASURE_GEN_PROGRAM_FOLDER)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--failure-path", default=DEFAULT_FAILURE_PATH)
    parser.add_argument("--overview-path", default=DEFAULT_OVERVIEW_PATH)
    parser.add_argument("--start-sketch", type=int, default=0)
    parser.add_argument("--end-sketch", type=int, default=None)
    parser.add_argument("--schedules-per-sketch", type=int, default=4000)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-attempt-factor", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flush-every", type=int, default=256)
    parser.add_argument("--failure-example-limit", type=int, default=8)
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run_generation(parse_args())
