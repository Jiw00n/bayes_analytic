"""Validate projected approximate GPU constraints against concrete GPU verification."""

import argparse
import json
import time

from modules.common import TO_MEASURE_PROGRAM_FOLDER
from modules.projected_gpu_validation import (
    collect_gpu_projection_diagnostics,
    collect_false_reject_diagnostics,
    ensure_parent_dir,
    load_sketch_lines,
    load_sketch_record,
    load_tasks_by_workload,
    build_schedule_generator,
)


def extract_params(state):
    params = {}
    for step_idx, step in enumerate(state.transform_steps):
        tk = step.type_key.split(".")[-1]
        if tk == "SplitStep":
            for li, length in enumerate(step.lengths):
                params[f"sp_{step_idx}_{li}"] = int(length)
        elif tk == "PragmaStep" and "auto_unroll_max_step$" in str(step.pragma_type):
            params[f"ur_{step_idx}"] = int(str(step.pragma_type).split("$")[1])
    return params


def run(args):
    _, tasks_by_wkey = load_tasks_by_workload(args.network_info_dir)
    sketch_lines = load_sketch_lines(args.sketches_path)

    total = 0
    mismatch_total = 0
    false_accept = 0
    false_reject = 0
    final_checker_mismatch = 0
    started = time.time()
    mismatch_records = []

    for idx, line in enumerate(sketch_lines):
        if idx < args.start:
            continue
        if args.limit is not None and total >= args.limit:
            break
        if not line:
            continue

        task, base_inp, base_res, state = load_sketch_record(line, tasks_by_wkey)
        gen = build_schedule_generator(task, state, base_inp=base_inp, base_res=base_res)
        params = extract_params(state)
        pruning_violations = gen.check_all_pruning(params)
        approx_ok = not pruning_violations
        concrete_result = gen.get_concrete_final_result(params)
        if concrete_result is None:
            raise RuntimeError(
                "Concrete final validation context is unavailable for validate_exact_gpu_constraints"
            )
        concrete_ok = bool(concrete_result["ok"])
        final_violations = gen.check_all_final(params)
        final_ok = not final_violations

        total += 1
        if args.print_every:
            print(idx, task.desc, approx_ok, concrete_ok, final_ok, flush=True)

        if final_ok != concrete_ok:
            final_checker_mismatch += 1

        if approx_ok != concrete_ok:
            mismatch_total += 1
            kind = "false_accept" if approx_ok and not concrete_ok else "false_reject"
            if kind == "false_accept":
                false_accept += 1
            else:
                false_reject += 1
            payload = {
                "index": idx,
                "task_desc": task.desc,
                "workload_key": task.workload_key,
                "kind": kind,
                "approx_ok": approx_ok,
                "concrete_ok": concrete_ok,
                "final_ok": final_ok,
                "params": params,
                "violations": pruning_violations,
                "pruning_violations": pruning_violations,
                "final_violations": final_violations,
                "concrete_final_result": concrete_result,
            }
            if kind == "false_accept":
                payload.update(collect_gpu_projection_diagnostics(gen, params))
            else:
                payload.update(collect_false_reject_diagnostics(gen, params, pruning_violations))
            mismatch_records.append(payload)
            print(json.dumps(payload), flush=True)
            if args.stop_on_mismatch:
                break
    summary = {
        "checked": total,
        "false_accept": false_accept,
        "false_reject": false_reject,
        "mismatch_total": mismatch_total,
        "final_checker_mismatch": final_checker_mismatch,
        "false_accept_rate": (false_accept / total) if total else 0.0,
        "false_reject_rate": (false_reject / total) if total else 0.0,
        "elapsed_sec": time.time() - started,
        "start": args.start,
        "limit": args.limit,
    }
    if args.mismatch_path:
        ensure_parent_dir(args.mismatch_path)
        with open(args.mismatch_path, "w") as f:
            for record in mismatch_records:
                f.write(json.dumps(record) + "\n")
    if args.summary_path:
        ensure_parent_dir(args.summary_path)
        with open(args.summary_path, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sketches-path",
        default=f"{TO_MEASURE_PROGRAM_FOLDER}/all_sketches.json",
    )
    parser.add_argument(
        "--network-info-dir",
        default="/root/work/tvm-ansor/gallery/dataset/network_info",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--print-every", action="store_true")
    parser.add_argument("--stop-on-mismatch", action="store_true")
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--mismatch-path", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
