"""Audit non-pruning checker correctness against concrete GPU verification."""

import argparse
import json
import time

from validate_exact_gpu_constraints import extract_params
from modules.common import TO_MEASURE_PROGRAM_FOLDER
from modules.param_manager import build_symbolic_state
from modules.projected_gpu_validation import (
    build_schedule_generator,
    ensure_parent_dir,
    load_sketch_lines,
    load_sketch_record,
    load_tasks_by_workload,
)
from modules.schedule_generator import ScheduleGenerator


def run(args):
    started = time.time()
    _, tasks_by_wkey = load_tasks_by_workload(args.network_info_dir)
    lines = load_sketch_lines(args.sketches_path)
    counts = {
        "checked": 0,
        "concrete_ok": 0,
        "concrete_bad": 0,
        "exact_mismatch": 0,
        "exact_false_reject": 0,
        "exact_false_accept": 0,
        "hybrid_mismatch": 0,
        "hybrid_false_reject": 0,
        "hybrid_false_accept": 0,
        "final_mismatch": 0,
        "final_false_reject": 0,
        "final_false_accept": 0,
    }

    mismatch_records = []
    for idx, line in enumerate(lines):
        if idx < args.start:
            continue
        if args.limit is not None and counts["checked"] >= args.limit:
            break
        if not line:
            continue

        task, base_inp, base_res, state = load_sketch_record(line, tasks_by_wkey)
        params = extract_params(state)

        if args.use_recorded_valid:
            sym = build_symbolic_state(task.compute_dag, state)
            gen = ScheduleGenerator(sym)
            concrete_ok = int(getattr(base_res, "error_no", 1)) == 0
            concrete = {
                "ok": concrete_ok,
                "violations": [] if concrete_ok else [f"base_res.error_no={base_res.error_no}"],
            }
        else:
            gen = build_schedule_generator(task, state, base_inp=base_inp, base_res=base_res)
            concrete = gen.get_concrete_final_result(params)
            if concrete is None:
                raise RuntimeError(f"Concrete final context missing at index {idx}")
            concrete_ok = bool(concrete["ok"])

        exact_violations = gen.check_all_exact(params)
        hybrid_violations = gen.check_all_hybrid(params)
        final_violations = gen.check_all_final(params)

        exact_ok = not exact_violations
        hybrid_ok = not hybrid_violations
        final_ok = not final_violations

        counts["checked"] += 1
        counts["concrete_ok" if concrete_ok else "concrete_bad"] += 1

        record = {
            "index": idx,
            "task_desc": task.desc,
            "workload_key": task.workload_key,
            "concrete_ok": concrete_ok,
            "concrete_violations": concrete.get("violations", []),
            "exact_ok": exact_ok,
            "exact_violations": exact_violations,
            "hybrid_ok": hybrid_ok,
            "hybrid_violations": hybrid_violations,
            "final_ok": final_ok,
            "final_violations": final_violations,
        }

        wrote = False
        for prefix, ok in (
            ("exact", exact_ok),
            ("hybrid", hybrid_ok),
            ("final", final_ok),
        ):
            if ok == concrete_ok:
                continue
            counts[f"{prefix}_mismatch"] += 1
            if ok and not concrete_ok:
                counts[f"{prefix}_false_accept"] += 1
            else:
                counts[f"{prefix}_false_reject"] += 1
            wrote = True

        if wrote:
            mismatch_records.append(record)

        if counts["checked"] % args.print_every == 0:
            print(
                json.dumps(
                    {
                        "progress": counts["checked"],
                        "elapsed_sec": time.time() - started,
                        "exact_mismatch": counts["exact_mismatch"],
                        "hybrid_mismatch": counts["hybrid_mismatch"],
                        "final_mismatch": counts["final_mismatch"],
                    }
                ),
                flush=True,
            )

    summary = dict(counts)
    summary["elapsed_sec"] = time.time() - started
    summary["start"] = args.start
    summary["limit"] = args.limit

    if args.mismatch_path:
        ensure_parent_dir(args.mismatch_path)
        with open(args.mismatch_path, "w") as f:
            for record in mismatch_records:
                f.write(json.dumps(record) + "\n")

    if args.summary_path:
        ensure_parent_dir(args.summary_path)
        with open(args.summary_path, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, sort_keys=True), flush=True)


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
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--mismatch-path", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--use-recorded-valid", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
