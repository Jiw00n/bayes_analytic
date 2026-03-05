#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure valid ratio by exhaustive DFS (no duplicate parameter combos)."""

import argparse
import copy
import json
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from constraint_formula import (
    DEFAULT_HW,
    build_task_map,
    build_system,
    eval_expr,
    get_divisors,
    lower_with_gpu_passes,
    record_to_task_state,
    verify_gpu_module,
)


def _load_records_grouped(log_path: str) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            wk = rec["i"][0][0]
            groups[wk].append(rec)
    return groups


def _verify_record_with_lowering(
    task_map: Dict[str, Any], rec: Dict[str, Any], hw: Dict[str, int]
) -> Tuple[bool, Optional[str]]:
    try:
        task, state = record_to_task_state(rec, task_map)
        mod = lower_with_gpu_passes(task, state)
        return verify_gpu_module(mod, hw), None
    except Exception as e:
        return False, str(e)


def _parse_unroll_values(s: str) -> List[int]:
    vals = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(int(tok))
    return vals


def _enumerate_sp_domain(extent: int, n_split: int, max_innermost: int) -> List[List[int]]:
    if n_split <= 0:
        return [[]]
    out: List[List[int]] = []
    cur = [1] * n_split

    def dfs(pos: int, rem: int) -> None:
        if pos == n_split - 1:
            for d in get_divisors(rem):
                d = int(d)
                if d <= int(max_innermost):
                    cur[pos] = d
                    out.append(list(cur))
            return
        for d in get_divisors(rem):
            d = int(d)
            cur[pos] = d
            dfs(pos + 1, rem // d)

    dfs(0, int(extent))
    return out


def _extract_knobs(
    base_record: Dict[str, Any],
    max_innermost: int,
    auto_unroll_values: Sequence[int],
) -> List[Dict[str, Any]]:
    knobs: List[Dict[str, Any]] = []
    steps = base_record["i"][1][1]
    for step_idx, s in enumerate(steps):
        if s[0] == "SP":
            extent = int(s[3])
            n_split = len(s[4])
            dom = _enumerate_sp_domain(extent, n_split, max_innermost)
            knobs.append(
                {
                    "kind": "SP",
                    "step_idx": int(step_idx),
                    "domain": dom,
                }
            )
        elif s[0] == "PR" and "auto_unroll_max_step" in str(s[3]):
            dom = sorted(set(int(x) for x in auto_unroll_values))
            try:
                base_v = int(str(s[3]).split("$", 1)[1])
                if base_v not in dom:
                    dom.append(base_v)
                    dom = sorted(set(dom))
            except Exception:
                pass
            knobs.append(
                {
                    "kind": "PR",
                    "step_idx": int(step_idx),
                    "domain": [int(x) for x in dom],
                }
            )
    return knobs


def _safe_total_combinations(knobs: List[Dict[str, Any]]) -> int:
    total = 1
    for k in knobs:
        total *= max(1, len(k["domain"]))
    return total


def _extract_thread_axis_limit_entries(system: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    constraints = system.get("constraints", {})
    kernels = constraints.get("kernels", {})
    for root, k in kernels.items():
        out.append(
            {
                "kernel": int(root),
                "x_exprs": list(k.get("thread_x_exprs", [])),
                "y_exprs": list(k.get("thread_y_exprs", [])),
                "z_exprs": list(k.get("thread_z_exprs", [])),
            }
        )
    return out


def _violates_thread_axis_limit(
    thread_axis_entries: List[Dict[str, Any]], steps: List[list], hw: Dict[str, int]
) -> bool:
    max_x = int(hw["max_thread_x"])
    max_y = int(hw["max_thread_y"])
    max_z = int(hw["max_thread_z"])
    for e in thread_axis_entries:
        for ex in e["x_exprs"]:
            if int(eval_expr(ex, steps)) > max_x:
                return True
        for ey in e["y_exprs"]:
            if int(eval_expr(ey, steps)) > max_y:
                return True
        for ez in e["z_exprs"]:
            if int(eval_expr(ez, steps)) > max_z:
                return True
    return False


def main() -> int:
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
    parser.add_argument("--task-idx", type=int, default=0)
    parser.add_argument("--schedule-idx", type=int, default=0)
    parser.add_argument(
        "--max-innermost-split-factor",
        type=int,
        default=DEFAULT_HW["max_innermost_split_factor"],
    )
    parser.add_argument(
        "--auto-unroll-values",
        type=str,
        default="0,16,64,512,1024",
    )
    parser.add_argument("--timeout-sec", type=float, default=0.0)
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=0,
        help="0 means unlimited until full DFS or timeout.",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-valid-jsonl",
        type=str,
        default="",
        help="Optional output for valid parameter combos.",
    )
    parser.add_argument(
        "--include-record",
        action="store_true",
        help="Include full record in output-valid-jsonl rows.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    hw = dict(DEFAULT_HW)

    groups = _load_records_grouped(args.log)
    workloads = sorted(groups.keys())
    if args.task_idx < 0 or args.task_idx >= len(workloads):
        raise ValueError("task-idx out of range: %d (num_workloads=%d)" % (args.task_idx, len(workloads)))
    wk = workloads[args.task_idx]
    records = groups[wk]
    if args.schedule_idx < 0 or args.schedule_idx >= len(records):
        raise ValueError(
            "schedule-idx out of range: %d (num_schedules=%d)" % (args.schedule_idx, len(records))
        )
    base_record = records[args.schedule_idx]
    steps = base_record["i"][1][1]

    task_map = build_task_map(args.tasks_pkl)
    base_task, _ = record_to_task_state(base_record, task_map)
    system = build_system(base_record, base_task, hw=hw, merge_report=None)
    thread_axis_entries = _extract_thread_axis_limit_entries(system)
    auto_unroll_values = _parse_unroll_values(args.auto_unroll_values)
    knobs = _extract_knobs(
        base_record=base_record,
        max_innermost=int(args.max_innermost_split_factor),
        auto_unroll_values=auto_unroll_values,
    )

    print("=" * 80)
    print("DFS valid-rate measurement (lower+verify ground truth)")
    print("=" * 80)
    print("task_idx            :", args.task_idx)
    print("schedule_idx        :", args.schedule_idx)
    print("workload_key        :", wk)
    print("knob_count          :", len(knobs))
    print("timeout_sec         :", args.timeout_sec)
    print("max_combinations    :", args.max_combinations)
    print("output_valid_jsonl  :", args.output_valid_jsonl or "(disabled)")
    print("-" * 80)
    for i, k in enumerate(knobs):
        print("knob %2d: %-2s step=%d domain=%d" % (i, k["kind"], k["step_idx"], len(k["domain"])))
    total_combos = _safe_total_combinations(knobs)
    print("theoretical_combos  :", total_combos)

    checked = 0
    valid = 0
    invalid = 0
    exceptions = 0
    masked_thread_axis_limit = 0
    timed_out = False
    stopped_by_cap = False
    start = time.perf_counter()

    fout = open(args.output_valid_jsonl, "w") if args.output_valid_jsonl else None
    rec_work = copy.deepcopy(base_record)
    rec_steps = rec_work["i"][1][1]

    def emit_valid() -> None:
        nonlocal valid
        valid += 1
        if fout is None:
            return
        sp_params: Dict[int, List[int]] = {}
        unroll_params: Dict[int, int] = {}
        for step_idx, s in enumerate(rec_steps):
            if s[0] == "SP":
                sp_params[int(step_idx)] = [int(x) for x in s[4]]
            elif s[0] == "PR" and "auto_unroll_max_step$" in str(s[3]):
                try:
                    unroll_params[int(step_idx)] = int(str(s[3]).split("$", 1)[1])
                except Exception:
                    pass
        row = {
            "task_idx": int(args.task_idx),
            "schedule_idx": int(args.schedule_idx),
            "workload_key": wk,
            "sp_params": sp_params,
            "unroll_params": unroll_params,
            "lower_valid": True,
        }
        if args.include_record:
            row["record"] = rec_work
        fout.write(json.dumps(row) + "\n")

    def over_limit() -> bool:
        nonlocal timed_out, stopped_by_cap
        if args.max_combinations > 0 and checked >= int(args.max_combinations):
            stopped_by_cap = True
            return True
        if args.timeout_sec > 0 and (time.perf_counter() - start) >= float(args.timeout_sec):
            timed_out = True
            return True
        return False

    def dfs(idx: int) -> bool:
        nonlocal checked, invalid, exceptions, masked_thread_axis_limit
        if over_limit():
            return False
        if idx == len(knobs):
            if _violates_thread_axis_limit(thread_axis_entries, rec_steps, hw):
                masked_thread_axis_limit += 1
                return not over_limit()
            checked += 1
            ok, err = _verify_record_with_lowering(task_map, rec_work, hw)
            if ok:
                emit_valid()
            else:
                if err is None:
                    invalid += 1
                else:
                    exceptions += 1
            if args.progress_every > 0 and checked % int(args.progress_every) == 0:
                elapsed = time.perf_counter() - start
                rate = checked / elapsed if elapsed > 0 else 0.0
                vr = valid / checked if checked > 0 else 0.0
                print(
                    "[progress] checked=%d valid=%d invalid=%d exc=%d valid_rate=%.4f speed=%.2f/s"
                    % (checked, valid, invalid, exceptions, vr, rate)
                )
            return not over_limit()

        knob = knobs[idx]
        step_idx = int(knob["step_idx"])
        step = rec_steps[step_idx]
        if knob["kind"] == "SP":
            old = list(step[4])
            for val in knob["domain"]:
                step[4] = [int(x) for x in val]
                if not dfs(idx + 1):
                    step[4] = old
                    return False
            step[4] = old
            return True
        if knob["kind"] == "PR":
            old = str(step[3])
            for val in knob["domain"]:
                step[3] = "auto_unroll_max_step$%d" % int(val)
                if not dfs(idx + 1):
                    step[3] = old
                    return False
            step[3] = old
            return True
        return dfs(idx + 1)

    completed = dfs(0)
    elapsed = time.perf_counter() - start
    if fout is not None:
        fout.close()

    status = "completed"
    if timed_out:
        status = "timeout"
    elif stopped_by_cap:
        status = "max_combinations_reached"
    elif not completed:
        status = "stopped"

    print("-" * 80)
    print("status              :", status)
    print("checked             :", checked)
    print("valid               :", valid)
    print("invalid             :", invalid)
    print("exceptions          :", exceptions)
    print("masked_thread_axis_limit:", masked_thread_axis_limit)
    print("valid_rate          : %.6f" % (float(valid) / checked if checked > 0 else 0.0))
    print("elapsed_sec         : %.3f" % elapsed)
    print("throughput          : %.2f / sec" % (checked / elapsed if elapsed > 0 else 0.0))
    if total_combos > 0:
        explored = checked + masked_thread_axis_limit
        print("coverage            : %.6f" % (float(explored) / float(total_combos)))
    if fout is not None:
        print("saved_valid_to      :", args.output_valid_jsonl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
