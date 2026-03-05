#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate valid split/unroll parameters for each schedule of each task."""

import argparse
import json
import random
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

from constraint_formula import (
    AUTO_UNROLL_CONFIGS,
    DEFAULT_HW,
    build_system,
    build_task_map,
    check_constraints_prefilter,
    evaluate_record,
    generate_valid_params,
    get_divisors,
    get_storage_rewrite_merge_report,
    lower_with_gpu_passes,
    randomize_record_params,
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


def _extract_params_from_record(rec: Dict[str, Any]) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    sp_params: Dict[int, List[int]] = {}
    unroll_params: Dict[int, int] = {}
    steps = rec["i"][1][1]
    for step_idx, s in enumerate(steps):
        if s[0] == "SP":
            sp_params[int(step_idx)] = [int(x) for x in s[4]]
        elif s[0] == "PR" and "auto_unroll_max_step$" in str(s[3]):
            try:
                v = int(str(s[3]).split("$", 1)[1])
                unroll_params[int(step_idx)] = v
            except Exception:
                pass
    return sp_params, unroll_params


def _param_key(sp_params: Dict[int, List[int]], unroll_params: Dict[int, int]) -> Tuple[Any, Any]:
    return (
        tuple((k, tuple(v)) for k, v in sorted(sp_params.items())),
        tuple(sorted(unroll_params.items())),
    )


def _merge_reject_type_counts(dst: Dict[str, int], src: Dict[str, int]) -> None:
    for k, v in src.items():
        dst[k] = int(dst.get(k, 0)) + int(v)


def _count_sp_combinations(extent: int, n_split: int, max_innermost: int) -> int:
    if n_split <= 0:
        return 1

    @lru_cache(None)
    def dp(pos: int, rem: int) -> int:
        if pos == n_split - 1:
            return 1 if int(rem) <= int(max_innermost) else 0
        total = 0
        for d in get_divisors(int(rem)):
            total += dp(pos + 1, int(rem // int(d)))
        return total

    return int(dp(0, int(extent)))


def _count_total_combinations(
    record: Dict[str, Any],
    max_innermost: int,
    auto_unroll_configs: Sequence[int] = AUTO_UNROLL_CONFIGS,
) -> Tuple[int, int, int]:
    """Return (total_combinations, sp_combinations_product, unroll_combinations_product)."""
    steps = record["i"][1][1]
    sp_prod = 1
    unroll_prod = 1

    for s in steps:
        if s[0] == "SP":
            extent = int(s[3])
            n_split = len(s[4])
            sp_prod *= _count_sp_combinations(extent, n_split, int(max_innermost))
        elif s[0] == "PR" and "auto_unroll_max_step" in str(s[3]):
            dom = sorted(set(int(x) for x in auto_unroll_configs))
            try:
                base_v = int(str(s[3]).split("$", 1)[1])
                if base_v not in dom:
                    dom.append(base_v)
            except Exception:
                pass
            unroll_prod *= max(1, len(set(dom)))

    return int(sp_prod * unroll_prod), int(sp_prod), int(unroll_prod)


def _diagnose_on_empty(
    base_rec: Dict[str, Any],
    system: Dict[str, Any],
    task_map: Dict[str, Any],
    hw: Dict[str, int],
    verify_lowering: bool,
    prefilter_mode: str,
    k: int,
    rng: random.Random,
) -> Dict[str, Any]:
    diag = {
        "samples": int(k),
        "prefilter_reject_by_type": {},
        "lower_invalid_count": 0,
        "lower_exception_count": 0,
        "prefilter_pass_count": 0,
    }
    for _ in range(int(k)):
        rec = randomize_record_params(base_rec, rng)
        pf = check_constraints_prefilter(system["constraints"], rec, hw, mode=prefilter_mode)
        if not pf["valid"]:
            for v in pf["violations"]:
                t = str(v.get("type", "unknown"))
                diag["prefilter_reject_by_type"][t] = int(diag["prefilter_reject_by_type"].get(t, 0)) + 1
            continue
        diag["prefilter_pass_count"] += 1
        if verify_lowering:
            ok, err = _verify_record_with_lowering(task_map, rec, hw)
            if not ok:
                if err:
                    diag["lower_exception_count"] += 1
                else:
                    diag["lower_invalid_count"] += 1
    return diag


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
    parser.add_argument("--max-tasks", type=int, default=24)
    parser.add_argument(
        "--max-schedules-per-task",
        type=int,
        default=1,
        help="How many base schedules per task to process (-1 means all).",
    )
    parser.add_argument("--samples-per-schedule", type=int, default=5)
    parser.add_argument(
        "--attempts-per-schedule",
        type=int,
        default=2000,
        help="Total candidate generation attempts budget per schedule.",
    )
    parser.add_argument("--max-candidates-per-split", type=int, default=256)
    parser.add_argument(
        "--prefilter-mode",
        type=str,
        default="relaxed",
        choices=["strict", "relaxed"],
    )
    parser.add_argument("--diagnose-on-empty", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    verify_group = parser.add_mutually_exclusive_group()
    verify_group.add_argument("--verify-lowering", dest="verify_lowering", action="store_true")
    verify_group.add_argument("--no-verify-lowering", dest="verify_lowering", action="store_false")
    parser.set_defaults(verify_lowering=True)
    include_base_group = parser.add_mutually_exclusive_group()
    include_base_group.add_argument("--include-base", dest="include_base", action="store_true")
    include_base_group.add_argument("--no-include-base", dest="include_base", action="store_false")
    parser.set_defaults(include_base=False)
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="/root/work/tvm-ansor/gallery/generated_params.jsonl",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    hw = dict(DEFAULT_HW)

    print("=" * 80)
    print("Valid split/unroll parameter generation")
    print("=" * 80)
    print("log                  :", args.log)
    print("tasks_pkl            :", args.tasks_pkl)
    print("max_tasks            :", args.max_tasks)
    print("max_scheds_per_task  :", args.max_schedules_per_task)
    print("samples_per_schedule :", args.samples_per_schedule)
    print("attempts_per_schedule:", args.attempts_per_schedule)
    print("verify_lowering      :", args.verify_lowering)
    print("prefilter_mode       :", args.prefilter_mode)
    print("include_base         :", args.include_base)
    print("diagnose_on_empty    :", args.diagnose_on_empty)
    print("output_jsonl         :", args.output_jsonl)

    task_map = build_task_map(args.tasks_pkl)
    groups = _load_records_grouped(args.log)
    workloads = sorted(groups.keys())[: args.max_tasks]
    if not workloads:
        print("No records found.")
        return 1

    total_sched = 0
    total_target = 0
    total_generated = 0
    total_build_fail = 0
    total_lower_fail = 0
    total_lower_exception = 0
    total_base_lower_valid = 0
    total_base_lower_valid_but_zero = 0

    with open(args.output_jsonl, "w") as fout:
        for task_idx, wk in enumerate(workloads):
            base_records = groups[wk]
            if args.max_schedules_per_task >= 0:
                base_records = base_records[: args.max_schedules_per_task]

            for sched_idx, base_rec in enumerate(base_records):
                total_sched += 1
                total_target += args.samples_per_schedule
                accepted_rows: List[Dict[str, Any]] = []
                accepted_keys = set()

                reject_stats: Dict[str, Any] = {
                    "prefilter_reject_by_type": {},
                    "lower_invalid_count": 0,
                    "lower_exception_count": 0,
                    "duplicate_skips": 0,
                    "prefilter_attempts": 0,
                    "prefilter_accepted": 0,
                }

                try:
                    task, state = record_to_task_state(base_rec, task_map)
                    merge_report = get_storage_rewrite_merge_report(task, state)
                    system = build_system(base_rec, task, hw=hw, merge_report=merge_report)
                except Exception as e:
                    total_build_fail += 1
                    print(
                        "Task %2d sched %2d: build_failed (%s)"
                        % (task_idx, sched_idx, str(e).splitlines()[0][:160])
                    )
                    continue

                total_combos, sp_combo_prod, unroll_combo_prod = _count_total_combinations(
                    record=base_rec,
                    max_innermost=int(hw["max_innermost_split_factor"]),
                )

                base_formula_valid = evaluate_record(system, base_rec)["valid"]
                base_pref = check_constraints_prefilter(
                    system["constraints"], base_rec, system["hw"], mode=args.prefilter_mode
                )
                base_prefilter_valid = base_pref["valid"]
                base_lower_valid = None
                base_lower_err = None
                if args.verify_lowering:
                    base_lower_valid, base_lower_err = _verify_record_with_lowering(task_map, base_rec, hw)
                    if base_lower_valid:
                        total_base_lower_valid += 1

                if args.include_base:
                    base_sp, base_unroll = _extract_params_from_record(base_rec)
                    base_key = _param_key(base_sp, base_unroll)
                    accepted = False
                    if args.verify_lowering:
                        accepted = bool(base_lower_valid)
                        if not base_lower_valid:
                            if base_lower_err:
                                reject_stats["lower_exception_count"] += 1
                                total_lower_exception += 1
                            else:
                                reject_stats["lower_invalid_count"] += 1
                                total_lower_fail += 1
                    else:
                        accepted = bool(base_prefilter_valid)
                        if not accepted:
                            for v in base_pref["violations"]:
                                t = str(v.get("type", "unknown"))
                                reject_stats["prefilter_reject_by_type"][t] = (
                                    int(reject_stats["prefilter_reject_by_type"].get(t, 0)) + 1
                                )
                    if accepted and base_key not in accepted_keys:
                        accepted_keys.add(base_key)
                        accepted_rows.append(
                            {
                                "sp_params": base_sp,
                                "unroll_params": base_unroll,
                                "record": base_rec,
                                "source": "base",
                                "lower_valid": (None if not args.verify_lowering else True),
                                "lower_error": base_lower_err,
                            }
                        )

                # Global-only generation phase (no base-biased sampling).
                phase_cfg = [
                    {
                        "name": "global",
                        "budget": max(1, int(args.attempts_per_schedule)),
                        "base_split_bias": 0.0,
                        "base_unroll_bias": 0.0,
                        "max_candidates": max(8, int(args.max_candidates_per_split)),
                    }
                ]

                for phase in phase_cfg:
                    phase_budget = int(phase["budget"])
                    while len(accepted_rows) < args.samples_per_schedule and phase_budget > 0:
                        need = args.samples_per_schedule - len(accepted_rows)
                        per_call_max_attempts = max(1, phase_budget // max(1, need))
                        local_rng = random.Random(rng.randint(0, 2**31 - 1))
                        gen = generate_valid_params(
                            system=system,
                            base_record=base_rec,
                            n=need,
                            rng=local_rng,
                            max_attempts=per_call_max_attempts,
                            max_candidates_per_split=int(phase["max_candidates"]),
                            prefilter_mode=args.prefilter_mode,
                            include_base=False,
                            return_stats=True,
                            base_split_bias=float(phase["base_split_bias"]),
                            base_unroll_bias=float(phase["base_unroll_bias"]),
                        )
                        rows = gen["rows"]
                        st = gen["stats"]

                        used_attempts = int(st.get("attempts", 0))
                        phase_budget -= max(1, used_attempts)
                        reject_stats["prefilter_attempts"] += int(st.get("attempts", 0))
                        reject_stats["prefilter_accepted"] += int(st.get("accepted", 0))
                        reject_stats["duplicate_skips"] += int(st.get("duplicate_skips", 0))
                        _merge_reject_type_counts(
                            reject_stats["prefilter_reject_by_type"],
                            st.get("prefilter_reject_by_type", {}),
                        )

                        if not rows:
                            if used_attempts <= 0:
                                break
                            continue

                        for row in rows:
                            key = _param_key(row["sp_params"], row["unroll_params"])
                            if key in accepted_keys:
                                reject_stats["duplicate_skips"] += 1
                                continue

                            lower_ok = None
                            lower_err = None
                            if args.verify_lowering:
                                lower_ok, lower_err = _verify_record_with_lowering(task_map, row["record"], hw)
                                if not lower_ok:
                                    if lower_err:
                                        reject_stats["lower_exception_count"] += 1
                                        total_lower_exception += 1
                                    else:
                                        reject_stats["lower_invalid_count"] += 1
                                        total_lower_fail += 1
                                    continue

                            accepted_keys.add(key)
                            row["lower_valid"] = (None if not args.verify_lowering else True)
                            row["lower_error"] = lower_err
                            accepted_rows.append(row)
                            if len(accepted_rows) >= args.samples_per_schedule:
                                break

                diag = None
                if len(accepted_rows) == 0 and int(args.diagnose_on_empty) > 0:
                    diag_rng = random.Random(rng.randint(0, 2**31 - 1))
                    diag = _diagnose_on_empty(
                        base_rec=base_rec,
                        system=system,
                        task_map=task_map,
                        hw=hw,
                        verify_lowering=args.verify_lowering,
                        prefilter_mode=args.prefilter_mode,
                        k=int(args.diagnose_on_empty),
                        rng=diag_rng,
                    )

                for sample_idx, row in enumerate(accepted_rows[: args.samples_per_schedule]):
                    rec = row["record"]
                    pred_strict = evaluate_record(system, rec)["valid"]
                    dump = {
                        "task_idx": int(task_idx),
                        "workload_key": wk,
                        "schedule_idx": int(sched_idx),
                        "sample_idx": int(sample_idx),
                        "sp_params": row["sp_params"],
                        "unroll_params": row["unroll_params"],
                        "accept_source": row.get("source", "mutated"),
                        "formula_valid": bool(pred_strict),
                        "lower_valid": row.get("lower_valid"),
                        "lower_error": row.get("lower_error"),
                        "reject_stats": reject_stats,
                        "record": rec,
                    }
                    if diag is not None:
                        dump["diagnose_on_empty"] = diag
                    fout.write(json.dumps(dump) + "\n")

                generated = min(len(accepted_rows), int(args.samples_per_schedule))
                total_generated += generated
                if bool(base_lower_valid) and generated == 0:
                    total_base_lower_valid_but_zero += 1

                diag_short = ""
                if diag is not None:
                    diag_short = " diag_pref_reject=%s" % json.dumps(diag["prefilter_reject_by_type"])
                print(
                    "Task %2d sched %2d: combos(total=%d sp=%d unroll=%d) base_formula(%s) base_pref(%s) base_lower(%s) generated %d/%d%s"
                    % (
                        task_idx,
                        sched_idx,
                        total_combos,
                        sp_combo_prod,
                        unroll_combo_prod,
                        base_formula_valid,
                        base_prefilter_valid,
                        base_lower_valid,
                        generated,
                        args.samples_per_schedule,
                        diag_short,
                    )
                )

    shortfall = total_target - total_generated
    print("-" * 80)
    print("Schedules processed             : %d" % total_sched)
    print("Requested samples              : %d" % total_target)
    print("Generated samples              : %d" % total_generated)
    print("Shortfall                      : %d" % shortfall)
    print("Build failures                 : %d" % total_build_fail)
    print("Lowering invalid count         : %d" % total_lower_fail)
    print("Lowering exception count       : %d" % total_lower_exception)
    print("Base lower valid schedules     : %d" % total_base_lower_valid)
    print("Base lower valid but zero gen  : %d" % total_base_lower_valid_but_zero)
    print("Saved to                       : %s" % args.output_jsonl)

    ok = shortfall == 0 and total_build_fail == 0 and total_base_lower_valid_but_zero == 0
    if args.strict and not ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
