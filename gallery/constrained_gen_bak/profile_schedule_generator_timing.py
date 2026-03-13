"""Profile ScheduleGenerator timing for a selected sketch."""

import argparse
import json
import random
import statistics
import time
from collections import Counter, defaultdict

from modules.common import TO_MEASURE_PROGRAM_FOLDER
from modules.constraint_set import ConstraintSet
from modules.param_manager import SymParamManager, build_symbolic_state
from modules.projected_gpu_validation import (
    build_schedule_generator,
    ensure_parent_dir,
    load_sketch_lines,
    load_sketch_record,
    load_tasks_by_workload,
)
from modules.schedule_generator import ScheduleGenerator
from modules.var_order_planner import VarOrderPlanner


def _summarize(values):
    if not values:
        return None

    values_ms = sorted(v * 1000.0 for v in values)

    def percentile(p):
        if len(values_ms) == 1:
            return values_ms[0]
        k = (len(values_ms) - 1) * p
        f = int(k)
        c = min(f + 1, len(values_ms) - 1)
        if f == c:
            return values_ms[f]
        return values_ms[f] * (c - k) + values_ms[c] * (k - f)

    return {
        "count": len(values_ms),
        "mean_ms": round(statistics.mean(values_ms), 3),
        "median_ms": round(statistics.median(values_ms), 3),
        "p95_ms": round(percentile(0.95), 3),
        "min_ms": round(min(values_ms), 3),
        "max_ms": round(max(values_ms), 3),
    }


def _timestamp_text():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _format_elapsed(seconds):
    if seconds >= 1.0:
        return f"{seconds:.3f}s"
    return f"{seconds * 1000.0:.3f}ms"


def _log_step_start(label, detail=None):
    suffix = f" | {detail}" if detail else ""
    print(f"[start {_timestamp_text()}] {label}{suffix}", flush=True)
    return time.perf_counter()


def _log_step_end(label, started_at, detail=None, failed=False):
    elapsed = time.perf_counter() - started_at
    status = "fail" if failed else "done"
    suffix = f" | {detail}" if detail else ""
    print(
        f"[{status} {_timestamp_text()}] {label} | elapsed={_format_elapsed(elapsed)}{suffix}",
        flush=True,
    )
    return elapsed


def _run_logged_step(label, fn, detail=None):
    started_at = _log_step_start(label, detail=detail)
    try:
        result = fn()
    except Exception:
        _log_step_end(label, started_at, failed=True)
        raise
    elapsed = _log_step_end(label, started_at)
    return result, elapsed


def _parse_int_csv(text):
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def _collect_matching_sketches(sketch_lines, tasks_by_wkey, task_desc_substr, limit=None):
    matches = []
    for idx, line in enumerate(sketch_lines):
        if not line:
            continue
        task, _, _, _ = load_sketch_record(line, tasks_by_wkey)
        if task_desc_substr in task.desc:
            matches.append(
                {
                    "index": idx,
                    "task_desc": task.desc,
                    "workload_key": task.workload_key,
                }
            )
            if limit is not None and len(matches) >= limit:
                break
    return matches


def _resolve_target(args, sketch_lines, tasks_by_wkey):
    if args.sketch_index is not None:
        idx = args.sketch_index
        if idx < 0 or idx >= len(sketch_lines):
            raise IndexError(f"Sketch index {idx} is out of range")
        line = sketch_lines[idx]
        if not line:
            raise ValueError(f"Sketch index {idx} points to an empty line")
        task, base_inp, base_res, state = load_sketch_record(line, tasks_by_wkey)
        return {
            "selection_mode": "sketch_index",
            "selected": {
                "index": idx,
                "task_desc": task.desc,
                "workload_key": task.workload_key,
            },
            "matches_preview": [
                {
                    "index": idx,
                    "task_desc": task.desc,
                    "workload_key": task.workload_key,
                }
            ],
            "task": task,
            "base_inp": base_inp,
            "base_res": base_res,
            "state": state,
        }

    matches = _collect_matching_sketches(
        sketch_lines,
        tasks_by_wkey,
        args.task_desc_substr,
        limit=None,
    )
    if not matches:
        raise ValueError(f"No sketches matched task-desc substring: {args.task_desc_substr!r}")
    if args.match_ordinal < 0 or args.match_ordinal >= len(matches):
        raise IndexError(
            f"match-ordinal {args.match_ordinal} is out of range for {len(matches)} matches"
        )

    chosen = matches[args.match_ordinal]
    task, base_inp, base_res, state = load_sketch_record(
        sketch_lines[chosen["index"]], tasks_by_wkey
    )
    return {
        "selection_mode": "task_desc_substr",
        "selected": chosen,
        "matches_preview": matches[: args.preview_matches],
        "match_count": len(matches),
        "task": task,
        "base_inp": base_inp,
        "base_res": base_res,
        "state": state,
    }


def _time_repeated(repeats, fn):
    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - t0)
    return timings


def _time_init_only_repeated(task, state, repeats, base_inp=None, base_res=None):
    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        build_schedule_generator(task, state, base_inp, base_res)
        timings.append(time.perf_counter() - t0)
    return timings


def _profile_single_init_serial(task, state):
    sym = build_symbolic_state(task, state)
    overall_started = _log_step_start("single_init_serial")

    gen = object.__new__(ScheduleGenerator)
    gen.s = sym
    gen.hw = dict(ScheduleGenerator.DEFAULT_HW_PARAM)
    gen.pm = SymParamManager(sym)
    gen._enabled = set(ScheduleGenerator.ALL_CONSTRAINT_KINDS)
    gen._constraints = []
    gen._var_constraints = {}
    gen._var_order = []
    gen._var_order_phase_entries = []
    gen._exact_gpu = None
    gen._projected_gpu = None
    gen._projected_gpu_context = None
    gen._vectorize_constraint_bundle = None
    gen._shared_memory_constraint_bundle = None
    gen._max_threads_constraint_bundle = None
    gen._max_vthread_constraint_bundle = None
    gen._split_structure_constraint_bundle = None
    gen._concrete_final_cache = {}
    gen.constraint_set = ConstraintSet(gen)
    gen.var_order_planner = VarOrderPlanner(gen)

    step_records = []

    def run_step(label, fn, detail=None):
        started = _log_step_start(f"single_init_serial:{label}", detail=detail)
        value = fn()
        elapsed = _log_step_end(f"single_init_serial:{label}", started)
        step_records.append(
            {
                "step": label,
                "elapsed_ms": round(elapsed * 1000.0, 3),
            }
        )
        return value

    sp_groups = run_step("pm._build_sp_groups", gen.pm._build_sp_groups)
    sp_extents = run_step("pm._build_sp_extents", lambda: gen.pm._build_sp_extents(sp_groups))

    def build_innermost_names():
        names = set()
        if "innermost_split" in gen._enabled:
            for _, group_names in sp_groups.items():
                names.add(group_names[-1])
        return names

    innermost_names = run_step("derive_innermost_names", build_innermost_names)

    def assign_base_metadata():
        gen._sp_groups = sp_groups
        gen._sp_extents = sp_extents
        gen._ur_names = [name for name in gen.s.sym_map if name.startswith("ur_")]
        gen._all_sp_names = []
        for step_idx in sorted(sp_groups.keys()):
            gen._all_sp_names.extend(sp_groups[step_idx])
        gen._innermost_names = innermost_names
        gen._exact_gpu = None
        gen._projected_gpu = None
        gen._projected_gpu_context = None
        gen._preferred_thread_vars = []

    run_step("assign_base_metadata", assign_base_metadata)
    run_step(
        "_collect_vthread_clamped_sp_names",
        lambda: setattr(
            gen,
            "_vthread_clamped_sp_names",
            gen.constraint_set._collect_vthread_clamped_sp_names(),
        ),
    )
    run_step("reset_constraint_storage", lambda: (gen._constraints.clear(), gen._var_constraints.clear()))

    constraint_keys = set()

    def add_constraint(
        expr_tree,
        rhs,
        kind,
        desc,
        is_upper=True,
        index_vars=True,
        display_text=None,
        display_rhs=None,
        alias_entries=None,
        fast_path=None,
    ):
        key = (
            kind,
            rhs,
            is_upper,
            display_text if display_text is not None else repr(expr_tree),
            display_rhs,
        )
        if key in constraint_keys:
            return
        constraint_keys.add(key)
        idx = len(gen._constraints)
        vars_in = expr_tree.variables()
        has_nonlinear = gen.constraint_set._has_nonlinear(expr_tree)
        product_form_meta = (
            gen.constraint_set._extract_product_form_meta(expr_tree)
            if is_upper
            else None
        )
        gen._constraints.append(
            {
                "tree": expr_tree,
                "rhs": rhs,
                "vars": vars_in,
                "kind": kind,
                "desc": desc,
                "is_upper": is_upper,
                "has_nonlinear": has_nonlinear,
                "display_text": display_text,
                "display_rhs": display_rhs,
                "alias_entries": list(alias_entries or []),
                "product_form_meta": product_form_meta,
                "fast_path": fast_path,
            }
        )
        if index_vars:
            for name in vars_in:
                gen._var_constraints.setdefault(name, []).append(idx)

    if "vectorize" in gen._enabled:
        vectorize = run_step("build_vectorize_constraints", gen.build_vectorize_constraints)

        def add_vectorize():
            for item in vectorize["items"]:
                add_constraint(
                    item["tree"],
                    item["limit"],
                    "vectorize",
                    item["desc"],
                    is_upper=True,
                    index_vars=True,
                )

        run_step("add_vectorize_constraints", add_vectorize)

    if "shared_memory" in gen._enabled:
        shared_memory = run_step("build_shared_memory_constraints", gen.build_shared_memory_constraints)
        run_step(
            "add_shared_memory_constraint",
            lambda: add_constraint(
                shared_memory["tree"],
                shared_memory["limit"],
                "shared_memory",
                shared_memory["desc"],
                is_upper=True,
                index_vars=True,
            ),
        )

    if "max_threads" in gen._enabled:
        max_threads = run_step("build_max_threads_constraints", gen.build_max_threads_constraints)
        run_step(
            "_collect_preferred_thread_vars",
            lambda: setattr(
                gen,
                "_preferred_thread_vars",
                gen.constraint_set._collect_preferred_thread_vars(max_threads["items"]),
            ),
        )

        def add_max_threads():
            for item in max_threads["items"]:
                add_constraint(
                    item["tree"],
                    item["limit"],
                    "max_threads",
                    item["desc"],
                    is_upper=True,
                    display_text=str(item["sym_extent"]),
                    alias_entries=item.get("alias_entries"),
                )

        run_step("add_max_threads_constraints", add_max_threads)

    if "max_vthread" in gen._enabled:
        max_vthread = run_step("build_max_vthread_constraints", gen.build_max_vthread_constraints)

        def add_max_vthread():
            for item in max_vthread["items"]:
                add_constraint(
                    item["tree"],
                    item["limit"],
                    "max_vthread",
                    item["desc"],
                    is_upper=True,
                    index_vars=True,
                )

        run_step("add_max_vthread_constraints", add_max_vthread)

    if "split_structure" in gen._enabled:
        split_structure = run_step("build_split_structure_constraints", gen.build_split_structure_constraints)

        def add_split_structure():
            for item in split_structure:
                add_constraint(
                    item["tree"],
                    item["limit"],
                    "split_structure",
                    item["desc"],
                    is_upper=True,
                    index_vars=True,
                    display_text=item["display_text"],
                    display_rhs=item["display_rhs"],
                    fast_path=gen.constraint_set._build_split_structure_fast_path(
                        item["sym_name"],
                        item["dependency_names"],
                        gen._sp_extents.get(item["step_idx"], item["limit"] + 1),
                    ),
                )

        run_step("add_split_structure_constraints", add_split_structure)

    legacy_order = run_step(
        "_compute_legacy_var_order",
        gen.var_order_planner._compute_legacy_var_order,
    )
    phase_entries = run_step(
        "_build_var_order_phase_entries",
        gen.var_order_planner._build_var_order_phase_entries,
    )

    def finalize_var_order():
        ordered = []
        seen = set()
        normalized_entries = []
        for entry in phase_entries:
            phase_vars = []
            for name in entry["vars"]:
                if name in seen:
                    continue
                seen.add(name)
                ordered.append(name)
                phase_vars.append(name)
            normalized_entries.append(
                {
                    "name": entry["name"],
                    "label": entry["label"],
                    "vars": phase_vars,
                }
            )
        for name in legacy_order:
            if name in seen:
                continue
            seen.add(name)
            ordered.append(name)
        gen._var_order_phase_entries = normalized_entries
        gen._var_order = ordered

    run_step("finalize_var_order", finalize_var_order)
    total = _log_step_end("single_init_serial", overall_started)
    step_total_ms = round(sum(item["elapsed_ms"] for item in step_records), 3)
    return {
        "overlap_check": {
            "overlap_detected_in_old_wrapper_profile": True,
            "reason": (
                "previous single_init_instrumented wrapped nested methods, so "
                "_preprocess/build_vectorize_constraints/_ensure_exact_gpu_constraints "
                "double-counted the same wall-clock interval"
            ),
        },
        "total_ms": round(total * 1000.0, 3),
        "step_total_ms": step_total_ms,
        "steps": step_records,
    }


def _profile_phase_prefixes(gen, phase_entries, repeats, max_retries):
    overall_started = _log_step_start(
        "profile_phase_prefixes",
        detail=f"phases={len(phase_entries)} repeats={repeats} max_retries={max_retries}",
    )
    result = {}
    for entry in phase_entries:
        phase_started = _log_step_start(
            f"profile_phase_prefix:{entry['name']}",
            detail=f"vars={len(entry['vars'])} repeats={repeats}",
        )
        times = []
        success = 0
        for seed in range(repeats):
            t0 = time.perf_counter()
            try:
                gen.randomize_params_prefix(
                    entry["name"],
                    rng=random.Random(seed),
                    max_retries=max_retries,
                )
                success += 1
            except Exception:
                pass
            times.append(time.perf_counter() - t0)
        result[entry["name"]] = {
            "var_count": len(entry["vars"]),
            "success": success,
            "attempts": repeats,
            "timing": _summarize(times),
        }
        _log_step_end(
            f"profile_phase_prefix:{entry['name']}",
            phase_started,
            detail=f"success={success}/{repeats}",
        )
    _log_step_end("profile_phase_prefixes", overall_started)
    return result


def _profile_randomize_runs(gen, repeats, max_retries_values):
    overall_started = _log_step_start(
        "profile_randomize_runs",
        detail=f"repeats={repeats} retry_buckets={list(max_retries_values)}",
    )
    result = {}
    for max_retries in max_retries_values:
        bucket_started = _log_step_start(
            f"profile_randomize_runs:max_retries={max_retries}",
            detail=f"repeats={repeats}",
        )
        times = []
        success = 0
        failures = 0
        last_error = None
        for seed in range(repeats):
            t0 = time.perf_counter()
            try:
                gen.randomize_params(rng=random.Random(seed), max_retries=max_retries)
                success += 1
            except Exception as err:  # pylint: disable=broad-except
                failures += 1
                last_error = f"{type(err).__name__}: {err}"
            times.append(time.perf_counter() - t0)
        result[f"max_retries_{max_retries}"] = {
            "success": success,
            "failures": failures,
            "attempts": repeats,
            "last_error": last_error,
            "timing": _summarize(times),
        }
        _log_step_end(
            f"profile_randomize_runs:max_retries={max_retries}",
            bucket_started,
            detail=f"success={success} failures={failures}",
        )
    _log_step_end("profile_randomize_runs", overall_started)
    return result


def _profile_randomize_internal(gen, repeats, max_retries):
    orig_filter = gen.domain_propagator.filter_by_constraints
    orig_propagate = gen.domain_propagator.propagate_domain
    orig_bisect_upper = gen.domain_propagator._bisect_upper
    orig_bisect_lower = gen.domain_propagator._bisect_lower
    orig_check_all_hybrid = gen.check_all_hybrid
    orig_check_all_exact = gen.check_all_exact
    orig_check_all_final = gen.check_all_final
    orig_get_concrete_final_result = gen.get_concrete_final_result
    orig_eval_exact_upper_bounds = gen.constraint_set._evaluate_exact_upper_bounds
    orig_divisors = gen.pm._divisors

    kind_by_index = {idx: constraint["kind"] for idx, constraint in enumerate(gen._constraints)}
    per_key_time = defaultdict(list)
    per_key_count = defaultdict(list)
    propagate_kind_time = defaultdict(list)
    propagate_kind_count = defaultdict(list)
    filter_kind_time = defaultdict(list)
    filter_kind_count = defaultdict(list)
    bisect_kind_time = defaultdict(list)
    bisect_kind_count = defaultdict(list)
    total_times = []
    other_times = []
    success = 0
    first_error = None

    overall_started = _log_step_start(
        "profile_randomize_internal",
        detail=f"repeats={repeats} max_retries={max_retries}",
    )
    for seed in range(repeats):
        seed_started = _log_step_start(
            f"profile_randomize_internal:seed={seed}",
            detail=f"max_retries={max_retries}",
        )
        gen._concrete_final_cache.clear()
        metrics = defaultdict(float)
        counts = Counter()
        propagate_metrics = defaultdict(float)
        propagate_counts = Counter()
        filter_metrics = defaultdict(float)
        filter_counts = Counter()
        bisect_metrics = defaultdict(float)
        bisect_counts = Counter()

        def wrap(orig, key):
            def inner(*args, **kwargs):
                t0 = time.perf_counter()
                try:
                    return orig(*args, **kwargs)
                finally:
                    metrics[key] += time.perf_counter() - t0
                    counts[key] += 1

            return inner

        def wrap_propagate(orig):
            def inner(assigned_name, domains):
                constraint_indices = gen._var_constraints.get(assigned_name, [])
                t0 = time.perf_counter()
                try:
                    return orig(assigned_name, domains)
                finally:
                    elapsed = time.perf_counter() - t0
                    metrics["propagate_domain"] += elapsed
                    counts["propagate_domain"] += 1
                    if constraint_indices:
                        per_constraint = elapsed / len(constraint_indices)
                        for ci in constraint_indices:
                            kind = kind_by_index[ci]
                            propagate_metrics[kind] += per_constraint
                            propagate_counts[kind] += 1

            return inner

        def wrap_filter(orig):
            def inner(var_name, candidates, constraint_indices, domains):
                t0 = time.perf_counter()
                try:
                    return orig(var_name, candidates, constraint_indices, domains)
                finally:
                    elapsed = time.perf_counter() - t0
                    metrics["filter_by_constraints"] += elapsed
                    counts["filter_by_constraints"] += 1
                    if constraint_indices:
                        per_constraint = elapsed / len(constraint_indices)
                        for ci in constraint_indices:
                            kind = kind_by_index[ci]
                            filter_metrics[kind] += per_constraint
                            filter_counts[kind] += 1

            return inner

        def wrap_bisect(orig, label):
            def inner(var_name, candidates, constraint, interval_domains):
                t0 = time.perf_counter()
                try:
                    return orig(var_name, candidates, constraint, interval_domains)
                finally:
                    elapsed = time.perf_counter() - t0
                    key = f"{label}:{constraint['kind']}"
                    bisect_metrics[key] += elapsed
                    bisect_counts[key] += 1

            return inner

        gen.domain_propagator.filter_by_constraints = wrap_filter(orig_filter)
        gen.domain_propagator.propagate_domain = wrap_propagate(orig_propagate)
        gen.domain_propagator._bisect_upper = wrap_bisect(orig_bisect_upper, "upper")
        gen.domain_propagator._bisect_lower = wrap_bisect(orig_bisect_lower, "lower")
        gen.check_all_hybrid = wrap(orig_check_all_hybrid, "check_all_hybrid")
        gen.check_all_exact = wrap(orig_check_all_exact, "check_all_exact")
        gen.check_all_final = wrap(orig_check_all_final, "check_all_final")
        gen.get_concrete_final_result = wrap(
            orig_get_concrete_final_result, "get_concrete_final_result"
        )
        gen.constraint_set._evaluate_exact_upper_bounds = wrap(
            orig_eval_exact_upper_bounds, "_evaluate_exact_upper_bounds"
        )
        gen.pm._divisors = wrap(orig_divisors, "divisors")

        status = "success"
        t0 = time.perf_counter()
        try:
            gen.randomize_params(rng=random.Random(seed), max_retries=max_retries)
            success += 1
        except Exception as err:  # pylint: disable=broad-except
            status = "fail"
            if first_error is None:
                first_error = f"{type(err).__name__}: {err}"
        total = time.perf_counter() - t0

        gen.domain_propagator.filter_by_constraints = orig_filter
        gen.domain_propagator.propagate_domain = orig_propagate
        gen.domain_propagator._bisect_upper = orig_bisect_upper
        gen.domain_propagator._bisect_lower = orig_bisect_lower
        gen.check_all_hybrid = orig_check_all_hybrid
        gen.check_all_exact = orig_check_all_exact
        gen.check_all_final = orig_check_all_final
        gen.get_concrete_final_result = orig_get_concrete_final_result
        gen.constraint_set._evaluate_exact_upper_bounds = orig_eval_exact_upper_bounds
        gen.pm._divisors = orig_divisors

        total_times.append(total)
        subtotal = 0.0
        for key, value in metrics.items():
            per_key_time[key].append(value)
            subtotal += value
        for key, value in counts.items():
            per_key_count[key].append(value)
        for key, value in propagate_metrics.items():
            propagate_kind_time[key].append(value)
            propagate_kind_count[key].append(propagate_counts[key])
        for key, value in filter_metrics.items():
            filter_kind_time[key].append(value)
            filter_kind_count[key].append(filter_counts[key])
        for key, value in bisect_metrics.items():
            bisect_kind_time[key].append(value)
            bisect_kind_count[key].append(bisect_counts[key])
        other_times.append(max(total - subtotal, 0.0))
        _log_step_end(
            f"profile_randomize_internal:seed={seed}",
            seed_started,
            detail=f"status={status}",
        )

    breakdown = {
        key: {
            "avg_total_ms_per_run": round(statistics.mean(values) * 1000.0, 3),
            "avg_calls_per_run": round(statistics.mean(per_key_count[key]), 3)
            if per_key_count[key]
            else 0.0,
        }
        for key, values in sorted(
            per_key_time.items(), key=lambda item: statistics.mean(item[1]), reverse=True
        )
    }
    breakdown["other_python_overhead"] = {
        "avg_total_ms_per_run": round(statistics.mean(other_times) * 1000.0, 3),
        "avg_calls_per_run": None,
    }

    propagate_by_kind = {
        key: {
            "avg_total_ms_per_run": round(statistics.mean(values) * 1000.0, 3),
            "avg_constraint_hits_per_run": round(statistics.mean(propagate_kind_count[key]), 3),
        }
        for key, values in sorted(
            propagate_kind_time.items(),
            key=lambda item: statistics.mean(item[1]),
            reverse=True,
        )
    }
    filter_by_kind = {
        key: {
            "avg_total_ms_per_run": round(statistics.mean(values) * 1000.0, 3),
            "avg_constraint_hits_per_run": round(statistics.mean(filter_kind_count[key]), 3),
        }
        for key, values in sorted(
            filter_kind_time.items(),
            key=lambda item: statistics.mean(item[1]),
            reverse=True,
        )
    }
    bisect_by_kind = {
        key: {
            "avg_total_ms_per_run": round(statistics.mean(values) * 1000.0, 3),
            "avg_calls_per_run": round(statistics.mean(bisect_kind_count[key]), 3),
        }
        for key, values in sorted(
            bisect_kind_time.items(),
            key=lambda item: statistics.mean(item[1]),
            reverse=True,
        )
    }

    _log_step_end("profile_randomize_internal", overall_started, detail=f"success={success}/{repeats}")
    return {
        "success": success,
        "attempts": repeats,
        "first_error": first_error,
        "total_timing": _summarize(total_times),
        "breakdown": breakdown,
        "propagate_by_constraint_kind": propagate_by_kind,
        "filter_by_constraint_kind": filter_by_kind,
        "bisect_by_constraint_kind": bisect_by_kind,
    }


def run(args):
    run_started = _log_step_start("profile_schedule_generator_timing.run")
    execution_steps = {}

    (tasks, tasks_by_wkey), elapsed = _run_logged_step(
        "load_tasks_by_workload",
        lambda: load_tasks_by_workload(args.network_info_dir),
        detail=args.network_info_dir,
    )
    execution_steps["load_tasks_by_workload_sec"] = round(elapsed, 6)

    sketch_lines, elapsed = _run_logged_step(
        "load_sketch_lines",
        lambda: load_sketch_lines(args.sketches_path),
        detail=args.sketches_path,
    )
    execution_steps["load_sketch_lines_sec"] = round(elapsed, 6)

    target, elapsed = _run_logged_step(
        "resolve_target",
        lambda: _resolve_target(args, sketch_lines, tasks_by_wkey),
        detail=(
            f"sketch_index={args.sketch_index}"
            if args.sketch_index is not None
            else f"task_desc_substr={args.task_desc_substr!r}"
        ),
    )
    execution_steps["resolve_target_sec"] = round(elapsed, 6)
    task = target["task"]
    base_inp = target["base_inp"]
    base_res = target["base_res"]
    state = target["state"]

    print(
        f"selected sketch={target['selected']['index']} task_desc={target['selected']['task_desc']}",
        flush=True,
    )

    sym, elapsed = _run_logged_step(
        "build_symbolic_state_for_metadata",
        lambda: build_symbolic_state(task, state),
    )
    execution_steps["build_symbolic_state_for_metadata_sec"] = round(elapsed, 6)

    gen, elapsed = _run_logged_step(
        "build_schedule_generator_for_metadata",
        lambda: ScheduleGenerator(sym, task=task, base_input=base_inp, base_result=base_res),
    )
    execution_steps["build_schedule_generator_for_metadata_sec"] = round(elapsed, 6)

    phase_entries, elapsed = _run_logged_step(
        "collect_var_order_phase_entries",
        gen.get_var_order_phase_entries,
    )
    execution_steps["collect_var_order_phase_entries_sec"] = round(elapsed, 6)

    constraint_records, elapsed = _run_logged_step(
        "collect_constraint_records",
        gen.get_constraint_records,
    )
    execution_steps["collect_constraint_records_sec"] = round(elapsed, 6)
    constraint_kind_counter = Counter(record["kind"] for record in constraint_records)

    result = {
        "selection": {
            "mode": target["selection_mode"],
            "selected": target["selected"],
            "matches_preview": target["matches_preview"],
        },
        "execution_steps": execution_steps,
        "meta": {
            "sp_var_count": len(gen._all_sp_names),
            "ur_var_count": len(gen._ur_names),
            "constraint_count": len(constraint_records),
            "constraint_kinds": dict(constraint_kind_counter),
            "phase_entries": [
                {
                    "name": entry["name"],
                    "var_count": len(entry["vars"]),
                    "vars": list(entry["vars"]),
                }
                for entry in phase_entries
            ],
        },
    }
    if "match_count" in target:
        result["selection"]["match_count"] = target["match_count"]

    timing_overview = {}
    value, elapsed = _run_logged_step(
        "timing_overview.load_sketch_record",
        lambda: _summarize(
            _time_repeated(
                args.build_repeats,
                lambda: _resolve_target(args, sketch_lines, tasks_by_wkey),
            )
        ),
        detail=f"repeats={args.build_repeats}",
    )
    timing_overview["load_sketch_record"] = value
    result["execution_steps"]["timing_overview.load_sketch_record_sec"] = round(elapsed, 6)

    value, elapsed = _run_logged_step(
        "timing_overview.build_symbolic_state",
        lambda: _summarize(
            _time_repeated(
                args.build_repeats,
                lambda: build_symbolic_state(task, state),
            )
        ),
        detail=f"repeats={args.build_repeats}",
    )
    timing_overview["build_symbolic_state"] = value
    result["execution_steps"]["timing_overview.build_symbolic_state_sec"] = round(elapsed, 6)

    value, elapsed = _run_logged_step(
        "timing_overview.schedule_generator_init_only",
        lambda: _summarize(
            _time_init_only_repeated(
                task,
                state,
                args.build_repeats,
                base_inp=base_inp,
                base_res=base_res,
            )
        ),
        detail=f"repeats={args.build_repeats}",
    )
    timing_overview["schedule_generator_init_only"] = value
    result["execution_steps"]["timing_overview.schedule_generator_init_only_sec"] = round(elapsed, 6)

    value = _profile_single_init_serial(task, state)
    timing_overview["single_init_serial"] = value
    result["execution_steps"]["timing_overview.single_init_serial_sec"] = round(
        value["total_ms"] / 1000.0, 6
    )
    result["timing_overview"] = timing_overview

    value, elapsed = _run_logged_step(
        "phase_prefix_timings",
        lambda: _profile_phase_prefixes(
            gen,
            phase_entries,
            repeats=args.phase_repeats,
            max_retries=args.phase_max_retries,
        ),
    )
    result["phase_prefix_timings"] = value
    result["execution_steps"]["phase_prefix_timings_sec"] = round(elapsed, 6)

    retry_values = _parse_int_csv(args.max_retries_values)
    value, elapsed = _run_logged_step(
        "randomize_params",
        lambda: _profile_randomize_runs(
            gen,
            repeats=args.randomize_repeats,
            max_retries_values=retry_values,
        ),
        detail=f"repeats={args.randomize_repeats}",
    )
    result["randomize_params"] = value
    result["execution_steps"]["randomize_params_sec"] = round(elapsed, 6)

    value, elapsed = _run_logged_step(
        "randomize_params_internal",
        lambda: _profile_randomize_internal(
            gen,
            repeats=args.internal_repeats,
            max_retries=args.internal_max_retries,
        ),
        detail=f"repeats={args.internal_repeats}",
    )
    result["randomize_params_internal"] = value
    result["execution_steps"]["randomize_params_internal_sec"] = round(elapsed, 6)
    result["execution_steps"]["run_elapsed_before_output_sec"] = round(
        time.perf_counter() - run_started, 6
    )

    if args.output_path:
        def _write_output():
            ensure_parent_dir(args.output_path)
            with open(args.output_path, "w") as f:
                json.dump(result, f, indent=2, sort_keys=True)

        _run_logged_step(
            "write_output_json",
            _write_output,
            detail=args.output_path,
        )
        print(f"wrote {args.output_path}", flush=True)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    _log_step_end("profile_schedule_generator_timing.run", run_started)


def parse_args():
    parser = argparse.ArgumentParser()
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--sketch-index", type=int, default=None)
    target.add_argument("--task-desc-substr", default=None)

    parser.add_argument("--match-ordinal", type=int, default=0)
    parser.add_argument("--preview-matches", type=int, default=12)
    parser.add_argument(
        "--sketches-path",
        default=f"{TO_MEASURE_PROGRAM_FOLDER}/all_sketches.json",
    )
    parser.add_argument(
        "--network-info-dir",
        default="/root/work/tvm-ansor/gallery/dataset/network_info",
    )
    parser.add_argument("--build-repeats", type=int, default=5)
    parser.add_argument("--phase-repeats", type=int, default=20)
    parser.add_argument("--phase-max-retries", type=int, default=1)
    parser.add_argument("--randomize-repeats", type=int, default=20)
    parser.add_argument("--max-retries-values", default="1,64")
    parser.add_argument("--internal-repeats", type=int, default=10)
    parser.add_argument("--internal-max-retries", type=int, default=64)
    parser.add_argument("--output-path", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
