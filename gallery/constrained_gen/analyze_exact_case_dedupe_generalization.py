"""Analyze exact GPU case-output dedupe opportunity across sketches."""

import argparse
import json
import os
import pickle
import time
from collections import Counter, defaultdict

from modules.projected_gpu_validation import (
    build_schedule_generator,
    ensure_parent_dir,
    load_sketch_lines,
    load_sketch_record,
    load_tasks_by_workload,
)


DEFAULT_ROOT = "/tmp/projected_gpu_full_validation/exact_case_dedupe_generalization"
DEFAULT_CACHE_DIR = f"{DEFAULT_ROOT}/generator_cache"
DEFAULT_OUTPUT_PATH = f"{DEFAULT_ROOT}/summary.json"
DEFAULT_REPORT_PATH = f"{DEFAULT_ROOT}/summary.md"
DEFAULT_REPRESENTATIVES_JSON = "/tmp/projected_gpu_full_validation/representatives_parallel_official.json"


def _safe_ratio(num, den):
    if not den:
        return None
    return round(float(num) / float(den), 3)


def _percentile(sorted_values, q):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    if lo == hi:
        return float(sorted_values[lo])
    return float(sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo))


def _summarize_numeric(values):
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    mean = sum(ordered) / len(ordered)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[mid]
    else:
        median = (ordered[mid - 1] + ordered[mid]) / 2.0
    return {
        "count": len(ordered),
        "mean": round(mean, 3),
        "median": round(median, 3),
        "min": round(ordered[0], 3),
        "p95": round(_percentile(ordered, 0.95), 3),
        "max": round(ordered[-1], 3),
    }


def _load_representative_indices(path):
    with open(path) as f:
        payload = json.load(f)

    by_index = {}
    for category in ("vectorize", "shared_memory", "max_vthread"):
        for item in payload.get(category, []):
            entry = by_index.setdefault(
                int(item["index"]),
                {
                    "index": int(item["index"]),
                    "task_desc": item["task_desc"],
                    "categories": set(),
                },
            )
            entry["categories"].add(category)

    ordered = []
    for index in sorted(by_index):
        entry = by_index[index]
        ordered.append(
            {
                "index": entry["index"],
                "task_desc": entry["task_desc"],
                "categories": sorted(entry["categories"]),
            }
        )
    return ordered


def _parse_indices(indices_csv):
    values = []
    for part in indices_csv.split(","):
        text = part.strip()
        if not text:
            continue
        values.append(int(text))
    return values


def _resolve_targets(args, sketch_lines, tasks_by_wkey):
    if args.indices:
        targets = []
        for idx in _parse_indices(args.indices):
            task, _, _, _ = load_sketch_record(sketch_lines[idx], tasks_by_wkey)
            targets.append(
                {
                    "index": idx,
                    "task_desc": task.desc,
                    "categories": ["manual"],
                }
            )
        return targets

    if args.representatives_json:
        return _load_representative_indices(args.representatives_json)

    targets = []
    for idx, line in enumerate(sketch_lines):
        if not line:
            continue
        task, _, _, _ = load_sketch_record(line, tasks_by_wkey)
        targets.append(
            {
                "index": idx,
                "task_desc": task.desc,
                "categories": ["all_sketches"],
            }
        )
        if args.limit and len(targets) >= args.limit:
            break
    return targets


def _generator_cache_path(cache_dir, index):
    return os.path.join(cache_dir, f"sketch_{index}.pkl")


def _expr_key(expr):
    return str(expr)


def _top_counter_entries(counter, limit=5):
    return [
        {"count": count, "key": key}
        for key, count in counter.most_common(limit)
    ]


def _analyze_exact_case_dedupe(gen):
    gen._ensure_exact_gpu_constraints()
    exact = gen._exact_gpu

    vector_cases = list(exact["vector_cases"])
    shared_cases = list(exact["shared_node"].cases)
    vthread_cases = list(exact["max_vthread_node"].cases)
    vector_expr_cases = list(exact["vector_node"].cases)

    case_count = len(vector_cases)
    if not (len(shared_cases) == len(vthread_cases) == len(vector_expr_cases) == case_count):
        raise RuntimeError(
            "Exact case arrays have mismatched lengths: "
            f"vector_cases={case_count} shared={len(shared_cases)} "
            f"vthread={len(vthread_cases)} vector={len(vector_expr_cases)}"
        )

    shared_counter = Counter(_expr_key(case["expr"]) for case in shared_cases)
    vthread_counter = Counter(_expr_key(case["expr"]) for case in vthread_cases)
    vector_counter = Counter(_expr_key(case["expr"]) for case in vector_expr_cases)
    combo_counter = Counter(
        (
            _expr_key(shared_case["expr"]),
            _expr_key(vthread_case["expr"]),
            _expr_key(vector_case["expr"]),
        )
        for shared_case, vthread_case, vector_case in zip(
            shared_cases, vthread_cases, vector_expr_cases
        )
    )

    vector_case_widths = [len(set(values)) for values in zip(*vector_cases)] if vector_cases and vector_cases[0] else []

    return {
        "selector_count": len(exact["selectors"]),
        "vector_case_count": case_count,
        "unique_selector_value_tuple_count": len(set(vector_cases)),
        "unique_shared_expr_count": len(shared_counter),
        "unique_vthread_expr_count": len(vthread_counter),
        "unique_vector_expr_count": len(vector_counter),
        "unique_output_combo_count": len(combo_counter),
        "duplicate_output_combo_count": case_count - len(combo_counter),
        "output_combo_duplicate_fraction": _safe_ratio(case_count - len(combo_counter), case_count),
        "output_combo_collapse_ratio": _safe_ratio(case_count, len(combo_counter)),
        "shared_constant": len(shared_counter) == 1,
        "vthread_constant": len(vthread_counter) == 1,
        "vector_constant": len(vector_counter) == 1,
        "selector_unique_value_counts": vector_case_widths,
        "largest_output_combo_bucket": max(combo_counter.values()) if combo_counter else 0,
        "top_output_combo_buckets": [
            {
                "count": count,
                "shared_expr": key[0],
                "vthread_expr": key[1],
                "vector_expr": key[2],
            }
            for key, count in combo_counter.most_common(5)
        ],
        "top_vector_expr_buckets": _top_counter_entries(vector_counter),
        "top_shared_expr_buckets": _top_counter_entries(shared_counter),
        "top_vthread_expr_buckets": _top_counter_entries(vthread_counter),
    }


def _load_or_build_generator(args, task, state, index):
    cache_path = _generator_cache_path(args.cache_dir, index)
    ensure_parent_dir(cache_path)

    timings = {}
    cache_status = {"cache_path": cache_path}
    dirty = False

    started = time.perf_counter()
    if os.path.exists(cache_path) and not args.rebuild_cache:
        print(f"    [load cache] {cache_path}", flush=True)
        with open(cache_path, "rb") as f:
            gen = pickle.load(f)
        cache_status["generator_cache_action"] = "load"
    else:
        print("    [build generator]", flush=True)
        gen = build_schedule_generator(task, state)
        cache_status["generator_cache_action"] = "build"
        dirty = True
    timings["generator_load_or_build_ms"] = round((time.perf_counter() - started) * 1000.0, 3)

    started = time.perf_counter()
    had_exact = gen._exact_gpu is not None
    print("    [ensure exact gpu constraints]", flush=True)
    gen._ensure_exact_gpu_constraints()
    timings["ensure_exact_gpu_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    cache_status["had_exact_in_cache"] = had_exact
    if not had_exact:
        dirty = True

    if dirty:
        started = time.perf_counter()
        print(f"    [write cache] {cache_path}", flush=True)
        with open(cache_path, "wb") as f:
            pickle.dump(gen, f, protocol=pickle.HIGHEST_PROTOCOL)
        timings["cache_write_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        cache_status["cache_write"] = True
    else:
        timings["cache_write_ms"] = 0.0
        cache_status["cache_write"] = False

    cache_status["cache_size_bytes"] = os.path.getsize(cache_path)
    return gen, cache_status, timings


def _aggregate_summary(results):
    with_cases = [item for item in results if item["dedupe"]["vector_case_count"] > 1]
    any_collapse = [item for item in with_cases if item["dedupe"]["unique_output_combo_count"] < item["dedupe"]["vector_case_count"]]
    shared_constant = [item for item in with_cases if item["dedupe"]["shared_constant"]]
    vthread_constant = [item for item in with_cases if item["dedupe"]["vthread_constant"]]
    vector_constant = [item for item in with_cases if item["dedupe"]["vector_constant"]]

    by_category = defaultdict(list)
    for item in results:
        for category in item["categories"]:
            by_category[category].append(item)

    category_summary = {}
    for category, items in sorted(by_category.items()):
        case_counts = [item["dedupe"]["vector_case_count"] for item in items]
        collapse_ratios = [
            item["dedupe"]["output_combo_collapse_ratio"]
            for item in items
            if item["dedupe"]["output_combo_collapse_ratio"] is not None
        ]
        category_summary[category] = {
            "sketch_count": len(items),
            "case_count": _summarize_numeric(case_counts),
            "collapse_ratio": _summarize_numeric(collapse_ratios),
            "collapsed_sketch_count": sum(
                1
                for item in items
                if item["dedupe"]["unique_output_combo_count"] < item["dedupe"]["vector_case_count"]
            ),
        }

    collapse_ratios = [
        item["dedupe"]["output_combo_collapse_ratio"]
        for item in with_cases
        if item["dedupe"]["output_combo_collapse_ratio"] is not None
    ]
    duplicate_fractions = [
        item["dedupe"]["output_combo_duplicate_fraction"]
        for item in with_cases
        if item["dedupe"]["output_combo_duplicate_fraction"] is not None
    ]
    generator_load_or_build_ms = [item["timings_ms"]["generator_load_or_build_ms"] for item in results]
    ensure_exact_ms = [item["timings_ms"]["ensure_exact_gpu_ms"] for item in results]
    cache_actions = Counter(item["cache"]["generator_cache_action"] for item in results)

    top_collapsed = sorted(
        results,
        key=lambda item: (
            item["dedupe"]["output_combo_collapse_ratio"] or 0.0,
            item["dedupe"]["vector_case_count"],
        ),
        reverse=True,
    )[:8]

    return {
        "processed_sketch_count": len(results),
        "multi_case_sketch_count": len(with_cases),
        "collapsed_sketch_count": len(any_collapse),
        "shared_constant_sketch_count": len(shared_constant),
        "vthread_constant_sketch_count": len(vthread_constant),
        "vector_constant_sketch_count": len(vector_constant),
        "collapse_ratio_summary": _summarize_numeric(collapse_ratios),
        "duplicate_fraction_summary": _summarize_numeric(duplicate_fractions),
        "generator_load_or_build_ms_summary": _summarize_numeric(generator_load_or_build_ms),
        "ensure_exact_gpu_ms_summary": _summarize_numeric(ensure_exact_ms),
        "cache_action_counts": dict(cache_actions),
        "category_summary": category_summary,
        "top_collapsed_sketches": [
            {
                "index": item["index"],
                "task_desc": item["task_desc"],
                "categories": item["categories"],
                "vector_case_count": item["dedupe"]["vector_case_count"],
                "unique_output_combo_count": item["dedupe"]["unique_output_combo_count"],
                "collapse_ratio": item["dedupe"]["output_combo_collapse_ratio"],
                "duplicate_fraction": item["dedupe"]["output_combo_duplicate_fraction"],
            }
            for item in top_collapsed
        ],
    }


def _render_report(payload):
    lines = []
    lines.append("# Exact Case Dedupe Generalization")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- processed sketches: {payload['summary']['processed_sketch_count']}")
    lines.append(f"- multi-case sketches: {payload['summary']['multi_case_sketch_count']}")
    lines.append(f"- collapsed sketches: {payload['summary']['collapsed_sketch_count']}")
    lines.append(
        f"- shared constant sketches: {payload['summary']['shared_constant_sketch_count']}"
    )
    lines.append(
        f"- vthread constant sketches: {payload['summary']['vthread_constant_sketch_count']}"
    )
    lines.append(
        f"- vector constant sketches: {payload['summary']['vector_constant_sketch_count']}"
    )
    lines.append("")
    lines.append("## Aggregate")
    for key in (
        "collapse_ratio_summary",
        "duplicate_fraction_summary",
        "generator_load_or_build_ms_summary",
        "ensure_exact_gpu_ms_summary",
        "cache_action_counts",
    ):
        value = payload["summary"].get(key)
        if value is None:
            continue
        lines.append(f"- {key}: {json.dumps(value, ensure_ascii=False)}")
    lines.append("")
    lines.append("## Top Collapsed Sketches")
    for item in payload["summary"]["top_collapsed_sketches"]:
        lines.append(
            "- "
            f"index={item['index']} "
            f"cases={item['vector_case_count']} "
            f"unique_combos={item['unique_output_combo_count']} "
            f"collapse_ratio={item['collapse_ratio']} "
            f"categories={','.join(item['categories'])} "
            f"task={item['task_desc']}"
        )
    lines.append("")
    lines.append("## Per Sketch")
    for item in payload["results"]:
        dedupe = item["dedupe"]
        lines.append(
            "- "
            f"index={item['index']} "
            f"categories={','.join(item['categories'])} "
            f"cases={dedupe['vector_case_count']} "
            f"unique_combos={dedupe['unique_output_combo_count']} "
            f"collapse_ratio={dedupe['output_combo_collapse_ratio']} "
            f"shared_unique={dedupe['unique_shared_expr_count']} "
            f"vthread_unique={dedupe['unique_vthread_expr_count']} "
            f"vector_unique={dedupe['unique_vector_expr_count']} "
            f"ensure_exact_gpu_ms={item['timings_ms']['ensure_exact_gpu_ms']} "
            f"task={item['task_desc']}"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-info-dir", default="/root/work/tvm-ansor/gallery/dataset/network_info")
    parser.add_argument("--sketches-path", default="/root/work/tvm-ansor/gallery/dataset/to_measure_programs/all_sketches.json")
    parser.add_argument("--representatives-json", default=DEFAULT_REPRESENTATIVES_JSON)
    parser.add_argument("--indices")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.time()

    print("[load tasks]", flush=True)
    _, tasks_by_wkey = load_tasks_by_workload(args.network_info_dir)
    print("[load sketch lines]", flush=True)
    sketch_lines = load_sketch_lines(args.sketches_path)
    targets = _resolve_targets(args, sketch_lines, tasks_by_wkey)
    if args.limit:
        targets = targets[: args.limit]

    print(f"[target count] {len(targets)}", flush=True)
    results = []

    for ordinal, target in enumerate(targets, start=1):
        index = int(target["index"])
        task_desc = target["task_desc"]
        categories = list(target["categories"])
        print(
            f"[{ordinal}/{len(targets)}] sketch={index} categories={','.join(categories)} task={task_desc}",
            flush=True,
        )

        line = sketch_lines[index]
        task, _, _, state = load_sketch_record(line, tasks_by_wkey)

        gen, cache_status, timings = _load_or_build_generator(args, task, state, index)

        started_item = time.perf_counter()
        dedupe = _analyze_exact_case_dedupe(gen)
        timings["dedupe_analysis_ms"] = round((time.perf_counter() - started_item) * 1000.0, 3)

        print(
            "    [done] "
            f"cases={dedupe['vector_case_count']} "
            f"unique_combos={dedupe['unique_output_combo_count']} "
            f"collapse_ratio={dedupe['output_combo_collapse_ratio']} "
            f"shared_unique={dedupe['unique_shared_expr_count']} "
            f"vthread_unique={dedupe['unique_vthread_expr_count']} "
            f"vector_unique={dedupe['unique_vector_expr_count']}",
            flush=True,
        )

        results.append(
            {
                "index": index,
                "task_desc": task.desc,
                "workload_key": task.workload_key,
                "categories": categories,
                "cache": cache_status,
                "timings_ms": timings,
                "dedupe": dedupe,
            }
        )

    payload = {
        "created_at_epoch_sec": round(time.time(), 3),
        "elapsed_sec": round(time.time() - started, 3),
        "args": {
            "network_info_dir": args.network_info_dir,
            "sketches_path": args.sketches_path,
            "representatives_json": args.representatives_json,
            "indices": args.indices,
            "limit": args.limit,
            "cache_dir": args.cache_dir,
            "rebuild_cache": args.rebuild_cache,
        },
        "summary": _aggregate_summary(results),
        "results": results,
    }

    ensure_parent_dir(args.output_path)
    with open(args.output_path, "w") as f:
        json.dump(payload, f, indent=2)

    ensure_parent_dir(args.report_path)
    with open(args.report_path, "w") as f:
        f.write(_render_report(payload))

    print(
        "[complete] "
        f"elapsed_sec={payload['elapsed_sec']} "
        f"output={args.output_path} "
        f"report={args.report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
