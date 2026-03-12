import argparse
import json
import re
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from dump_symbolic_state_projected_constraints import run as run_dump
from modules.projected_gpu_validation import (
    build_schedule_generator,
    load_sketch_lines,
    load_sketch_record,
    load_tasks_by_workload,
)


SECTION_BAR = "=" * 100
_SUMMARY_TASKS_BY_WKEY = None
_SUMMARY_SKETCH_LINES = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sketches-path",
        default="/root/work/tvm-ansor/gallery/dataset/to_measure_programs/all_sketches.json",
    )
    parser.add_argument(
        "--network-info-dir",
        default="/root/work/tvm-ansor/gallery/dataset/network_info",
    )
    parser.add_argument(
        "--output-path",
        default="/tmp/projected_gpu_full_validation/all_sketches_prefix_through_non_product_gate_vars.txt",
    )
    parser.add_argument(
        "--summary-json-path",
        default="/tmp/projected_gpu_full_validation/all_sketches_prefix_through_non_product_gate_vars_leftover_summary.json",
    )
    parser.add_argument(
        "--summary-md-path",
        default="/tmp/projected_gpu_full_validation/all_sketches_prefix_through_non_product_gate_vars_leftover_summary.md",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--generation-max-retries", type=int, default=64)
    parser.add_argument("--generation-stop-after-phase", default="non_product_gate_vars")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def build_dump(args):
    sketch_lines = load_sketch_lines(args.sketches_path)
    run_dump(
        SimpleNamespace(
            indices=",".join(str(i) for i in range(len(sketch_lines))),
            output_path=args.output_path,
            group_by="none",
            include_raw_exact=False,
            generation_stop_after_phase=args.generation_stop_after_phase,
            generation_max_retries=args.generation_max_retries,
            seed_base=args.seed_base,
            workers=args.workers,
            network_info_dir=args.network_info_dir,
            sketches_path=args.sketches_path,
        )
    )


def _normalize_family(task_desc):
    return re.sub(r"_\d+$", "", task_desc)


def _format_simplified_expr(expr):
    text = str(expr)
    text = text.replace("ceiling(", "ceil(")
    text = text.replace("floor(", "floor(")
    text = text.replace("Min(", "min(")
    text = text.replace("Max(", "max(")
    text = text.replace("Mod(", "mod(")
    return text


@lru_cache(maxsize=32768)
def _simplify_math_text(text):
    normalized = " ".join(str(text).split())
    if not normalized:
        return normalized

    try:
        from sympy import Max, Min, Mod, ceiling, floor
        from sympy.parsing.sympy_parser import parse_expr
    except Exception:
        return normalized

    try:
        expr = parse_expr(
            normalized,
            local_dict={
                "ceil": ceiling,
                "floor": floor,
                "min": Min,
                "max": Max,
                "mod": Mod,
            },
            evaluate=True,
        )
    except Exception:
        return normalized

    return _format_simplified_expr(expr)


def _simplify_constraint_text(text):
    kind, sep, rest = text.partition(":")
    if not sep:
        return _simplify_math_text(text)

    rest = rest.strip()
    if " <= " not in rest:
        return f"{kind}:{' ' if rest else ''}{_simplify_math_text(rest)}".rstrip()

    lhs, rhs = rest.rsplit(" <= ", 1)
    lhs = _simplify_math_text(lhs)
    rhs = _simplify_math_text(rhs)
    return f"{kind}: {lhs} <= {rhs}"


def _init_summary_worker(network_info_dir, sketches_path):
    global _SUMMARY_TASKS_BY_WKEY, _SUMMARY_SKETCH_LINES
    _, tasks_by_wkey = load_tasks_by_workload(network_info_dir)
    _SUMMARY_TASKS_BY_WKEY = tasks_by_wkey
    _SUMMARY_SKETCH_LINES = load_sketch_lines(sketches_path)


def _simplify_constraint_items(items):
    simplified = []
    for item in items:
        simplified.append(
            {
                "kind": item["kind"],
                "constraint": _simplify_constraint_text(item["constraint"]),
                **({"vars": item["vars"]} if item.get("vars") else {}),
                **({"domains": item["domains"]} if item.get("domains") else {}),
            }
        )
    return simplified


def _summarize_task(sketch_index, seed_base, generation_max_retries, generation_stop_after_phase):
    if _SUMMARY_TASKS_BY_WKEY is None or _SUMMARY_SKETCH_LINES is None:
        raise RuntimeError("Summary worker is not initialized")

    line = _SUMMARY_SKETCH_LINES[sketch_index]
    task, _, _, state = load_sketch_record(line, _SUMMARY_TASKS_BY_WKEY)
    gen = build_schedule_generator(task, state)
    snapshot = gen.randomize_params_prefix(
        generation_stop_after_phase,
        rng=random.Random(seed_base + sketch_index),
        max_retries=generation_max_retries,
    )

    leftover_constraints = _simplify_constraint_items(snapshot["leftover_constraints"])
    resolved_false_constraints = _simplify_constraint_items(
        snapshot["resolved_false_constraints"]
    )

    return {
        "representative_index": sketch_index,
        "task_desc": task.desc,
        "task_family": _normalize_family(task.desc),
        "workload_key": task.workload_key,
        "leftover_constraint_count": len(leftover_constraints),
        "resolved_false_constraint_count": len(resolved_false_constraints),
        "leftover_constraints": leftover_constraints,
        "resolved_false_constraints": resolved_false_constraints,
        "propagated_singleton_count": len(snapshot.get("fixed_values", {})) - len(snapshot.get("params", {})),
        "remaining_domain_count": len(snapshot.get("remaining_domains", {})),
    }


def summarize_leftovers(args):
    sketch_lines = load_sketch_lines(args.sketches_path)
    tasks = []

    if args.workers == 1:
        _init_summary_worker(args.network_info_dir, args.sketches_path)
        for sketch_index in range(len(sketch_lines)):
            tasks.append(
                _summarize_task(
                    sketch_index,
                    args.seed_base,
                    args.generation_max_retries,
                    args.generation_stop_after_phase,
                )
            )
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_summary_worker,
            initargs=(args.network_info_dir, args.sketches_path),
        ) as executor:
            futures = {
                executor.submit(
                    _summarize_task,
                    sketch_index,
                    args.seed_base,
                    args.generation_max_retries,
                    args.generation_stop_after_phase,
                ): sketch_index
                for sketch_index in range(len(sketch_lines))
            }
            for future in as_completed(futures):
                tasks.append(future.result())

    tasks.sort(key=lambda item: item["representative_index"])
    tasks_with_findings = [
        item for item in tasks
        if item["leftover_constraint_count"] or item["resolved_false_constraint_count"]
    ]

    return {
        "source_path": args.sketches_path,
        "generation_stop_after_phase": args.generation_stop_after_phase,
        "section_count": len(sketch_lines),
        "tasks_with_leftovers": sum(1 for item in tasks if item["leftover_constraint_count"]),
        "tasks_with_resolved_false": sum(1 for item in tasks if item["resolved_false_constraint_count"]),
        "tasks_with_findings": len(tasks_with_findings),
        "fully_covered_sections": len(sketch_lines) - len(tasks_with_findings),
        "leftover_constraint_occurrences": sum(item["leftover_constraint_count"] for item in tasks),
        "resolved_false_constraint_occurrences": sum(
            item["resolved_false_constraint_count"] for item in tasks
        ),
        "tasks": tasks_with_findings,
    }


def write_summary_files(summary, json_path, md_path):
    Path(json_path).write_text(json.dumps(summary, indent=2) + "\n")

    md_lines = [
        "# Constraint Summary",
        "",
        f"- Source: `{summary['source_path']}`",
        f"- Sections: {summary['section_count']}",
        f"- Tasks with leftovers: {summary['tasks_with_leftovers']}",
        f"- Tasks with resolved_false: {summary['tasks_with_resolved_false']}",
        f"- Tasks with findings: {summary['tasks_with_findings']}",
        f"- Fully covered sections: {summary['fully_covered_sections']}",
        f"- Leftover constraint occurrences: {summary['leftover_constraint_occurrences']}",
        f"- Resolved_false constraint occurrences: {summary['resolved_false_constraint_occurrences']}",
        "",
        "## Tasks",
        "",
    ]
    for item in summary["tasks"]:
        md_lines.append(
            f"- `index={item['representative_index']}` `{item['task_desc']}` "
            f"(leftover={item['leftover_constraint_count']}, "
            f"resolved_false={item['resolved_false_constraint_count']})"
        )
        for leftover in item["leftover_constraints"]:
            md_lines.append(f"  - `{leftover['constraint']}`")
        for resolved_false in item["resolved_false_constraints"]:
            md_lines.append(f"  - `[resolved_false] {resolved_false['constraint']}`")

    Path(md_path).write_text("\n".join(md_lines) + "\n")


def main():
    args = parse_args()
    if not args.summary_only:
        build_dump(args)
    summary = summarize_leftovers(args)
    write_summary_files(summary, args.summary_json_path, args.summary_md_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
