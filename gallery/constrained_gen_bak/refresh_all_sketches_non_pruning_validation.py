import argparse
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from modules.projected_gpu_validation import (
    build_schedule_generator,
    ensure_parent_dir,
    load_sketch_lines,
    load_sketch_record,
    load_tasks_by_workload,
)
from validate_exact_gpu_constraints import extract_params


SECTION_BAR = "=" * 100
_WORKER_TASKS_BY_WKEY = None
_WORKER_SKETCH_LINES = None


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
        default="/tmp/projected_gpu_full_validation/all_sketches_non_pruning_validation.txt",
    )
    parser.add_argument(
        "--summary-json-path",
        default="/tmp/projected_gpu_full_validation/all_sketches_non_pruning_validation_summary.json",
    )
    parser.add_argument(
        "--summary-md-path",
        default="/tmp/projected_gpu_full_validation/all_sketches_non_pruning_validation_summary.md",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--indices", default=None)
    parser.add_argument("--indices-file", default=None)
    parser.add_argument("--include-raw-exact-for-issues", action="store_true")
    parser.add_argument("--skip-concrete-reverify", action="store_true")
    return parser.parse_args()


def _parse_indices_text(text):
    indices = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        indices.append(int(part))
    return indices


def _resolve_indices(args, sketch_lines):
    if args.indices and args.indices_file:
        raise ValueError("Use only one of --indices or --indices-file")

    if args.indices:
        return _parse_indices_text(args.indices)

    if args.indices_file:
        text = Path(args.indices_file).read_text()
        indices = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            indices.extend(_parse_indices_text(line))
        return indices

    indices = list(range(args.start, len(sketch_lines)))
    if args.limit is not None:
        indices = indices[:args.limit]
    return indices


def _normalize_family(task_desc):
    return re.sub(r"_\d+$", "", task_desc)


def _bool_text(value):
    if value is None:
        return "n/a"
    return "true" if value else "false"


def _format_key_values(mapping):
    if not mapping:
        return "(none)"
    return ", ".join(f"{name}={mapping[name]}" for name in sorted(mapping))


def _append_list_block(lines, label, values):
    lines.append(f"{label}:")
    if values is None:
        lines.append("  (skipped)")
        return
    if values:
        for item in values:
            lines.append(f"  - {item}")
    else:
        lines.append("  (none)")


def _issue_kinds(recorded_valid, concrete_ok, exact_violations, hybrid_violations, final_violations):
    issue_kinds = []
    if concrete_ok is not None and concrete_ok != recorded_valid:
        issue_kinds.append("concrete_reverify_mismatch")

    if recorded_valid:
        if exact_violations:
            issue_kinds.append("exact_false_reject")
        if hybrid_violations:
            issue_kinds.append("hybrid_false_reject")
        if final_violations:
            issue_kinds.append("final_false_reject")
    else:
        if not exact_violations:
            issue_kinds.append("exact_false_accept")
        if hybrid_violations is not None and not hybrid_violations:
            issue_kinds.append("hybrid_false_accept")
        if final_violations is not None and not final_violations:
            issue_kinds.append("final_false_accept")
    return issue_kinds


def _status_from_violations(violations):
    if violations is None:
        return None
    return not violations


def _render_section(task, sketch_index, gen, params, base_error_no, concrete_result, exact_violations, hybrid_violations, final_violations, include_raw_exact):
    recorded_valid = base_error_no == 0
    concrete_ok = None if concrete_result is None else bool(concrete_result.get("ok"))
    exact_ok = _status_from_violations(exact_violations)
    hybrid_ok = _status_from_violations(hybrid_violations)
    final_ok = _status_from_violations(final_violations)
    issue_kinds = _issue_kinds(
        recorded_valid,
        concrete_ok,
        exact_violations,
        hybrid_violations,
        final_violations,
    )

    lines = [
        SECTION_BAR,
        f"task_desc: {task.desc}",
        f"task_family: {_normalize_family(task.desc)}",
        f"representative_index: {sketch_index}",
        f"representative_task_desc: {task.desc}",
        f"workload_key: {task.workload_key}",
        "",
        "[Recorded Measurement]",
        f"base_error_no: {base_error_no}",
        f"recorded_valid: {_bool_text(recorded_valid)}",
        "",
        "[Validation Status]",
        f"concrete_reverify_ok: {_bool_text(concrete_ok)}",
        f"exact_ok: {_bool_text(exact_ok)}",
        f"hybrid_ok: {_bool_text(hybrid_ok)}",
        f"final_ok: {_bool_text(final_ok)}",
        "issue_kinds: " + (", ".join(issue_kinds) if issue_kinds else "(none)"),
        "",
        "[Violations]",
    ]
    _append_list_block(lines, "exact", exact_violations)
    _append_list_block(lines, "hybrid", hybrid_violations)
    _append_list_block(lines, "final", final_violations)
    _append_list_block(
        lines,
        "concrete_reverify",
        [] if concrete_result is None else list(concrete_result.get("violations", [])),
    )

    lines.extend(
        [
            "",
            "[Recorded Params]",
            _format_key_values(params),
            "",
            "[Structural Highlights]",
            gen.get_structural_highlights_str(include_vars=True),
            "",
            "[Symbolic State]",
            str(gen.s),
            "",
            "[Projected Constraints]",
            gen.get_constraints_str(include_vars=True),
            "",
            "[Projected Constraints With Recorded Values]",
            gen.get_constraints_with_assignment_str(sym_map=params, include_vars=True),
        ]
    )

    if include_raw_exact and issue_kinds:
        lines.extend(
            [
                "",
                "[Raw Exact Constraints]",
                gen.get_raw_exact_constraints_str(include_vars=True),
            ]
        )

    return "\n".join(lines), issue_kinds


def _build_issue_record(task, sketch_index, params, base_error_no, concrete_result, exact_violations, hybrid_violations, final_violations, issue_kinds):
    if not issue_kinds:
        return None

    return {
        "representative_index": sketch_index,
        "task_desc": task.desc,
        "task_family": _normalize_family(task.desc),
        "workload_key": task.workload_key,
        "base_error_no": int(base_error_no),
        "recorded_valid": int(base_error_no) == 0,
        "concrete_reverify_ok": None if concrete_result is None else bool(concrete_result.get("ok")),
        "issue_kinds": issue_kinds,
        "params": params,
        "exact_violation_count": len(exact_violations),
        "hybrid_violation_count": None if hybrid_violations is None else len(hybrid_violations),
        "final_violation_count": None if final_violations is None else len(final_violations),
        "concrete_reverify_violation_count": 0 if concrete_result is None else len(concrete_result.get("violations", [])),
        "exact_violations": exact_violations,
        "hybrid_violations": hybrid_violations,
        "final_violations": final_violations,
        "concrete_reverify_violations": [] if concrete_result is None else list(concrete_result.get("violations", [])),
    }


def _init_worker(network_info_dir, sketches_path):
    global _WORKER_TASKS_BY_WKEY, _WORKER_SKETCH_LINES
    _, tasks_by_wkey = load_tasks_by_workload(network_info_dir)
    _WORKER_TASKS_BY_WKEY = tasks_by_wkey
    _WORKER_SKETCH_LINES = load_sketch_lines(sketches_path)


def _process_sketch(sketch_index, include_raw_exact_for_issues, skip_concrete_reverify):
    if _WORKER_TASKS_BY_WKEY is None or _WORKER_SKETCH_LINES is None:
        raise RuntimeError("Worker is not initialized")

    line = _WORKER_SKETCH_LINES[sketch_index]
    if not line:
        raise ValueError(f"Sketch index {sketch_index} points to an empty line")

    task, base_inp, base_res, state = load_sketch_record(line, _WORKER_TASKS_BY_WKEY)
    if skip_concrete_reverify:
        gen = build_schedule_generator(task, state)
    else:
        gen = build_schedule_generator(task, state, base_inp=base_inp, base_res=base_res)
    params = extract_params(state)

    exact_violations = gen.check_all_exact(params)
    if skip_concrete_reverify:
        final_violations = None
        hybrid_violations = None
        concrete_result = None
    else:
        final_violations = gen.check_all_final(params)
        concrete_result = gen.get_concrete_final_result(params)
        hybrid_violations = list(final_violations)

    section, issue_kinds = _render_section(
        task,
        sketch_index,
        gen,
        params,
        int(getattr(base_res, "error_no", 1)),
        concrete_result,
        exact_violations,
        hybrid_violations,
        final_violations,
        include_raw_exact_for_issues,
    )
    issue_record = _build_issue_record(
        task,
        sketch_index,
        params,
        int(getattr(base_res, "error_no", 1)),
        concrete_result,
        exact_violations,
        hybrid_violations,
        final_violations,
        issue_kinds,
    )
    return sketch_index, section, issue_record


def _summarize_issues(sketches_path, issue_records, total_sections, start, limit, skip_concrete_reverify):
    issue_records = [item for item in issue_records if item is not None]
    issue_records.sort(key=lambda item: item["representative_index"])

    issue_kind_counts = Counter()
    recorded_valid_sections = 0
    recorded_invalid_sections = 0
    for item in issue_records:
        issue_kind_counts.update(item["issue_kinds"])
    for item in issue_records:
        if item["recorded_valid"]:
            recorded_valid_sections += 1
        else:
            recorded_invalid_sections += 1

    return {
        "source_path": sketches_path,
        "start": start,
        "limit": limit,
        "skip_concrete_reverify": skip_concrete_reverify,
        "section_count": total_sections,
        "recorded_valid_issue_sections": recorded_valid_sections,
        "recorded_invalid_issue_sections": recorded_invalid_sections,
        "tasks_with_issues": len(issue_records),
        "fully_clean_sections": total_sections - len(issue_records),
        "issue_kind_counts": dict(sorted(issue_kind_counts.items())),
        "exact_false_reject_sections": sum(
            1 for item in issue_records if "exact_false_reject" in item["issue_kinds"]
        ),
        "hybrid_false_reject_sections": sum(
            1 for item in issue_records if "hybrid_false_reject" in item["issue_kinds"]
        ),
        "final_false_reject_sections": sum(
            1 for item in issue_records if "final_false_reject" in item["issue_kinds"]
        ),
        "concrete_reverify_mismatch_sections": sum(
            1 for item in issue_records if "concrete_reverify_mismatch" in item["issue_kinds"]
        ),
        "tasks": issue_records,
    }


def _write_summary_files(summary, json_path, md_path):
    ensure_parent_dir(json_path)
    ensure_parent_dir(md_path)
    Path(json_path).write_text(json.dumps(summary, indent=2) + "\n")

    md_lines = [
        "# Non-Pruning Validation Summary",
        "",
        f"- Source: `{summary['source_path']}`",
        f"- Skip concrete reverify: {summary['skip_concrete_reverify']}",
        f"- Sections: {summary['section_count']}",
        f"- Tasks with issues: {summary['tasks_with_issues']}",
        f"- Fully clean sections: {summary['fully_clean_sections']}",
        f"- Exact false reject sections: {summary['exact_false_reject_sections']}",
        f"- Hybrid false reject sections: {summary['hybrid_false_reject_sections']}",
        f"- Final false reject sections: {summary['final_false_reject_sections']}",
        f"- Concrete reverify mismatch sections: {summary['concrete_reverify_mismatch_sections']}",
        "",
        "## Issue Kind Counts",
        "",
    ]

    if summary["issue_kind_counts"]:
        for kind, count in summary["issue_kind_counts"].items():
            md_lines.append(f"- `{kind}`: {count}")
    else:
        md_lines.append("- (none)")

    md_lines.extend(["", "## Tasks", ""])
    if summary["tasks"]:
        for item in summary["tasks"]:
            md_lines.append(
                f"- `index={item['representative_index']}` `{item['task_desc']}` "
                f"issues={item['issue_kinds']} "
                f"(exact={item['exact_violation_count']}, "
                f"hybrid={item['hybrid_violation_count']}, "
                f"final={item['final_violation_count']}, "
                f"concrete={item['concrete_reverify_violation_count']})"
            )
    else:
        md_lines.append("- (none)")

    Path(md_path).write_text("\n".join(md_lines) + "\n")


def main():
    args = parse_args()
    sketch_lines = load_sketch_lines(args.sketches_path)
    indices = _resolve_indices(args, sketch_lines)

    sections_by_index = {}
    issue_records = []

    if args.workers == 1:
        _init_worker(args.network_info_dir, args.sketches_path)
        for ordinal, sketch_index in enumerate(indices, start=1):
            _, section, issue_record = _process_sketch(
                sketch_index,
                args.include_raw_exact_for_issues,
                args.skip_concrete_reverify,
            )
            sections_by_index[sketch_index] = section
            if issue_record is not None:
                issue_records.append(issue_record)
            print(f"[{ordinal}/{len(indices)}] audited sketch_index={sketch_index}", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(args.network_info_dir, args.sketches_path),
        ) as executor:
            futures = {
                executor.submit(
                    _process_sketch,
                    sketch_index,
                    args.include_raw_exact_for_issues,
                    args.skip_concrete_reverify,
                ): sketch_index
                for sketch_index in indices
            }
            for ordinal, future in enumerate(as_completed(futures), start=1):
                sketch_index, section, issue_record = future.result()
                sections_by_index[sketch_index] = section
                if issue_record is not None:
                    issue_records.append(issue_record)
                print(f"[{ordinal}/{len(indices)}] audited sketch_index={sketch_index}", flush=True)

    sections = [sections_by_index[idx] for idx in indices]
    ensure_parent_dir(args.output_path)
    Path(args.output_path).write_text("\n".join(sections) + "\n")

    summary = _summarize_issues(
        args.sketches_path,
        issue_records,
        len(indices),
        args.start,
        args.limit,
        args.skip_concrete_reverify,
    )
    _write_summary_files(summary, args.summary_json_path, args.summary_md_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
