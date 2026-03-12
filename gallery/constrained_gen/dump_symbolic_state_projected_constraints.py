"""Dump symbolic state and projected constraints for representative sketch indices."""

import argparse
import random
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

from modules.projected_gpu_validation import (
    build_schedule_generator,
    ensure_parent_dir,
    load_sketch_lines,
    load_sketch_record,
    load_tasks_by_workload,
)


SECTION_BAR = "=" * 100
_WORKER_TASKS_BY_WKEY = None
_WORKER_SKETCH_LINES = None


def _parse_indices(text):
    indices = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        indices.append(int(part))
    if not indices:
        raise ValueError("Expected at least one sketch index")
    return indices


def _normalize_family(task_desc):
    return re.sub(r"_\d+$", "", task_desc)


def _build_family_meta(tasks):
    family_map = {}
    for task in tasks:
        family = _normalize_family(task.desc)
        family_map.setdefault(family, set()).add(task.desc)
    return {
        family: {
            "variant_count": len(task_descs),
            "variant_task_descs": sorted(task_descs),
        }
        for family, task_descs in family_map.items()
    }


def _build_section_header(task, sketch_index, group_by, family_meta):
    lines = [SECTION_BAR]
    if group_by == "family":
        family = _normalize_family(task.desc)
        meta = family_meta[family]
        lines.append(f"task_family: {family}")
        lines.append(f"variant_count: {meta['variant_count']}")
        lines.append(f"variant_task_descs: {meta['variant_task_descs']}")
    else:
        lines.append(f"task_desc: {task.desc}")
        lines.append(f"task_family: {_normalize_family(task.desc)}")
    lines.append(f"representative_index: {sketch_index}")
    lines.append(f"representative_task_desc: {task.desc}")
    lines.append(f"workload_key: {task.workload_key}")
    return lines


def _render_section(task, sketch_index, gen, group_by, family_meta, include_raw_exact):
    lines = _build_section_header(task, sketch_index, group_by, family_meta)
    prefix_snapshot = None
    if getattr(gen, "_dump_generation_prefix_snapshot", None) is not None:
        prefix_snapshot = gen._dump_generation_prefix_snapshot
        lines.append("")
        lines.append("[Generation Prefix]")
        lines.append(f"stop_after_phase: {prefix_snapshot['stop_after_phase']}")
        if prefix_snapshot.get("resolved_stop_phase_name") is not None:
            lines.append(f"resolved_stop_phase_name: {prefix_snapshot['resolved_stop_phase_name']}")
        if prefix_snapshot.get("resolved_stop_phase_family") is not None:
            lines.append(f"resolved_stop_phase_family: {prefix_snapshot['resolved_stop_phase_family']}")
        lines.append("phase_order:")
        for phase in prefix_snapshot["phases"]:
            vars_text = ", ".join(phase["vars"]) if phase["vars"] else "(none)"
            scope_label = phase.get("grid_scope_label", "(no-scope)")
            lines.append(
                f"  {phase['name']} [{phase.get('family', 'unknown')}] "
                f"@ {scope_label}: {vars_text}"
            )
        var_order_text = ", ".join(prefix_snapshot["var_order"]) if prefix_snapshot["var_order"] else "(none)"
        lines.append(f"generated_var_order: {var_order_text}")
        ordered_params = [
            f"{name}={prefix_snapshot['params'][name]}"
            for name in prefix_snapshot["var_order"]
            if name in prefix_snapshot["params"]
        ]
        lines.append(
            "generated_params: "
            + (", ".join(ordered_params) if ordered_params else "(none)")
        )
        fixed_values = [
            f"{name}={prefix_snapshot['fixed_values'][name]}"
            for name in sorted(prefix_snapshot.get("fixed_values", {}).keys())
            if name not in prefix_snapshot["params"]
        ]
        lines.append(
            "propagated_singleton_params: "
            + (", ".join(fixed_values) if fixed_values else "(none)")
        )
        remaining_domains = [
            f"{name}=[{bounds[0]},{bounds[1]}]"
            for name, bounds in sorted(prefix_snapshot.get("remaining_domains", {}).items())
        ]
        lines.append(
            "remaining_domains: "
            + (", ".join(remaining_domains) if remaining_domains else "(none)")
        )
        lines.append(
            "constraint_status: "
            f"leftover={len(prefix_snapshot.get('leftover_constraints', []))}, "
            f"resolved_false={len(prefix_snapshot.get('resolved_false_constraints', []))}, "
            f"resolved_true={prefix_snapshot.get('resolved_true_count', 0)}"
        )
    lines.append("")
    lines.append("[Structural Highlights]")
    lines.append(gen.get_structural_highlights_str(include_vars=True))
    lines.append("")
    lines.append("[Symbolic State]")
    lines.append(str(gen.s))
    lines.append("")
    lines.append("[Projected Constraints]")
    lines.append(gen.get_constraints_str(include_vars=True))
    if prefix_snapshot is not None:
        lines.append("")
        lines.append("[Projected Constraints With Generated Values]")
        lines.append(gen.get_constraints_with_assignment_str(include_vars=True))
    if include_raw_exact:
        lines.append("")
        lines.append("[Raw Exact Constraints]")
        lines.append(gen.get_raw_exact_constraints_str(include_vars=True))
    return "\n".join(lines)


def _init_worker(network_info_dir, sketches_path):
    global _WORKER_TASKS_BY_WKEY, _WORKER_SKETCH_LINES

    _, tasks_by_wkey = load_tasks_by_workload(network_info_dir)
    _WORKER_TASKS_BY_WKEY = tasks_by_wkey
    _WORKER_SKETCH_LINES = load_sketch_lines(sketches_path)


def _render_section_worker(
    sketch_index,
    group_by,
    family_meta,
    include_raw_exact,
    generation_stop_after_phase,
    generation_max_retries,
    seed_base,
):
    if _WORKER_TASKS_BY_WKEY is None or _WORKER_SKETCH_LINES is None:
        raise RuntimeError("Worker is not initialized")

    if sketch_index >= len(_WORKER_SKETCH_LINES):
        raise IndexError(f"Sketch index {sketch_index} is out of range")
    line = _WORKER_SKETCH_LINES[sketch_index]
    if not line:
        raise ValueError(f"Sketch index {sketch_index} points to an empty line")

    task, _, _, state = load_sketch_record(line, _WORKER_TASKS_BY_WKEY)
    gen = build_schedule_generator(task, state)
    if generation_stop_after_phase:
        rng = random.Random(seed_base + sketch_index)
        gen._dump_generation_prefix_snapshot = gen.randomize_params_prefix(  # pylint: disable=protected-access
            generation_stop_after_phase,
            rng=rng,
            max_retries=generation_max_retries,
        )
    return sketch_index, _render_section(
        task,
        sketch_index,
        gen,
        group_by,
        family_meta,
        include_raw_exact,
    )


def run(args):
    indices = _parse_indices(args.indices)
    tasks, tasks_by_wkey = load_tasks_by_workload(args.network_info_dir)
    family_meta = _build_family_meta(tasks)
    sketch_lines = load_sketch_lines(args.sketches_path)

    sections_by_index = {}
    if args.workers == 1:
        for ordinal, sketch_index in enumerate(indices, start=1):
            if sketch_index >= len(sketch_lines):
                raise IndexError(f"Sketch index {sketch_index} is out of range")
            line = sketch_lines[sketch_index]
            if not line:
                raise ValueError(f"Sketch index {sketch_index} points to an empty line")
            task, _, _, state = load_sketch_record(line, tasks_by_wkey)
            gen = build_schedule_generator(task, state)
            if args.generation_stop_after_phase:
                rng = random.Random(args.seed_base + sketch_index)
                gen._dump_generation_prefix_snapshot = gen.randomize_params_prefix(  # pylint: disable=protected-access
                    args.generation_stop_after_phase,
                    rng=rng,
                    max_retries=args.generation_max_retries,
                )
            sections_by_index[sketch_index] = _render_section(
                task,
                sketch_index,
                gen,
                args.group_by,
                family_meta,
                args.include_raw_exact,
            )
            print(
                f"[{ordinal}/{len(indices)}] rendered sketch_index={sketch_index}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(args.network_info_dir, args.sketches_path),
        ) as executor:
            futures = {
                executor.submit(
                    _render_section_worker,
                    sketch_index,
                    args.group_by,
                    family_meta,
                    args.include_raw_exact,
                    args.generation_stop_after_phase,
                    args.generation_max_retries,
                    args.seed_base,
                ): sketch_index
                for sketch_index in indices
            }
            for ordinal, future in enumerate(as_completed(futures), start=1):
                sketch_index, section = future.result()
                sections_by_index[sketch_index] = section
                print(
                    f"[{ordinal}/{len(indices)}] rendered sketch_index={sketch_index}",
                    flush=True,
                )

    sections = [sections_by_index[sketch_index] for sketch_index in indices]

    ensure_parent_dir(args.output_path)
    with open(args.output_path, "w") as f:
        f.write("\n".join(sections) + "\n")

    print(
        f"dump_done sections={len(sections)} group_by={args.group_by} output={args.output_path}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", required=True, help="Comma-separated sketch indices")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--group-by", choices=("none", "family"), default="none")
    parser.add_argument("--include-raw-exact", action="store_true")
    parser.add_argument(
        "--generation-stop-after-phase",
        choices=(
            "pure_product_max_threads",
            "pure_product_max_vthread",
            "split_structure_max_threads",
            "split_structure_max_vthread",
            "scaled_product_upper_bound",
            "non_product_direct_arm",
            "non_product_gate_vars",
        ),
        default=None,
    )
    parser.add_argument("--generation-max-retries", type=int, default=64)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--network-info-dir", default="/root/work/tvm-ansor/gallery/dataset/network_info")
    parser.add_argument(
        "--sketches-path",
        default="/root/work/tvm-ansor/gallery/dataset/to_measure_programs/all_sketches.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
