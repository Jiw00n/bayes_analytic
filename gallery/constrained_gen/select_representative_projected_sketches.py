"""Select representative sketches for projected GPU constraint validation."""

import argparse
import json
import time

from modules.common import TO_MEASURE_PROGRAM_FOLDER
from modules.projected_gpu_validation import (
    build_schedule_generator,
    ensure_parent_dir,
    load_sketch_lines,
    load_sketch_record,
    load_tasks_by_workload,
)


def _build_base_entry(idx, task, gen):
    return {
        "index": idx,
        "task_desc": task.desc,
        "workload_key": task.workload_key,
        "projected_constraints": gen.get_constraints_str(include_vars=True),
        "raw_exact_constraints": gen.get_raw_exact_constraints_str(include_vars=True),
    }


def run(args):
    started = time.time()
    _, tasks_by_wkey = load_tasks_by_workload(args.network_info_dir)
    sketch_lines = load_sketch_lines(args.sketches_path)

    vectorize = []
    shared_memory = []
    max_vthread = []
    scanned = 0

    for idx, line in enumerate(sketch_lines):
        if (
            len(vectorize) >= args.vectorize_quota
            and len(shared_memory) >= args.shared_quota
            and len(max_vthread) >= args.max_vthread_quota
        ):
            break
        if not line:
            continue
        scanned += 1
        if args.print_every and scanned % args.print_every == 0:
            print(
                f"[scan={scanned}] vectorize={len(vectorize)} "
                f"shared_memory={len(shared_memory)} max_vthread={len(max_vthread)}",
                flush=True,
            )

        task, _, _, state = load_sketch_record(line, tasks_by_wkey)
        gen = build_schedule_generator(task, state)

        vector_node = gen.build_vectorize_constraints()["tree"]
        shared_node = gen.build_shared_memory_constraints()["tree"]
        vthread_items = gen.build_max_vthread_constraints()["items"]
        vector_tree = str(vector_node)
        shared_tree = str(shared_node)
        vthread_tree = " | ".join(str(item["sym_extent"]) for item in vthread_items)
        base_entry = None

        def make_entry(category, category_tree):
            nonlocal base_entry
            if base_entry is None:
                base_entry = _build_base_entry(idx, task, gen)
            entry = dict(base_entry)
            entry["category"] = category
            entry["category_tree"] = category_tree
            return entry

        if len(vectorize) < args.vectorize_quota and vector_tree != "0":
            vectorize.append(make_entry("vectorize", vector_tree))
        if len(shared_memory) < args.shared_quota and shared_node.variables():
            shared_memory.append(make_entry("shared_memory", shared_tree))
        if len(max_vthread) < args.max_vthread_quota and vthread_items and vthread_tree not in ("0", "1"):
            max_vthread.append(make_entry("max_vthread", vthread_tree))

    payload = {
        "sketches_path": args.sketches_path,
        "network_info_dir": args.network_info_dir,
        "vectorize_quota": args.vectorize_quota,
        "shared_quota": args.shared_quota,
        "max_vthread_quota": args.max_vthread_quota,
        "scanned_sketches": scanned,
        "selected_counts": {
            "vectorize": len(vectorize),
            "shared_memory": len(shared_memory),
            "max_vthread": len(max_vthread),
        },
        "elapsed_sec": time.time() - started,
        "vectorize": vectorize,
        "shared_memory": shared_memory,
        "max_vthread": max_vthread,
    }

    ensure_parent_dir(args.output_path)
    with open(args.output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(
        "representative_selection_done "
        f"vectorize={len(vectorize)} "
        f"shared_memory={len(shared_memory)} "
        f"max_vthread={len(max_vthread)} "
        f"scanned={scanned} "
        f"elapsed_sec={payload['elapsed_sec']:.2f}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-info-dir", default="/root/work/tvm-ansor/gallery/dataset/network_info")
    parser.add_argument(
        "--sketches-path",
        default=f"{TO_MEASURE_PROGRAM_FOLDER}/all_sketches.json",
    )
    parser.add_argument("--vectorize-quota", type=int, default=12)
    parser.add_argument("--shared-quota", type=int, default=12)
    parser.add_argument("--max-vthread-quota", type=int, default=8)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
