"""Re-measure the top-p% (capped at top-k) records from each *.json file in
``--input-dir``. Output measurement files are written to ``--output-dir``,
preserving the original filename. Best-cost selection is done on the *stored*
measurements; the actual re-measurement uses a fresh ProgramMeasurer with
RecordToFile, so the output files have the same auto_scheduler record-line
format as the input files.

Usage:
    python remeasure_topk.py \
        --input-dir  .../t4 \
        --output-dir .../t4_remeasure \
        --top-pct 5 --top-k 10
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Tuple

from tvm import auto_scheduler

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from modules.task_paths import load_and_register_tasks  # noqa: E402


RUN_TIMEOUT = 5
NUMBER = 1
VERBOSE = 1
DEFAULT_NETWORK_INFO_FOLDER = (
    "/workspace/tvm_gits/tvm-ansor/gallery/dataset/network_info_all"
)


def _task_repeat(task) -> int:
    return 3


def _make_measurer(task, log_filename: str):
    builder = auto_scheduler.measure.LocalBuilder()
    runner = auto_scheduler.measure.LocalRunner(
        timeout=RUN_TIMEOUT,
        repeat=_task_repeat(task),
        number=NUMBER,
        enable_cpu_cache_flush=False,
    )
    return auto_scheduler.measure.ProgramMeasurer(
        builder,
        runner,
        [auto_scheduler.RecordToFile(log_filename)],
        verbose=VERBOSE,
    )


def _record_mean_cost(result) -> float:
    """Mean seconds for a successful measurement; ``inf`` for errors / empty."""
    no_error = int(auto_scheduler.measure.MeasureErrorNo.NO_ERROR)
    if int(result.error_no) != no_error:
        return float("inf")
    costs = [float(c) for c in result.costs]
    if not costs:
        return float("inf")
    return float(sum(costs) / len(costs))


def _select_top_records(
    inputs,
    results,
    *,
    top_pct: float | None,
    top_k: int | None,
) -> List[Tuple]:
    """Sort by stored mean cost (ascending) and keep the top selection —
    failures sink to the end so they are never selected unless forced by an
    extreme cap.

    Exactly one of ``top_pct`` / ``top_k`` must be set; the other is ignored.
    """
    n = len(inputs)
    if n == 0:
        return []
    paired = list(zip(inputs, results))
    paired.sort(key=lambda ir: _record_mean_cost(ir[1]))
    if top_pct is not None:
        keep = math.ceil(n * (float(top_pct) / 100.0))
    else:
        keep = int(top_k)
    keep = max(0, min(keep, n))
    return paired[:keep]


def remeasure_file(
    input_path: Path,
    output_dir: Path,
    *,
    top_pct: float | None,
    top_k: int | None,
    tasks_by_workload_key: dict,
) -> bool:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / input_path.name

    try:
        inputs, results = auto_scheduler.RecordReader(str(input_path)).read_lines()
        inputs = list(inputs)
        results = list(results)
    except Exception as err:  # pylint: disable=broad-except
        print(f"[remeasure] fail load {input_path}: {type(err).__name__}: {err}")
        return False

    if not inputs:
        print(f"[remeasure] empty {input_path}")
        return False
    if len(inputs) != len(results):
        print(
            f"[remeasure] mismatched count {input_path}: "
            f"inputs={len(inputs)} results={len(results)}"
        )
        return False

    workload_key = inputs[0].task.workload_key
    for inp in inputs[1:]:
        if inp.task.workload_key != workload_key:
            print(
                f"[remeasure] mixed workload_keys in {input_path.name} — "
                f"skipping"
            )
            return False

    task = tasks_by_workload_key.get(workload_key)
    if task is None:
        # Fall back to the task carried by the record. Workload registration
        # done at startup should make this measurable even without lookup.
        task = inputs[0].task

    selected = _select_top_records(inputs, results, top_pct=top_pct, top_k=top_k)
    if not selected:
        print(f"[remeasure] no records selected {input_path}")
        return False

    measure_inputs = [
        auto_scheduler.MeasureInput(task, inp.state) for inp, _ in selected
    ]

    try:
        measurer = _make_measurer(task, str(output_path))
        empty_policy = auto_scheduler.search_policy.EmptyPolicy(task)
        new_results = measurer.measure(task, empty_policy, measure_inputs)
    except Exception as err:  # pylint: disable=broad-except
        print(f"[remeasure] fail measure {input_path}: {type(err).__name__}: {err}")
        return False

    no_error = int(auto_scheduler.measure.MeasureErrorNo.NO_ERROR)
    errors = sum(int(r.error_no) != no_error for r in new_results)
    print(
        f"[remeasure] done {input_path.name}: "
        f"selected={len(selected)}/{len(inputs)} "
        f"measured={len(new_results)} errors={errors} → {output_path}"
    )
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Re-measure the top records from each json file in --input-dir; "
            "save results to --output-dir with preserved filenames. Exactly "
            "one of --top-pct / --top-k must be given."
        )
    )
    p.add_argument("--input-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    selector = p.add_mutually_exclusive_group(required=True)
    selector.add_argument(
        "--top-pct",
        type=float,
        default=None,
        help="Top percentile to keep per file (e.g. 5 = top 5%%).",
    )
    selector.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of records to keep per file.",
    )
    p.add_argument(
        "--network-info-folder",
        type=str,
        default=DEFAULT_NETWORK_INFO_FOLDER,
        help="Folder containing all_tasks.pkl for workload registration.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.top_pct is not None and args.top_pct <= 0:
        print("[remeasure] --top-pct must be > 0")
        sys.exit(1)
    if args.top_k is not None and args.top_k <= 0:
        print("[remeasure] --top-k must be > 0")
        sys.exit(1)

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"[remeasure] not a directory: {input_dir}")
        sys.exit(1)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"[remeasure] no json files in {input_dir}")
        sys.exit(1)

    selector = (
        f"top_pct={args.top_pct}" if args.top_pct is not None
        else f"top_k={args.top_k}"
    )
    print(
        f"[remeasure] start dir={input_dir} files={len(json_files)} "
        f"{selector} → {args.output_dir}"
    )

    try:
        tasks = load_and_register_tasks(args.network_info_folder)
    except Exception as err:  # pylint: disable=broad-except
        print(
            f"[remeasure] fail to load/register tasks from "
            f"{args.network_info_folder}: {type(err).__name__}: {err}"
        )
        tasks = []

    tasks_by_workload_key = {task.workload_key: task for task in tasks}

    ok = 0
    for idx, path in enumerate(json_files, start=1):
        print(f"[remeasure] [{idx}/{len(json_files)}] {path.name}")
        if remeasure_file(
            path,
            Path(args.output_dir),
            top_pct=args.top_pct,
            top_k=args.top_k,
            tasks_by_workload_key=tasks_by_workload_key,
        ):
            ok += 1
    print(
        f"[remeasure] done files={len(json_files)} "
        f"ok={ok} fail={len(json_files) - ok}"
    )
    if ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
