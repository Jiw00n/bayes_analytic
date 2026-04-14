"""Generate one valid constrained record per selected task.

This is the first-pass research entrypoint for the cleaned-up constrained-gen
workflow. It uses the active concrete-sketch -> symbolic generator ->
params-to-state -> measure-record path without introducing a second workflow.
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
from collections import Counter
from contextlib import nullcontext

from tvm import auto_scheduler
from tvm.auto_scheduler import SketchPolicy
from tqdm import tqdm

sys.path.append("/root/work/tvm-ansor/gallery/constrained_gen")

from modules.task_paths import get_to_measure_gen_filename, load_and_register_tasks
from modules.schedule_generator import ScheduleGenerator


TASK_NETWORK_INFO_FOLDER = "/root/work/tvm-ansor/gallery/dataset/network_info_all"
_WORKER_TASKS = None
ENABLED_CONSTRAINTS_NO_VECTORIZE = (
    'shared_memory',
    'max_threads',
    'max_vthread',
    'innermost_split',
    'split_structure',
    # 'max_threads_per_block',
    # 'max_vector_bytes',
)



def _task_prefix(task_index, task):
    return f"[task {task_index}] {task.desc}"


def _short_output_path(output_path):
    try:
        return os.path.relpath(output_path, start=os.getcwd())
    except ValueError:
        return output_path


def _emit_task_log(task_index, task, message):
    print(f"{_task_prefix(task_index, task)} {message}")


def _maybe_emit_task_log(task_index, task, emit_logs, message):
    if emit_logs:
        _emit_task_log(task_index, task, message)


def _emit_failure(task_index, task, stage, sketch_index, err):
    """실패 정보를 사람이 읽기 쉬운 한 줄 로그로 출력한다."""
    error_text = f"{type(err).__name__}: {err}" if isinstance(err, Exception) else str(err)
    sketch_text = "-" if sketch_index is None else str(sketch_index)
    _emit_task_log(
        task_index,
        task,
        f"FAIL stage={stage} sketch={sketch_text} error={error_text}",
    )


def generate_concrete_sketches(task):
    """SketchPolicy로 task에 대한 구체 스케치(State) 목록을 생성해 반환한다."""
    policy = SketchPolicy(
        task,
        params={"sample_init_no_invalid": 1},
        verbose=False,
    )
    return list(policy.generate_concrete_sketches())


def build_measure_record(task, state):
    """Concrete State를 MeasureInput/MeasureResult 쌍으로 만든다."""
    # breakpoint()
    measure_input = auto_scheduler.MeasureInput(task, state)
    measure_result = auto_scheduler.MeasureResult([0.0], 0, "", 0, time.time())
    return measure_input, measure_result


def save_records_batch(output_path, measure_inputs, measure_results):
    """MeasureInput/MeasureResult 목록을 한 파일에 저장한다."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    auto_scheduler.save_records(output_path, measure_inputs, measure_results)


def dedupe_measure_records(task, measure_inputs, measure_results):
    """MeasureInput/MeasureResult 목록을 concrete-state fingerprint 기준으로 최종 dedupe한다."""
    from modules.concrete_gpu_verify import concrete_state_fingerprint

    seen_fingerprints = set()
    deduped_inputs = []
    deduped_results = []
    dropped = 0

    for measure_input, measure_result in zip(measure_inputs, measure_results):
        fingerprint = concrete_state_fingerprint(task, measure_input.state)
        if fingerprint in seen_fingerprints:
            dropped += 1
            continue
        seen_fingerprints.add(fingerprint)
        deduped_inputs.append(measure_input)
        deduped_results.append(measure_result)

    return deduped_inputs, deduped_results, dropped


def load_existing_output_fingerprints(task, output_path):
    """기존 output file에 저장된 concrete state fingerprint 집합을 반환한다."""
    from modules.concrete_gpu_verify import concrete_state_fingerprint

    if not os.path.exists(output_path):
        return set()

    existing_inputs, _ = auto_scheduler.RecordReader(output_path).read_lines()
    return {
        concrete_state_fingerprint(task, measure_input.state)
        for measure_input in existing_inputs
    }


def process_task(task_index, task, records_per_task=1, emit_logs=True, show_record_bar=False):
    """한 task에 대해 unique concrete schedule을 records_per_task개까지 모아 저장한다."""
    output_path = get_to_measure_gen_filename(task, task_index=task_index, output_dir=TO_MEASURE_GEN_PROGRAM_FOLDER)
    output_label = _short_output_path(output_path)
    target_count = max(1, int(records_per_task))

    if os.path.exists(output_path):
        _maybe_emit_task_log(
            task_index,
            task,
            emit_logs,
            f"SKIP existing_output={output_label}",
        )
        return {
            "ok": True,
            "stage": "skipped_existing_output",
            "task_index": task_index,
            "workload_key": task.workload_key,
            "output_path": output_path,
            "record_count": 0,
            "duplicates_skipped": 0,
            "sketches_exhausted": 0,
            "search_exhausted": False,
            "final_dedup_dropped": 0,
            "skipped_existing_output": True,
        }

    try:
        sketches = generate_concrete_sketches(task)
    except Exception as err:  # pylint: disable=broad-except
        _emit_failure(task_index, task, "generate_concrete_sketches", None, err)
        return {
            "ok": False,
            "stage": "generate_concrete_sketches",
            "task_index": task_index,
            "workload_key": task.workload_key,
        }

    if not sketches:
        _emit_failure(task_index, task, "zero_sketches", None, "no concrete sketches")
        return {
            "ok": False,
            "stage": "zero_sketches",
            "task_index": task_index,
            "workload_key": task.workload_key,
        }

    try:
        seen_fingerprints = load_existing_output_fingerprints(task, output_path)
    except Exception as err:  # pylint: disable=broad-except
        _emit_failure(task_index, task, "load_existing_output", None, err)
        return {
            "ok": False,
            "stage": "load_existing_output",
            "task_index": task_index,
            "workload_key": task.workload_key,
        }

    existing_seed_count = len(seen_fingerprints)
    _maybe_emit_task_log(
        task_index,
        task,
        emit_logs,
        f"START target={target_count} sketches={len(sketches)} seeded={existing_seed_count}",
    )

    collected_inputs = []
    collected_results = []
    total_duplicates_skipped = 0
    exhausted_sketches = 0
    record_bar_ctx = (
        tqdm(total=target_count, desc=f"task {task_index} records", unit="record", leave=False)
        if show_record_bar
        else nullcontext()
    )

    with record_bar_ctx as record_bar:
        for sketch_index, state in enumerate(sketches):
            if len(collected_inputs) >= target_count:
                break

            try:
                gen = ScheduleGenerator.from_task_state(task, state, enabled_constraints=ENABLED_CONSTRAINTS_NO_VECTORIZE)
            except Exception as err:  # pylint: disable=broad-except
                _emit_failure(task_index, task, "construct_schedule_generator", sketch_index, err)
                continue

            start_stats = gen.get_unique_search_stats()
            while len(collected_inputs) < target_count:
                try:
                    payload = gen.next_unique_schedule(seen_fingerprints)
                except Exception as err:  # pylint: disable=broad-except
                    _emit_failure(task_index, task, "next_unique_schedule", sketch_index, err)
                    payload = None
                    break

                if payload is None:
                    exhausted_sketches += 1
                    break

                seen_fingerprints.add(payload["fingerprint"])
                try:
                    measure_input, measure_result = build_measure_record(task, payload["state"])
                except Exception as err:  # pylint: disable=broad-except
                    _emit_failure(task_index, task, "build_measure_record", sketch_index, err)
                    break

                collected_inputs.append(measure_input)
                collected_results.append(measure_result)
                if record_bar is not None:
                    record_bar.update(1)
                if len(collected_inputs) >= target_count:
                    _maybe_emit_task_log(
                        task_index,
                        task,
                        emit_logs,
                        (
                            f"PROGRESS unique={len(collected_inputs)}/{target_count} "
                            f"seeded={existing_seed_count} dup={total_duplicates_skipped}"
                        ),
                    )

            end_stats = gen.get_unique_search_stats()
            sketch_duplicates = end_stats["duplicates_skipped"] - start_stats["duplicates_skipped"]
            total_duplicates_skipped += sketch_duplicates
            if len(collected_inputs) < target_count:
                _maybe_emit_task_log(
                    task_index,
                    task,
                    emit_logs,
                    (
                        f"SKETCH_DONE sketch={sketch_index} status=exhausted "
                        f"unique={len(collected_inputs)}/{target_count} "
                        f"sketch_dup={sketch_duplicates}"
                    ),
                )

    if collected_inputs:
        deduped_inputs, deduped_results, dropped = dedupe_measure_records(
            task,
            collected_inputs,
            collected_results,
        )
        if dropped:
            _maybe_emit_task_log(
                task_index,
                task,
                emit_logs,
                f"FINAL_DEDUPE dropped={dropped} kept={len(deduped_inputs)}",
            )
        try:
            save_records_batch(output_path, deduped_inputs, deduped_results)
        except Exception as err:  # pylint: disable=broad-except
            _emit_failure(task_index, task, "save_records", None, err)
            return {
                "ok": False,
                "stage": "save_records",
                "task_index": task_index,
                "workload_key": task.workload_key,
            }

        search_exhausted = len(deduped_inputs) < target_count
        status = "EXHAUSTED" if search_exhausted else "OK"
        _maybe_emit_task_log(
            task_index,
            task,
            emit_logs,
            (
                f"{status} unique={len(deduped_inputs)}/{target_count} "
                f"seeded={existing_seed_count} dup={total_duplicates_skipped} "
                f"sketches_exhausted={exhausted_sketches} saved={output_label}"
            ),
        )
        return {
            "ok": True,
            "stage": "ok",
            "task_index": task_index,
            "workload_key": task.workload_key,
            "sketch_count": len(sketches),
            "output_path": output_path,
            "record_count": len(deduped_inputs),
            "duplicates_skipped": total_duplicates_skipped,
            "sketches_exhausted": exhausted_sketches,
            "search_exhausted": search_exhausted,
            "final_dedup_dropped": dropped,
        }

    _maybe_emit_task_log(
        task_index,
        task,
        emit_logs,
        (
            f"EXHAUSTED unique=0/{target_count} seeded={existing_seed_count} "
            f"dup={total_duplicates_skipped} sketches_exhausted={exhausted_sketches}"
        ),
    )
    return {
        "ok": True,
        "stage": "exhausted",
        "task_index": task_index,
        "workload_key": task.workload_key,
        "sketch_count": len(sketches),
        "output_path": output_path,
        "record_count": 0,
        "duplicates_skipped": total_duplicates_skipped,
        "sketches_exhausted": exhausted_sketches,
        "search_exhausted": True,
        "final_dedup_dropped": 0,
        "existing_seed_count": existing_seed_count,
    }

def _init_task_worker(network_info_folder=TASK_NETWORK_INFO_FOLDER):
    """워커 프로세스마다 task 목록을 한 번만 로드해 전역 캐시에 보관한다."""
    global _WORKER_TASKS
    _WORKER_TASKS = load_and_register_tasks(network_info_folder)


def _process_task_in_worker(task_index, records_per_task=1, emit_logs=False):
    """초기화된 워커 전역 task 캐시에서 task를 꺼내 process_task를 실행한다."""
    if _WORKER_TASKS is None:
        _init_task_worker()
    return process_task(
        task_index,
        _WORKER_TASKS[task_index],
        records_per_task=records_per_task,
        emit_logs=emit_logs,
    )


# def _print_selected_task_indices(selected_tasks):
#     indices = [str(task_index) for task_index, _ in selected_tasks]
#     if len(indices) > 40:
#         shown = ",".join(indices[:40])
#         print(f"task_idxs {shown},+{len(indices) - 40}")
#         return
#     indices = ",".join(indices)
#     print(f"task_idxs {indices}")


def run_selected_tasks(
    selected_tasks,
    workers,
    records_per_task=1,
    quiet_task_logs=False,
    suppress_progress_output=False,
):
    """선택된 task들을 workers 수만큼 병렬로 처리하고 결과 목록을 반환한다 (1이면 순차)."""
    if workers < 1:
        raise ValueError(f"--workers must be >= 1: {workers}")

    total_tasks = len(selected_tasks)
    # if not suppress_progress_output:
    #     _print_selected_task_indices(selected_tasks)

    if workers == 1 or len(selected_tasks) <= 1:
        results = []
        progress_ctx = (
            tqdm(total=total_tasks, desc="tasks", unit="task", smoothing=0.1)
            if not suppress_progress_output
            else nullcontext()
        )
        with progress_ctx as pbar:
            for task_index, task in selected_tasks:
                results.append(
                    process_task(
                        task_index,
                        task,
                        records_per_task,
                        emit_logs=False,
                        show_record_bar=not suppress_progress_output,
                    )
                )
                if pbar is not None:
                    pbar.update(1)
        return results

    results = []
    max_workers = min(workers, len(selected_tasks))
    future_map = {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_task_worker,
        initargs=(TASK_NETWORK_INFO_FOLDER,),
    ) as executor:
        for task_index, task in selected_tasks:
            future = executor.submit(
                _process_task_in_worker,
                task_index,
                records_per_task,
                False,
            )
            future_map[future] = (task_index, task)

        progress_ctx = (
            tqdm(total=total_tasks, desc="tasks", unit="task", smoothing=0.1)
            if not suppress_progress_output
            else nullcontext()
        )
        with progress_ctx as pbar:
            for future in concurrent.futures.as_completed(future_map):
                task_index, task = future_map[future]
                try:
                    results.append(future.result())
                except Exception as err:  # pylint: disable=broad-except
                    _emit_failure(task_index, task, "worker_process", None, err)
                    results.append(
                        {
                            "ok": False,
                            "stage": "worker_process",
                            "task_index": task_index,
                            "workload_key": task.workload_key,
                        }
                    )
                if pbar is not None:
                    pbar.update(1)
    return results


def select_tasks(tasks, args):
    """인자(task-index / workload-key / --all)에 따라 생성할 (인덱스, task) 목록을 반환한다."""
    if args.task_index is not None:
        if args.task_index < 0 or args.task_index >= len(tasks):
            raise ValueError(
                f"--task-index out of range: {args.task_index} (num_tasks={len(tasks)})"
            )
        return [(args.task_index, tasks[args.task_index])]

    if args.workload_key is None:
        return list(enumerate(tasks))

    matches = [
        (idx, task)
        for idx, task in enumerate(tasks)
        if task.workload_key == args.workload_key
    ]
    if not matches:
        raise ValueError(f"--workload-key not found: {args.workload_key}")
    return matches



def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate one constrained record per selected task."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task-index", type=int)
    group.add_argument("--workload-key", type=str)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--records-per-task", type=int, default=1, metavar="N",
        help="한 task당 생성할 유효 레코드 개수 (기본 1)",
    )
    parser.add_argument("--_quiet-task-logs", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_emit-result-json", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()



def main():
    args = parse_args()
    if args.output_dir:
        global TO_MEASURE_GEN_PROGRAM_FOLDER
        TO_MEASURE_GEN_PROGRAM_FOLDER = args.output_dir
    tasks = load_and_register_tasks(TASK_NETWORK_INFO_FOLDER)
    selected_tasks = select_tasks(tasks, args)

    results = run_selected_tasks(
        selected_tasks,
        args.workers,
        records_per_task=args.records_per_task,
        quiet_task_logs=args._quiet_task_logs,
        suppress_progress_output=args._emit_result_json,
    )

    success_count = 0
    exhausted_count = 0
    skipped_count = 0
    failure_stages = Counter()
    for result in results:
        if result["ok"]:
            success_count += 1
            if result.get("search_exhausted"):
                exhausted_count += 1
            if result.get("skipped_existing_output"):
                skipped_count += 1
        else:
            failure_stages[result["stage"]] += 1

    failure_count = len(selected_tasks) - success_count
    if args._emit_result_json:
        if len(results) != 1:
            raise ValueError("--_emit-result-json requires exactly one selected task")
        print(json.dumps({"result_type": "task_result", **results[0]}, sort_keys=True))
        if failure_count:
            sys.exit(1)
        return

    print()
    print(
        f"generate_programs_summary selected={len(selected_tasks)} "
        f"ok={success_count} skipped={skipped_count} "
        f"exhausted={exhausted_count} failed={failure_count}"
    )
    if failure_stages:
        print("failure_stages")
        for stage, count in sorted(failure_stages.items()):
            print(f"  {stage}: {count}")
    if failure_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
