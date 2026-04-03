"""Measure generated constrained-gen records for registered tasks."""

import argparse
import os
import random
import sys

from tvm import auto_scheduler

from ..modules.task_paths import (
    get_measure_record_filename,
    get_to_measure_gen_filename,
    load_and_register_tasks,
)


RUN_TIMEOUT = 5
NUMBER = 1
VERBOSE = 1




def _emit_failure(source, stage, error):
    """측정 실패 정보를 짧은 한 줄 로그로 출력한다."""
    message = f"{type(error).__name__}: {error}" if isinstance(error, Exception) else str(error)
    print(f"[measure] fail stage={stage} source={source} error={message}")


def _task_repeat(task):
    """task의 FLOPS에 따라 측정 반복 횟수를 정한다."""
    # if task.compute_dag.flop_ct >= 2416443392.0:
    #     return 4
    # if task.compute_dag.flop_ct >= 834928640.0:
    #     return 6
    # if task.compute_dag.flop_ct <= 2097152.0:
    #     return 10
    return 3


def _make_measurer(task, log_filename):
    """task용 ProgramMeasurer(Builder+Runner+RecordToFile)를 만들어 반환한다."""
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


def select_input_files(tasks, task_index, explicit_input_paths=None):
    """명시 입력 또는 task_index에 따라 실제 존재하는 입력 파일 목록을 반환한다."""
    if explicit_input_paths:
        input_paths = [os.path.abspath(path) for path in explicit_input_paths]
        missing_paths = [path for path in input_paths if not os.path.isfile(path)]
        if missing_paths:
            raise FileNotFoundError(f"input file not found: {missing_paths[0]}")
        return input_paths

    if task_index is None:
        input_paths = []
        for task in tasks:
            input_path = os.path.abspath(get_to_measure_gen_filename(task))
            if os.path.isfile(input_path):
                input_paths.append(input_path)
        return input_paths

    if task_index < 0 or task_index >= len(tasks):
        raise ValueError(f"--task-index out of range: {task_index} (num_tasks={len(tasks)})")

    selected_task = tasks[task_index]
    input_path = os.path.abspath(get_to_measure_gen_filename(selected_task))
    if not os.path.isfile(input_path):
        raise FileNotFoundError(
            f"no generated record file for --task-index {task_index}: expected {input_path}"
        )
    return [input_path]


def measure_file(input_path, tasks_by_workload_key, records_per_task, output_dir=None):
    """한 레코드 파일을 로드·검증 후 측정하고 결과를 저장한 뒤 결과 딕셔너리를 반환한다."""
    try:
        raw_inputs, _ = auto_scheduler.RecordReader(input_path).read_lines()
        raw_inputs = list(raw_inputs)
    except Exception as err:  # pylint: disable=broad-except
        _emit_failure(input_path, "load_records", err)
        return {
            "ok": False,
            "stage": "load_records",
            "input_path": input_path,
        }

    try:
        if not raw_inputs:
            raise ValueError("record file is empty")

        first_workload_key = raw_inputs[0].task.workload_key
        if first_workload_key not in tasks_by_workload_key:
            raise KeyError(f"unknown workload_key: {first_workload_key}")

        for inp in raw_inputs[1:]:
            if inp.task.workload_key != first_workload_key:
                raise ValueError("record file contains multiple workload_keys")

        task = tasks_by_workload_key[first_workload_key]
    except Exception as err:  # pylint: disable=broad-except
        _emit_failure(input_path, "validate_input_records", err)
        return {
            "ok": False,
            "stage": "validate_input_records",
            "input_path": input_path,
        }

    if records_per_task is not None and records_per_task < len(raw_inputs):
        raw_inputs = random.sample(raw_inputs, records_per_task)

    measure_inputs = [auto_scheduler.MeasureInput(task, inp.state) for inp in raw_inputs]
    output_path = get_measure_record_filename(task, task.target, output_dir=output_dir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        measurer = _make_measurer(task, output_path)
        empty_policy = auto_scheduler.search_policy.EmptyPolicy(task)
        results = measurer.measure(task, empty_policy, measure_inputs)
    except Exception as err:  # pylint: disable=broad-except
        _emit_failure(input_path, "measure_records", err)
        return {
            "ok": False,
            "stage": "measure_records",
            "input_path": input_path,
        }

    no_error = int(auto_scheduler.measure.MeasureErrorNo.NO_ERROR)
    measure_errors = sum(int(result.error_no) != no_error for result in results)
    usable_measurement = measure_errors == 0
    return {
        "ok": True,
        "usable_measurement": usable_measurement,
        "measured": len(results),
        "measure_errors": measure_errors,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure generated constrained-gen records for selected task(s)."
    )
    parser.add_argument("--task-index", type=int)
    parser.add_argument(
        "--input",
        dest="input_paths",
        action="append",
        default=None,
        help="직접 측정할 입력 record JSON 파일 경로. 여러 번 지정 가능.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--records-per-task", type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.records_per_task is not None and args.records_per_task < 1:
        _emit_failure("args", "parse_args", "--records-per-task must be >= 1")
        sys.exit(1)
    if args.task_index is not None and args.input_paths:
        _emit_failure("args", "parse_args", "--task-index and --input cannot be used together")
        sys.exit(1)

    tasks = load_and_register_tasks()
    try:
        input_paths = select_input_files(tasks, args.task_index, args.input_paths)
    except Exception as err:  # pylint: disable=broad-except
        _emit_failure("tasks", "select_task", err)
        sys.exit(1)

    if not input_paths:
        _emit_failure("tasks", "discover_inputs", "no generated record files found")
        sys.exit(1)

    tasks_by_workload_key = {task.workload_key: task for task in tasks}
    results = [
        measure_file(
            input_path,
            tasks_by_workload_key,
            args.records_per_task,
            output_dir=args.output_dir,
        )
        for input_path in input_paths
    ]

    success_count = 0
    usable_measurement_count = 0
    nonzero_measure_error_files = 0
    measured_total = 0
    measure_errors_total = 0
    for result in results:
        if result["ok"]:
            success_count += 1
            if result["usable_measurement"]:
                usable_measurement_count += 1
            else:
                nonzero_measure_error_files += 1
            measured_total += result["measured"]
            measure_errors_total += result["measure_errors"]

    failure_count = len(input_paths) - success_count
    print(
        f"[measure] done files={len(input_paths)} "
        f"success={success_count} failure={failure_count} "
        f"usable={usable_measurement_count} "
        f"with_errors={nonzero_measure_error_files} "
        f"measured={measured_total} errors={measure_errors_total}"
    )
    if failure_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
