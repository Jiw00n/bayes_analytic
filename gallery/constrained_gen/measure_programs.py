"""Measure generated constrained-gen records.

This entrypoint reuses the repo's existing TVM record-reading and measurement
flow. It accepts either one generated record file or a directory of generated
record files and writes measured records to the standard measured-gen output
path.
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from collections import Counter

from tvm import auto_scheduler

from modules.task_paths import get_measure_record_filename, load_and_register_tasks


RUN_TIMEOUT = 5
NUMBER = 1
VERBOSE = 1
MEASURE_ERROR_NAMES = {
    value: name
    for name, value in auto_scheduler.measure.MeasureErrorNo.__dict__.items()
    if name.isupper() and isinstance(value, int)
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure generated constrained-gen record file(s)."
    )
    parser.add_argument("input_path")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--_emit-result-json", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _emit_failure(input_path, stage, error):
    """측정 실패 정보를 JSON 한 줄로 stdout에 출력한다."""
    print(
        json.dumps(
            {
                "input_path": input_path,
                "stage": stage,
                "error": f"{type(error).__name__}: {error}" if isinstance(error, Exception) else str(error),
            },
            sort_keys=True,
            default=str,
        )
    )


def _format_measure_error_name(error_no):
    """측정 에러 코드를 사람이 읽기 쉬운 이름 문자열로 바꾼다."""
    return MEASURE_ERROR_NAMES.get(error_no, f"ERROR_{error_no}")


def _task_repeat(task):
    """task의 FLOPS에 따라 측정 반복 횟수를 정한다."""
    if task.compute_dag.flop_ct >= 2416443392.0:
        return 4
    if task.compute_dag.flop_ct >= 834928640.0:
        return 6
    if task.compute_dag.flop_ct <= 2097152.0:
        return 10
    return 8


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


def iter_input_files(input_path):
    """입력 경로(파일 또는 디렉터리)에서 측정 대상 .json 파일 경로 목록을 반환한다."""
    if os.path.isfile(input_path):
        return [os.path.abspath(input_path)]

    if not os.path.isdir(input_path):
        raise FileNotFoundError(input_path)

    files = []
    for root, dirnames, filenames in os.walk(input_path):
        dirnames.sort()
        for filename in sorted(filenames):
            if filename.endswith(".json"):
                files.append(os.path.abspath(os.path.join(root, filename)))
    return files


def load_source_inputs(input_path):
    """JSON 레코드 파일에서 MeasureInput 목록을 읽어 반환한다."""
    inputs, _ = auto_scheduler.RecordReader(input_path).read_lines()
    return list(inputs)


def validate_task_inputs(task_inputs, tasks_by_workload_key):
    """레코드가 비어 있지 않고 단일 workload인지 검사한 뒤 대응하는 task를 반환한다."""
    if not task_inputs:
        raise ValueError("record file is empty")

    first_workload_key = task_inputs[0].task.workload_key
    if first_workload_key not in tasks_by_workload_key:
        raise KeyError(f"unknown workload_key: {first_workload_key}")

    for inp in task_inputs[1:]:
        if inp.task.workload_key != first_workload_key:
            raise ValueError("record file contains multiple workload_keys")

    return tasks_by_workload_key[first_workload_key]


def measure_file(input_path, tasks_by_workload_key):
    """한 레코드 파일을 로드·검증 후 측정하고 결과를 저장한 뒤 결과 딕셔너리를 반환한다."""
    print(f"[measure] start {input_path}")

    try:
        raw_inputs = load_source_inputs(input_path)
    except Exception as err:  # pylint: disable=broad-except
        _emit_failure(input_path, "load_records", err)
        return {
            "ok": False,
            "stage": "load_records",
            "input_path": input_path,
        }

    try:
        task = validate_task_inputs(raw_inputs, tasks_by_workload_key)
    except Exception as err:  # pylint: disable=broad-except
        _emit_failure(input_path, "validate_input_records", err)
        return {
            "ok": False,
            "stage": "validate_input_records",
            "input_path": input_path,
        }

    measure_inputs = [auto_scheduler.MeasureInput(task, inp.state) for inp in raw_inputs]
    output_path = get_measure_record_filename(task, task.target)
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

    error_histogram = Counter()
    for result in results:
        error_histogram[_format_measure_error_name(int(result.error_no))] += 1

    measure_errors = len(results) - error_histogram.get("NO_ERROR", 0)
    usable_measurement = measure_errors == 0
    print(
        f"[measure] OK pipeline=1 usable={int(usable_measurement)} "
        f"inputs={len(measure_inputs)} errors={measure_errors} "
        f"source={input_path} saved={output_path}"
    )
    return {
        "ok": True,
        "stage": "ok",
        "input_path": input_path,
        "output_path": output_path,
        "pipeline_ok": True,
        "usable_measurement": usable_measurement,
        "measured": len(results),
        "measure_errors": measure_errors,
        "measure_error_histogram": dict(sorted(error_histogram.items())),
    }


def _run_file_subprocess(input_path):
    """해당 파일만 측정하는 워커 서브프로세를 실행해 완료 결과를 반환한다."""
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        input_path,
        "--_emit-result-json",
    ]
    return subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _extract_subprocess_result(stdout):
    """서브프로세 stdout에서 JSON 결과(file_result)와 그 외 출력 줄을 분리해 반환한다."""
    relay_lines = []
    result = None
    for line in stdout.splitlines():
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            relay_lines.append(line)
            continue
        if payload.get("result_type") == "file_result":
            result = payload
        else:
            relay_lines.append(line)
    return relay_lines, result


def _collect_subprocess_result(input_path, completed):
    """서브프로세 완료 객체에서 결과를 추출하고, 없으면 실패 딕셔너리를 반환한다."""
    relay_lines, result = _extract_subprocess_result(completed.stdout)
    for line in relay_lines:
        print(line)

    stderr = completed.stderr.strip()
    if stderr:
        print(stderr, file=sys.stderr)

    if result is not None:
        result.pop("result_type", None)
        return result

    error = f"worker subprocess exited with code {completed.returncode}"
    if stderr:
        error = f"{error}: {stderr.splitlines()[-1]}"
    _emit_failure(input_path, "worker_process", error)
    return {
        "ok": False,
        "stage": "worker_process",
        "input_path": input_path,
    }


def run_selected_files(input_paths, tasks_by_workload_key, workers):
    """입력 파일 목록을 workers 수만큼 병렬로 측정하고 결과 목록을 반환한다 (1이면 순차)."""
    if workers < 1:
        raise ValueError(f"--workers must be >= 1: {workers}")

    if workers == 1 or len(input_paths) <= 1:
        return [measure_file(input_path, tasks_by_workload_key) for input_path in input_paths]

    results = []
    max_workers = min(workers, len(input_paths))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_run_file_subprocess, input_path): input_path
            for input_path in input_paths
        }
        for future in concurrent.futures.as_completed(future_map):
            input_path = future_map[future]
            try:
                completed = future.result()
            except Exception as err:  # pylint: disable=broad-except
                _emit_failure(input_path, "worker_process", err)
                results.append(
                    {
                        "ok": False,
                        "stage": "worker_process",
                        "input_path": input_path,
                    }
                )
                continue
            results.append(_collect_subprocess_result(input_path, completed))
    return results


def main():
    args = parse_args()

    try:
        input_paths = iter_input_files(args.input_path)
    except Exception as err:  # pylint: disable=broad-except
        _emit_failure(args.input_path, "discover_inputs", err)
        sys.exit(1)

    if not input_paths:
        _emit_failure(args.input_path, "discover_inputs", "no .json record files found")
        sys.exit(1)

    tasks = load_and_register_tasks()
    tasks_by_workload_key = {task.workload_key: task for task in tasks}
    results = run_selected_files(input_paths, tasks_by_workload_key, args.workers)

    success_count = 0
    failure_stages = Counter()
    usable_measurement_count = 0
    nonzero_measure_error_files = 0
    measured_total = 0
    measure_errors_total = 0
    measure_error_histogram = Counter()
    for result in results:
        if result["ok"]:
            success_count += 1
            if result["usable_measurement"]:
                usable_measurement_count += 1
            else:
                nonzero_measure_error_files += 1
            measured_total += result["measured"]
            measure_errors_total += result["measure_errors"]
            measure_error_histogram.update(result["measure_error_histogram"])
        else:
            failure_stages[result["stage"]] += 1

    failure_count = len(input_paths) - success_count
    if args._emit_result_json:
        if len(results) != 1:
            raise ValueError("--_emit-result-json requires exactly one input file")
        print(json.dumps({"result_type": "file_result", **results[0]}, sort_keys=True))
        if failure_count:
            sys.exit(1)
        return

    print(
        f"measure_programs_summary input_files={len(input_paths)} "
        f"pipeline_successes={success_count} pipeline_failures={failure_count} "
        f"usable_measurements={usable_measurement_count} "
        f"files_with_measure_errors={nonzero_measure_error_files} "
        f"measured={measured_total} measure_errors={measure_errors_total}"
    )
    if failure_stages:
        print("failure_stage_histogram")
        for stage, count in sorted(failure_stages.items()):
            print(f"  {stage}={count}")
    if measure_error_histogram:
        print("measure_error_histogram")
        for error_name, count in sorted(measure_error_histogram.items()):
            print(f"  {error_name}={count}")
    if failure_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
