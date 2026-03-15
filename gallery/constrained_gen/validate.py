"""Minimal research/debug validation entrypoint for constrained generation.

This validates the active concrete-sketch -> symbolic generator -> observability
report path without introducing a second workflow.
"""

import argparse
import json
import sys
from collections import Counter

from tvm.auto_scheduler import SketchPolicy

from modules.task_paths import load_and_register_tasks
from modules.schedule_generator import ScheduleGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate constrained-gen observability on selected task(s)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task-index", type=int)
    group.add_argument("--workload-key", type=str)
    group.add_argument("--all", action="store_true")
    return parser.parse_args()


def select_tasks(tasks, args):
    """인자(task-index / workload-key / --all)에 따라 검사할 (인덱스, task) 목록을 반환한다."""
    if args.task_index is not None:
        if args.task_index < 0 or args.task_index >= len(tasks):
            raise ValueError(
                f"--task-index out of range: {args.task_index} (num_tasks={len(tasks)})"
            )
        return [(args.task_index, tasks[args.task_index])]

    if args.workload_key is not None:
        matches = [
            (idx, task)
            for idx, task in enumerate(tasks)
            if task.workload_key == args.workload_key
        ]
        if not matches:
            raise ValueError(f"--workload-key not found: {args.workload_key}")
        return matches

    return list(enumerate(tasks))


def generate_concrete_sketches(task):
    """SketchPolicy로 task에 대한 구체 스케치(State) 목록을 생성해 반환한다."""
    policy = SketchPolicy(
        task,
        params={"sample_init_no_invalid": 1},
        verbose=False,
    )
    return list(policy.generate_concrete_sketches())


def build_failure_report(task_index, task, stage, payload):
    """실패 시 task 정보·stage·추가 payload를 묶은 딕셔너리를 만든다."""
    return {
        "task_index": task_index,
        "task_desc": task.desc,
        "workload_key": task.workload_key,
        "stage": stage,
        **payload,
    }


def _print_failure(report):
    """실패 리포트를 JSON으로 포맷해 stdout에 출력한다."""
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


def _summarize_prefix_report(prefix_report):
    """prefix 샘플링 결과를 한 줄 요약 문자열로 만든다."""
    phase = prefix_report["phase_selection"]
    assignment = prefix_report["assignment"]["params"]
    constraints = prefix_report["constraints"]
    return (
        f"prefix phase={phase['resolved_phase_name']} "
        f"params={len(assignment)} "
        f"remaining_domains={len(prefix_report['domains']['remaining'])} "
        f"leftover_constraints={len(constraints['leftover'])} "
        f"resolved_false={len(constraints['resolved_false'])}"
    )


def _summarize_full_sample(params, pruning_violations, exact_violations, final_violations):
    """전체 샘플·위반 개수를 한 줄 요약 문자열로 만든다."""
    return (
        f"sampled_params={len(params)} "
        f"pruning={len(pruning_violations)} "
        f"exact={len(exact_violations)} "
        f"final={len(final_violations)}"
    )


def _print_task_failure_status(report):
    """실패한 task의 인덱스·stage·설명을 한 줄로 출력한다."""
    extra = []
    if "sketch_count" in report:
        extra.append(f"sketches={report['sketch_count']}")
    if "selected_sketch_index" in report:
        extra.append(f"selected_sketch={report['selected_sketch_index']}")
    suffix = f" {' '.join(extra)}" if extra else ""
    print(
        f"[task {report['task_index']}] FAIL stage={report['stage']}{suffix} "
        f"{report['task_desc']}"
    )


def _print_task_success(task_index, task, sketch_count, var_order_report, prefix_report, sampled_params,
                        pruning_violations, exact_violations, final_violations):
    """검증 통과한 task의 요약과 prefix·전체 샘플 요약을 출력한다."""
    print(
        f"[task {task_index}] OK {task.desc} sketches={sketch_count} "
        f"selected_sketch=0 phase_count={var_order_report['phase_count']}"
    )
    print(f"  {_summarize_prefix_report(prefix_report)}")
    print(f"  {_summarize_full_sample(sampled_params, pruning_violations, exact_violations, final_violations)}")


def success_result(task_index, task):
    """검증 성공 시 공통 결과 딕셔너리를 만든다."""
    return {
        "ok": True,
        "task_index": task_index,
        "task_desc": task.desc,
        "workload_key": task.workload_key,
        "stage": "ok",
    }


def failure_result(report):
    """실패 리포트를 공통 결과 딕셔너리 형태로 감싼다."""
    return {
        "ok": False,
        "task_index": report["task_index"],
        "task_desc": report["task_desc"],
        "workload_key": report["workload_key"],
        "stage": report["stage"],
        "report": report,
    }


def validate_task(task_index, task):
    """한 task에 대해 스케치 생성·제너레이터·prefix/전체 샘플·체커를 돌리고 성공/실패 결과를 반환한다."""
    try:
        sketches = generate_concrete_sketches(task)
    except Exception as err:  # pylint: disable=broad-except
        report = build_failure_report(
            task_index,
            task,
            "generate_concrete_sketches",
            {"error": f"{type(err).__name__}: {err}"},
        )
        _print_task_failure_status(report)
        _print_failure(report)
        return failure_result(report)

    if not sketches:
        report = build_failure_report(
            task_index,
            task,
            "zero_sketches",
            {
                "error": "SketchPolicy.generate_concrete_sketches() returned no states",
                "sketch_count": 0,
            },
        )
        _print_task_failure_status(report)
        _print_failure(report)
        return failure_result(report)

    state = sketches[0]
    try:
        gen = ScheduleGenerator.from_task_state(task, state)
    except Exception as err:  # pylint: disable=broad-except
        report = build_failure_report(
            task_index,
            task,
            "construct_schedule_generator",
            {
                "error": f"{type(err).__name__}: {err}",
                "selected_sketch_index": 0,
                "sketch_count": len(sketches),
            },
        )
        _print_task_failure_status(report)
        _print_failure(report)
        return failure_result(report)
    var_order_report = gen.get_full_var_order_entries()
    phases = var_order_report["phases"]
    if not phases:
        report = build_failure_report(
            task_index,
            task,
            "get_full_var_order_entries",
            {
                "error": "No var-order phases were produced",
                "var_order_report": var_order_report,
            },
        )
        _print_task_failure_status(report)
        _print_failure(report)
        return failure_result(report)

    first_phase_name = phases[0]["phase_name"]
    try:
        prefix_report = gen.randomize_params_prefix(first_phase_name)
    except Exception as err:  # pylint: disable=broad-except
        report = build_failure_report(
            task_index,
            task,
            "randomize_params_prefix",
            {
                "error": f"{type(err).__name__}: {err}",
                "var_order_report": var_order_report,
                "assignment_report": gen.get_constraints_under_assignment({}),
            },
        )
        _print_task_failure_status(report)
        _print_failure(report)
        return failure_result(report)

    try:
        sampled_params = gen.randomize_params()
    except Exception as err:  # pylint: disable=broad-except
        report = build_failure_report(
            task_index,
            task,
            "randomize_params",
            {
                "error": f"{type(err).__name__}: {err}",
                "var_order_report": var_order_report,
                "prefix_report": prefix_report,
                "assignment_report": gen.get_constraints_under_assignment({}),
            },
        )
        _print_task_failure_status(report)
        _print_failure(report)
        return failure_result(report)

    pruning_violations = gen.check_all_pruning(sampled_params)
    exact_violations = gen.check_all_exact(sampled_params)
    final_violations = gen.check_all_final(sampled_params)

    _print_task_success(
        task_index,
        task,
        len(sketches),
        var_order_report,
        prefix_report,
        sampled_params,
        pruning_violations,
        exact_violations,
        final_violations,
    )

    if pruning_violations or exact_violations or final_violations:
        report = build_failure_report(
            task_index,
            task,
            "checkers",
            {
                "params": sampled_params,
                "pruning_violations": pruning_violations,
                "exact_violations": exact_violations,
                "final_violations": final_violations,
                "prefix_report": prefix_report,
                "assignment_report": gen.get_constraints_under_assignment(sampled_params),
            },
        )
        _print_task_failure_status(report)
        _print_failure(report)
        return failure_result(report)

    return success_result(task_index, task)


def print_final_summary(results):
    """검증 결과 목록의 성공/실패 개수와 stage별 실패 히스토그램을 출력하고 요약 딕셔너리를 반환한다."""
    selected_count = len(results)
    success_count = sum(1 for result in results if result["ok"])
    failure_count = selected_count - success_count
    failure_stage_histogram = Counter(
        result["stage"] for result in results if not result["ok"]
    )

    print(
        "validation_summary "
        f"selected_tasks={selected_count} "
        f"successes={success_count} "
        f"failures={failure_count}"
    )
    if failure_stage_histogram:
        print("failure_stage_histogram")
        for stage, count in sorted(failure_stage_histogram.items()):
            print(f"  {stage}={count}")

    return {
        "selected_tasks": selected_count,
        "success_count": success_count,
        "failure_count": failure_count,
        "failure_stage_histogram": dict(sorted(failure_stage_histogram.items())),
    }


def main():
    args = parse_args()
    tasks = load_and_register_tasks()
    selected_tasks = select_tasks(tasks, args)

    results = []
    for task_index, task in selected_tasks:
        results.append(validate_task(task_index, task))

    summary = print_final_summary(results)

    if summary["failure_count"]:
        print("validation_failed", file=sys.stderr)
        sys.exit(1)

    print("validation_ok")


if __name__ == "__main__":
    main()
