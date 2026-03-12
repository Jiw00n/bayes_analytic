"""Helpers for projected GPU constraint validation and triage."""

import os

import tvm
from tvm import auto_scheduler
from tvm import tir
from tvm.auto_scheduler.measure_record import load_record_from_string

from .common import TO_MEASURE_PROGRAM_FOLDER, load_and_register_tasks
from .param_manager import build_symbolic_state
from .schedule_generator import ScheduleGenerator
from .tvm_verify import GPU_VERIFY_CONSTRAINTS, verify_gpu_func_errors


ROOT_CAUSE_RUNTIME_PROJECTION = "runtime_projection_upper_bound_insufficient"
ROOT_CAUSE_VECTORIZE_PROJECTION = "vectorize_selector_projection_insufficient"
ROOT_CAUSE_VALIDATOR_DRIVER = "validator_or_driver_state_or_lowering_difference"
ROOT_CAUSE_SYMBOLIC_THREAD_BINDING = "symbolic_thread_binding_semantics_mismatch"
ROOT_CAUSE_EXACT_INTERVAL_UNKNOWN = "exact_interval_unknown"
ROOT_CAUSE_EXACT_SYMBOLIC_CASE_STAT = "exact_symbolic_case_stat_mismatch"
ROOT_CAUSE_CUSTOM_POST_VECTORIZE_LOWERING = "custom_post_vectorize_lowering_mismatch"
_SENTINEL_UPPER_BOUND = 1 << 60


_LOWER_SYMBOLIC_POST_VECTORIZE = tvm.get_global_func(
    "constrained_gen.lower_symbolic_post_vectorize"
)


def ensure_parent_dir(path):
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_tasks_by_workload(network_info_dir):
    tasks = load_and_register_tasks(network_info_dir)
    return tasks, {task.workload_key: task for task in tasks}


def load_sketch_lines(sketches_path=None):
    if sketches_path is None:
        sketches_path = f"{TO_MEASURE_PROGRAM_FOLDER}/all_sketches.json"
    with open(sketches_path) as f:
        return [line.strip() for line in f]


def load_sketch_record(line, tasks_by_wkey):
    base_inp, base_res = load_record_from_string(line)
    recovered = auto_scheduler.measure.recover_measure_input(base_inp)
    task = tasks_by_wkey[recovered.task.workload_key]
    return task, base_inp, base_res, recovered.state


def build_schedule_generator(task, state, base_inp=None, base_res=None):
    sym = build_symbolic_state(task.compute_dag, state)
    return ScheduleGenerator(
        sym,
        task=task,
        base_input=base_inp,
        base_result=base_res,
    )


def _kind_snapshot(gen, params, kind, exact_tree, projected_tree, limit):
    projected_value = projected_tree.evaluate(params)
    exact_value = gen.constraint_set._exact_upper_bound(exact_tree, params)
    return {
        "kind": kind,
        "limit": limit,
        "projected_value": projected_value,
        "exact_value": exact_value,
        "projected_ok": projected_value <= limit,
        "exact_ok": exact_value is None or exact_value <= limit,
    }


def _gpu_error_kind(message):
    if (
        "threads per block" in message
        or "threadIdx.x" in message
        or "threadIdx.y" in message
        or "threadIdx.z" in message
    ):
        return "max_threads"
    if "shared memory per block" in message:
        return "shared_memory"
    if "vthread" in message:
        return "max_vthread"
    if "vector bytes" in message or "Number of lanes" in message:
        return "vectorize"
    if "local memory per block" in message:
        return "local_memory"
    if "launched kernels" in message:
        return "max_kernels"
    return "other"


def _simplify_primfunc(func):
    mod = tvm.IRModule({"main": func})
    mod = tir.transform.Simplify()(mod)
    return next(iter(mod.functions.values()))


def _substitute_params_in_primfunc(func, params):
    subst = {}

    def visit(node):
        if isinstance(node, tvm.tir.Var):
            value = params.get(str(node.name))
            if value is not None:
                subst[node] = tvm.tir.IntImm(node.dtype, int(value))

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    if not subst:
        return func

    body = tvm.tir.stmt_functor.substitute(func.body, subst)
    return tvm.tir.PrimFunc(
        func.params,
        body,
        func.ret_type,
        func.buffer_map,
        func.attrs,
        func.span,
    )


def _finite_interval_upper(expr, domains):
    upper = int(expr.interval(domains)[1])
    if upper >= _SENTINEL_UPPER_BOUND:
        return None
    return upper


def _collect_exact_lowering_differential(gen, params, max_cases=8):
    if not gen.has_concrete_final_context():
        return None

    gen._ensure_exact_gpu_constraints()
    exact = gen._exact_gpu
    vector_node = exact["vector_node"]
    shared_node = exact["shared_node"]
    max_vthread_node = exact["max_vthread_node"]
    domains = vector_node._augment_domains(dict(params))

    feasible_cases = []
    for case in vector_node.cases:
        values = tuple(int(v) for v in case["values"])
        if not vector_node._case_feasible(values, domains):
            continue

        shared_expr = shared_node._case_map.get(values, shared_node.default)
        max_vthread_expr = max_vthread_node._case_map.get(values, max_vthread_node.default)
        shared_bytes = _finite_interval_upper(shared_expr, domains)
        max_vthread = _finite_interval_upper(max_vthread_expr, domains)
        row = {
            "values": list(values),
            "vector_bytes": int(case["expr"].evaluate(params)),
            "shared_bytes": shared_bytes,
            "max_vthread": max_vthread,
        }
        score_terms = [
            row["vector_bytes"] / max(1, gen.hw["max_vector_bytes"]),
        ]
        if shared_bytes is not None:
            score_terms.append(shared_bytes / max(1, gen.hw["max_shared_memory_per_block"]))
        if max_vthread is not None:
            score_terms.append(max_vthread / max(1, gen.hw["max_vthread_extent"]))
        row["score"] = max(score_terms)
        feasible_cases.append(row)

    if not feasible_cases:
        return {
            "selector_exprs": [str(selector) for selector in exact["selectors"]],
            "feasible_case_count": 0,
            "top_cases": [],
        }

    feasible_cases.sort(key=lambda item: (-item["score"], tuple(item["values"])))
    top_cases = feasible_cases[:max_cases]

    concrete_errors = []
    concrete_result = gen.get_concrete_final_result(params)
    if concrete_result is not None:
        concrete_errors = list(concrete_result.get("violations", []))

    for row in top_cases:
        try:
            func = _LOWER_SYMBOLIC_POST_VECTORIZE(exact["pre_func"], row["values"])
            func = _substitute_params_in_primfunc(func, params)
            func = _simplify_primfunc(func)
            verify_errors = verify_gpu_func_errors(func, GPU_VERIFY_CONSTRAINTS)
        except Exception as err:  # pylint: disable=broad-except
            verify_errors = [f"{type(err).__name__}: {err}"]
        row["verify_errors"] = verify_errors
        row["verify_kinds"] = sorted({_gpu_error_kind(msg) for msg in verify_errors})
        row.pop("score", None)

    return {
        "selector_exprs": [str(selector) for selector in exact["selectors"]],
        "feasible_case_count": len(feasible_cases),
        "concrete_verify_errors": concrete_errors,
        "concrete_verify_kinds": sorted({_gpu_error_kind(msg) for msg in concrete_errors}),
        "top_cases": top_cases,
    }


def _classify_exact_false_reject(kind, exact_lowering_differential):
    if exact_lowering_differential is None:
        return ROOT_CAUSE_EXACT_SYMBOLIC_CASE_STAT

    top_cases = exact_lowering_differential.get("top_cases") or []
    symbolic_verify_kinds = {
        verify_kind
        for row in top_cases
        for verify_kind in row.get("verify_kinds", [])
    }
    concrete_verify_kinds = set(exact_lowering_differential.get("concrete_verify_kinds", []))
    if kind in symbolic_verify_kinds and kind not in concrete_verify_kinds:
        return ROOT_CAUSE_CUSTOM_POST_VECTORIZE_LOWERING

    if kind == "shared_memory":
        if any(row.get("shared_bytes") is None for row in top_cases):
            return ROOT_CAUSE_EXACT_INTERVAL_UNKNOWN
    elif kind == "max_vthread":
        if any(row.get("max_vthread") is None for row in top_cases):
            return ROOT_CAUSE_EXACT_INTERVAL_UNKNOWN

    return ROOT_CAUSE_EXACT_SYMBOLIC_CASE_STAT


def collect_gpu_projection_diagnostics(gen, params):
    gen._ensure_exact_gpu_constraints()
    gen._ensure_projected_gpu_constraints()
    snapshots = []
    snapshots.append(
        _kind_snapshot(
            gen,
            params,
            "vectorize",
            gen._exact_gpu["vector_node"],
            gen._projected_gpu["vector_node"],
            gen.hw["max_vector_bytes"],
        )
    )
    snapshots.append(
        _kind_snapshot(
            gen,
            params,
            "shared_memory",
            gen._exact_gpu["shared_node"],
            gen._projected_gpu["shared_node"],
            gen.hw["max_shared_memory_per_block"],
        )
    )
    snapshots.append(
        _kind_snapshot(
            gen,
            params,
            "max_vthread",
            gen._exact_gpu["max_vthread_node"],
            gen._projected_gpu["max_vthread_node"],
            gen.hw["max_vthread_extent"],
        )
    )

    exact_violations = [item["kind"] for item in snapshots if not item["exact_ok"]]
    projected_violations = [item["kind"] for item in snapshots if not item["projected_ok"]]

    if "vectorize" in exact_violations:
        root_cause = ROOT_CAUSE_VECTORIZE_PROJECTION
    elif exact_violations:
        root_cause = ROOT_CAUSE_RUNTIME_PROJECTION
    else:
        root_cause = ROOT_CAUSE_VALIDATOR_DRIVER

    return {
        "projected_constraints": gen.get_constraints_str(include_vars=True),
        "raw_exact_constraints": gen.get_raw_exact_constraints_str(include_vars=True),
        "snapshots": snapshots,
        "exact_violations": exact_violations,
        "projected_violations": projected_violations,
        "root_cause": root_cause,
    }


def collect_false_reject_diagnostics(gen, params, violations):
    gen._ensure_exact_gpu_constraints()
    gen._ensure_projected_gpu_constraints()
    snapshots = []
    snapshot_by_kind = {}

    for kind, exact_tree, projected_tree, limit in (
        (
            "vectorize",
            gen._exact_gpu["vector_node"],
            gen._projected_gpu["vector_node"],
            gen.hw["max_vector_bytes"],
        ),
        (
            "shared_memory",
            gen._exact_gpu["shared_node"],
            gen._projected_gpu["shared_node"],
            gen.hw["max_shared_memory_per_block"],
        ),
        (
            "max_vthread",
            gen._exact_gpu["max_vthread_node"],
            gen._projected_gpu["max_vthread_node"],
            gen.hw["max_vthread_extent"],
        ),
    ):
        snap = _kind_snapshot(gen, params, kind, exact_tree, projected_tree, limit)
        snapshots.append(snap)
        snapshot_by_kind[kind] = snap

    exact_lowering_differential = None
    if any(not snap["exact_ok"] for snap in snapshots):
        exact_lowering_differential = _collect_exact_lowering_differential(gen, params)

    root_causes = set()
    for violation in violations:
        if violation.startswith("shared_memory:"):
            snap = snapshot_by_kind["shared_memory"]
            if snap["exact_ok"] and not snap["projected_ok"]:
                root_causes.add(ROOT_CAUSE_RUNTIME_PROJECTION)
            else:
                root_causes.add(
                    _classify_exact_false_reject("shared_memory", exact_lowering_differential)
                )
        elif violation.startswith("vectorize"):
            snap = snapshot_by_kind["vectorize"]
            if snap["exact_ok"] and not snap["projected_ok"]:
                root_causes.add(ROOT_CAUSE_VECTORIZE_PROJECTION)
            elif not snap["exact_ok"]:
                root_causes.add(
                    _classify_exact_false_reject("vectorize", exact_lowering_differential)
                )
            else:
                root_causes.add(ROOT_CAUSE_VALIDATOR_DRIVER)
        elif violation.startswith("vthread extent"):
            snap = snapshot_by_kind["max_vthread"]
            if not snap["exact_ok"]:
                root_causes.add(
                    _classify_exact_false_reject("max_vthread", exact_lowering_differential)
                )
            elif not snap["projected_ok"]:
                root_causes.add(ROOT_CAUSE_RUNTIME_PROJECTION)
            else:
                root_causes.add(ROOT_CAUSE_VALIDATOR_DRIVER)
        elif violation.startswith("threads per block"):
            root_causes.add(ROOT_CAUSE_SYMBOLIC_THREAD_BINDING)
        else:
            root_causes.add(ROOT_CAUSE_VALIDATOR_DRIVER)

    if not root_causes:
        root_causes.add(ROOT_CAUSE_VALIDATOR_DRIVER)

    concrete_result = gen.get_concrete_final_result(params)

    return {
        "snapshots": snapshots,
        "root_causes": sorted(root_causes),
        "concrete_final_result": concrete_result,
        "exact_lowering_differential": exact_lowering_differential,
    }
