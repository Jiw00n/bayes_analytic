# -*- coding: utf-8 -*-
"""
Provenance-based exact GPU constraint formulas.

Lowering은 기존처럼 상수로 한 번만 돌리되, State(infer_bound)에서 각 iter의 extent가
어떤 split 파라미터(step_idx, length_pos)에 의존하는지 probing으로 추적하고,
thread_per_block / vthread 등에 대한 정확한 symbolic 제약식을 복원한다.

Usage:
    from constraint_provenance import build_provenance_formulas, eval_thread_formula, eval_vthread_formula

    formulas = build_provenance_formulas(task, state, record, record_to_task_and_state)
    thread_val = eval_thread_formula(formulas, record)  # 단일 커널일 때만 의미 있는 “전체 thread” 곱
    vthread_val = eval_vthread_formula(formulas, record)

Note:
    다중 커널(예: Winograd)에서는 thread_per_block이 커널별로 다름. eval_thread_formula는
    모든 thread-bound iter extent의 곱을 주므로, 검증 시에는 TIR lowering으로 커널별 값을
    확인하거나 formula의 iter_deps/base_extents로 커널별 식을 구성해야 함.
"""

from __future__ import absolute_import

import copy
from collections import defaultdict

try:
    from tvm.topi.utils import get_const_int
except ImportError:
    import tvm

    def get_const_int(expr):
        if isinstance(expr, (int,)):
            return int(expr)
        ana = tvm.arith.Analyzer()
        expr = ana.simplify(expr)
        if not isinstance(expr, tvm.tir.IntImm):
            raise ValueError("Expect constant int")
        return int(expr.value)


# IteratorAnnotation: 4=vthread, 5=blockIdx.x, 6=threadIdx.x, 7=blockIdx.y, 8=threadIdx.y, 9=blockIdx.z, 10=threadIdx.z
# thread_per_block = product of threadIdx.x/y/z and vthread (not blockIdx)
ANNOTATION_THREAD = (4, 6, 8, 10)  # vthread, threadIdx.x, threadIdx.y, threadIdx.z
ANNOTATION_VTHREAD = (4,)  # vthread only


def _get_state_iter_info(state_with_bound):
    """
    From State after infer_bound_from_state, collect (stage_idx, iter_idx, extent_int, annotation).

    Returns list of (stage_idx, iter_idx, extent, annotation).
    """
    result = []
    try:
        stages = state_with_bound.stages
    except AttributeError:
        return result
    for stage_idx, stage in enumerate(stages):
        iters = getattr(stage, "iters", [])
        for iter_idx, it in enumerate(iters):
            r = getattr(it, "range", None)
            if r is None or getattr(r, "extent", None) is None:
                continue
            try:
                extent = get_const_int(r.extent)
            except (ValueError, TypeError):
                continue
            ann = getattr(it, "annotation", 0)
            try:
                ann = int(ann) if hasattr(ann, "__int__") else ann
            except (TypeError, ValueError):
                ann = 0
            result.append((stage_idx, iter_idx, extent, ann))
    return result


def _get_schedule_iter_info(schedule):
    """
    From a te.Schedule, collect (op_name, iter_var_name, extent_int, thread_tag) for each leaf iter.

    Returns
    -------
    list of (op_name, iter_name, extent, thread_tag)
    """
    result = []
    try:
        stage_map = schedule.stage_map
    except AttributeError:
        return result
    for op in stage_map:
        stage = schedule[op]
        op_name = op.name if hasattr(op, "name") else str(op)
        for iv in stage.leaf_iter_vars:
            if getattr(iv, "dom", None) is None or getattr(iv.dom, "extent", None) is None:
                continue
            try:
                extent = get_const_int(iv.dom.extent)
            except (ValueError, TypeError):
                continue
            thread_tag = getattr(iv, "thread_tag", "") or ""
            iter_name = iv.var.name_hint if hasattr(iv.var, "name_hint") else str(iv.var)
            result.append((op_name, iter_name, extent, thread_tag))
    return result


def _get_sp_step_lengths(record):
    """
    From JSON record, get list of (step_idx, lengths_list) for each SP (Split) step.
    """
    steps = record["i"][1][1]
    out = []
    for step_idx, s in enumerate(steps):
        if s[0] != "SP":
            continue
        lengths = list(s[4])  # [l0, l1, ...]
        out.append((step_idx, lengths))
    return out


def _inject_param(record, step_idx, length_pos, value):
    """Return a new record with step step_idx's length at length_pos set to value."""
    rec = copy.deepcopy(record)
    s = rec["i"][1][1][step_idx]
    if s[0] != "SP":
        return rec
    lengths = list(s[4])
    if 0 <= length_pos < len(lengths):
        lengths[length_pos] = value
        s[4] = lengths
    return rec


def _state_iter_key(stage_idx, iter_idx):
    return (stage_idx, iter_idx)


def build_provenance_formulas(task, state, record, record_to_task_and_state):
    """
    Build exact formulas for thread_per_block and vthread extent from State (after infer_bound).

    Uses State's iterator annotations (threadIdx.x etc.) and extents; one probe per (step_idx, length_pos)
    to see which iter extents change. thread_per_block = product of extents of iters with
    thread annotation (4=vthread, 6=threadIdx.x, 8=threadIdx.y, 10=threadIdx.z).

    Parameters
    ----------
    task : SearchTask
    state : State (from record)
    record : dict, JSON record (for step structure and base lengths)
    record_to_task_and_state : callable(record) -> (task, state)

    Returns
    -------
    dict with keys:
      - "thread_per_block_iters": list of (stage_idx, iter_idx) that contribute to thread_per_block
      - "vthread_extent_iters": list of (stage_idx, iter_idx) for vthread
      - "iter_deps": dict mapping (stage_idx, iter_idx) -> list of (step_idx, length_pos)
      - "base_extents": dict (stage_idx, iter_idx) -> base extent
      - "base_iter_info": list of (stage_idx, iter_idx, extent, annotation)
      - "sp_steps": list of (step_idx, lengths) for SP steps
    """
    # State with bounds (infer_bound_from_state)
    try:
        state_with_bound = task.compute_dag.infer_bound_from_state(state)
    except Exception:
        return {
            "thread_per_block_iters": [],
            "vthread_extent_iters": [],
            "iter_deps": {},
            "base_extents": {},
            "base_iter_info": [],
            "sp_steps": [],
        }
    base_iters = _get_state_iter_info(state_with_bound)
    if not base_iters:
        return {
            "thread_per_block_iters": [],
            "vthread_extent_iters": [],
            "iter_deps": {},
            "base_extents": {},
            "base_iter_info": [],
            "sp_steps": [],
        }

    base_key_to_extent = {}
    for stage_idx, iter_idx, extent, ann in base_iters:
        base_key_to_extent[_state_iter_key(stage_idx, iter_idx)] = (extent, ann)

    sp_steps = _get_sp_step_lengths(record)
    steps_flat = []
    for step_idx, lengths in sp_steps:
        for length_pos in range(len(lengths)):
            steps_flat.append((step_idx, length_pos))

    # Probe: for each (step_idx, length_pos), mutate length and see which state iter extents change
    iter_deps = defaultdict(list)
    for step_idx, length_pos in steps_flat:
        base_length = record["i"][1][1][step_idx][4][length_pos]
        probe_val = 2 if base_length != 2 else 3
        rec_probe = _inject_param(record, step_idx, length_pos, probe_val)
        try:
            _, state_probe = record_to_task_and_state(rec_probe)
            state_probe_bound = task.compute_dag.infer_bound_from_state(state_probe)
        except Exception:
            continue
        probe_iters = _get_state_iter_info(state_probe_bound)
        probe_key_to_extent = {}
        for si, ii, ext, _ in probe_iters:
            probe_key_to_extent[_state_iter_key(si, ii)] = ext
        for key, (base_extent, _) in base_key_to_extent.items():
            if key not in probe_key_to_extent:
                continue
            if probe_key_to_extent[key] != base_extent:
                iter_deps[key].append((step_idx, length_pos))

    thread_bound_iters = [
        (stage_idx, iter_idx)
        for stage_idx, iter_idx, _extent, ann in base_iters
        if ann in ANNOTATION_THREAD
    ]
    vthread_bound_iters = [
        (stage_idx, iter_idx)
        for stage_idx, iter_idx, _extent, ann in base_iters
        if ann in ANNOTATION_VTHREAD
    ]
    base_extents = dict(
        (_state_iter_key(si, ii), ext) for si, ii, ext, _ in base_iters
    )

    return {
        "thread_per_block_iters": thread_bound_iters,
        "vthread_extent_iters": vthread_bound_iters,
        "iter_deps": dict(iter_deps),
        "base_extents": base_extents,
        "base_iter_info": base_iters,
        "sp_steps": sp_steps,
    }


def _eval_iter_extent(iter_key, iter_deps, base_extents, record):
    """Extent of one iter = product of length[step_idx][length_pos] for (s,p) in iter_deps; if no deps, use base_extents."""
    deps = iter_deps.get(iter_key, [])
    if not deps:
        return base_extents.get(iter_key, 1)
    steps = record["i"][1][1]
    val = 1
    for step_idx, length_pos in deps:
        s = steps[step_idx]
        if s[0] != "SP" or length_pos >= len(s[4]):
            continue
        val *= int(s[4][length_pos])
    return val


def eval_thread_formula(formulas, record):
    """
    Evaluate thread_per_block from formulas and record.

    thread_per_block = product over thread_bound_iters of extent(iter);
    extent(iter) = product of length[s][p] for (s,p) in iter_deps[iter], or base_extents if no deps.
    """
    iter_deps = formulas.get("iter_deps", {})
    base_extents = formulas.get("base_extents", {})
    thread_iters = formulas.get("thread_per_block_iters", [])
    val = 1
    for iter_key in thread_iters:
        key = iter_key if isinstance(iter_key, tuple) and len(iter_key) == 2 else tuple(iter_key)
        val *= _eval_iter_extent(key, iter_deps, base_extents, record)
    return val


def eval_vthread_formula(formulas, record):
    """Vthread extent = product over vthread_bound_iters of extent(iter)."""
    iter_deps = formulas.get("iter_deps", {})
    base_extents = formulas.get("base_extents", {})
    vthread_iters = formulas.get("vthread_extent_iters", [])
    val = 1
    for iter_key in vthread_iters:
        key = iter_key if isinstance(iter_key, tuple) and len(iter_key) == 2 else tuple(iter_key)
        val *= _eval_iter_extent(key, iter_deps, base_extents, record)
    return val
