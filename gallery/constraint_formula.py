# -*- coding: utf-8 -*-
"""Step-simulator based GPU constraint formulas for auto-scheduler records.

This module builds symbolic-ish formulas from one base record and evaluates
record validity without lowering.
"""

from __future__ import absolute_import

import copy
import json
import math
import pickle
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import tvm
from tvm import auto_scheduler, tir
from tvm.auto_scheduler.measure_record import load_record_from_string

try:
    from tvm.topi.utils import get_const_int
except ImportError:

    def get_const_int(expr):
        if isinstance(expr, int):
            return int(expr)
        ana = tvm.arith.Analyzer()
        expr = ana.simplify(expr)
        if not isinstance(expr, tvm.tir.IntImm):
            raise ValueError("Expect constant int")
        return int(expr.value)


DEFAULT_HW = {
    "max_threads_per_block": 1024,
    "max_thread_x": 1024,
    "max_thread_y": 1024,
    "max_thread_z": 64,
    "max_shared_memory_per_block": 49152,
    "max_local_memory_per_block": 2**31 - 1,
    "max_vthread": 8,
    "max_vector_bytes": 16,
    "max_innermost_split_factor": 64,
    "warp_size": 32,
}

AUTO_UNROLL_CONFIGS = (0, 16, 64, 512, 1024)

ANNOTATION_VTHREAD = 4
ANNOTATION_BLOCK_X = 5
ANNOTATION_THREAD_X = 6
ANNOTATION_BLOCK_Y = 7
ANNOTATION_THREAD_Y = 8
ANNOTATION_BLOCK_Z = 9
ANNOTATION_THREAD_Z = 10
ANNOTATION_VECTORIZE = 2


@dataclass(frozen=True)
class Expr:
    op: str
    args: Tuple[Any, ...]


def const_expr(v: int) -> Expr:
    return Expr("const", (int(v),))


def param_expr(step_idx: int, length_pos: int) -> Expr:
    return Expr("param", (int(step_idx), int(length_pos)))


def _flatten_expr(op: str, items: Iterable[Expr]) -> List[Expr]:
    flat: List[Expr] = []
    for item in items:
        if item.op == op:
            flat.extend(item.args)  # type: ignore[arg-type]
        else:
            flat.append(item)
    return flat


def mul_expr(*items: Expr) -> Expr:
    flat = _flatten_expr("mul", items)
    out: List[Expr] = []
    c = 1
    for x in flat:
        if x.op == "const":
            c *= int(x.args[0])
        else:
            out.append(x)
    if c == 0:
        return const_expr(0)
    if c != 1:
        out.insert(0, const_expr(c))
    if not out:
        return const_expr(1)
    if len(out) == 1:
        return out[0]
    return Expr("mul", tuple(out))


def add_expr(*items: Expr) -> Expr:
    flat = _flatten_expr("add", items)
    out: List[Expr] = []
    c = 0
    for x in flat:
        if x.op == "const":
            c += int(x.args[0])
        else:
            out.append(x)
    if c != 0:
        out.insert(0, const_expr(c))
    if not out:
        return const_expr(0)
    if len(out) == 1:
        return out[0]
    return Expr("add", tuple(out))


def floordiv_expr(a: Expr, b: Expr) -> Expr:
    if b.op == "const" and int(b.args[0]) == 1:
        return a
    return Expr("floordiv", (a, b))


def ceildiv_expr(a: Expr, b: Expr) -> Expr:
    if b.op == "const" and int(b.args[0]) == 1:
        return a
    return Expr("ceildiv", (a, b))


def max_expr(items: Sequence[Expr]) -> Expr:
    if not items:
        return const_expr(0)
    if len(items) == 1:
        return items[0]
    return Expr("max", tuple(items))


def eval_expr(expr: Expr, steps: List[list]) -> int:
    op = expr.op
    if op == "const":
        return int(expr.args[0])
    if op == "param":
        step_idx, length_pos = expr.args
        s = steps[int(step_idx)]
        if s[0] != "SP":
            return 1
        arr = s[4]
        if int(length_pos) >= len(arr):
            return 1
        return int(arr[int(length_pos)])
    if op == "mul":
        val = 1
        for x in expr.args:
            val *= eval_expr(x, steps)
        return val
    if op == "add":
        val = 0
        for x in expr.args:
            val += eval_expr(x, steps)
        return val
    if op == "floordiv":
        a = eval_expr(expr.args[0], steps)
        b = eval_expr(expr.args[1], steps)
        if b == 0:
            return 0
        return a // b
    if op == "ceildiv":
        a = eval_expr(expr.args[0], steps)
        b = eval_expr(expr.args[1], steps)
        if b == 0:
            return 0
        return (a + b - 1) // b
    if op == "max":
        vals = [eval_expr(x, steps) for x in expr.args]
        return max(vals) if vals else 0
    raise ValueError("Unknown expr op: %s" % op)


def expr_to_str(expr: Expr) -> str:
    op = expr.op
    if op == "const":
        return str(expr.args[0])
    if op == "param":
        return "SP[%d].l[%d]" % (expr.args[0], expr.args[1])
    if op == "mul":
        return "(" + " * ".join(expr_to_str(x) for x in expr.args) + ")"
    if op == "add":
        return "(" + " + ".join(expr_to_str(x) for x in expr.args) + ")"
    if op == "floordiv":
        return "(%s // %s)" % (expr_to_str(expr.args[0]), expr_to_str(expr.args[1]))
    if op == "ceildiv":
        return "ceildiv(%s, %s)" % (expr_to_str(expr.args[0]), expr_to_str(expr.args[1]))
    if op == "max":
        return "max(" + ", ".join(expr_to_str(x) for x in expr.args) + ")"
    return repr(expr)


@dataclass
class IterInfo:
    name: str
    extent_expr: Expr
    annotation: int = 0
    iter_kind: int = 0


@dataclass
class StageInfo:
    op_name: str
    op_type: int
    compute_at: int
    scope: str
    dtype_bytes: int
    iters: List[IterInfo] = field(default_factory=list)
    compute_at_target: Optional[Tuple[int, int]] = None
    inline: bool = False


@dataclass
class SimContext:
    stages: List[StageInfo]
    steps: List[Dict[str, Any]]
    sp_param_exprs: Dict[int, List[Expr]]
    stage_size_exprs: Dict[int, Expr] = field(default_factory=dict)


def _stage_scope_from_name(op_name: str) -> str:
    if ".shared" in op_name:
        return "shared"
    if ".local" in op_name:
        return "local"
    return "global"


def _dtype_bytes(dtype: str) -> int:
    dt = tvm.DataType(dtype)
    return max(1, (int(dt.bits) // 8) * int(dt.lanes))


def build_dag_info(task) -> List[StageInfo]:
    init_state = task.compute_dag.get_init_state()
    bound = task.compute_dag.infer_bound_from_state(init_state)
    stages: List[StageInfo] = []
    for st in bound.stages:
        op_name = str(st.op.name)
        op_type = int(st.op_type)
        compute_at = int(st.compute_at)
        dtype_bytes = 4
        try:
            if st.op.num_outputs > 0:
                dtype_bytes = _dtype_bytes(st.op.output(0).dtype)
        except Exception:
            pass
        iters: List[IterInfo] = []
        for it in st.iters:
            ext_val = 1
            if it.range is not None and getattr(it.range, "extent", None) is not None:
                try:
                    ext_val = get_const_int(it.range.extent)
                except Exception:
                    ext_val = 1
            iters.append(
                IterInfo(
                    name=str(it.name),
                    extent_expr=const_expr(ext_val),
                    annotation=int(it.annotation),
                    iter_kind=int(it.iter_kind),
                )
            )
        stages.append(
            StageInfo(
                op_name=op_name,
                op_type=op_type,
                compute_at=compute_at,
                scope=_stage_scope_from_name(op_name),
                dtype_bytes=dtype_bytes,
                iters=iters,
                compute_at_target=None,
                inline=(compute_at == 1),
            )
        )
    return stages


def _infer_bound_from_record(task, record: Dict[str, Any]):
    inp, _ = load_record_from_string(json.dumps(record))
    return task.compute_dag.infer_bound_from_state(inp.state)


def parse_steps(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_steps = record["i"][1][1]
    steps: List[Dict[str, Any]] = []
    for idx, s in enumerate(raw_steps):
        kind = s[0]
        d: Dict[str, Any] = {"idx": idx, "kind": kind, "raw": s}
        if kind == "SP":
            d.update(
                {
                    "stage_id": int(s[1]),
                    "iter_id": int(s[2]),
                    "extent": int(s[3]),
                    "lengths": [int(x) for x in s[4]],
                    "inner_to_outer": bool(int(s[5])),
                }
            )
        elif kind == "AN":
            d.update({"stage_id": int(s[1]), "iter_id": int(s[2]), "annotation": int(s[3])})
        elif kind == "FU":
            d.update({"stage_id": int(s[1]), "fused_ids": [int(x) for x in s[2]]})
        elif kind == "RE":
            d.update({"stage_id": int(s[1]), "after_ids": [int(x) for x in s[2]]})
        elif kind == "FSP":
            d.update(
                {
                    "stage_id": int(s[1]),
                    "iter_id": int(s[2]),
                    "src_step_id": int(s[3]),
                    "n_split": int(s[4]),
                }
            )
        elif kind == "FFSP":
            d.update(
                {
                    "stage_id": int(s[1]),
                    "iter_id": int(s[2]),
                    "src_step_ids": [int(x) for x in s[3]],
                    "level": int(s[4]),
                    "factor_or_nparts": bool(int(s[5])),
                }
            )
        elif kind == "PR":
            d.update({"stage_id": int(s[1]), "iter_id": int(s[2]), "pragma": str(s[3])})
        elif kind == "CA":
            d.update(
                {
                    "stage_id": int(s[1]),
                    "target_stage_id": int(s[2]),
                    "target_iter_id": int(s[3]),
                }
            )
        elif kind == "CI":
            d.update({"stage_id": int(s[1])})
        elif kind == "CR":
            d.update({"stage_id": int(s[1])})
        elif kind == "CHR":
            d.update(
                {
                    "stage_id": int(s[1]),
                    "scope_name": str(s[2]),
                    "reader_stage_ids": [int(x) for x in s[3]],
                }
            )
        elif kind == "CHW":
            d.update({"stage_id": int(s[1]), "scope_name": str(s[2])})
        elif kind == "SA":
            d.update(
                {
                    "stage_id": int(s[1]),
                    "iter_id": int(s[2]),
                    "factor": int(s[3]),
                    "offset": int(s[4]),
                }
            )
        steps.append(d)
    return steps


def _shift_compute_at_targets(stages: List[StageInfo], insert_at: int, delta: int) -> None:
    for st in stages:
        if st.compute_at_target is None:
            continue
        ts, ti = st.compute_at_target
        if ts >= insert_at:
            st.compute_at_target = (ts + delta, ti)


def _clone_iters_for_cache(src_stage: StageInfo) -> List[IterInfo]:
    if src_stage.iters:
        return [
            IterInfo(name=it.name, extent_expr=it.extent_expr, annotation=0, iter_kind=it.iter_kind)
            for it in src_stage.iters
        ]
    return [IterInfo(name="ax0", extent_expr=const_expr(1), annotation=0, iter_kind=0)]


def _apply_split(stage: StageInfo, iter_id: int, lengths: List[Expr], inner_to_outer: bool) -> None:
    if iter_id < 0 or iter_id >= len(stage.iters):
        return
    base_it = stage.iters[iter_id]
    tosplit = base_it.extent_expr
    outs: List[IterInfo] = []
    n = len(lengths)

    for i in range(n):
        if inner_to_outer:
            l = lengths[n - i - 1]
            nm = "%s.%d" % (base_it.name, n - i)
        else:
            l = lengths[i]
            nm = "%s.%d" % (base_it.name, i)
        outs.append(IterInfo(name=nm, extent_expr=l, annotation=0, iter_kind=base_it.iter_kind))
        tosplit = ceildiv_expr(tosplit, l)

    if inner_to_outer:
        outs.append(IterInfo(name="%s.0" % base_it.name, extent_expr=tosplit, annotation=0))
        outs = list(reversed(outs))
    else:
        outs.append(IterInfo(name="%s.%d" % (base_it.name, n), extent_expr=tosplit, annotation=0))

    stage.iters = stage.iters[:iter_id] + outs + stage.iters[iter_id + 1 :]


def simulate_steps(steps: List[Dict[str, Any]], dag_info: List[StageInfo]) -> SimContext:
    stages = copy.deepcopy(dag_info)
    sp_param_exprs: Dict[int, List[Expr]] = {}

    for step in steps:
        kind = step["kind"]
        if kind == "SP":
            stage_id = step["stage_id"]
            if stage_id < 0 or stage_id >= len(stages):
                continue
            n = len(step["lengths"])
            lengths = [param_expr(step["idx"], i) for i in range(n)]
            sp_param_exprs[step["idx"]] = lengths
            _apply_split(stages[stage_id], step["iter_id"], lengths, step["inner_to_outer"])
        elif kind == "AN":
            stage_id = step["stage_id"]
            iter_id = step["iter_id"]
            if 0 <= stage_id < len(stages) and 0 <= iter_id < len(stages[stage_id].iters):
                stages[stage_id].iters[iter_id].annotation = step["annotation"]
        elif kind == "FU":
            stage_id = step["stage_id"]
            if not (0 <= stage_id < len(stages)):
                continue
            fused_ids = list(step["fused_ids"])
            if not fused_ids:
                continue
            if min(fused_ids) < 0 or max(fused_ids) >= len(stages[stage_id].iters):
                continue
            iters = stages[stage_id].iters
            selected = [iters[i] for i in fused_ids]
            new_extent = mul_expr(*[it.extent_expr for it in selected])
            new_name = "@".join([it.name for it in selected])
            new_iter = IterInfo(name=new_name, extent_expr=new_extent, annotation=0)
            a = fused_ids[0]
            b = fused_ids[-1]
            stages[stage_id].iters = iters[:a] + [new_iter] + iters[b + 1 :]
        elif kind == "RE":
            stage_id = step["stage_id"]
            if not (0 <= stage_id < len(stages)):
                continue
            ids = step["after_ids"]
            iters = stages[stage_id].iters
            if len(ids) != len(iters):
                continue
            if min(ids) < 0 or max(ids) >= len(iters):
                continue
            stages[stage_id].iters = [iters[i] for i in ids]
        elif kind == "FSP":
            stage_id = step["stage_id"]
            iter_id = step["iter_id"]
            src = sp_param_exprs.get(step["src_step_id"], [])
            n_split = step["n_split"]
            if n_split <= 0:
                continue
            lengths: List[Expr] = []
            for i in range(max(0, n_split - 1)):
                lengths.append(src[i] if i < len(src) else const_expr(1))
            tail = src[n_split - 1 :]
            lengths.append(mul_expr(*tail) if tail else const_expr(1))
            if 0 <= stage_id < len(stages):
                _apply_split(stages[stage_id], iter_id, lengths, True)
        elif kind == "FFSP":
            stage_id = step["stage_id"]
            iter_id = step["iter_id"]
            level = step["level"]
            factors: List[Expr] = []
            for src_step_id in step["src_step_ids"]:
                src = sp_param_exprs.get(src_step_id, [])
                if 0 <= level < len(src):
                    factors.append(src[level])
            factor = mul_expr(*factors) if factors else const_expr(1)
            if 0 <= stage_id < len(stages):
                _apply_split(stages[stage_id], iter_id, [factor], step["factor_or_nparts"])
        elif kind == "CA":
            sid = step["stage_id"]
            if 0 <= sid < len(stages):
                stages[sid].compute_at = 2
                stages[sid].inline = False
                stages[sid].compute_at_target = (step["target_stage_id"], step["target_iter_id"])
        elif kind == "CR":
            sid = step["stage_id"]
            if 0 <= sid < len(stages):
                stages[sid].compute_at = 0
                stages[sid].inline = False
                stages[sid].compute_at_target = None
        elif kind == "CI":
            sid = step["stage_id"]
            if 0 <= sid < len(stages):
                stages[sid].compute_at = 1
                stages[sid].inline = True
                stages[sid].compute_at_target = None
        elif kind == "CHR":
            sid = step["stage_id"]
            if not (0 <= sid < len(stages)):
                continue
            src_stage = stages[sid]
            scope = step["scope_name"]
            new_name = "%s.%s" % (src_stage.op_name, scope)
            new_stage = StageInfo(
                op_name=new_name,
                op_type=1,
                compute_at=0,
                scope=scope,
                dtype_bytes=src_stage.dtype_bytes,
                iters=_clone_iters_for_cache(src_stage),
            )
            insert_at = sid + 1
            _shift_compute_at_targets(stages, insert_at, 1)
            stages.insert(insert_at, new_stage)
        elif kind == "CHW":
            sid = step["stage_id"]
            if not (0 <= sid < len(stages)):
                continue
            src_stage = stages[sid]
            scope = step["scope_name"]
            new_name = "%s.%s" % (src_stage.op_name, scope)
            new_stage = StageInfo(
                op_name=new_name,
                op_type=1,
                compute_at=0,
                scope=scope,
                dtype_bytes=src_stage.dtype_bytes,
                iters=_clone_iters_for_cache(src_stage),
            )
            insert_at = sid
            _shift_compute_at_targets(stages, insert_at, 1)
            stages.insert(insert_at, new_stage)
        else:
            # PR/SA and other no-op steps for this simulator.
            pass

    return SimContext(stages=stages, steps=steps, sp_param_exprs=sp_param_exprs, stage_size_exprs={})


def _calibrate_sim_context_with_bound(task, base_record: Dict[str, Any], sim_ctx: SimContext) -> SimContext:
    try:
        base_bound = _infer_bound_from_record(task, base_record)
    except Exception:
        return sim_ctx

    stages = sim_ctx.stages
    n_stage = min(len(base_bound.stages), len(stages))
    if n_stage <= 0:
        return sim_ctx

    base_steps = base_record["i"][1][1]
    base_extent: Dict[Tuple[int, int], int] = {}
    base_ann: Dict[Tuple[int, int], int] = {}
    valid_keys = set()
    for sid in range(n_stage):
        st = base_bound.stages[sid]
        n_iter = min(len(st.iters), len(stages[sid].iters))
        for iid in range(n_iter):
            it = st.iters[iid]
            ext = 1
            if it.range is not None and getattr(it.range, "extent", None) is not None:
                try:
                    ext = get_const_int(it.range.extent)
                except Exception:
                    ext = 1
            base_extent[(sid, iid)] = int(ext)
            base_ann[(sid, iid)] = int(it.annotation)
            valid_keys.add((sid, iid))

    params: List[Tuple[int, int]] = []
    for step in sim_ctx.steps:
        if step["kind"] != "SP":
            continue
        for pos in range(len(step["lengths"])):
            params.append((step["idx"], pos))

    iter_deps: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    stage_size_base: Dict[int, int] = {}
    stage_size_deps: Dict[int, List[Tuple[int, int]]] = {}

    for sid in range(n_stage):
        st = base_bound.stages[sid]
        n_iter = min(len(st.iters), len(stages[sid].iters))
        prod = 1
        for iid in range(n_iter):
            prod *= base_extent.get((sid, iid), 1)
        stage_size_base[sid] = int(prod * stages[sid].dtype_bytes)
    for step_idx, pos in params:
        cur = int(base_steps[step_idx][4][pos])
        probe = 2 if cur != 2 else 3
        rec_probe = copy.deepcopy(base_record)
        rec_probe["i"][1][1][step_idx][4][pos] = probe
        try:
            probe_bound = _infer_bound_from_record(task, rec_probe)
        except Exception:
            continue
        n_stage_probe = min(len(probe_bound.stages), n_stage)
        for sid in range(n_stage_probe):
            st = probe_bound.stages[sid]
            n_iter = min(len(st.iters), len(base_bound.stages[sid].iters), len(stages[sid].iters))
            prod = 1
            for iid in range(n_iter):
                if (sid, iid) not in valid_keys:
                    continue
                it = st.iters[iid]
                pext = 1
                if it.range is not None and getattr(it.range, "extent", None) is not None:
                    try:
                        pext = get_const_int(it.range.extent)
                    except Exception:
                        pext = 1
                if pext != base_extent[(sid, iid)]:
                    iter_deps.setdefault((sid, iid), []).append((step_idx, pos))
                prod *= pext
            probe_stage_size = int(prod * stages[sid].dtype_bytes)
            if sid in stage_size_base and probe_stage_size != stage_size_base[sid]:
                stage_size_deps.setdefault(sid, []).append((step_idx, pos))

    for sid, st in enumerate(stages[:n_stage]):
        for iid, it in enumerate(st.iters):
            key = (sid, iid)
            if key not in valid_keys:
                continue

            it.annotation = base_ann[key]

            target_base = base_extent.get(key, 1)
            try:
                cur_base = eval_expr(it.extent_expr, base_steps)
            except Exception:
                cur_base = None

            if cur_base == target_base:
                continue

            deps = iter_deps.get(key, [])
            if not deps:
                it.extent_expr = const_expr(target_base)
                continue

            dep_expr = mul_expr(*[param_expr(a, b) for a, b in deps])
            dep_base = eval_expr(dep_expr, base_steps)
            if dep_base == 0:
                it.extent_expr = const_expr(target_base)
            elif target_base % dep_base == 0:
                scale = target_base // dep_base
                it.extent_expr = mul_expr(const_expr(scale), dep_expr)
            else:
                # Fallback to base constant if the simple multiplicative model fails.
                it.extent_expr = const_expr(target_base)

    for sid in range(n_stage):
        base_sz = stage_size_base.get(sid)
        if base_sz is None:
            continue
        deps = stage_size_deps.get(sid, [])
        if not deps:
            sim_ctx.stage_size_exprs[sid] = const_expr(base_sz)
            continue
        dep_expr = mul_expr(*[param_expr(a, b) for a, b in deps])
        dep_base = eval_expr(dep_expr, base_steps)
        if dep_base == 0:
            sim_ctx.stage_size_exprs[sid] = const_expr(base_sz)
        elif base_sz % dep_base == 0:
            sim_ctx.stage_size_exprs[sid] = mul_expr(const_expr(base_sz // dep_base), dep_expr)
        else:
            sim_ctx.stage_size_exprs[sid] = const_expr(base_sz)

    return sim_ctx


def _kernel_root_id(stages: List[StageInfo], stage_id: int, memo: Dict[int, Optional[int]]) -> Optional[int]:
    if stage_id in memo:
        return memo[stage_id]
    cur = stage_id
    seen = set()
    while True:
        if cur in seen:
            memo[stage_id] = cur
            return cur
        seen.add(cur)
        if cur < 0 or cur >= len(stages):
            memo[stage_id] = stage_id
            return stage_id
        st = stages[cur]
        if st.compute_at != 2 or st.compute_at_target is None:
            memo[stage_id] = cur
            return cur
        parent_sid = st.compute_at_target[0]
        if parent_sid < 0 or parent_sid >= len(stages):
            memo[stage_id] = cur
            return cur
        cur = parent_sid


def _normalize_name(name: str) -> str:
    out = []
    for c in name:
        if c.isalnum() or c == "_":
            out.append(c)
        else:
            out.append("_")
    return "".join(out)


def _to_py_obj(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, tvm.runtime.String):
        return str(x)
    if isinstance(x, tvm.ir.container.Map):
        return {str(k): _to_py_obj(v) for k, v in x.items()}
    if isinstance(x, tvm.ir.container.Array):
        return [_to_py_obj(v) for v in x]
    return x


def _map_merge_groups_to_stage_ids(
    scope: str,
    alloc_groups: List[List[str]],
    kernel_stage_ids: List[int],
    stages: List[StageInfo],
) -> Tuple[List[List[int]], bool]:
    stage_ids_in_scope = [sid for sid in kernel_stage_ids if stages[sid].scope == scope]
    norm_to_stage: Dict[int, str] = {sid: _normalize_name(stages[sid].op_name) for sid in stage_ids_in_scope}

    mapped_groups: List[List[int]] = []
    ambiguous = False

    for group in alloc_groups:
        mapped: List[int] = []
        for alloc_name in group:
            an = _normalize_name(str(alloc_name))
            candidates = []
            for sid, sn in norm_to_stage.items():
                if an == sn or an.startswith(sn + "_") or sn.startswith(an + "_"):
                    candidates.append(sid)
            candidates = sorted(set(candidates))
            if len(candidates) == 1:
                mapped.append(candidates[0])
            elif len(candidates) > 1:
                ambiguous = True
            else:
                ambiguous = True
        mapped = sorted(set(mapped))
        if mapped:
            mapped_groups.append(mapped)

    covered = set()
    for g in mapped_groups:
        covered.update(g)
    for sid in stage_ids_in_scope:
        if sid not in covered:
            mapped_groups.append([sid])

    return mapped_groups, ambiguous


def extract_constraints(
    sim_ctx: SimContext,
    hw: Optional[Dict[str, int]] = None,
    merge_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    hw = dict(DEFAULT_HW if hw is None else hw)
    stages = sim_ctx.stages
    steps = sim_ctx.steps

    root_cache: Dict[int, Optional[int]] = {}
    kernels: Dict[int, Dict[str, Any]] = {}

    for sid, st in enumerate(stages):
        if st.inline or st.compute_at == 1:
            continue
        if st.op_type == 0:
            continue
        root = _kernel_root_id(stages, sid, root_cache)
        if root is None:
            continue
        k = kernels.setdefault(
            root,
            {
                "stage_ids": [],
                "thread_x_exprs": [],
                "thread_y_exprs": [],
                "thread_z_exprs": [],
                "vthread_exprs": [],
                "vector_checks": [],
                "shared_stage_sizes": {},
                "local_stage_sizes": {},
            },
        )
        k["stage_ids"].append(sid)

        for it in st.iters:
            if it.annotation == ANNOTATION_THREAD_X:
                k["thread_x_exprs"].append(it.extent_expr)
            elif it.annotation == ANNOTATION_THREAD_Y:
                k["thread_y_exprs"].append(it.extent_expr)
            elif it.annotation == ANNOTATION_THREAD_Z:
                k["thread_z_exprs"].append(it.extent_expr)
            elif it.annotation == ANNOTATION_VTHREAD:
                k["vthread_exprs"].append(it.extent_expr)
            elif it.annotation == ANNOTATION_VECTORIZE:
                k["vector_checks"].append(
                    {
                        "stage_id": sid,
                        "expr": it.extent_expr,
                        "dtype_bytes": st.dtype_bytes,
                    }
                )

        if sid in sim_ctx.stage_size_exprs:
            stage_size = sim_ctx.stage_size_exprs[sid]
        else:
            stage_extent = const_expr(1)
            for it in st.iters:
                stage_extent = mul_expr(stage_extent, it.extent_expr)
            stage_size = mul_expr(const_expr(st.dtype_bytes), stage_extent)

        if st.scope == "shared":
            k["shared_stage_sizes"][sid] = stage_size
        elif st.scope == "local":
            k["local_stage_sizes"][sid] = stage_size

    divisibility_constraints = []
    innermost_constraints = []
    for step in steps:
        if step["kind"] != "SP":
            continue
        sp_idx = int(step["idx"])
        extent = int(step["extent"])
        n = len(step["lengths"])
        params = [(sp_idx, i) for i in range(n)]
        if n > 0:
            innermost_constraints.append({"step_idx": sp_idx, "length_pos": n - 1})
        divisibility_constraints.append({"step_idx": sp_idx, "extent": extent, "params": params})

    report = _to_py_obj(merge_report or {})
    if not isinstance(report, dict):
        report = {}
    one_func_report = {}
    if report:
        first_key = sorted(report.keys())[0]
        val = report[first_key]
        if isinstance(val, dict):
            one_func_report = val

    for root, k in kernels.items():
        tx = k["thread_x_exprs"]
        ty = k["thread_y_exprs"]
        tz = k["thread_z_exprs"]
        vt = k["vthread_exprs"]

        x_axis = tx[0] if tx else const_expr(1)
        y_axis = ty[0] if ty else const_expr(1)
        z_axis = tz[0] if tz else const_expr(1)
        vthread_prod = mul_expr(*vt) if vt else const_expr(1)

        k["thread_per_block_expr"] = mul_expr(x_axis, y_axis, z_axis, vthread_prod)
        k["vthread_prod_expr"] = vthread_prod

        shared_sizes: Dict[int, Expr] = k["shared_stage_sizes"]
        local_sizes: Dict[int, Expr] = k["local_stage_sizes"]

        shared_groups = one_func_report.get("shared", []) if isinstance(one_func_report, dict) else []
        local_groups = one_func_report.get("local", []) if isinstance(one_func_report, dict) else []

        mapped_shared, shared_amb = _map_merge_groups_to_stage_ids(
            "shared", shared_groups, k["stage_ids"], stages
        )
        mapped_local, local_amb = _map_merge_groups_to_stage_ids(
            "local", local_groups, k["stage_ids"], stages
        )

        if shared_amb:
            k["shared_expr"] = add_expr(*list(shared_sizes.values())) if shared_sizes else const_expr(0)
            k["shared_merge_fallback"] = True
        else:
            parts = []
            for g in mapped_shared:
                exprs = [shared_sizes[sid] for sid in g if sid in shared_sizes]
                if exprs:
                    parts.append(max_expr(exprs))
            k["shared_expr"] = add_expr(*parts) if parts else const_expr(0)
            k["shared_merge_fallback"] = False

        if local_amb:
            k["local_expr"] = add_expr(*list(local_sizes.values())) if local_sizes else const_expr(0)
            k["local_merge_fallback"] = True
        else:
            parts = []
            for g in mapped_local:
                exprs = [local_sizes[sid] for sid in g if sid in local_sizes]
                if exprs:
                    parts.append(max_expr(exprs))
            k["local_expr"] = add_expr(*parts) if parts else const_expr(0)
            k["local_merge_fallback"] = False

    return {
        "hw": hw,
        "steps": steps,
        "kernels": kernels,
        "divisibility_constraints": divisibility_constraints,
        "innermost_constraints": innermost_constraints,
    }


def check_constraints(
    constraints: Dict[str, Any],
    record: Dict[str, Any],
    hw: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    hw = dict(constraints.get("hw", DEFAULT_HW)) if hw is None else dict(hw)
    steps = record["i"][1][1]

    violations: List[Dict[str, Any]] = []
    values: Dict[str, Any] = {"kernels": {}}

    for c in constraints["divisibility_constraints"]:
        prod = 1
        for step_idx, length_pos in c["params"]:
            s = steps[step_idx]
            if s[0] != "SP" or length_pos >= len(s[4]):
                continue
            prod *= int(s[4][length_pos])
        extent = int(c["extent"])
        if prod <= 0 or extent % prod != 0:
            violations.append(
                {
                    "type": "divisibility",
                    "step_idx": c["step_idx"],
                    "extent": extent,
                    "product": prod,
                }
            )

    max_inner = int(hw.get("max_innermost_split_factor", DEFAULT_HW["max_innermost_split_factor"]))
    for c in constraints["innermost_constraints"]:
        s = steps[c["step_idx"]]
        pos = c["length_pos"]
        if s[0] != "SP" or pos >= len(s[4]):
            continue
        val = int(s[4][pos])
        if val > max_inner:
            violations.append(
                {
                    "type": "innermost_split",
                    "step_idx": c["step_idx"],
                    "value": val,
                    "limit": max_inner,
                }
            )

    for root, k in constraints["kernels"].items():
        kv: Dict[str, Any] = {}

        x_vals = [eval_expr(e, steps) for e in k["thread_x_exprs"]]
        y_vals = [eval_expr(e, steps) for e in k["thread_y_exprs"]]
        z_vals = [eval_expr(e, steps) for e in k["thread_z_exprs"]]

        kv["thread_x_values"] = x_vals
        kv["thread_y_values"] = y_vals
        kv["thread_z_values"] = z_vals

        def _axis_check(axis_name: str, vals: List[int], max_v: int) -> None:
            if not vals:
                return
            if len(set(vals)) > 1:
                violations.append(
                    {
                        "type": "thread_axis_mismatch",
                        "kernel": int(root),
                        "axis": axis_name,
                        "values": vals,
                    }
                )
            axis_val = vals[0]
            if axis_val > max_v:
                violations.append(
                    {
                        "type": "thread_axis_limit",
                        "kernel": int(root),
                        "axis": axis_name,
                        "value": axis_val,
                        "limit": max_v,
                    }
                )

        _axis_check("x", x_vals, int(hw["max_thread_x"]))
        _axis_check("y", y_vals, int(hw["max_thread_y"]))
        _axis_check("z", z_vals, int(hw["max_thread_z"]))

        vthread_prod = eval_expr(k["vthread_prod_expr"], steps)
        kv["vthread_prod"] = vthread_prod
        if vthread_prod > int(hw["max_vthread"]):
            violations.append(
                {
                    "type": "vthread_limit",
                    "kernel": int(root),
                    "value": vthread_prod,
                    "limit": int(hw["max_vthread"]),
                }
            )

        thread_per_block = eval_expr(k["thread_per_block_expr"], steps)
        kv["thread_per_block"] = thread_per_block
        if thread_per_block > int(hw["max_threads_per_block"]):
            violations.append(
                {
                    "type": "thread_per_block",
                    "kernel": int(root),
                    "value": thread_per_block,
                    "limit": int(hw["max_threads_per_block"]),
                }
            )

        shared_val = eval_expr(k["shared_expr"], steps)
        local_val = eval_expr(k["local_expr"], steps)
        kv["shared_bytes"] = shared_val
        kv["local_bytes"] = local_val
        if shared_val > int(hw["max_shared_memory_per_block"]):
            violations.append(
                {
                    "type": "shared_memory",
                    "kernel": int(root),
                    "value": shared_val,
                    "limit": int(hw["max_shared_memory_per_block"]),
                    "merge_fallback": bool(k.get("shared_merge_fallback", False)),
                }
            )
        if local_val > int(hw["max_local_memory_per_block"]):
            violations.append(
                {
                    "type": "local_memory",
                    "kernel": int(root),
                    "value": local_val,
                    "limit": int(hw["max_local_memory_per_block"]),
                    "merge_fallback": bool(k.get("local_merge_fallback", False)),
                }
            )

        vector_rows = []
        for v in k["vector_checks"]:
            lanes = eval_expr(v["expr"], steps)
            bytes_used = lanes * int(v["dtype_bytes"])
            row = {"stage_id": int(v["stage_id"]), "lanes": lanes, "bytes": bytes_used}
            vector_rows.append(row)
            if bytes_used > int(hw["max_vector_bytes"]):
                violations.append(
                    {
                        "type": "vector_bytes",
                        "kernel": int(root),
                        "stage_id": int(v["stage_id"]),
                        "lanes": lanes,
                        "value": bytes_used,
                        "limit": int(hw["max_vector_bytes"]),
                    }
                )
        kv["vector_checks"] = vector_rows

        values["kernels"][int(root)] = kv

    return {
        "valid": len(violations) == 0,
        "violations": violations,
        "values": values,
    }


def check_constraints_prefilter(
    constraints: Dict[str, Any],
    record: Dict[str, Any],
    hw: Optional[Dict[str, int]] = None,
    mode: str = "relaxed",
) -> Dict[str, Any]:
    """Check constraints with a configurable hard-fail subset for generation."""
    res = check_constraints(constraints, record, hw)
    if mode == "strict":
        out = dict(res)
        out["mode"] = "strict"
        out["all_violations"] = list(res["violations"])
        return out
    if mode != "relaxed":
        raise ValueError("Unknown prefilter mode: %s" % mode)

    hard_types = {"divisibility", "innermost_split", "thread_axis_limit"}
    hard_violations = [v for v in res["violations"] if v.get("type") in hard_types]
    return {
        "valid": len(hard_violations) == 0,
        "violations": hard_violations,
        "all_violations": list(res["violations"]),
        "values": res["values"],
        "mode": "relaxed",
    }


def build_system(
    base_record: Dict[str, Any],
    task,
    hw: Optional[Dict[str, int]] = None,
    merge_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    hw_use = dict(DEFAULT_HW if hw is None else hw)
    steps = parse_steps(base_record)
    dag_info = build_dag_info(task)
    sim_ctx = simulate_steps(steps, dag_info)
    # Probing-based formula calibration is disabled.
    # sim_ctx = _calibrate_sim_context_with_bound(task, base_record, sim_ctx)
    constraints = extract_constraints(sim_ctx, hw_use, merge_report)
    return {
        "base_record": base_record,
        "hw": hw_use,
        "steps": steps,
        "sim_ctx": sim_ctx,
        "constraints": constraints,
    }


def evaluate_record(system: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    return check_constraints(system["constraints"], record, system["hw"])


def build_task_map(tasks_pkl_path: str) -> Dict[str, Any]:
    with open(tasks_pkl_path, "rb") as f:
        tasks, _ = pickle.load(f)
    return {t.workload_key: t for t in tasks}


def record_to_task_state(record: Dict[str, Any], task_map: Dict[str, Any]):
    line = json.dumps(record)
    inp, _ = load_record_from_string(line)
    wk = inp.task.workload_key
    if wk not in task_map:
        raise KeyError("workload_key not found in task map: %s" % wk)
    return task_map[wk], inp.state


_s2m = tvm.get_global_func("driver.schedule_to_module")
GPU_PASSES = tvm.transform.Sequential(
    [
        tir.transform.InjectPrefetch(),
        tir.transform.StorageFlatten(64, False),
        tir.transform.NarrowDataType(32),
        tir.transform.Simplify(),
        tir.transform.VectorizeLoop(True),
        tir.transform.InjectVirtualThread(),
        tir.transform.StorageRewrite(),
        tir.transform.Simplify(),
    ]
)


def lower_with_gpu_passes(task, state):
    sch, tensors = task.compute_dag.apply_steps_from_state(state)
    mod = _s2m(sch, tensors, "main", {})
    return GPU_PASSES(mod)


def verify_gpu_module(mod, hw: Optional[Dict[str, int]] = None) -> bool:
    hw_use = dict(DEFAULT_HW if hw is None else hw)
    verify = tvm.get_global_func("tir.analysis.verify_gpu_code")
    constraints = {
        "max_shared_memory_per_block": int(hw_use["max_shared_memory_per_block"]),
        "max_local_memory_per_block": int(hw_use["max_local_memory_per_block"]),
        "max_threads_per_block": int(hw_use["max_threads_per_block"]),
        "max_thread_x": int(hw_use["max_thread_x"]),
        "max_thread_y": int(hw_use["max_thread_y"]),
        "max_thread_z": int(hw_use["max_thread_z"]),
        "max_vthread": int(hw_use["max_vthread"]),
        "max_vector_bytes": int(hw_use["max_vector_bytes"]),
        "max_kernels": 2**31 - 1,
    }
    for _, f in mod.functions.items():
        if isinstance(f, tvm.tir.PrimFunc):
            if not verify(f, constraints):
                return False
    return True


def get_storage_rewrite_merge_report(task, state) -> Dict[str, Any]:
    try:
        from tvm.tir import analysis as tir_analysis

        tir_analysis.clear_storage_rewrite_report()
        _ = lower_with_gpu_passes(task, state)
        rep = tir_analysis.get_storage_rewrite_report()
        py_rep = _to_py_obj(rep)
        if isinstance(py_rep, dict):
            return py_rep
    except Exception:
        pass
    return {}


def get_divisors(n: int) -> List[int]:
    out = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            out.append(i)
            if i * i != n:
                out.append(n // i)
        i += 1
    return sorted(out)


def _extract_sp_and_unroll_knobs(record: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[int]]:
    sp_knobs: List[Dict[str, Any]] = []
    unroll_steps: List[int] = []
    steps = record["i"][1][1]
    for step_idx, s in enumerate(steps):
        if s[0] == "SP":
            sp_knobs.append(
                {
                    "step_idx": int(step_idx),
                    "extent": int(s[3]),
                    "n_split": len(s[4]),
                }
            )
        elif s[0] == "PR" and "auto_unroll_max_step" in str(s[3]):
            unroll_steps.append(int(step_idx))
    return sp_knobs, unroll_steps


def _parse_auto_unroll_value(pragma: Any) -> Optional[int]:
    s = str(pragma)
    if not s.startswith("auto_unroll_max_step$"):
        return None
    try:
        return int(s.split("$", 1)[1])
    except Exception:
        return None


def _sample_split_lengths(
    extent: int,
    n_split: int,
    max_innermost_split_factor: int,
    max_candidates: int,
    rng: random.Random,
) -> List[List[int]]:
    if n_split <= 0:
        return [[]]

    out: List[List[int]] = []
    cur = [1] * n_split

    def dfs(pos: int, rem: int) -> None:
        if len(out) >= max_candidates:
            return
        if pos == n_split - 1:
            if rem <= max_innermost_split_factor:
                cur[pos] = int(rem)
                out.append(list(cur))
            return
        divs = get_divisors(rem)
        rng.shuffle(divs)
        for d in divs:
            cur[pos] = int(d)
            dfs(pos + 1, rem // d)
            if len(out) >= max_candidates:
                return

    dfs(0, int(extent))
    return out


def _set_split_and_unroll_params(
    record: Dict[str, Any],
    sp_params: Dict[int, List[int]],
    unroll_params: Dict[int, int],
) -> Dict[str, Any]:
    rec = copy.deepcopy(record)
    steps = rec["i"][1][1]
    for step_idx, vals in sp_params.items():
        s = steps[int(step_idx)]
        if s[0] == "SP":
            s[4] = [int(x) for x in vals]
    for step_idx, v in unroll_params.items():
        s = steps[int(step_idx)]
        if s[0] == "PR":
            s[3] = "auto_unroll_max_step$%d" % int(v)
    return rec


def generate_valid_params(
    system: Dict[str, Any],
    base_record: Dict[str, Any],
    n: int = 1,
    rng: Optional[random.Random] = None,
    max_attempts: int = 1000,
    max_innermost_split_factor: int = DEFAULT_HW["max_innermost_split_factor"],
    max_candidates_per_split: int = 256,
    auto_unroll_configs: Sequence[int] = AUTO_UNROLL_CONFIGS,
    prefilter_mode: str = "relaxed",
    include_base: bool = True,
    return_stats: bool = False,
    base_split_bias: float = 0.6,
    base_unroll_bias: float = 0.7,
) -> Any:
    """Generate valid split/unroll parameters under current formula constraints.

    Returns a list of dicts:
      {
        "sp_params": {step_idx: [lengths...]},
        "unroll_params": {step_idx: value},
        "record": mutated_record
      }
    """
    if rng is None:
        rng = random.Random()
    target_n = max(1, int(n))
    max_attempts = max(1, int(max_attempts))
    max_candidates_per_split = max(1, int(max_candidates_per_split))

    sp_knobs, unroll_steps = _extract_sp_and_unroll_knobs(base_record)
    base_steps = base_record["i"][1][1]
    base_sp_params: Dict[int, List[int]] = {}
    base_unroll_params: Dict[int, int] = {}
    for k in sp_knobs:
        step_idx = int(k["step_idx"])
        base_sp_params[step_idx] = [int(x) for x in base_steps[step_idx][4]]
    for step_idx in unroll_steps:
        v = _parse_auto_unroll_value(base_steps[step_idx][3])
        if v is not None:
            base_unroll_params[int(step_idx)] = int(v)

    split_domains: Dict[int, List[List[int]]] = {}
    for k in sp_knobs:
        step_idx = int(k["step_idx"])
        cand = _sample_split_lengths(
            extent=int(k["extent"]),
            n_split=int(k["n_split"]),
            max_innermost_split_factor=int(max_innermost_split_factor),
            max_candidates=int(max_candidates_per_split),
            rng=rng,
        )
        base_cand = base_sp_params.get(step_idx)
        if base_cand is not None and base_cand not in cand:
            cand = [base_cand] + cand
        if not cand:
            empty = {
                "rows": [],
                "stats": {
                    "prefilter_mode": prefilter_mode,
                    "attempts": 0,
                    "accepted": 0,
                    "duplicate_skips": 0,
                    "prefilter_reject_by_type": {},
                },
            }
            return empty if return_stats else []
        split_domains[step_idx] = cand

    unroll_domain: Dict[int, List[int]] = {}
    for step_idx in unroll_steps:
        vals = [int(x) for x in auto_unroll_configs]
        base_v = base_unroll_params.get(step_idx)
        if base_v is not None and base_v not in vals:
            vals = [base_v] + vals
        unroll_domain[step_idx] = vals

    def _params_key(sp_params: Dict[int, List[int]], unroll_params: Dict[int, int]) -> Tuple[Any, Any]:
        return (
            tuple((k, tuple(v)) for k, v in sorted(sp_params.items())),
            tuple(sorted(unroll_params.items())),
        )

    stats: Dict[str, Any] = {
        "prefilter_mode": prefilter_mode,
        "attempts": 0,
        "accepted": 0,
        "duplicate_skips": 0,
        "prefilter_reject_by_type": {},
    }
    results: List[Dict[str, Any]] = []
    seen = set()
    per_goal_attempts = max_attempts * target_n

    if include_base:
        base_key = _params_key(base_sp_params, base_unroll_params)
        seen.add(base_key)
        pf = check_constraints_prefilter(system["constraints"], base_record, system["hw"], mode=prefilter_mode)
        if pf["valid"]:
            results.append(
                {
                    "sp_params": {int(k): [int(x) for x in v] for k, v in sorted(base_sp_params.items())},
                    "unroll_params": {int(k): int(v) for k, v in sorted(base_unroll_params.items())},
                    "record": copy.deepcopy(base_record),
                    "source": "base",
                }
            )
            stats["accepted"] += 1
        else:
            for v in pf["violations"]:
                t = str(v.get("type", "unknown"))
                stats["prefilter_reject_by_type"][t] = stats["prefilter_reject_by_type"].get(t, 0) + 1
        if len(results) >= target_n:
            out = {"rows": results, "stats": stats}
            return out if return_stats else results

    for _ in range(per_goal_attempts):
        if len(results) >= target_n:
            break
        stats["attempts"] += 1

        sp_params: Dict[int, List[int]] = {}
        for step_idx, cand in split_domains.items():
            use_base = bool(base_sp_params) and rng.random() < float(base_split_bias)
            if use_base and step_idx in base_sp_params:
                sp_params[int(step_idx)] = list(base_sp_params[step_idx])
            else:
                sp_params[int(step_idx)] = list(rng.choice(cand))

        unroll_params: Dict[int, int] = {}
        for step_idx in unroll_steps:
            use_base = bool(base_unroll_params) and rng.random() < float(base_unroll_bias)
            if use_base and step_idx in base_unroll_params:
                unroll_params[int(step_idx)] = int(base_unroll_params[step_idx])
            else:
                unroll_params[int(step_idx)] = int(rng.choice(unroll_domain[step_idx]))

        key = _params_key(sp_params, unroll_params)
        if key in seen:
            stats["duplicate_skips"] += 1
            continue
        seen.add(key)

        rec = _set_split_and_unroll_params(base_record, sp_params, unroll_params)
        pf = check_constraints_prefilter(system["constraints"], rec, system["hw"], mode=prefilter_mode)
        if pf["valid"]:
            results.append(
                {
                    "sp_params": {int(k): [int(x) for x in v] for k, v in sorted(sp_params.items())},
                    "unroll_params": {int(k): int(v) for k, v in sorted(unroll_params.items())},
                    "record": rec,
                    "source": "mutated",
                }
            )
            stats["accepted"] += 1
        else:
            for v in pf["violations"]:
                t = str(v.get("type", "unknown"))
                stats["prefilter_reject_by_type"][t] = stats["prefilter_reject_by_type"].get(t, 0) + 1

    out = {"rows": results, "stats": stats}
    return out if return_stats else results


def randomize_record_params(
    record: Dict[str, Any],
    rng: Optional[random.Random] = None,
    max_innermost_split_factor: int = DEFAULT_HW["max_innermost_split_factor"],
    auto_unroll_configs: Sequence[int] = AUTO_UNROLL_CONFIGS,
) -> Dict[str, Any]:
    if rng is None:
        rng = random.Random()
    rec = copy.deepcopy(record)
    steps = rec["i"][1][1]

    for s in steps:
        if s[0] == "SP":
            extent = int(s[3])
            n = len(s[4])
            lengths = []
            rem = extent
            for i in range(n):
                divs = get_divisors(rem)
                if i == n - 1:
                    divs = [d for d in divs if d <= max_innermost_split_factor] or [1]
                v = int(rng.choice(divs))
                lengths.append(v)
                rem //= v
            s[4] = lengths
        elif s[0] == "PR" and "auto_unroll_max_step" in str(s[3]):
            v = int(rng.choice(list(auto_unroll_configs)))
            s[3] = "auto_unroll_max_step$%d" % v
    return rec


def generate_valid_record(
    system: Dict[str, Any],
    base_record: Dict[str, Any],
    rng: Optional[random.Random] = None,
    max_attempts: int = 300,
) -> Optional[Dict[str, Any]]:
    rows = generate_valid_params(
        system=system,
        base_record=base_record,
        n=1,
        rng=rng,
        max_attempts=max_attempts,
        prefilter_mode="strict",
    )
    if not rows:
        return None
    return rows[0]["record"]


__all__ = [
    "DEFAULT_HW",
    "AUTO_UNROLL_CONFIGS",
    "parse_steps",
    "simulate_steps",
    "extract_constraints",
    "check_constraints",
    "check_constraints_prefilter",
    "build_system",
    "evaluate_record",
    "build_task_map",
    "record_to_task_state",
    "lower_with_gpu_passes",
    "verify_gpu_module",
    "get_storage_rewrite_merge_report",
    "generate_valid_params",
    "randomize_record_params",
    "generate_valid_record",
    "expr_to_str",
]
