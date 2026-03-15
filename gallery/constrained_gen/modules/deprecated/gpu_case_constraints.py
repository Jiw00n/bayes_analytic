"""
gpu_case_constraints — exact symbolic shared/vector/vthread/max_threads constraints.

This module owns exact case-table materialization and post-vectorize exact
lowering helpers. Projected pruning helpers now live in
`gpu_projection_constraints.py`.
"""

from itertools import product

import tvm

from ..expr_nodes import CaseSplitNode, ConstNode, PrimExprNode
from ..gpu_projection_constraints import build_projected_gpu_context as _build_projected_gpu_context
_LOWER_SYMBOLIC_POST_VECTORIZE = tvm.get_global_func(
    "constrained_gen.lower_symbolic_post_vectorize"
)
_EXTRACT_GPU_CASE_STATS = tvm.get_global_func("constrained_gen.extract_gpu_case_stats")
_EXTRACT_ALL_GPU_CASE_STATS = tvm.get_global_func(
    "constrained_gen.extract_all_gpu_case_stats"
)


# Process-local reuse for repeated exact-case extraction on structurally
# identical pre-vectorize TIR and the same selector-case table.
_EXACT_GPU_CASE_STATS_CACHE = {}


# ------------------------------------------------------------------
# Deprecated
# ------------------------------------------------------------------


def _normalize_vector_cases(vector_cases):
    return tuple(tuple(int(v) for v in values) for values in vector_cases)


def _extract_all_gpu_case_stats_cached(pre_func, vector_cases):
    normalized_cases = _normalize_vector_cases(vector_cases)
    if not normalized_cases:
        return []

    cache_key = (str(pre_func), normalized_cases)
    cached_stats = _EXACT_GPU_CASE_STATS_CACHE.get(cache_key)
    if cached_stats is not None:
        return [list(stats) for stats in cached_stats]

    extracted_stats = tuple(
        tuple(stats)
        for stats in _EXTRACT_ALL_GPU_CASE_STATS(
            pre_func,
            [list(values) for values in normalized_cases],
        )
    )
    _EXACT_GPU_CASE_STATS_CACHE[cache_key] = extracted_stats
    return [list(stats) for stats in extracted_stats]


def _divisors(n):
    if n <= 0:
        return [1]
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)


def _enumerate_group_assignments(step_idx, group_names, sp_extents, innermost_names, innermost_limit):
    extent = sp_extents.get(step_idx)
    if extent is None:
        return [{}]

    results = []

    def dfs(name_idx, remaining, current):
        name = group_names[name_idx]
        candidates = _divisors(remaining)
        if name in innermost_names:
            candidates = [c for c in candidates if c <= innermost_limit]
        for chosen in candidates:
            current[name] = chosen
            if name_idx + 1 == len(group_names):
                results.append(dict(current))
            else:
                next_remaining = (remaining + chosen - 1) // chosen
                dfs(name_idx + 1, next_remaining, current)
            current.pop(name, None)

    dfs(0, extent, {})
    return results


def _build_sp_domains(sp_groups, sp_extents, innermost_names, innermost_limit):
    domains = {}
    for step_idx, names in sp_groups.items():
        extent = sp_extents.get(step_idx)
        if extent is None:
            continue
        for name in names:
            hi = extent
            if name in innermost_names:
                hi = min(hi, innermost_limit)
            domains[name] = [1, hi]
    return domains


def _collect_runtime_extent_exprs(pre_func):
    analyzer = tvm.arith.Analyzer()
    extent_exprs = {}

    def visit(node):
        if not isinstance(node, tvm.tir.AttrStmt):
            return
        if node.attr_key not in ("thread_extent", "virtual_thread"):
            return
        iter_var = node.node
        if not hasattr(iter_var, "var"):
            return
        name = str(iter_var.var.name)
        extent_exprs.setdefault(name, analyzer.simplify(node.value))

    tvm.tir.stmt_functor.post_order_visit(pre_func.body, visit)
    return extent_exprs


def _collect_vector_loop_scalar_bytes(pre_func):
    scalar_bytes = []

    def walk(stmt):
        if isinstance(stmt, tvm.tir.For) and stmt.kind == tvm.tir.ForKind.VECTORIZED:
            max_bytes = 1

            def visit(node):
                nonlocal max_bytes
                dtype = None
                if isinstance(node, tvm.tir.Allocate):
                    dtype = tvm.DataType(node.dtype)
                elif isinstance(node, tvm.tir.Cast):
                    dtype = tvm.DataType(node.dtype)
                elif isinstance(node, tvm.tir.BufferLoad):
                    dtype = tvm.DataType(node.dtype)
                elif isinstance(node, tvm.tir.BufferStore):
                    dtype = tvm.DataType(node.value.dtype)
                if dtype is not None and dtype.lanes == 1:
                    max_bytes = max(max_bytes, dtype.bits // 8)

            tvm.tir.stmt_functor.post_order_visit(stmt.body, visit)
            scalar_bytes.append(max_bytes)
        for child in stmt.body.seq if isinstance(stmt, tvm.tir.SeqStmt) else []:
            walk(child)

    def visit_stmt(stmt):
        if isinstance(stmt, tvm.tir.SeqStmt):
            for child in stmt.seq:
                visit_stmt(child)
            return
        walk(stmt)
        if isinstance(stmt, tvm.tir.AttrStmt):
            visit_stmt(stmt.body)
        elif isinstance(stmt, tvm.tir.For):
            visit_stmt(stmt.body)
        elif isinstance(stmt, tvm.tir.LetStmt):
            visit_stmt(stmt.body)
        elif isinstance(stmt, tvm.tir.IfThenElse):
            visit_stmt(stmt.then_case)
            if stmt.else_case is not None:
                visit_stmt(stmt.else_case)
        elif isinstance(stmt, tvm.tir.While):
            visit_stmt(stmt.body)
        elif isinstance(stmt, tvm.tir.Allocate):
            visit_stmt(stmt.body)

    visit_stmt(pre_func.body)
    return scalar_bytes


def _build_runtime_domains(pre_func):
    analyzer = tvm.arith.Analyzer()
    runtime_domains = {}

    def add_domain(name, lo_expr, hi_expr):
        runtime_domains.setdefault(
            name,
            (
                PrimExprNode(analyzer.simplify(lo_expr)),
                PrimExprNode(analyzer.simplify(hi_expr)),
            ),
        )

    def visit_stmt(stmt):
        if isinstance(stmt, tvm.tir.SeqStmt):
            for child in stmt.seq:
                visit_stmt(child)
            return

        if isinstance(stmt, tvm.tir.AttrStmt):
            if stmt.attr_key in ("thread_extent", "virtual_thread"):
                iter_var = stmt.node
                if hasattr(iter_var, "var"):
                    add_domain(
                        str(iter_var.var.name),
                        tvm.tir.IntImm("int32", 0),
                        analyzer.simplify(stmt.value - 1),
                    )
            visit_stmt(stmt.body)
            return

        if isinstance(stmt, tvm.tir.For):
            add_domain(
                str(stmt.loop_var.name),
                stmt.min,
                analyzer.simplify(stmt.min + stmt.extent - 1),
            )
            visit_stmt(stmt.body)
            return

        if isinstance(stmt, tvm.tir.LetStmt):
            add_domain(str(stmt.var.name), stmt.value, stmt.value)
            visit_stmt(stmt.body)
            return

        if isinstance(stmt, tvm.tir.IfThenElse):
            visit_stmt(stmt.then_case)
            if stmt.else_case is not None:
                visit_stmt(stmt.else_case)
            return

        if isinstance(stmt, tvm.tir.While):
            visit_stmt(stmt.body)
            return

        if isinstance(stmt, tvm.tir.Allocate):
            visit_stmt(stmt.body)
            return

        block_realize = getattr(tvm.tir, "BlockRealize", None)
        if block_realize is not None and isinstance(stmt, block_realize):
            visit_stmt(stmt.block)
            return

        block = getattr(tvm.tir, "Block", None)
        if block is not None and isinstance(stmt, block):
            if stmt.init is not None:
                visit_stmt(stmt.init)
            visit_stmt(stmt.body)

    visit_stmt(pre_func.body)
    return runtime_domains


def _enumerate_selector_value_tuples(selector_nodes, sp_groups, sp_extents, innermost_names, innermost_limit, runtime_domains, max_case_value):
    if not selector_nodes:
        return [tuple()]

    domains = _build_sp_domains(sp_groups, sp_extents, innermost_names, innermost_limit)
    resolved = dict(domains)
    for name, (lo_expr, hi_expr) in runtime_domains.items():
        lo = lo_expr.interval(resolved)[0]
        hi = hi_expr.interval(resolved)[1]
        if hi < lo:
            hi = lo
        resolved[name] = [lo, hi]

    ranges = []
    for selector in selector_nodes:
        _, hi = selector.interval(resolved)
        hi = max(1, min(int(hi), int(max_case_value)))
        ranges.append(range(1, hi + 1))

    return [tuple(values) for values in product(*ranges)]


def _make_case_node(selectors, cases, default_limit, runtime_domains):
    return CaseSplitNode(
        selectors,
        cases,
        default=ConstNode(int(default_limit) + 1),
        extra_domains=runtime_domains,
    )


def build_exact_constraint_nodes(
    sym_state,
    hw,
    sp_groups,
    sp_extents,
    innermost_names,
    innermost_limit,
    projected_context=None,
):
    """exact GPU 케이스 테이블(vector/shared/max_vthread/max_threads)을 담은 노드 딕셔너리를 만든다."""
    if projected_context is None:
        projected_context = _build_projected_gpu_context(sym_state)

    pre_func = projected_context["pre_func"]
    selectors = projected_context["selectors"]
    vector_scalar_bytes = list(projected_context["vector_scalar_bytes"])
    runtime_domains = projected_context["runtime_domains"]
    vector_cases = _enumerate_selector_value_tuples(
        selectors,
        sp_groups,
        sp_extents,
        innermost_names,
        innermost_limit,
        runtime_domains,
        hw["max_vector_bytes"],
    )

    vector_case_nodes = []
    shared_case_nodes = []
    max_vthread_case_nodes = []
    max_threads_case_nodes = []
    case_expr_vars = {
        "vectorize": set(),
        "shared_memory": set(),
        "max_vthread": set(),
        "max_threads": set(),
    }

    case_stats_list = _extract_all_gpu_case_stats_cached(pre_func, vector_cases)
    if len(case_stats_list) != len(vector_cases):
        raise RuntimeError(
            "Expected one GPU case stats row per selector case. "
            f"cases={len(vector_cases)} stats={len(case_stats_list)}"
        )

    for values, stats in zip(vector_cases, case_stats_list):
        if len(stats) != 4:
            raise RuntimeError(f"Expected 4 case stats, got {len(stats)}")

        shared_expr, max_vthread_expr, max_threads_expr, max_vector_bytes = stats
        vector_case_nodes.append(
            {
                "values": tuple(int(v) for v in values),
                "expr": ConstNode(int(max_vector_bytes)),
            }
        )
        case_expr_vars["vectorize"] |= vector_case_nodes[-1]["expr"].variables()
        shared_case_nodes.append(
            {
                "values": tuple(int(v) for v in values),
                "expr": PrimExprNode(shared_expr),
            }
        )
        case_expr_vars["shared_memory"] |= shared_case_nodes[-1]["expr"].variables()
        max_vthread_case_nodes.append(
            {
                "values": tuple(int(v) for v in values),
                "expr": PrimExprNode(max_vthread_expr),
            }
        )
        case_expr_vars["max_vthread"] |= max_vthread_case_nodes[-1]["expr"].variables()
        max_threads_case_nodes.append(
            {
                "values": tuple(int(v) for v in values),
                "expr": PrimExprNode(max_threads_expr),
            }
        )
        case_expr_vars["max_threads"] |= max_threads_case_nodes[-1]["expr"].variables()

    raw_vector_node = _make_case_node(
        selectors, vector_case_nodes, hw["max_vector_bytes"], runtime_domains
    )
    raw_shared_node = _make_case_node(
        selectors,
        shared_case_nodes,
        hw["max_shared_memory_per_block"],
        runtime_domains,
    )
    raw_max_vthread_node = _make_case_node(
        selectors,
        max_vthread_case_nodes,
        hw["max_vthread_extent"],
        runtime_domains,
    )
    raw_max_threads_node = _make_case_node(
        selectors,
        max_threads_case_nodes,
        hw["max_threads_per_block"],
        runtime_domains,
    )

    return {
        "pre_func": pre_func,
        "selectors": selectors,
        "vector_scalar_bytes": vector_scalar_bytes,
        "runtime_domains": runtime_domains,
        "vector_node": raw_vector_node,
        "shared_node": raw_shared_node,
        "max_vthread_node": raw_max_vthread_node,
        "max_threads_node": raw_max_threads_node,
        "vector_cases": vector_cases,
        "case_expr_vars": case_expr_vars,
    }
def lower_symbolic_post_vectorize_case(pre_func, selector_values):
    """selector 값 조합에 해당하는 한 케이스로 post-vectorize TIR PrimFunc를 낮춰 반환한다."""
    return _LOWER_SYMBOLIC_POST_VECTORIZE(pre_func, list(map(int, selector_values)))
