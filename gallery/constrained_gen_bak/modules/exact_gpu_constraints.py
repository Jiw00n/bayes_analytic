"""
exact_gpu_constraints — exact symbolic shared/vector/vthread/max_threads constraints.

This module materializes an exact case table from symbolic pre-vectorize TIR.
Each case corresponds to a concrete tuple of vectorized loop extents, while the
case expressions remain symbolic in the split knobs (`sp_*`).
"""

from itertools import product

import tvm

from .expr_nodes import CaseSplitNode, ConstNode, MaxNode, PrimExprNode, parse_expr_tree


_LOWER_SYMBOLIC_PRE_VECTORIZE = tvm.get_global_func(
    "constrained_gen.lower_symbolic_pre_vectorize"
)
_LIST_VECTORIZED_LOOP_EXTENTS = tvm.get_global_func(
    "constrained_gen.list_vectorized_loop_extents"
)
_LOWER_SYMBOLIC_POST_VECTORIZE = tvm.get_global_func(
    "constrained_gen.lower_symbolic_post_vectorize"
)
_EXTRACT_GPU_CASE_STATS = tvm.get_global_func("constrained_gen.extract_gpu_case_stats")
_EXTRACT_ALL_GPU_CASE_STATS = tvm.get_global_func(
    "constrained_gen.extract_all_gpu_case_stats"
)


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


def _collect_projected_gpu_metadata(pre_func):
    analyzer = tvm.arith.Analyzer()
    runtime_domains = {}
    vector_scalar_bytes = []

    def add_domain(name, lo_expr, hi_expr):
        runtime_domains.setdefault(
            name,
            (
                PrimExprNode(analyzer.simplify(lo_expr)),
                PrimExprNode(analyzer.simplify(hi_expr)),
            ),
        )

    def collect_vector_loop_scalar_bytes(stmt):
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
        vector_scalar_bytes.append(max_bytes)

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
            if stmt.kind == tvm.tir.ForKind.VECTORIZED:
                collect_vector_loop_scalar_bytes(stmt)
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
    return vector_scalar_bytes, runtime_domains


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


def _expr_node_to_primexpr(node):
    if isinstance(node, PrimExprNode):
        return node.expr
    if isinstance(node, ConstNode):
        return tvm.tir.IntImm("int32", node.val)
    raise TypeError(f"Unsupported ExprNode for PrimExpr conversion: {type(node)!r}")


def _has_noninteger_var(expr):
    found = False

    def visit(node):
        nonlocal found
        if isinstance(node, tvm.tir.Var):
            dtype = str(node.dtype)
            if not (dtype.startswith("int") or dtype.startswith("uint") or dtype == "bool"):
                found = True

    tvm.tir.stmt_functor.post_order_visit(expr, visit)
    return found


def _project_runtime_upper(expr_node, runtime_domains):
    if not isinstance(expr_node, PrimExprNode):
        return expr_node

    analyzer = tvm.arith.Analyzer()
    dom_map = {}
    for name, (lo_expr, hi_expr) in runtime_domains.items():
        var = expr_node._var_map.get(name)
        if var is None:
            continue
        dom_map[var] = tvm.arith.IntervalSet(
            _expr_node_to_primexpr(lo_expr),
            _expr_node_to_primexpr(hi_expr),
        )

    if not dom_map:
        return PrimExprNode(analyzer.simplify(expr_node.expr))

    interval = analyzer.int_set(expr_node.expr, dom_map)
    projected = analyzer.simplify(interval.max_value)
    if not _has_noninteger_var(projected) and "pos_inf" not in str(projected) and "neg_inf" not in str(projected):
        return PrimExprNode(projected)

    subst = {}
    for name, (_, hi_expr) in runtime_domains.items():
        var = expr_node._var_map.get(name)
        if var is not None:
            subst[var] = _expr_node_to_primexpr(hi_expr)
    projected = analyzer.simplify(tvm.tir.stmt_functor.substitute(expr_node.expr, subst))
    if not _has_noninteger_var(projected) and "pos_inf" not in str(projected) and "neg_inf" not in str(projected):
        return PrimExprNode(projected)

    return ConstNode(1 << 60)


def _wrap_primexpr(expr):
    return PrimExprNode(tvm.arith.Analyzer().simplify(expr))


def _validate_projected_free_vars(node, constraint_name, allowed_var_names):
    if allowed_var_names is None:
        return

    unexpected = sorted(set(node.variables()) - set(allowed_var_names))
    if not unexpected:
        return

    raise RuntimeError(
        f"Projected {constraint_name} constraint still has non-symbolic free vars: "
        f"{unexpected}. expr={node!r}"
    )


def _collapse_max(children):
    if not children:
        return ConstNode(0)
    if len(children) == 1:
        return children[0]
    return MaxNode(children)


def _dedupe_nodes(nodes):
    deduped = []
    seen = set()
    for node in nodes:
        key = repr(node)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(node)
    return deduped


def _to_pruning_expr_node(node, constraint_name):
    if isinstance(node, ConstNode):
        return node
    if isinstance(node, MaxNode):
        return MaxNode(
            [_to_pruning_expr_node(child, constraint_name) for child in node.children]
        )
    if isinstance(node, PrimExprNode):
        text = str(node).replace("T.min(", "min(").replace("T.max(", "max(")
        try:
            return parse_expr_tree(text)
        except ValueError as err:
            raise RuntimeError(
                f"Unsupported projected {constraint_name} expression for pruning: {text}"
            ) from err
    raise TypeError(f"Unsupported projected {constraint_name} node type: {type(node)!r}")


def build_projected_gpu_context(sym_state):
    if sym_state._state is None:
        raise RuntimeError("SymbolicState._state is required for projected GPU constraint lowering")

    pre_func = _LOWER_SYMBOLIC_PRE_VECTORIZE(sym_state.compute_dag, sym_state._state)
    selector_exprs = list(_LIST_VECTORIZED_LOOP_EXTENTS(pre_func))
    if len(selector_exprs) > 2:
        raise RuntimeError(
            f"Unsupported sketch with {len(selector_exprs)} vectorized loops; expected at most 2"
        )

    selectors = [PrimExprNode(expr) for expr in selector_exprs]
    vector_scalar_bytes, runtime_domains = _collect_projected_gpu_metadata(pre_func)
    if len(vector_scalar_bytes) < len(selectors):
        vector_scalar_bytes.extend([1] * (len(selectors) - len(vector_scalar_bytes)))

    return {
        "pre_func": pre_func,
        "selectors": selectors,
        "vector_scalar_bytes": vector_scalar_bytes,
        "runtime_domains": runtime_domains,
    }


def build_projected_vectorize_constraint_node(projection_context, hw, allowed_var_names=None):
    runtime_domains = projection_context["runtime_domains"]
    analyzer = tvm.arith.Analyzer()
    selectors = projection_context["selectors"]

    if selectors:
        vector_terms = []
        for selector, scalar_bytes in zip(
            selectors,
            projection_context["vector_scalar_bytes"],
        ):
            projected_selector = _project_runtime_upper(selector, runtime_domains)
            if isinstance(projected_selector, PrimExprNode):
                expr = analyzer.simplify(projected_selector.expr * int(scalar_bytes))
                vector_terms.append(_wrap_primexpr(expr))
            else:
                vector_terms.append(ConstNode(projected_selector.val * int(scalar_bytes)))
    else:
        vector_terms = [ConstNode(0)]

    vector_node = _collapse_max(_dedupe_nodes(vector_terms))
    _validate_projected_free_vars(vector_node, "vectorize", allowed_var_names)
    return vector_node


def build_projected_shared_memory_constraint_node(
    projection_context,
    hw,
    allowed_var_names=None,
):
    runtime_domains = projection_context["runtime_domains"]
    analyzer = tvm.arith.Analyzer()
    total_shared_bytes = tvm.tir.IntImm("int32", 0)

    def visit(node):
        nonlocal total_shared_bytes
        if not isinstance(node, tvm.tir.Allocate):
            return
        storage_scope = getattr(node.buffer_var.type_annotation, "storage_scope", "")
        if str(storage_scope) != "shared":
            return

        alloc_count = tvm.tir.IntImm("int32", 1)
        for extent in node.extents:
            alloc_count = analyzer.simplify(alloc_count * extent)

        dtype = tvm.DataType(node.dtype)
        elem_bytes = dtype.bits // 8
        total_shared_bytes = analyzer.simplify(
            total_shared_bytes + alloc_count * int(elem_bytes * dtype.lanes)
        )

    tvm.tir.stmt_functor.post_order_visit(projection_context["pre_func"].body, visit)
    shared_node = _project_runtime_upper(PrimExprNode(total_shared_bytes), runtime_domains)
    _validate_projected_free_vars(shared_node, "shared_memory", allowed_var_names)
    return shared_node


def build_exact_constraint_nodes(
    sym_state,
    hw,
    sp_groups,
    sp_extents,
    innermost_names,
    innermost_limit,
    projected_context=None,
):
    if projected_context is None:
        projected_context = build_projected_gpu_context(sym_state)

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

    case_stats_list = [
        list(stats)
        for stats in _EXTRACT_ALL_GPU_CASE_STATS(
            pre_func,
            [list(map(int, values)) for values in vector_cases],
        )
    ]
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


def build_projected_constraint_nodes(exact_nodes, hw, allowed_var_names=None):
    runtime_domains = exact_nodes["runtime_domains"]

    shared_terms = _dedupe_nodes(
        [_project_runtime_upper(case["expr"], runtime_domains) for case in exact_nodes["shared_node"].cases]
    )
    max_vthread_terms = _dedupe_nodes(
        [_project_runtime_upper(case["expr"], runtime_domains) for case in exact_nodes["max_vthread_node"].cases]
    )

    vector_node = build_projected_vectorize_constraint_node(
        {
            "selectors": exact_nodes["selectors"],
            "vector_scalar_bytes": exact_nodes["vector_scalar_bytes"],
            "runtime_domains": runtime_domains,
        },
        hw,
        allowed_var_names=allowed_var_names,
    )
    shared_node = _collapse_max(shared_terms)
    max_vthread_node = _collapse_max(_dedupe_nodes(max_vthread_terms))
    max_vthread_node = _to_pruning_expr_node(max_vthread_node, "max_vthread")

    _validate_projected_free_vars(shared_node, "shared_memory", allowed_var_names)
    _validate_projected_free_vars(max_vthread_node, "max_vthread", allowed_var_names)

    return {
        "vector_node": vector_node,
        "shared_node": shared_node,
        "max_vthread_node": max_vthread_node,
    }


def lower_symbolic_post_vectorize_case(pre_func, selector_values):
    return _LOWER_SYMBOLIC_POST_VECTORIZE(pre_func, list(map(int, selector_values)))
