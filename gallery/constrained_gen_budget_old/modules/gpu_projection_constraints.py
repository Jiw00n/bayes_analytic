"""
gpu_projection_constraints — projected GPU pruning constraints derived from symbolic TIR.

"""

import tvm

from .expr_nodes import ConstNode, MaxNode, PrimExprNode, parse_expr_tree


_LOWER_SYMBOLIC_PRE_VECTORIZE = tvm.get_global_func(
    "constrained_gen.lower_symbolic_pre_vectorize"
)
_LIST_VECTORIZED_LOOP_EXTENTS = tvm.get_global_func(
    "constrained_gen.list_vectorized_loop_extents"
)


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

    def visit_stmt(stmt, block_scope_ord=()):
        if isinstance(stmt, tvm.tir.SeqStmt):
            for child in stmt.seq:
                visit_stmt(child, block_scope_ord)
            return

        if isinstance(stmt, tvm.tir.AttrStmt):
            if stmt.attr_key in ("thread_extent", "virtual_thread"):
                iter_var = stmt.node
                if hasattr(iter_var, "var"):
                    var_name = str(iter_var.var.name)
                    add_domain(
                        var_name,
                        tvm.tir.IntImm("int32", 0),
                        analyzer.simplify(stmt.value - 1),
                    )
            visit_stmt(stmt.body, block_scope_ord)
            return

        if isinstance(stmt, tvm.tir.For):
            add_domain(
                str(stmt.loop_var.name),
                stmt.min,
                analyzer.simplify(stmt.min + stmt.extent - 1),
            )
            if stmt.kind == tvm.tir.ForKind.VECTORIZED:
                collect_vector_loop_scalar_bytes(stmt)
            visit_stmt(stmt.body, block_scope_ord)
            return

        if isinstance(stmt, tvm.tir.LetStmt):
            add_domain(str(stmt.var.name), stmt.value, stmt.value)
            visit_stmt(stmt.body, block_scope_ord)
            return

        if isinstance(stmt, tvm.tir.IfThenElse):
            visit_stmt(stmt.then_case, block_scope_ord)
            if stmt.else_case is not None:
                visit_stmt(stmt.else_case, block_scope_ord)
            return

        if isinstance(stmt, tvm.tir.While):
            visit_stmt(stmt.body, block_scope_ord)
            return

        if isinstance(stmt, tvm.tir.Allocate):
            visit_stmt(stmt.body, block_scope_ord)
            return

        block_realize = getattr(tvm.tir, "BlockRealize", None)
        if block_realize is not None and isinstance(stmt, block_realize):
            visit_stmt(stmt.block, block_scope_ord)
            return

        block = getattr(tvm.tir, "Block", None)
        if block is not None and isinstance(stmt, block):
            if stmt.init is not None:
                visit_stmt(stmt.init, block_scope_ord)
            visit_stmt(stmt.body, block_scope_ord)

    visit_stmt(pre_func.body)
    return vector_scalar_bytes, runtime_domains


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


def build_projected_gpu_context(sym_state):
    """심볼 state를 pre-vectorize TIR로 낮춘 뒤 selector·runtime 도메인 등 projection 컨텍스트를 반환한다."""
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
    """projection 컨텍스트로부터 vectorize 상한 제약용 ExprNode 트리를 만든다."""
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
    """projection 컨텍스트로부터 shared memory 상한 제약용 ExprNode를 만든다."""
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




# ------------------------------------------------------------------
# Deprecated
# ------------------------------------------------------------------
