"""Per-tensor, per-dim semantic labels derived from ``task.compute_dag``.

For each tensor in the compute DAG we emit one label per dim:

- **ComputeOp tensors** with matching axis count → use ``op.axis[i].var.name``
  (e.g. ``n``, ``h``, ``w``, ``co``).
- **PlaceholderOp tensors** → walk the DAG bodies, find every ProducerLoad that
  reads this placeholder, and for each dim collect the iter-var names used in
  the index expression. A single var → that var's name; multiple vars →
  sorted and joined with ``+``.

Both paths derive labels from op-level identifiers (iter_var names), which are
fixed by the op code and do *not* change when tensor shapes change. Records
sharing the same compute DAG structure therefore share the same label list.
"""

from __future__ import annotations

from typing import List, Set


def _extract_var_names(expr) -> List[str]:
    """Collect iter-var-like Var names referenced inside a TIR expression.

    We keep the post-order visit ordering for determinism, then dedupe.
    """
    from tvm import tir

    names: List[str] = []

    def _visit(node):
        if isinstance(node, tir.Var):
            names.append(str(node.name))

    tir.stmt_functor.post_order_visit(expr, _visit)
    seen: Set[str] = set()
    deduped: List[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def _placeholder_dim_labels(dag, placeholder_tensor) -> List[str]:
    """Label each dim of ``placeholder_tensor`` by the iter-vars that index it."""
    from tvm import tir

    rank = len(placeholder_tensor.shape)
    per_dim: List[Set[str]] = [set() for _ in range(rank)]
    target_name = str(placeholder_tensor.op.name)
    visited_names: Set[str] = set()

    def _walk(tensor) -> None:
        op = tensor.op
        op_name = str(op.name)
        if op_name in visited_names:
            return
        visited_names.add(op_name)
        body_seq = getattr(op, "body", None) or ()
        for body in body_seq:
            def _visit(node, _per_dim=per_dim, _target=target_name):
                if isinstance(node, tir.ProducerLoad):
                    producer = getattr(node, "producer", None)
                    producer_op = getattr(producer, "op", None)
                    if producer_op is not None and str(producer_op.name) == _target:
                        for dim_idx, idx_expr in enumerate(node.indices):
                            for var_name in _extract_var_names(idx_expr):
                                _per_dim[dim_idx].add(var_name)
            tir.stmt_functor.post_order_visit(body, _visit)
        for input_tensor in getattr(op, "input_tensors", []) or []:
            _walk(input_tensor)

    for tensor in dag.tensors:
        _walk(tensor)

    labels: List[str] = []
    for dim_idx, names in enumerate(per_dim):
        if not names:
            labels.append(f"{placeholder_tensor.name}_d{dim_idx}")
        elif len(names) == 1:
            labels.append(next(iter(names)))
        else:
            labels.append("+".join(sorted(names)))
    return labels


def semantic_labels_for_task(task) -> List[List[str]]:
    """Return ``labels[tensor_idx][dim_idx]`` for every tensor in the DAG."""
    dag = task.compute_dag
    labels: List[List[str]] = []
    for tensor in dag.tensors:
        rank = len(tensor.shape)
        op = tensor.op
        op_type = type(op).__name__
        if op_type == "PlaceholderOp":
            tensor_labels = _placeholder_dim_labels(dag, tensor)
            if len(tensor_labels) != rank:
                tensor_labels = [f"{tensor.name}_d{dim_idx}" for dim_idx in range(rank)]
        else:
            axis_seq = getattr(op, "axis", None)
            if axis_seq is not None and len(axis_seq) == rank:
                tensor_labels = [str(ax.var.name) for ax in axis_seq]
            else:
                tensor_labels = [f"{tensor.name}_d{dim_idx}" for dim_idx in range(rank)]
        labels.append(tensor_labels)
    return labels


def flatten_labels(labels: List[List[str]]) -> List[str]:
    flat: List[str] = []
    for tensor_idx, tensor_labels in enumerate(labels):
        for dim_idx, name in enumerate(tensor_labels):
            flat.append(f"t{tensor_idx}:{name}")
    return flat
