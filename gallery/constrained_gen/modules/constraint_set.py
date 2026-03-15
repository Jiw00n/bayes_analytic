import re

from .sym_types import ANNOTATION_STR, CA_INLINED, CA_ITER
from .expr_nodes import (
    AddNode,
    CaseSplitNode,
    CeilDivNode,
    ConstNode,
    MaxNode,
    MinNode,
    MulNode,
    PrimExprNode,
    ScaleMulNode,
    SubNode,
    SumNode,
    VarNode,
    parse_expr_tree,
)
from .deprecated.gpu_case_constraints import build_exact_constraint_nodes
from .gpu_projection_constraints import (
    build_projected_constraint_nodes,
    build_projected_gpu_context,
    build_projected_shared_memory_constraint_node,
    build_projected_vectorize_constraint_node,
)


class ConstraintSet:
    def __init__(self, gen):
        """ScheduleGenerator를 받아 제약 묶음·검사 로직을 담당한다."""
        self.gen = gen

    # ------------------------------------------------------------------
    # Constraint-family bundle builders
    # ------------------------------------------------------------------

    def _build_vectorize_constraints(self):
        g = self.gen
        if g._vectorize_constraint_bundle is not None:
            return g._vectorize_constraint_bundle
        self._ensure_projected_gpu_constraints(('vectorize',))
        tree = g._projected_gpu['vector_node']
        limit = g.hw['max_vector_bytes']
        desc = "vectorize: runtime-projected selector upper bound ≤ max_vector_bytes"
        items = []
        for idx, child in enumerate(self._flatten_max_terms(tree)):
            items.append({
                'tree': child,
                'limit': limit,
                'desc': f"vectorize term {idx + 1}: runtime-projected selector upper bound ≤ max_vector_bytes",
            })
        g._vectorize_constraint_bundle = {
            'tree': tree,
            'limit': limit,
            'desc': desc,
            'items': self._dedupe_constraint_items(items, self._vectorize_item_key),
        }
        return g._vectorize_constraint_bundle

    # ------------------------------------------------------------------
    # Constraint-family pruning checks
    # ------------------------------------------------------------------

    def _check_vectorize(self, sym_map=None):
        g = self.gen
        if sym_map is None:
            sym_map = g.s.sym_map
        c = self._build_vectorize_constraints()
        violations = []
        params = g._normalize_concrete_params(sym_map)
        concrete_result = None
        for item in c['items']:
            val = item['tree'].evaluate(sym_map)
            if val > item['limit']:
                if params is not None:
                    if concrete_result is None:
                        concrete_result = g._get_concrete_final_result(params)
                    if concrete_result is not None and bool(concrete_result.get('ok')):
                        continue
                violations.append(f"{item['desc']}: actual={val}")
        return violations

    def _check_vectorize_exact(self, sym_map=None):
        g = self.gen
        if sym_map is None:
            sym_map = g.s.sym_map
        self._ensure_exact_gpu_constraints()
        total = self._exact_upper_bound(g._exact_gpu['vector_node'], sym_map)
        if total is None:
            return []
        if total > g.hw['max_vector_bytes']:
            return [
                "vectorize: exact vector bytes upper bound ≤ limit: "
                f"actual={total}"
            ]
        return []

    def _build_shared_memory_constraints(self):
        g = self.gen
        if g._shared_memory_constraint_bundle is not None:
            return g._shared_memory_constraint_bundle
        self._ensure_projected_gpu_constraints(('shared_memory',))
        g._shared_memory_constraint_bundle = {
            'tree': g._projected_gpu['shared_node'],
            'limit': g.hw['max_shared_memory_per_block'],
            'desc': "shared_memory: runtime-projected shared bytes upper bound ≤ limit",
        }
        return g._shared_memory_constraint_bundle

    def _check_shared_memory(self, sym_map=None):
        g = self.gen
        if sym_map is None:
            sym_map = g.s.sym_map
        c = self._build_shared_memory_constraints()
        total = c['tree'].evaluate(sym_map)
        if total > c['limit']:
            params = g._normalize_concrete_params(sym_map)
            if params is not None:
                concrete_result = g._get_concrete_final_result(params)
                if concrete_result is not None and bool(concrete_result.get('ok')):
                    return []
            return [f"{c['desc']}: actual={total}"]
        return []

    def _check_shared_memory_exact(self, sym_map=None):
        g = self.gen
        if sym_map is None:
            sym_map = g.s.sym_map
        self._ensure_exact_gpu_constraints()
        total = self._exact_upper_bound(g._exact_gpu['shared_node'], sym_map)
        if total is None:
            return []
        if total > g.hw['max_shared_memory_per_block']:
            return [
                "shared_memory: exact shared bytes upper bound ≤ limit: "
                f"actual={total}"
            ]
        return []

    def _build_max_threads_constraints(self):
        g = self.gen
        if g._max_threads_constraint_bundle is not None:
            return g._max_threads_constraint_bundle
        thread_items = self._collect_thread_binding_axes()
        thread_items = self._canonicalize_thread_binding_axes(thread_items)

        vthread_items = self._collect_vthread_binding_axes()
        vthread_items = self._canonicalize_thread_binding_axes(vthread_items)

        items = []
        for block_scope, scoped_items in self._group_binding_items_by_block_scope(
            thread_items + vthread_items
        ):
            total_item = self._build_thread_per_block_constraint_item(block_scope, scoped_items)
            if total_item is not None:
                items.append(total_item)
        items.extend(thread_items)
        g._max_threads_constraint_bundle = {
            'items': self._dedupe_constraint_items(items, self._thread_extent_item_key)
        }
        return g._max_threads_constraint_bundle

    def _check_max_threads(self, sym_map=None):
        g = self.gen
        if sym_map is None:
            sym_map = g.s.sym_map
        c = self._build_max_threads_constraints()
        if not c['items']:
            return []
        violations = []
        params = g._normalize_concrete_params(sym_map)
        concrete_result = None
        for item in c['items']:
            val = item['tree'].evaluate(sym_map)
            if not isinstance(val, int):
                violations.append(f"{item['desc']}: actual={item['axis_name']}={val}")
                continue
            if val > item['limit']:
                if item['axis_name'] == 'threads per block' and params is not None:
                    if concrete_result is None:
                        concrete_result = g._get_concrete_final_result(params)
                    if concrete_result is not None and bool(concrete_result.get('ok')):
                        continue
                violations.append(f"{item['desc']}: actual={val}")
        return violations

    def _check_max_threads_exact(self, sym_map=None):
        g = self.gen
        if sym_map is None:
            sym_map = g.s.sym_map

        violations = []
        for item in self._build_max_threads_constraints()['items']:
            if item['axis_name'] == 'threads per block':
                continue
            val = item['tree'].evaluate(sym_map)
            if not isinstance(val, int):
                violations.append(f"{item['desc']}: actual={item['axis_name']}={val}")
                continue
            if val > item['limit']:
                violations.append(f"{item['desc']}: actual={val}")

        self._ensure_exact_gpu_constraints()
        total = self._exact_upper_bound(g._exact_gpu['max_threads_node'], sym_map)
        if total is None:
            return violations
        if total > g.hw['max_threads_per_block']:
            violations.append(
                "max_threads: exact threads per block upper bound ≤ limit: "
                f"actual={total}"
            )
        return violations

    def _collect_thread_binding_axes(self):
        g = self.gen
        axis_items = []
        for ann in (6, 8, 10):
            candidates = []
            for sid, stage in enumerate(g.s.stages):
                if stage.compute_at == CA_INLINED:
                    continue
                for iid, it in enumerate(stage.iters):
                    if it.annotation != ann or it.extent is None:
                        continue
                    candidates.append({
                        'stage_id': sid,
                        'iter_id': iid,
                        'annotation': ann,
                        'block_scope': self._resolve_block_scope(sid, iid),
                        'op_name': stage.op_name,
                        'compute_at': stage.compute_at,
                        'iter_name': it.name,
                        'sym_extent': it.extent,
                        'tree': parse_expr_tree(str(it.extent)),
                    })
            if not candidates:
                continue
            axis_name = ANNOTATION_STR[ann]
            limit = self._thread_axis_limit(ann)
            for candidate in candidates:
                anchor_stage_id, anchor_iter_id = self._resolve_thread_axis_anchor(
                    candidate['stage_id'], candidate['iter_id'], ann
                )
                axis_items.append({
                    **candidate,
                    'axis_name': axis_name,
                    'anchor_stage_id': anchor_stage_id,
                    'anchor_iter_id': anchor_iter_id,
                    'limit': limit,
                    'desc': (
                        f"{axis_name} extent "
                        f"{candidate['op_name']}:{candidate['iter_name']} ≤ {limit}"
                    ),
                })
        return axis_items

    def _collect_vthread_binding_axes(self):
        g = self.gen
        axis_items = []
        ann = 4
        for sid, stage in enumerate(g.s.stages):
            if stage.compute_at == CA_INLINED:
                continue
            for iid, it in enumerate(stage.iters):
                if it.annotation != ann or it.extent is None:
                    continue
                anchor_stage_id, anchor_iter_id = self._resolve_thread_axis_anchor(sid, iid, ann)
                axis_items.append({
                    'stage_id': sid,
                    'iter_id': iid,
                    'annotation': ann,
                    'block_scope': self._resolve_block_scope(sid, iid),
                    'axis_name': ANNOTATION_STR[ann],
                    'op_name': stage.op_name,
                    'compute_at': stage.compute_at,
                    'iter_name': it.name,
                    'sym_extent': it.extent,
                    'tree': parse_expr_tree(str(it.extent)),
                    'anchor_stage_id': anchor_stage_id,
                    'anchor_iter_id': anchor_iter_id,
                    'limit': g.hw['max_vthread_extent'],
                    'desc': (
                        f"{ANNOTATION_STR[ann]} extent "
                        f"{stage.op_name}:{it.name} ≤ {g.hw['max_vthread_extent']}"
                    ),
                })
        return axis_items

    def _resolve_thread_axis_anchor(self, stage_id, iter_id, ann):
        g = self.gen
        cur_stage_id = stage_id
        search_iid = iter_id - 1

        while True:
            stage = g.s.stages[cur_stage_id]
            for iid in range(search_iid, -1, -1):
                if stage.iters[iid].annotation == ann:
                    return (cur_stage_id, iid)
            if stage.compute_at != CA_ITER or stage.attach_stage_id is None:
                break
            search_iid = stage.attach_iter_id
            cur_stage_id = stage.attach_stage_id

        return (stage_id, iter_id)

    def _resolve_block_scope(self, stage_id, iter_id):
        g = self.gen
        scope_rev = []
        seen = set()
        cur_stage_id = stage_id
        search_iid = iter_id - 1

        while True:
            stage = g.s.stages[cur_stage_id]
            for iid in range(search_iid, -1, -1):
                ann = stage.iters[iid].annotation
                if ann not in (5, 7, 9):
                    continue
                anchor_stage_id, anchor_iter_id = self._resolve_thread_axis_anchor(
                    cur_stage_id, iid, ann
                )
                entry = (ANNOTATION_STR[ann], anchor_stage_id, anchor_iter_id)
                if entry in seen:
                    continue
                seen.add(entry)
                scope_rev.append(entry)
            if stage.compute_at != CA_ITER or stage.attach_stage_id is None:
                break
            search_iid = stage.attach_iter_id
            cur_stage_id = stage.attach_stage_id

        return tuple(reversed(scope_rev))

    @staticmethod
    def _binding_item_order_key(item):
        ann_order = {4: 0, 6: 1, 8: 2, 10: 3}
        return (
            ann_order.get(item.get('annotation'), 99),
            item['stage_id'],
            item['iter_id'],
        )

    @staticmethod
    def _group_binding_items_by_block_scope(items):
        grouped = {}
        order = []
        for item in items:
            key = item.get('block_scope', tuple())
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(item)
        return [(key, grouped[key]) for key in order]

    def _canonicalize_block_scope_binding_items(self, items):
        grouped = {}
        order = []
        for item in sorted(items, key=self._binding_item_order_key):
            key = item['axis_name']
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(item)

        canonical = []
        for key in order:
            group = grouped[key]
            chosen = min(group, key=self._thread_extent_preference_key)
            canonical.append(chosen)
        return canonical

    @staticmethod
    def _format_block_scope(block_scope):
        if not block_scope:
            return "kernel root"
        return " > ".join(
            f"{axis}@s{stage_id}.i{iter_id}"
            for axis, stage_id, iter_id in block_scope
        )

    def _build_thread_per_block_constraint_item(self, block_scope, scoped_items):
        g = self.gen
        factors = []
        for item in self._canonicalize_block_scope_binding_items(scoped_items):
            factors.append({
                'axis_name': item['axis_name'],
                'sym_extent': item['sym_extent'],
                'tree': item['tree'],
            })

        if not factors:
            return None

        tree = None
        for factor in factors:
            tree = factor['tree'] if tree is None else MulNode(tree, factor['tree'])

        sym_extent = g._format_expr(tree, top_level=True)
        factor_label = " * ".join(factor['axis_name'] for factor in factors)
        scope_label = self._format_block_scope(block_scope)
        return {
            'axis_name': 'threads per block',
            'sym_extent': sym_extent,
            'tree': tree,
            'limit': g.hw['max_threads_per_block'],
            'desc': (
                f"threads per block under {scope_label} ({factor_label}) "
                f"≤ {g.hw['max_threads_per_block']}"
            ),
            'block_scope': block_scope,
        }

    def _canonicalize_thread_binding_axes(self, items):
        grouped = {}
        group_order = []
        for item in items:
            key = (
                item['axis_name'],
                item.get('block_scope', tuple()),
                item['anchor_stage_id'],
                item['anchor_iter_id'],
            )
            if key not in grouped:
                grouped[key] = []
                group_order.append(key)
            grouped[key].append(item)

        canonical_items = []
        for key in group_order:
            group = grouped[key]
            chosen = min(group, key=self._thread_extent_preference_key)
            alias_count = len(group)
            alias_entries = self._build_thread_alias_entries(group, chosen)
            canonical_items.append({
                **chosen,
                'alias_group_size': alias_count,
                'alias_entries': alias_entries,
                'desc': (
                    chosen['desc']
                    if alias_count == 1
                    else f"{chosen['desc']} (canonicalized from {alias_count} aliases)"
                ),
            })
        return canonical_items

    @staticmethod
    def _build_thread_alias_entries(group, chosen):
        chosen_id = (chosen['stage_id'], chosen['iter_id'])
        ordered = [chosen] + [
            item for item in group
            if (item['stage_id'], item['iter_id']) != chosen_id
        ]
        entries = []
        for idx, item in enumerate(ordered):
            entries.append({
                'label': f"{item['op_name']}:{item['iter_name']}",
                'expr': str(item['sym_extent']),
                'is_canonical': idx == 0,
            })
        return entries

    def _thread_extent_preference_key(self, item):
        tree = item['tree']
        pure_product_vars = self._extract_pure_product_vars(tree)
        exotic_nodes, node_count = self._thread_tree_metrics(tree)
        return (
            0 if pure_product_vars is not None else 1,
            exotic_nodes,
            node_count,
            len(tree.variables()),
            len(repr(tree)),
            item['stage_id'],
            item['iter_id'],
        )

    def _thread_tree_metrics(self, node):
        if isinstance(node, (ConstNode, VarNode, PrimExprNode)):
            exotic = 0 if isinstance(node, (ConstNode, VarNode)) else 1
            return exotic, 1
        if isinstance(node, MulNode):
            left_exotic, left_nodes = self._thread_tree_metrics(node.left)
            right_exotic, right_nodes = self._thread_tree_metrics(node.right)
            return left_exotic + right_exotic, left_nodes + right_nodes + 1
        if isinstance(node, (AddNode, SubNode, MinNode, CeilDivNode)):
            left_exotic, left_nodes = self._thread_tree_metrics(node.left)
            right_exotic, right_nodes = self._thread_tree_metrics(node.right)
            return left_exotic + right_exotic + 1, left_nodes + right_nodes + 1
        if isinstance(node, ScaleMulNode):
            child_exotic, child_nodes = self._thread_tree_metrics(node.child)
            return child_exotic + 1, child_nodes + 1
        if isinstance(node, (SumNode, MaxNode)):
            exotic = 1
            node_count = 1
            for child in node.children:
                child_exotic, child_nodes = self._thread_tree_metrics(child)
                exotic += child_exotic
                node_count += child_nodes
            return exotic, node_count
        if isinstance(node, CaseSplitNode):
            exotic = 1
            node_count = 1
            for selector in node.selectors:
                child_exotic, child_nodes = self._thread_tree_metrics(selector)
                exotic += child_exotic
                node_count += child_nodes
            for case in node.cases:
                child_exotic, child_nodes = self._thread_tree_metrics(case['expr'])
                exotic += child_exotic
                node_count += child_nodes
            child_exotic, child_nodes = self._thread_tree_metrics(node.default)
            return exotic + child_exotic, node_count + child_nodes
        return 1, 1

    def _extract_pure_product_vars(self, node):
        if isinstance(node, VarNode):
            return [node.name]
        if isinstance(node, ConstNode):
            return [] if node.val == 1 else None
        if isinstance(node, MulNode):
            left = self._extract_pure_product_vars(node.left)
            right = self._extract_pure_product_vars(node.right)
            if left is None or right is None:
                return None
            return left + right
        return None

    def _coerce_product_form_tree(self, node):
        if isinstance(node, PrimExprNode):
            text = str(node).replace("T.min(", "min(").replace("T.max(", "max(")
            try:
                node = parse_expr_tree(text)
            except ValueError:
                return None
        return self._normalize_legal_product_tree(node)

    def _normalize_legal_product_tree(self, node):
        g = self.gen
        if isinstance(node, MinNode):
            left = self._normalize_legal_product_tree(node.left)
            right = self._normalize_legal_product_tree(node.right)
            if isinstance(left, VarNode) and left.name in g._vthread_clamped_sp_names:
                return left
            if isinstance(right, VarNode) and right.name in g._vthread_clamped_sp_names:
                return right
            return MinNode(left, right)
        if isinstance(node, MulNode):
            return MulNode(
                self._normalize_legal_product_tree(node.left),
                self._normalize_legal_product_tree(node.right),
            )
        if isinstance(node, AddNode):
            return AddNode(
                self._normalize_legal_product_tree(node.left),
                self._normalize_legal_product_tree(node.right),
            )
        if isinstance(node, SubNode):
            return SubNode(
                self._normalize_legal_product_tree(node.left),
                self._normalize_legal_product_tree(node.right),
            )
        if isinstance(node, CeilDivNode):
            return CeilDivNode(
                self._normalize_legal_product_tree(node.left),
                self._normalize_legal_product_tree(node.right),
            )
        if isinstance(node, ScaleMulNode):
            return ScaleMulNode(
                self._normalize_legal_product_tree(node.child),
                node.scale,
            )
        if isinstance(node, SumNode):
            return SumNode([self._normalize_legal_product_tree(child) for child in node.children])
        if isinstance(node, MaxNode):
            return MaxNode([self._normalize_legal_product_tree(child) for child in node.children])
        return node

    def _extract_product_form_vars(self, node):
        node = self._coerce_product_form_tree(node)
        if node is None:
            return None
        if isinstance(node, VarNode):
            return [node.name]
        if isinstance(node, ConstNode):
            return []
        if isinstance(node, ScaleMulNode):
            return self._extract_product_form_vars(node.child)
        if isinstance(node, MulNode):
            left = self._extract_product_form_vars(node.left)
            right = self._extract_product_form_vars(node.right)
            if left is None or right is None:
                return None
            return left + right
        return None

    def _extract_product_form_meta(self, node):
        node = self._coerce_product_form_tree(node)
        if node is None:
            return None
        if isinstance(node, VarNode):
            return {'factors': (node.name,), 'scale': 1}
        if isinstance(node, ConstNode):
            return {'factors': tuple(), 'scale': node.val}
        if isinstance(node, ScaleMulNode):
            child = self._extract_product_form_meta(node.child)
            if child is None:
                return None
            return {
                'factors': child['factors'],
                'scale': child['scale'] * node.scale,
            }
        if isinstance(node, MulNode):
            left = self._extract_product_form_meta(node.left)
            right = self._extract_product_form_meta(node.right)
            if left is None or right is None:
                return None
            return {
                'factors': left['factors'] + right['factors'],
                'scale': left['scale'] * right['scale'],
            }
        return None

    def _ordered_tree_variables(self, node):
        if isinstance(node, VarNode):
            return [node.name]
        if isinstance(node, ConstNode):
            return []
        if isinstance(node, ScaleMulNode):
            return self._ordered_tree_variables(node.child)
        if isinstance(node, (MulNode, AddNode, SubNode, MinNode, CeilDivNode)):
            return self._ordered_tree_variables(node.left) + self._ordered_tree_variables(node.right)
        if isinstance(node, (SumNode, MaxNode)):
            ordered = []
            for child in node.children:
                ordered.extend(self._ordered_tree_variables(child))
            return ordered
        if isinstance(node, PrimExprNode):
            return sorted(node.variables())
        if isinstance(node, CaseSplitNode):
            ordered = []
            for selector in node.selectors:
                ordered.extend(self._ordered_tree_variables(selector))
            for case in node.cases:
                ordered.extend(self._ordered_tree_variables(case['expr']))
            ordered.extend(self._ordered_tree_variables(node.default))
            return ordered
        return sorted(node.variables())

    def _ordered_unique_tree_variables(self, node):
        ordered = []
        seen = set()
        for name in self._ordered_tree_variables(node):
            if name in seen:
                continue
            seen.add(name)
            ordered.append(name)
        return ordered

    @staticmethod
    def _append_unique_vars(target, names):
        seen = set(target)
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            target.append(name)

    def _collect_preferred_thread_vars(self, items):
        ordered = []
        seen = set()
        for item in items:
            product_vars = self._extract_pure_product_vars(item['tree'])
            names = (
                product_vars
                if product_vars is not None
                else self._ordered_unique_tree_variables(item['tree'])
            )
            for name in names:
                if name in seen:
                    continue
                seen.add(name)
                ordered.append(name)
        return ordered

    def _thread_axis_limit(self, ann):
        g = self.gen
        if ann == 6:
            return g.hw['max_thread_x']
        if ann == 8:
            return g.hw['max_thread_y']
        if ann == 10:
            return g.hw['max_thread_z']
        return g.hw['max_threads_per_block']

    def _flatten_max_terms(self, node):
        if isinstance(node, MaxNode):
            terms = []
            for child in node.children:
                terms.extend(self._flatten_max_terms(child))
            return terms
        return [node]

    @staticmethod
    def _dedupe_constraint_items(items, key_fn):
        deduped = []
        seen = set()
        for item in items:
            key = key_fn(item)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    @staticmethod
    def _vectorize_item_key(item):
        return ('vectorize', repr(item['tree']), item['limit'])

    @staticmethod
    def _thread_extent_item_key(item):
        return (
            'max_threads',
            item['axis_name'],
            str(item['sym_extent']),
            item['limit'],
            item.get('block_scope', tuple()),
        )

    @staticmethod
    def _vthread_item_key(item):
        return (
            'max_vthread',
            item['axis_name'],
            str(item['sym_extent']),
            item['limit'],
            item.get('block_scope', tuple()),
        )

    @staticmethod
    def _split_structure_item_key(item):
        return ('split_structure', item['sym_name'], item['display_rhs'])

    def _build_split_bound_denominator(self, names, extent):
        product_tree = None
        for name in names:
            node = VarNode(name)
            product_tree = node if product_tree is None else MulNode(product_tree, node)
        if product_tree is None:
            product_tree = ConstNode(1)
        return MinNode(product_tree, ConstNode(extent))

    @staticmethod
    def _build_split_structure_fast_path(sym_name, dependency_names, extent):
        return {
            'kind': 'split_structure',
            'sym_name': sym_name,
            'dependency_names': tuple(dependency_names),
            'extent': int(extent),
        }

    def _build_vthread_split_display_rhs(self, names, extent, pos, inner_to_outer):
        if inner_to_outer:
            dependency_names = names[pos + 1:]
            if not dependency_names:
                return ConstNode(extent)
            head_name = dependency_names[0]
            tail_names = dependency_names[1:]
        else:
            dependency_names = names[:pos]
            if not dependency_names:
                return ConstNode(extent)
            head_name = dependency_names[-1]
            tail_names = dependency_names[:-1]

        if tail_names:
            tail_rhs = CeilDivNode(
                ConstNode(extent),
                self._build_split_bound_denominator(tail_names, extent),
            )
        else:
            tail_rhs = ConstNode(extent)

        return CeilDivNode(tail_rhs, MinNode(VarNode(head_name), tail_rhs))

    def _flatten_mul_nodes(self, node):
        if isinstance(node, MulNode):
            return self._flatten_mul_nodes(node.left) + self._flatten_mul_nodes(node.right)
        return [node]

    def _collect_vthread_clamped_sp_names(self):
        g = self.gen
        names = set()
        for _, _, extent in g.s.get_vthread_extents():
            if extent is None:
                continue
            try:
                tree = parse_expr_tree(str(extent))
            except ValueError:
                continue
            for factor in self._flatten_mul_nodes(tree):
                if isinstance(factor, MinNode) and isinstance(factor.left, VarNode):
                    if re.fullmatch(r"sp_\d+_\d+", factor.left.name):
                        names.add(factor.left.name)
        return names

    def _build_max_vthread_constraints(self):
        g = self.gen
        if g._max_vthread_constraint_bundle is not None:
            return g._max_vthread_constraint_bundle
        items = self._collect_vthread_binding_axes()
        items = self._canonicalize_thread_binding_axes(items)
        g._max_vthread_constraint_bundle = {
            'items': self._dedupe_constraint_items(items, self._vthread_item_key)
        }
        return g._max_vthread_constraint_bundle

    def _check_max_vthread(self, sym_map=None):
        g = self.gen
        if sym_map is None:
            sym_map = g.s.sym_map
        c = self._build_max_vthread_constraints()
        violations = []
        params = g._normalize_concrete_params(sym_map)
        concrete_result = None
        for item in c['items']:
            val = item['tree'].evaluate(sym_map)
            if val > item['limit']:
                if params is not None:
                    if concrete_result is None:
                        concrete_result = g._get_concrete_final_result(params)
                    if concrete_result is not None and bool(concrete_result.get('ok')):
                        continue
                violations.append(f"{item['desc']}: actual={val}")
        return violations

    def _check_max_vthread_exact(self, sym_map=None):
        g = self.gen
        if sym_map is None:
            sym_map = g.s.sym_map
        self._ensure_exact_gpu_constraints()
        total = self._exact_upper_bound(g._exact_gpu['max_vthread_node'], sym_map)
        if total is None:
            return []
        if total > g.hw['max_vthread_extent']:
            return [
                "max_vthread: exact vthread extent upper bound ≤ limit: "
                f"actual={total}"
            ]
        return []

    def _build_innermost_split_constraints(self):
        g = self.gen
        limit = g.hw['max_innermost_split_factor']
        sp_groups = g.pm._build_sp_groups()
        constraints = []
        for step_idx, names in sorted(sp_groups.items()):
            last_name = names[-1]
            constraints.append({
                'sym_name': last_name,
                'step_idx': step_idx,
                'limit': limit,
                'desc': f"max_innermost_split {last_name}: value ≤ {limit}",
            })
        return constraints

    def _build_split_structure_constraints(self):
        g = self.gen
        if g._split_structure_constraint_bundle is not None:
            return g._split_structure_constraint_bundle
        constraints = []
        if g.s._state is None:
            g._split_structure_constraint_bundle = constraints
            return g._split_structure_constraint_bundle

        steps = g.s._state.transform_steps
        for step_idx, names in sorted(g._sp_groups.items()):
            if step_idx >= len(steps):
                continue

            step = steps[step_idx]
            if step.type_key.split(".")[-1] != "SplitStep":
                continue

            extent = g._sp_extents.get(step_idx)
            if extent is None or extent <= 1 or len(names) <= 1:
                continue

            inner_to_outer = bool(step.inner_to_outer)
            for pos, sym_name in enumerate(names):
                dependency_names = names[pos + 1:] if inner_to_outer else names[:pos]
                if not dependency_names:
                    continue

                denom_tree = self._build_split_bound_denominator(dependency_names, extent)
                if sym_name in g._vthread_clamped_sp_names:
                    rhs_tree = self._build_vthread_split_display_rhs(
                        names, extent, pos, inner_to_outer
                    )
                else:
                    rhs_tree = CeilDivNode(ConstNode(extent), denom_tree)
                display_rhs = g._format_expr(rhs_tree, top_level=True)
                constraints.append({
                    'sym_name': sym_name,
                    'step_idx': step_idx,
                    'dependency_names': tuple(dependency_names),
                    'tree': MulNode(SubNode(VarNode(sym_name), ConstNode(1)), denom_tree),
                    'limit': extent - 1,
                    'display_text': sym_name,
                    'display_rhs': display_rhs,
                    'desc': (
                        f"split_structure {sym_name}: legal SplitStep upper bound "
                        f"derived from extent {extent}"
                    ),
                })

        g._split_structure_constraint_bundle = self._dedupe_constraint_items(
            constraints, self._split_structure_item_key
        )
        return g._split_structure_constraint_bundle

    def _check_innermost_split(self, sym_map=None):
        g = self.gen
        if sym_map is None:
            sym_map = g.s.sym_map
        violations = []
        for c in self._build_innermost_split_constraints():
            val = sym_map.get(c['sym_name'])
            if val is not None and isinstance(val, int) and val > c['limit']:
                violations.append(f"{c['desc']}: actual={val}")
        return violations

    def _check_split_structure(self, sym_map=None):
        g = self.gen
        if sym_map is None:
            sym_map = g.s.sym_map
        violations = []
        for c in self._build_split_structure_constraints():
            val = c['tree'].evaluate(sym_map)
            if val > c['limit']:
                actual = sym_map.get(c['sym_name'])
                violations.append(
                    f"{c['display_text']} <= {c['display_rhs']}: actual={c['sym_name']}={actual}"
                )
        return violations

    # ------------------------------------------------------------------
    # Deprecated
    # ------------------------------------------------------------------

    def check_all_pruning(self, sym_map=None):
        """projected/심볼릭 pruning 제약만 검사해 위반 문자열 목록을 반환한다."""
        g = self.gen
        violations = []
        if 'vectorize' in g._enabled:
            violations.extend(self._check_vectorize(sym_map))
        if 'shared_memory' in g._enabled:
            violations.extend(self._check_shared_memory(sym_map))
        if 'max_threads' in g._enabled:
            violations.extend(self._check_max_threads(sym_map))
        if 'max_vthread' in g._enabled:
            violations.extend(self._check_max_vthread(sym_map))
        if 'innermost_split' in g._enabled:
            violations.extend(self._check_innermost_split(sym_map))
        if 'split_structure' in g._enabled:
            violations.extend(self._check_split_structure(sym_map))
        return violations

    def check_all_exact(self, sym_map=None):
        """exact GPU 케이스 제약까지 검사해 위반 문자열 목록을 반환한다."""
        g = self.gen
        if sym_map is None:
            sym_map = g.s.sym_map

        violations = []
        exact_upper_bounds = self._evaluate_exact_upper_bounds(sym_map)

        if 'vectorize' in g._enabled:
            total = exact_upper_bounds.get('vectorize')
            if total is not None and total > g.hw['max_vector_bytes']:
                violations.append(
                    "vectorize: exact vector bytes upper bound ≤ limit: "
                    f"actual={total}"
                )

        if 'shared_memory' in g._enabled:
            total = exact_upper_bounds.get('shared_memory')
            if total is not None and total > g.hw['max_shared_memory_per_block']:
                violations.append(
                    "shared_memory: exact shared bytes upper bound ≤ limit: "
                    f"actual={total}"
                )

        if 'max_threads' in g._enabled:
            for item in self._build_max_threads_constraints()['items']:
                if item['axis_name'] == 'threads per block':
                    continue
                val = item['tree'].evaluate(sym_map)
                if not isinstance(val, int):
                    violations.append(f"{item['desc']}: actual={item['axis_name']}={val}")
                    continue
                if val > item['limit']:
                    violations.append(f"{item['desc']}: actual={val}")

            total = exact_upper_bounds.get('max_threads')
            if total is not None and total > g.hw['max_threads_per_block']:
                violations.append(
                    "max_threads: exact threads per block upper bound ≤ limit: "
                    f"actual={total}"
                )

        if 'max_vthread' in g._enabled:
            total = exact_upper_bounds.get('max_vthread')
            if total is not None and total > g.hw['max_vthread_extent']:
                violations.append(
                    "max_vthread: exact vthread extent upper bound ≤ limit: "
                    f"actual={total}"
                )

        if 'innermost_split' in g._enabled:
            violations.extend(self._check_innermost_split(sym_map))
        if 'split_structure' in g._enabled:
            violations.extend(self._check_split_structure(sym_map))
        return violations

    @staticmethod
    def _exact_upper_bound(node, sym_map):
        _, hi = node.interval(dict(sym_map))
        hi = int(hi)
        if hi >= (1 << 60):
            return None
        return hi

    @staticmethod
    def _exact_upper_bound_from_interval(interval):
        hi = int(interval[1])
        if hi >= (1 << 60):
            return None
        return hi

    def _evaluate_exact_upper_bounds(self, sym_map):
        g = self.gen
        result = {}
        enabled_exact_kinds = {
            kind
            for kind in ('vectorize', 'shared_memory', 'max_threads', 'max_vthread')
            if kind in g._enabled
        }
        if not enabled_exact_kinds:
            return result

        self._ensure_exact_gpu_constraints()
        exact_nodes = {
            'vectorize': g._exact_gpu['vector_node'],
            'shared_memory': g._exact_gpu['shared_node'],
            'max_threads': g._exact_gpu['max_threads_node'],
            'max_vthread': g._exact_gpu['max_vthread_node'],
        }

        case_nodes = [
            exact_nodes[kind]
            for kind in enabled_exact_kinds
            if isinstance(exact_nodes[kind], CaseSplitNode)
        ]
        if len(case_nodes) != len(enabled_exact_kinds):
            for kind in enabled_exact_kinds:
                result[kind] = self._exact_upper_bound(exact_nodes[kind], sym_map)
            return result

        shared_selectors = tuple(repr(selector) for selector in case_nodes[0].selectors)
        shared_domains = tuple(sorted(case_nodes[0].extra_domains.keys()))
        if any(
            tuple(repr(selector) for selector in node.selectors) != shared_selectors
            or tuple(sorted(node.extra_domains.keys())) != shared_domains
            for node in case_nodes[1:]
        ):
            for kind in enabled_exact_kinds:
                result[kind] = self._exact_upper_bound(exact_nodes[kind], sym_map)
            return result

        domains = case_nodes[0]._augment_domains(dict(sym_map))
        feasible_case_values = case_nodes[0].feasible_case_values(domains)
        if self._can_evaluate_exact_cases_concretely(g._exact_gpu, enabled_exact_kinds, sym_map):
            # For fully assigned params, many exact case expressions evaluate to concrete ints
            # even when interval() falls back to the sentinel upper bound.
            result.update(
                self._evaluate_exact_upper_bounds_concretely(
                    exact_nodes,
                    enabled_exact_kinds,
                    feasible_case_values,
                    sym_map,
                )
            )
            return result
        for kind in enabled_exact_kinds:
            interval = exact_nodes[kind].interval_with_feasible_cases(domains, feasible_case_values)
            result[kind] = self._exact_upper_bound_from_interval(interval)
        return result

    @staticmethod
    def _can_evaluate_exact_cases_concretely(exact_gpu, enabled_exact_kinds, sym_map):
        case_expr_vars = exact_gpu.get('case_expr_vars')
        if not case_expr_vars:
            return False
        concrete_names = {
            name
            for name, value in sym_map.items()
            if value is not None and not isinstance(value, list)
        }
        return all(case_expr_vars.get(kind, set()) <= concrete_names for kind in enabled_exact_kinds)

    def _evaluate_exact_upper_bounds_concretely(
        self,
        exact_nodes,
        enabled_exact_kinds,
        feasible_case_values,
        sym_map,
    ):
        result = {}
        if not feasible_case_values:
            for kind in enabled_exact_kinds:
                result[kind] = self._exact_upper_bound_from_interval(
                    exact_nodes[kind].default.interval(dict(sym_map))
                )
            return result

        maxima = {kind: None for kind in enabled_exact_kinds}
        for values in feasible_case_values:
            for kind in enabled_exact_kinds:
                expr = exact_nodes[kind]._case_map.get(values, exact_nodes[kind].default)
                val = int(expr.evaluate(sym_map))
                best = maxima[kind]
                maxima[kind] = val if best is None else max(best, val)

        for kind, val in maxima.items():
            result[kind] = None if val is None or val >= (1 << 60) else val
        return result

    def _ensure_exact_gpu_constraints(self):
        g = self.gen
        if g._exact_gpu is not None:
            return
        self._ensure_projected_gpu_context()
        g._exact_gpu = build_exact_constraint_nodes(
            g.s,
            g.hw,
            g._sp_groups,
            g._sp_extents,
            g._innermost_names,
            g.hw['max_innermost_split_factor'],
            projected_context=g._projected_gpu_context,
        )

    def _ensure_projected_gpu_context(self):
        g = self.gen
        if g._projected_gpu_context is not None:
            return
        g._projected_gpu_context = build_projected_gpu_context(g.s)

    def _ensure_projected_gpu_constraints(self, kinds=None):
        g = self.gen
        if kinds is None:
            kinds = ('vectorize', 'shared_memory', 'max_vthread')
        if g._projected_gpu is None:
            g._projected_gpu = {}

        allowed_var_names = set(g.s.sym_map.keys())
        requested = set(kinds)

        if 'vectorize' in requested and 'vector_node' not in g._projected_gpu:
            self._ensure_projected_gpu_context()
            g._projected_gpu['vector_node'] = build_projected_vectorize_constraint_node(
                g._projected_gpu_context,
                g.hw,
                allowed_var_names=allowed_var_names,
            )

        if 'shared_memory' in requested and 'shared_node' not in g._projected_gpu:
            self._ensure_projected_gpu_context()
            g._projected_gpu['shared_node'] = build_projected_shared_memory_constraint_node(
                g._projected_gpu_context,
                g.hw,
                allowed_var_names=allowed_var_names,
            )

        need_exact_projected = (
            'max_vthread' in requested and 'max_vthread_node' not in g._projected_gpu
        )
        if not need_exact_projected:
            return

        self._ensure_exact_gpu_constraints()
        projected = build_projected_constraint_nodes(
            g._exact_gpu,
            g.hw,
            allowed_var_names=allowed_var_names,
        )
        if 'max_vthread' in requested and 'max_vthread_node' not in g._projected_gpu:
            g._projected_gpu['max_vthread_node'] = projected['max_vthread_node']

    # ------------------------------------------------------------------
    # Preprocess pipeline
    # ------------------------------------------------------------------

    def preprocess(self):
        """제약 묶음·변수별 제약 인덱스·변수 순서·phase 정보를 한 번에 계산해 제너레이터에 채운다."""
        g = self.gen
        sp_groups = g.pm._build_sp_groups()
        sp_extents = g.pm._build_sp_extents(sp_groups)

        innermost_names = set()
        if 'innermost_split' in g._enabled:
            for _, names in sp_groups.items():
                innermost_names.add(names[-1])

        g._sp_groups = sp_groups
        g._sp_extents = sp_extents
        g._ur_names = [n for n in g.s.sym_map if n.startswith("ur_")]
        g._all_sp_names = []
        for step_idx in sorted(sp_groups.keys()):
            g._all_sp_names.extend(sp_groups[step_idx])

        g._innermost_names = innermost_names
        g._exact_gpu = None
        g._projected_gpu = None
        g._projected_gpu_context = None
        g._preferred_thread_vars = []
        g._vthread_clamped_sp_names = self._collect_vthread_clamped_sp_names()

        g._constraints = []
        g._var_constraints = {}
        constraint_keys = set()

        def _add_constraint(
            expr_tree,
            rhs,
            kind,
            desc,
            is_upper=True,
            index_vars=True,
            display_text=None,
            display_rhs=None,
            alias_entries=None,
            fast_path=None,
        ):
            key = (
                kind,
                rhs,
                is_upper,
                display_text if display_text is not None else repr(expr_tree),
                display_rhs,
            )
            if key in constraint_keys:
                return
            constraint_keys.add(key)
            idx = len(g._constraints)
            vars_in = expr_tree.variables()
            unexpected_vars = sorted(v for v in vars_in if v not in g.s.sym_map)
            if unexpected_vars:
                raise RuntimeError(
                    f"Constraint {kind} references non-symbolic vars: {unexpected_vars}. "
                    f"expr={expr_tree!r}"
                )
            has_nonlinear = self._has_nonlinear(expr_tree)
            product_form_meta = self._extract_product_form_meta(expr_tree) if is_upper else None
            g._constraints.append({
                'tree': expr_tree,
                'rhs': rhs,
                'vars': vars_in,
                'kind': kind,
                'desc': desc,
                'is_upper': is_upper,
                'has_nonlinear': has_nonlinear,
                'display_text': display_text,
                'display_rhs': display_rhs,
                'alias_entries': list(alias_entries or []),
                'product_form_meta': product_form_meta,
                'fast_path': fast_path,
            })
            if index_vars:
                for v in vars_in:
                    g._var_constraints.setdefault(v, []).append(idx)

        if 'vectorize' in g._enabled:
            c = self._build_vectorize_constraints()
            for item in c['items']:
                _add_constraint(
                    item['tree'],
                    item['limit'],
                    'vectorize',
                    item['desc'],
                    is_upper=True,
                    index_vars=True,
                )

        if 'shared_memory' in g._enabled:
            sm = self._build_shared_memory_constraints()
            _add_constraint(
                sm['tree'],
                sm['limit'],
                'shared_memory',
                sm['desc'],
                is_upper=True,
                index_vars=True,
            )

        if 'max_threads' in g._enabled:
            c = self._build_max_threads_constraints()
            g._preferred_thread_vars = self._collect_preferred_thread_vars(c['items'])
            for item in c['items']:
                _add_constraint(
                    item['tree'],
                    item['limit'],
                    'max_threads',
                    item['desc'],
                    is_upper=True,
                    display_text=str(item['sym_extent']),
                    alias_entries=item.get('alias_entries'),
                )

        if 'max_vthread' in g._enabled:
            c = self._build_max_vthread_constraints()
            for item in c['items']:
                product_meta = self._extract_product_form_meta(item['tree'])
                display_text = str(item['sym_extent'])
                if product_meta is not None and int(product_meta['scale']) == 1:
                    factors = list(product_meta['factors'])
                    if factors:
                        display_text = " * ".join(factors)
                _add_constraint(
                    item['tree'],
                    item['limit'],
                    'max_vthread',
                    item['desc'],
                    is_upper=True,
                    display_text=display_text,
                    alias_entries=item.get('alias_entries'),
                )

        if 'split_structure' in g._enabled:
            for c in self._build_split_structure_constraints():
                _add_constraint(
                    c['tree'],
                    c['limit'],
                    'split_structure',
                    c['desc'],
                    is_upper=True,
                    index_vars=True,
                    display_text=c['display_text'],
                    display_rhs=c['display_rhs'],
                    fast_path=self._build_split_structure_fast_path(
                        c['sym_name'],
                        c['dependency_names'],
                        g._sp_extents.get(c['step_idx'], c['limit'] + 1),
                    ),
                )

        g.var_order_planner.compute_var_order()

    def _has_nonlinear(self, node):
        if isinstance(node, (MinNode, CeilDivNode, PrimExprNode, CaseSplitNode, MaxNode)):
            return True
        if isinstance(node, (MulNode, AddNode, SubNode)):
            return self._has_nonlinear(node.left) or self._has_nonlinear(node.right)
        if isinstance(node, ScaleMulNode):
            return self._has_nonlinear(node.child)
        if isinstance(node, SumNode):
            return any(self._has_nonlinear(c) for c in node.children)
        return False
