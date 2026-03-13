"""
schedule_generator — ScheduleGenerator facade over constraint, planning, propagation,
and sampling components.

제약 목록:
  1) max_vectorize_bytes
  2) max_shared_memory
  3) max_threads
  4) max_vthread
  5) max_innermost_split
  6) split_structure
"""
import re

from .sym_types import ANNOTATION_STR, CA_INLINED, CA_ITER, SymExpr, eval_sym_extent
from .param_manager import SymParamManager
from .expr_nodes import (
    ExprNode, ConstNode, VarNode, MulNode, AddNode, SubNode,
    MinNode, CeilDivNode, ScaleMulNode, SumNode, PrimExprNode, CaseSplitNode, MaxNode,
    parse_expr_tree,
)
from .constraint_set import ConstraintSet
from .var_order_planner import VarOrderPlanner
from .domain_propagator import DomainPropagator
from .param_sampler import ParamSampler


class ScheduleGenerator:
    """
    HW_PARAM 기반 제약식을 구축하고, 제약을 만족하는 파라미터를 생성하는 생성기.
    """

    DEFAULT_HW_PARAM = {
        'max_vector_bytes': 16,
        'max_shared_memory_per_block': 49152,
        'max_threads_per_block': 1024,
        'max_thread_x': 1024,
        'max_thread_y': 1024,
        'max_thread_z': 64,
        'max_vthread_extent': 8,
        'max_innermost_split_factor': 64,
    }

    ALL_CONSTRAINT_KINDS = (
        'vectorize', 'shared_memory', 'max_threads',
        'max_vthread', 'innermost_split', 'split_structure',
    )
    VAR_ORDER_PHASE_FAMILIES = (
        ('pure_product_max_threads', 'pure-product upper bound: max_threads'),
        ('pure_product_max_vthread', 'pure-product upper bound: max_vthread'),
        ('split_structure_max_threads', 'split_structure: max_threads-linked'),
        ('split_structure_max_vthread', 'split_structure: max_vthread-linked'),
        ('scaled_product_upper_bound', 'scaled-product upper bound'),
        ('non_product_direct_arm', 'non-product direct-arm'),
        ('non_product_gate_vars', 'non-product gate-vars'),
    )
    _FORMAT_WRAP_LIMIT = 100

    def __init__(
        self,
        sym_state,
        hw_param=None,
        enabled_constraints=None,
        task=None,
        base_input=None,
        base_result=None,
        base_state=None,
    ):
        self.s = sym_state
        self.hw = dict(self.DEFAULT_HW_PARAM)
        if hw_param is not None:
            self.hw.update(hw_param)
        self.pm = SymParamManager(sym_state)
        self._task = task
        self._base_input = base_input
        self._base_result = base_result
        self._base_state = base_state

        # 활성 제약 종류
        if enabled_constraints is None:
            self._enabled = set(self.ALL_CONSTRAINT_KINDS)
        else:
            unknown = set(enabled_constraints) - set(self.ALL_CONSTRAINT_KINDS)
            if unknown:
                raise ValueError(f"Unknown constraint kinds: {unknown}")
            self._enabled = set(enabled_constraints)

        self._constraints = []
        self._var_constraints = {}
        self._var_order = []
        self._var_order_phase_entries = []
        self._exact_gpu = None
        self._projected_gpu = None
        self._projected_gpu_context = None
        self._vectorize_constraint_bundle = None
        self._shared_memory_constraint_bundle = None
        self._max_threads_constraint_bundle = None
        self._max_vthread_constraint_bundle = None
        self._split_structure_constraint_bundle = None
        self._concrete_final_cache = {}
        self.constraint_set = ConstraintSet(self)
        self.var_order_planner = VarOrderPlanner(self)
        self.domain_propagator = DomainPropagator(self)
        self.param_sampler = ParamSampler(self)

        self.constraint_set.preprocess()

    # ═══════════════════════════════════════════════════════════
    # Public API: constraints and validation
    # ═══════════════════════════════════════════════════════════

    def build_vectorize_constraints(self):
        return self.constraint_set.build_vectorize_constraints()

    def check_vectorize(self, sym_map=None):
        return self.constraint_set.check_vectorize(sym_map)

    def check_vectorize_exact(self, sym_map=None):
        return self.constraint_set.check_vectorize_exact(sym_map)

    def build_shared_memory_constraints(self):
        return self.constraint_set.build_shared_memory_constraints()

    def check_shared_memory(self, sym_map=None):
        return self.constraint_set.check_shared_memory(sym_map)

    def check_shared_memory_exact(self, sym_map=None):
        return self.constraint_set.check_shared_memory_exact(sym_map)

    def build_max_threads_constraints(self):
        return self.constraint_set.build_max_threads_constraints()

    def check_max_threads(self, sym_map=None):
        return self.constraint_set.check_max_threads(sym_map)

    def check_max_threads_exact(self, sym_map=None):
        return self.constraint_set.check_max_threads_exact(sym_map)

    def build_max_vthread_constraints(self):
        return self.constraint_set.build_max_vthread_constraints()

    def check_max_vthread(self, sym_map=None):
        return self.constraint_set.check_max_vthread(sym_map)

    def check_max_vthread_exact(self, sym_map=None):
        return self.constraint_set.check_max_vthread_exact(sym_map)

    def build_innermost_split_constraints(self):
        return self.constraint_set.build_innermost_split_constraints()

    def build_split_structure_constraints(self):
        return self.constraint_set.build_split_structure_constraints()

    def check_innermost_split(self, sym_map=None):
        return self.constraint_set.check_innermost_split(sym_map)

    def check_split_structure(self, sym_map=None):
        return self.constraint_set.check_split_structure(sym_map)

    def check_all_pruning(self, sym_map=None):
        return self.constraint_set.check_all_pruning(sym_map)

    def check_all_exact(self, sym_map=None):
        return self.constraint_set.check_all_exact(sym_map)

    def _check_all_final_with_concrete_result(self, sym_map, concrete_result):
        violations = []
        if 'innermost_split' in self._enabled:
            violations.extend(self.check_innermost_split(sym_map))
        if 'split_structure' in self._enabled:
            violations.extend(self.check_split_structure(sym_map))

        if concrete_result is not None:
            violations.extend(concrete_result.get('violations', []))
            return violations

        if violations:
            return violations
        return self.check_all_exact(sym_map)

    def check_all_hybrid(self, sym_map=None):
        concrete_result = self.get_concrete_final_result(sym_map)
        if concrete_result is not None:
            return self._check_all_final_with_concrete_result(sym_map, concrete_result)
        return self.check_all_exact(sym_map)

    def _normalize_concrete_params(self, sym_map=None):
        if sym_map is None:
            sym_map = self.s.sym_map
        params = {}
        for name, current in self.s.sym_map.items():
            if not (name.startswith("sp_") or name.startswith("ur_")):
                continue
            value = sym_map.get(name, current)
            if value is None or not isinstance(value, int):
                return None
            params[name] = int(value)
        return params

    def has_concrete_final_context(self):
        if self._task is None:
            return False
        has_record_context = self._base_input is not None and self._base_result is not None
        return has_record_context or self._base_state is not None

    def get_concrete_final_result(self, sym_map=None):
        if not self.has_concrete_final_context():
            return None

        params = self._normalize_concrete_params(sym_map)
        if params is None:
            return None

        cache_key = tuple(sorted(params.items()))
        cached = self._concrete_final_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        from .tvm_verify import (
            lower_with_gpu_passes,
            params_to_state_from_record,
            params_to_state_from_state,
            verify_gpu_module_errors,
        )

        try:
            if self._base_input is not None and self._base_result is not None:
                state = params_to_state_from_record(
                    self._task,
                    self._base_input,
                    self._base_result,
                    params,
                )
            else:
                state = params_to_state_from_state(
                    self._task,
                    self._base_state,
                    params,
                )
            mod = lower_with_gpu_passes(self._task, state)
            violations = verify_gpu_module_errors(mod)
            ok = not violations
            result = {
                'ok': ok,
                'error': None if ok else "; ".join(violations),
                'violations': violations,
            }
        except Exception as err:  # pylint: disable=broad-except
            result = {
                'ok': False,
                'error': f"{type(err).__name__}: {err}",
                'violations': [f"concrete_gpu_verify: {type(err).__name__}: {err}"],
            }

        self._concrete_final_cache[cache_key] = dict(result)
        return result

    def check_all_final(self, sym_map=None):
        concrete_result = self.get_concrete_final_result(sym_map)
        return self._check_all_final_with_concrete_result(sym_map, concrete_result)

    def check_all(self, sym_map=None):
        return self.check_all_pruning(sym_map)

    def get_constraint_records(self):
        """현재 활성화된 제약식을 사람이 읽기 쉬운 dict 목록으로 반환."""
        records = []
        for idx, constraint in enumerate(self._constraints):
            op = "<=" if constraint['is_upper'] else ">="
            vars_in = sorted(constraint['vars'])
            records.append({
                'index': idx,
                'kind': constraint['kind'],
                'expr': self._format_constraint_expr(constraint),
                'op': op,
                'rhs': constraint['rhs'],
                'display_rhs': (
                    constraint['display_rhs']
                    if constraint.get('display_rhs') is not None
                    else constraint['rhs']
                ),
                'desc': constraint['desc'],
                'vars': vars_in,
                'has_nonlinear': constraint['has_nonlinear'],
                'alias_entries': list(constraint.get('alias_entries', [])),
            })
        return records

    def _format_alias_entry_lines(self, entry):
        label = entry['label']
        if entry.get('is_canonical'):
            label = f"{label} (canonical)"
        expr_lines = self._format_display_text(entry['expr']).splitlines()
        if len(expr_lines) == 1:
            return [f"{label} = {expr_lines[0]}"]
        lines = [f"{label} ="]
        lines.extend(self._indent_lines(expr_lines, prefix="  "))
        return lines

    def _format_constraint_expr(self, constraint):
        display_text = constraint.get('display_text')
        if display_text:
            return self._format_display_text(display_text)
        return self._format_tree(constraint['tree'])

    def get_constraints_str(self, include_vars=False, include_meta=False):
        """현재 활성화된 모든 제약식을 multi-line 문자열로 포맷한다."""
        lines = []
        for record in self.get_constraint_records():
            expr_lines = record['expr'].splitlines() if record['expr'] else [""]
            has_aliases = bool(record.get('alias_entries'))
            if len(expr_lines) == 1 and not has_aliases:
                lines.append(
                    f"{record['kind']}: "
                    f"{expr_lines[0]} {record['op']} {record['display_rhs']}"
                )
            else:
                lines.append(f"{record['kind']}:")
                for expr_line in expr_lines:
                    lines.append(f"  {expr_line}")
                if has_aliases:
                    lines.append("  aliases:")
                    for entry in record['alias_entries']:
                        for alias_line in self._format_alias_entry_lines(entry):
                            lines.append(f"    {alias_line}")
                lines.append(f"  {record['op']} {record['display_rhs']}")
            if include_meta:
                lines.append(f"  desc: {record['desc']}")
                lines.append(f"  nonlinear: {record['has_nonlinear']}")
            if include_vars:
                vars_text = ", ".join(record['vars']) if record['vars'] else "(none)"
                lines.append(f"  vars: {vars_text}")
        return "\n".join(lines)

    def get_constraints_with_assignment_str(self, sym_map=None, include_vars=False, include_eval=True):
        if sym_map is None:
            sym_map = self.s.sym_map

        lines = []
        for constraint, record in zip(self._constraints, self.get_constraint_records()):
            instantiated_record = dict(record)
            instantiated_record['expr'] = self._simplify_constraint_expr_text(
                constraint, record, sym_map
            )
            instantiated_record['display_rhs'] = self._simplify_constraint_rhs_text(
                constraint, record, sym_map
            )
            instantiated_record['alias_entries'] = [
                {
                    **entry,
                    'expr': self._simplify_expr_text(entry['expr'], sym_map),
                }
                for entry in record.get('alias_entries', [])
            ]
            lines.extend(
                self._format_constraint_record_lines(
                    instantiated_record,
                    include_aliases=True,
                    include_vars=include_vars,
                )
            )
            if include_eval:
                lhs_val = self._evaluate_display_lhs_value(constraint, sym_map)
                rhs_val = self._evaluate_display_rhs_value(constraint, sym_map)
                lines.append(
                    f"  eval: {lhs_val} {instantiated_record['op']} {rhs_val}"
                )
        return "\n".join(lines)

    def _format_constraint_record_lines(self, record, include_aliases=True, include_vars=False, indent=""):
        lines = []
        expr_lines = record['expr'].splitlines() if record['expr'] else [""]
        has_aliases = include_aliases and bool(record.get('alias_entries'))
        if len(expr_lines) == 1 and not has_aliases:
            lines.append(
                f"{indent}{record['kind']}: "
                f"{expr_lines[0]} {record['op']} {record['display_rhs']}"
            )
        else:
            lines.append(f"{indent}{record['kind']}:")
            for expr_line in expr_lines:
                lines.append(f"{indent}  {expr_line}")
            if has_aliases:
                lines.append(f"{indent}  aliases:")
                for entry in record['alias_entries']:
                    for alias_line in self._format_alias_entry_lines(entry):
                        lines.append(f"{indent}    {alias_line}")
            lines.append(f"{indent}  {record['op']} {record['display_rhs']}")
        if include_vars:
            vars_text = ", ".join(record['vars']) if record['vars'] else "(none)"
            lines.append(f"{indent}  vars: {vars_text}")
        return lines

    @staticmethod
    def _substitute_text(text, sym_map):
        substituted = str(text)
        for sym_name in sorted(sym_map.keys(), key=len, reverse=True):
            value = sym_map[sym_name]
            if value is None:
                continue
            substituted = substituted.replace(sym_name, str(value))
        return substituted

    def _simplify_constraint_expr_text(self, constraint, record, sym_map):
        display_text = constraint.get('display_text')
        if display_text is not None:
            return self._simplify_expr_text(display_text, sym_map)
        simplified = self._simplify_expr_node(constraint['tree'], sym_map)
        return self._format_tree(simplified)

    def _simplify_constraint_rhs_text(self, constraint, record, sym_map):
        rhs = constraint.get('display_rhs')
        if rhs is None:
            rhs = record['display_rhs']
        if isinstance(rhs, int):
            return str(rhs)
        return self._simplify_expr_text(rhs, sym_map)

    def _simplify_expr_text(self, text, sym_map):
        substituted = self._substitute_text(text, sym_map)
        normalized = " ".join(substituted.split())
        try:
            parsed = parse_expr_tree(normalized)
        except ValueError:
            return substituted
        simplified = self._simplify_expr_node(parsed, {})
        return self._format_expr(simplified, top_level=True)

    def _simplify_expr_node(self, node, sym_map):
        if isinstance(node, ConstNode):
            return node
        if isinstance(node, VarNode):
            if node.name in sym_map and sym_map[node.name] is not None:
                return ConstNode(sym_map[node.name])
            return node
        if isinstance(node, MulNode):
            left = self._simplify_expr_node(node.left, sym_map)
            right = self._simplify_expr_node(node.right, sym_map)
            if isinstance(left, ConstNode) and isinstance(right, ConstNode):
                return ConstNode(left.val * right.val)
            if isinstance(left, ConstNode):
                if left.val == 0:
                    return ConstNode(0)
                if left.val == 1:
                    return right
            if isinstance(right, ConstNode):
                if right.val == 0:
                    return ConstNode(0)
                if right.val == 1:
                    return left
            return MulNode(left, right)
        if isinstance(node, AddNode):
            left = self._simplify_expr_node(node.left, sym_map)
            right = self._simplify_expr_node(node.right, sym_map)
            if isinstance(left, ConstNode) and isinstance(right, ConstNode):
                return ConstNode(left.val + right.val)
            if isinstance(left, ConstNode) and left.val == 0:
                return right
            if isinstance(right, ConstNode) and right.val == 0:
                return left
            return AddNode(left, right)
        if isinstance(node, SubNode):
            left = self._simplify_expr_node(node.left, sym_map)
            right = self._simplify_expr_node(node.right, sym_map)
            if isinstance(left, ConstNode) and isinstance(right, ConstNode):
                return ConstNode(left.val - right.val)
            if isinstance(right, ConstNode) and right.val == 0:
                return left
            return SubNode(left, right)
        if isinstance(node, MinNode):
            left = self._simplify_expr_node(node.left, sym_map)
            right = self._simplify_expr_node(node.right, sym_map)
            if isinstance(left, ConstNode) and isinstance(right, ConstNode):
                return ConstNode(min(left.val, right.val))
            return MinNode(left, right)
        if isinstance(node, MaxNode):
            children = [self._simplify_expr_node(child, sym_map) for child in node.children]
            if all(isinstance(child, ConstNode) for child in children):
                return ConstNode(max(child.val for child in children) if children else 0)
            return MaxNode(children)
        if isinstance(node, CeilDivNode):
            left = self._simplify_expr_node(node.left, sym_map)
            right = self._simplify_expr_node(node.right, sym_map)
            if isinstance(left, ConstNode) and isinstance(right, ConstNode):
                denom = max(right.val, 1)
                return ConstNode((left.val + denom - 1) // denom)
            if isinstance(right, ConstNode) and right.val == 1:
                return left
            return CeilDivNode(left, right)
        if isinstance(node, ScaleMulNode):
            child = self._simplify_expr_node(node.child, sym_map)
            if isinstance(child, ConstNode):
                return ConstNode(child.val * node.scale)
            if node.scale == 0:
                return ConstNode(0)
            if node.scale == 1:
                return child
            return ScaleMulNode(child, node.scale)
        if isinstance(node, SumNode):
            children = [self._simplify_expr_node(child, sym_map) for child in node.children]
            const_sum = 0
            new_children = []
            for child in children:
                if isinstance(child, ConstNode):
                    const_sum += child.val
                else:
                    new_children.append(child)
            if const_sum:
                new_children.append(ConstNode(const_sum))
            if not new_children:
                return ConstNode(0)
            if len(new_children) == 1:
                return new_children[0]
            return SumNode(new_children)
        if isinstance(node, PrimExprNode):
            return self._simplify_prim_expr_node(node, sym_map)
        if isinstance(node, CaseSplitNode):
            selectors = [self._simplify_expr_node(selector, sym_map) for selector in node.selectors]
            if selectors and all(isinstance(selector, ConstNode) for selector in selectors):
                values = tuple(selector.val for selector in selectors)
                expr = node._case_map.get(values, node.default)  # pylint: disable=protected-access
                return self._simplify_expr_node(expr, sym_map)
            cases = [
                {
                    'values': case['values'],
                    'expr': self._simplify_expr_node(case['expr'], sym_map),
                }
                for case in node.cases
            ]
            default = self._simplify_expr_node(node.default, sym_map)
            return CaseSplitNode(selectors, cases, default=default, extra_domains=node.extra_domains)
        return node

    @staticmethod
    def _simplify_prim_expr_node(node, sym_map):
        import tvm

        expr = node.expr
        subst = {}
        for name, var in node._var_map.items():  # pylint: disable=protected-access
            if name in sym_map and sym_map[name] is not None:
                subst[var] = sym_map[name]
        if subst:
            expr = tvm.tir.stmt_functor.substitute(expr, subst)
        simplified = tvm.arith.Analyzer().simplify(expr)

        var_map = {}

        def visit(cur):
            if isinstance(cur, tvm.tir.Var):
                var_map.setdefault(str(cur.name), cur)

        tvm.tir.stmt_functor.post_order_visit(simplified, visit)
        if not var_map:
            return ConstNode(int(simplified))
        return PrimExprNode(simplified)

    @staticmethod
    def _evaluate_display_rhs_value(constraint, sym_map):
        rhs = constraint.get('display_rhs')
        if rhs is None:
            return constraint['rhs']
        if isinstance(rhs, int):
            return rhs
        return eval_sym_extent(SymExpr(str(rhs)), sym_map)

    @staticmethod
    def _evaluate_display_lhs_value(constraint, sym_map):
        display_text = constraint.get('display_text')
        if display_text is None:
            return constraint['tree'].evaluate(sym_map)
        return eval_sym_extent(SymExpr(str(display_text)), sym_map)

    def _format_stage_iter_extent_lines(self, stage_id, iter_id, extent, indent=""):
        stage = self.s.stages[stage_id]
        it = stage.iters[iter_id]
        label = f"{stage.op_name}:{it.name}"
        expr_lines = self._format_display_text(str(extent)).splitlines()
        if len(expr_lines) == 1:
            return [f"{indent}{label} = {expr_lines[0]}"]
        lines = [f"{indent}{label} ="]
        for expr_line in expr_lines:
            lines.append(f"{indent}  {expr_line}")
        return lines

    def get_structural_highlights_str(self, include_vars=True):
        records = self.get_constraint_records()
        max_thread_records = [record for record in records if record["kind"] == "max_threads"]
        max_vthread_records = [record for record in records if record["kind"] == "max_vthread"]
        split_records = [record for record in records if record["kind"] == "split_structure"]
        interesting_max_vthread_records = [
            record for record in max_vthread_records
            if record["vars"] or record["expr"].strip() not in ("0", "1")
        ]

        thread_vars = set()
        for record in max_thread_records:
            thread_vars.update(record["vars"])

        vthread_vars = set()
        for record in max_vthread_records:
            vthread_vars.update(record["vars"])

        thread_split_records = [record for record in split_records if record["expr"] in thread_vars]
        vthread_split_records = [record for record in split_records if record["expr"] in vthread_vars]

        lines = []

        lines.append("[Thread Highlights]")
        if max_thread_records:
            for idx, record in enumerate(max_thread_records, start=1):
                lines.append(f"group_{idx}:")
                lines.extend(
                    self._format_constraint_record_lines(
                        record, include_aliases=True, include_vars=include_vars, indent="  "
                    )
                )
        else:
            lines.append("(none)")

        if thread_split_records:
            lines.append("thread_linked_split_bounds:")
            for record in thread_split_records:
                lines.extend(
                    self._format_constraint_record_lines(
                        record, include_aliases=False, include_vars=include_vars, indent="  "
                    )
                )

        lines.append("")
        lines.append("[VThread Highlights]")
        vthread_extents = list(self.s.get_vthread_extents())
        if vthread_extents:
            lines.append("symbolic_extents:")
            for stage_id, iter_id, extent in vthread_extents:
                lines.extend(self._format_stage_iter_extent_lines(stage_id, iter_id, extent, indent="  "))
        elif not interesting_max_vthread_records and not vthread_split_records:
            lines.append("(none)")

        if interesting_max_vthread_records:
            lines.append("raw_product_bound:")
            for record in interesting_max_vthread_records:
                lines.extend(
                    self._format_constraint_record_lines(
                        record, include_aliases=False, include_vars=include_vars, indent="  "
                    )
                )

        if vthread_split_records:
            lines.append("vthread_linked_split_bounds:")
            for record in vthread_split_records:
                lines.extend(
                    self._format_constraint_record_lines(
                        record, include_aliases=False, include_vars=include_vars, indent="  "
                    )
                )

        return "\n".join(lines)

    def get_raw_exact_constraints_str(self, include_vars=False):
        self._ensure_exact_gpu_constraints()
        items = (
            ("vectorize", self._exact_gpu["vector_node"], self.hw["max_vector_bytes"]),
            ("shared_memory", self._exact_gpu["shared_node"], self.hw["max_shared_memory_per_block"]),
            ("max_threads", self._exact_gpu["max_threads_node"], self.hw["max_threads_per_block"]),
            ("max_vthread", self._exact_gpu["max_vthread_node"], self.hw["max_vthread_extent"]),
        )
        lines = []
        for kind, tree, rhs in items:
            expr_lines = self._format_tree(tree).splitlines()
            if len(expr_lines) == 1:
                lines.append(f"{kind}: {expr_lines[0]} <= {rhs}")
            else:
                lines.append(f"{kind}:")
                for expr_line in expr_lines:
                    lines.append(f"  {expr_line}")
                lines.append(f"  <= {rhs}")
            if include_vars:
                vars_text = ", ".join(sorted(tree.variables())) if tree.variables() else "(none)"
                lines.append(f"  vars: {vars_text}")
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════
    # 2) 전처리
    # ═══════════════════════════════════════════════════════════

    def _preprocess(self):
        self.constraint_set.preprocess()

    @staticmethod
    def _has_nonlinear(node):
        if isinstance(node, (MinNode, CeilDivNode, PrimExprNode, CaseSplitNode, MaxNode)):
            return True
        if isinstance(node, (MulNode, AddNode, SubNode)):
            return ScheduleGenerator._has_nonlinear(node.left) or ScheduleGenerator._has_nonlinear(node.right)
        if isinstance(node, ScaleMulNode):
            return ScheduleGenerator._has_nonlinear(node.child)
        if isinstance(node, SumNode):
            return any(ScheduleGenerator._has_nonlinear(c) for c in node.children)
        return False

    def _format_tree(self, node):
        if isinstance(node, CaseSplitNode):
            if not node.selectors and len(node.cases) == 1 and node.cases[0]['values'] == tuple():
                return self._format_tree(node.cases[0]['expr'])

            lines = []
            if node.selectors:
                lines.append("selectors:")
                for idx, selector in enumerate(node.selectors):
                    lines.append(f"  s{idx} = {self._format_expr(selector)}")
            lines.append("cases:")
            for case in node.cases:
                if node.selectors:
                    pred = " and ".join(
                        f"s{idx} == {target}"
                        for idx, target in enumerate(case['values'])
                    )
                else:
                    pred = "default"
                expr_text = self._format_tree(case['expr'])
                expr_lines = expr_text.splitlines()
                if len(expr_lines) == 1:
                    lines.append(f"  when {pred}: {expr_lines[0]}")
                else:
                    lines.append(f"  when {pred}:")
                    lines.extend(self._indent_lines(expr_lines, prefix="    "))
            return "\n".join(lines)
        return self._format_expr(node, top_level=True)

    def _format_display_text(self, text):
        parts = self._split_top_level(text, "*")
        if len(parts) <= 1:
            return text
        compact = " * ".join(parts)
        if len(compact) <= self._FORMAT_WRAP_LIMIT:
            return compact
        return "\n".join([parts[0]] + [f"* {part}" for part in parts[1:]])

    def _split_top_level(self, text, sep):
        parts = []
        depth = 0
        start = 0
        for idx, ch in enumerate(text):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == sep and depth == 0:
                parts.append(text[start:idx].strip())
                start = idx + 1
        parts.append(text[start:].strip())
        return [part for part in parts if part]

    def _format_expr(self, node, top_level=False):
        if isinstance(node, ConstNode):
            return str(node.val)
        if isinstance(node, VarNode):
            return node.name
        if isinstance(node, PrimExprNode):
            return self._format_prim_expr(node.expr, top_level=top_level)
        if isinstance(node, MinNode):
            args = [self._format_expr(node.left), self._format_expr(node.right)]
            return self._format_call("min", args, top_level=top_level)
        if isinstance(node, MaxNode):
            args = [self._format_expr(child) for child in node.children]
            return self._format_call("max", args, top_level=top_level)
        if isinstance(node, CeilDivNode):
            left = self._format_div_operand(node.left, is_denominator=False)
            right = self._format_div_operand(node.right, is_denominator=True)
            return f"ceil({left} / {right})"
        if isinstance(node, ScaleMulNode):
            parts = self._flatten_mul_parts(node)
            return self._format_joined(parts, " * ", top_level=top_level)
        if isinstance(node, MulNode):
            parts = self._flatten_mul_parts(node)
            return self._format_joined(parts, " * ", top_level=top_level)
        if isinstance(node, SumNode):
            parts = [self._format_expr(child) for child in node.children]
            return self._format_joined(parts, " + ", top_level=top_level)
        if isinstance(node, AddNode):
            parts = self._flatten_add_parts(node)
            return self._format_joined(parts, " + ", top_level=top_level)
        if isinstance(node, SubNode):
            left = self._format_expr(node.left)
            right = self._format_expr(node.right)
            return f"{left} - {right}"
        return self._normalize_expr_text(str(node))

    def _flatten_mul_parts(self, node):
        if isinstance(node, MulNode):
            return self._flatten_mul_parts(node.left) + self._flatten_mul_parts(node.right)
        if isinstance(node, ScaleMulNode):
            return self._flatten_mul_parts(node.child) + [str(node.scale)]
        return [self._format_mul_operand(node)]

    def _flatten_add_parts(self, node):
        if isinstance(node, AddNode):
            return self._flatten_add_parts(node.left) + self._flatten_add_parts(node.right)
        return [self._format_expr(node)]

    def _format_mul_operand(self, node):
        text = self._format_expr(node)
        if isinstance(node, (AddNode, SubNode, SumNode)):
            return f"({text})"
        return text

    def _format_div_operand(self, node, is_denominator):
        text = self._format_expr(node)
        if isinstance(node, (AddNode, SubNode, SumNode)):
            return f"({text})"
        if is_denominator and isinstance(node, (MulNode, ScaleMulNode)):
            return f"({text})"
        return text

    def _format_joined(self, parts, separator, top_level=False):
        compact = separator.join(parts)
        if not top_level or len(parts) <= 1 or len(compact) <= self._FORMAT_WRAP_LIMIT:
            return compact
        lead = separator.strip()
        return "\n".join([parts[0]] + [f"{lead} {part}" for part in parts[1:]])

    def _format_call(self, name, args, top_level=False):
        compact = f"{name}(" + ", ".join(args) + ")"
        if not top_level or len(compact) <= self._FORMAT_WRAP_LIMIT:
            return compact
        lines = [f"{name}("]
        for idx, arg in enumerate(args):
            suffix = "," if idx + 1 < len(args) else ""
            arg_lines = arg.splitlines()
            lines.extend(
                self._indent_lines(
                    [
                        f"{line}{suffix if line_idx == len(arg_lines) - 1 else ''}"
                        for line_idx, line in enumerate(arg_lines)
                    ],
                    prefix="  ",
                )
            )
        lines.append(")")
        return "\n".join(lines)

    @staticmethod
    def _indent_lines(lines, prefix="  "):
        return [f"{prefix}{line}" for line in lines]

    @staticmethod
    def _normalize_expr_text(text):
        text = text.replace("T.min(", "min(")
        text = text.replace("T.max(", "max(")
        text = re.sub(r",\s*", ", ", text)
        return text.strip()

    def _format_prim_expr(self, expr, top_level=False):
        import tvm

        if isinstance(expr, tvm.tir.IntImm):
            return str(int(expr))
        if isinstance(expr, tvm.tir.FloatImm):
            return str(expr)
        if isinstance(expr, tvm.tir.Var):
            return str(expr)
        if isinstance(expr, tvm.tir.Cast):
            return self._format_prim_expr(expr.value, top_level=top_level)

        ceil_chain = self._extract_prim_ceil_chain(expr)
        if ceil_chain is not None:
            base_text, denoms = ceil_chain
            text = base_text
            for denom in denoms:
                text = f"ceil({text} / {self._format_prim_div_operand(denom, is_denominator=True)})"
            return text

        if isinstance(expr, tvm.tir.Mul):
            parts = self._flatten_prim_mul(expr)
            return self._format_joined(parts, " * ", top_level=top_level)
        if isinstance(expr, tvm.tir.Add):
            parts = self._flatten_prim_add(expr)
            return self._format_joined(parts, " + ", top_level=top_level)
        if isinstance(expr, tvm.tir.Sub):
            left = self._format_prim_expr(expr.a)
            right = self._format_prim_expr(expr.b)
            return f"{left} - {right}"
        if isinstance(expr, tvm.tir.FloorDiv):
            left = self._format_prim_div_operand(expr.a, is_denominator=False)
            right = self._format_prim_div_operand(expr.b, is_denominator=True)
            return f"floor({left} / {right})"
        if isinstance(expr, tvm.tir.FloorMod):
            args = [
                self._format_prim_div_operand(expr.a, is_denominator=False),
                self._format_prim_div_operand(expr.b, is_denominator=True),
            ]
            return self._format_call("mod", args, top_level=top_level)
        if isinstance(expr, tvm.tir.Min):
            args = [self._format_prim_expr(expr.a), self._format_prim_expr(expr.b)]
            return self._format_call("min", args, top_level=top_level)
        if isinstance(expr, tvm.tir.Max):
            args = [self._format_prim_expr(expr.a), self._format_prim_expr(expr.b)]
            return self._format_call("max", args, top_level=top_level)

        return self._normalize_expr_text(str(expr))

    def _flatten_prim_mul(self, expr):
        import tvm

        if isinstance(expr, tvm.tir.Mul):
            return self._flatten_prim_mul(expr.a) + self._flatten_prim_mul(expr.b)
        return [self._format_prim_mul_operand(expr)]

    def _flatten_prim_add(self, expr):
        import tvm

        if isinstance(expr, tvm.tir.Add):
            return self._flatten_prim_add(expr.a) + self._flatten_prim_add(expr.b)
        return [self._format_prim_expr(expr)]

    def _extract_prim_ceil_chain(self, expr):
        import tvm

        if not isinstance(expr, tvm.tir.Add):
            return None

        if self._is_prim_int_one(expr.a):
            current = expr.b
        elif self._is_prim_int_one(expr.b):
            current = expr.a
        else:
            return None

        denoms = []
        while isinstance(current, tvm.tir.FloorDiv):
            denoms.append(current.b)
            current = current.a

        base_text = self._format_prim_ceil_base(current)
        if base_text is None:
            return None

        denoms.reverse()
        return base_text, denoms

    def _format_prim_ceil_base(self, expr):
        import tvm

        if isinstance(expr, tvm.tir.Sub) and self._is_prim_int_one(expr.b):
            return self._format_prim_expr(expr.a)
        if isinstance(expr, tvm.tir.IntImm):
            return str(int(expr) + 1)
        return None

    @staticmethod
    def _is_prim_int_one(expr):
        import tvm

        return isinstance(expr, tvm.tir.IntImm) and int(expr) == 1

    def _format_prim_mul_operand(self, expr):
        import tvm

        text = self._format_prim_expr(expr)
        if isinstance(expr, (tvm.tir.Add, tvm.tir.Sub)):
            return f"({text})"
        return text

    def _format_prim_div_operand(self, expr, is_denominator):
        import tvm

        text = self._format_prim_expr(expr)
        if isinstance(expr, (tvm.tir.Add, tvm.tir.Sub)):
            return f"({text})"
        if is_denominator and isinstance(expr, (tvm.tir.Mul, tvm.tir.FloorDiv)):
            return f"({text})"
        return text

    def _ensure_exact_gpu_constraints(self):
        self.constraint_set._ensure_exact_gpu_constraints()

    def _ensure_projected_gpu_constraints(self, kinds=None):
        self.constraint_set._ensure_projected_gpu_constraints(kinds)

    def _compute_var_order(self):
        self.var_order_planner.compute_var_order()

    def get_var_order_phase_entries(self):
        return self.var_order_planner.get_var_order_phase_entries()

    def _resolve_var_order_stop_index(self, stop_after_phase):
        return self.var_order_planner._resolve_var_order_stop_index(stop_after_phase)

    def get_var_order_prefix(self, stop_after_phase):
        return self.var_order_planner.get_var_order_prefix(stop_after_phase)

    # ═══════════════════════════════════════════════════════════
    # 3) 제약 만족 파라미터 생성
    # ═══════════════════════════════════════════════════════════

    def _randomize_params_with_order(
        self,
        var_order,
        rng=None,
        max_retries=1,
        assign_unroll=True,
        require_full_validation=True,
        return_domains=False,
    ):
        return self.param_sampler._randomize_params_with_order(
            var_order,
            rng=rng,
            max_retries=max_retries,
            assign_unroll=assign_unroll,
            require_full_validation=require_full_validation,
            return_domains=return_domains,
        )

    def _snapshot_domains(self, domains):
        return self.domain_propagator._snapshot_domains(domains)

    def _fixed_and_remaining_from_domains(self, domains):
        return self.domain_propagator._fixed_and_remaining_from_domains(domains)

    def analyze_constraints_under_domains(self, domains):
        return self.domain_propagator.analyze_constraints_under_domains(domains)

    def _analyze_constraint_bounds(self, constraint, expr_text, fixed_values, remaining_domains):
        return self.domain_propagator._analyze_constraint_bounds(
            constraint,
            expr_text,
            fixed_values,
            remaining_domains,
        )

    def randomize_params(self, rng=None, max_retries=1):
        return self.param_sampler.randomize_params(rng=rng, max_retries=max_retries)

    def randomize_params_prefix(self, stop_after_phase, rng=None, max_retries=1):
        return self.param_sampler.randomize_params_prefix(
            stop_after_phase,
            rng=rng,
            max_retries=max_retries,
        )

    def _apply_upper_bound_to_domain(self, dom, hi_allowed):
        return self.domain_propagator._apply_upper_bound_to_domain(dom, hi_allowed)

    def _get_sym_value(self, sym_map, name):
        return self.domain_propagator._get_sym_value(sym_map, name)

    def _propagate_domain(self, assigned_name, domains):
        self.domain_propagator.propagate_domain(assigned_name, domains)

    def _filter_by_constraints(self, var_name, candidates, constraint_indices, domains):
        return self.domain_propagator.filter_by_constraints(
            var_name,
            candidates,
            constraint_indices,
            domains,
        )

    def enumerate_all_params(self, max_results=100_000):
        return self.param_sampler.enumerate_all_params(max_results=max_results)
