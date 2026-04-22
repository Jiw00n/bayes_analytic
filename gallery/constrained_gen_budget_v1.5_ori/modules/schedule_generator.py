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
  7) min_thread_extent (optional heuristic)
"""
import re

from .sym_types import ANNOTATION_STR, CA_INLINED, CA_ITER, SymExpr, eval_sym_extent
from .symbolic_state_bridge import SymParamManager
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
        'warp_size': 32,
    }

    ALL_CONSTRAINT_KINDS = (
        'vectorize', 'shared_memory', 'max_threads',
        'max_vthread', 'innermost_split', 'split_structure', 'min_thread_extent',
    )
    DEFAULT_ENABLED_CONSTRAINT_KINDS = (
        'vectorize',
        'shared_memory',
        'max_threads',
        'max_vthread',
        'innermost_split',
        'split_structure',
    )
    VAR_ORDER_PHASE_FAMILIES = (
        ('execution_max_threads_pure_product', 'execution: max_threads pure-product'),
        (
            'execution_max_vthread',
            'execution: max_vthread',
        ),
        ('execution_split_assignment', 'execution: split_structure assignment'),
        ('execution_non_product_direct_arm', 'execution: non-product direct-arm'),
        ('execution_non_product_gate_vars', 'execution: non-product gate-vars'),
        ('memory_split_structure', 'memory: shared-memory-linked split_structure'),
        ('instruction_vectorize_unroll', 'instruction: vectorize + unroll'),
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
        """심볼 상태·HW 파라미터·제약 종류를 받아 제너레이터와 제약/플래너/샘플러를 초기화한다."""
        self.s = sym_state
        self.hw = dict(self.DEFAULT_HW_PARAM)
        if task is not None:
            hardware_params = getattr(task, "hardware_params", None)
            if hardware_params is not None:
                try:
                    warp_size = getattr(hardware_params, "warp_size")
                except AttributeError:
                    warp_size = None
                if warp_size is not None:
                    self.hw['warp_size'] = int(warp_size)
        if hw_param is not None:
            self.hw.update(hw_param)
        self.pm = SymParamManager(sym_state)
        self._task = task
        self._base_input = base_input
        self._base_result = base_result
        self._base_state = base_state

        # 활성 제약 종류
        if enabled_constraints is None:
            # Measured-program generation excludes the init-population-only heuristic.
            self._enabled = set(self.DEFAULT_ENABLED_CONSTRAINT_KINDS)
        else:
            unknown = set(enabled_constraints) - set(self.ALL_CONSTRAINT_KINDS)
            if unknown:
                raise ValueError(f"Unknown constraint kinds: {unknown}")
            self._enabled = set(enabled_constraints)

        self._constraints = []
        self._var_constraints = {}
        self._var_order = []
        self._var_order_phase_entries = []
        self._projected_gpu = None
        self._projected_gpu_context = None
        self._grid_scope_context = None
        self._split_step_scope_map = None
        self._vectorize_constraint_bundle = None
        self._shared_memory_constraint_bundle = None
        self._max_threads_constraint_bundle = None
        self._max_threads_per_block_constraint_bundle = None
        self._max_vthread_constraint_bundle = None
        self._split_structure_constraint_bundle = None
        self._min_thread_extent_constraint_bundle = None
        self._budget_specs = []
        self._budget_spec_by_name = {}
        self._budget_spec_by_factor = {}
        self._all_budget_names = []
        self._concrete_final_cache = {}
        self.constraint_set = ConstraintSet(self)
        self.var_order_planner = VarOrderPlanner(self)
        self.domain_propagator = DomainPropagator(self)
        self.param_sampler = ParamSampler(self)
        self._inspector = _ScheduleGeneratorInspector(self)
        self._vectorize_split_step_indices = self._find_vectorize_split_step_indices()

        self.constraint_set.preprocess()

    @classmethod
    def from_task_state(
        cls,
        task,
        state,
        hw_param=None,
        enabled_constraints=None,
        base_input=None,
        base_result=None,
    ):
        """task와 concrete state로부터 SymbolicState를 만들고 ScheduleGenerator 인스턴스를 반환한다."""
        from .symbolic_state_bridge import build_symbolic_state

        sym_state = build_symbolic_state(task, state)
        return cls(
            sym_state,
            hw_param=hw_param,
            enabled_constraints=enabled_constraints,
            task=task,
            base_input=base_input,
            base_result=base_result,
            base_state=state if base_input is None or base_result is None else None,
        )

    # ═══════════════════════════════════════════════════════════
    # Public workflow API
    # ═══════════════════════════════════════════════════════════

    def _check_all_final_with_concrete_result(self, sym_map, concrete_result):
        """심볼릭 최종 검사와 concrete lowering 검사를 합쳐 위반 목록을 만든다."""
        violations = []
        if 'innermost_split' in self._enabled:
            violations.extend(self.constraint_set._check_innermost_split(sym_map))
        if 'split_structure' in self._enabled:
            violations.extend(self.constraint_set._check_split_structure(sym_map))
        if 'min_thread_extent' in self._enabled:
            violations.extend(self.constraint_set._check_min_thread_extent(sym_map))

        if concrete_result is not None:
            violations.extend(concrete_result.get('violations', []))
            return violations

        if violations:
            return violations
        return self.constraint_set.check_all_pruning(sym_map)

    def check_all_hybrid(self, sym_map=None):
        """가능하면 concrete 검증까지, 아니면 pruning 제약까지만 검사한다."""
        concrete_result = self._get_concrete_final_result(sym_map)
        if concrete_result is not None:
            return self._check_all_final_with_concrete_result(sym_map, concrete_result)
        return self.constraint_set.check_all_pruning(sym_map)

    def _normalize_concrete_params(self, sym_map=None):
        """sym_map에서 concrete한 sp_/ur_ 값만 추려 정규화한다."""
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

    def _has_concrete_final_context(self):
        """concrete final 검증에 필요한 task/record/state 문맥이 있는지 확인한다."""
        if self._task is None:
            return False
        has_record_context = self._base_input is not None and self._base_result is not None
        return has_record_context or self._base_state is not None

    def _get_concrete_final_result(self, sym_map=None):
        """현재 파라미터를 concrete state로 내려 GPU 오류 검증 결과를 캐시해 반환한다."""
        if not self._has_concrete_final_context():
            return None

        params = self._normalize_concrete_params(sym_map)
        if params is None:
            return None

        cache_key = tuple(sorted(params.items()))
        cached = self._concrete_final_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        from .concrete_gpu_verify import lower_with_gpu_passes, verify_gpu_module_errors

        try:
            state = self.params_to_state(params)
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


    def _find_vectorize_split_step_indices(self):
        """SplitStep 바로 뒤에 vectorize AnnotationStep이 오는 step index 집합을 반환한다."""
        indices = set()
        if self.s._state is None:
            return indices
        steps = self.s._state.transform_steps
        for i in range(len(steps) - 1):
            s = steps[i]
            ns = steps[i + 1]
            if (s.type_key.split(".")[-1] == "SplitStep"
                    and ns.type_key.split(".")[-1] == "AnnotationStep"
                    and int(ns.annotation) == 2):
                indices.add(i)
        return indices

    def _build_split_domains(self):
        """모든 split 변수의 초기 도메인 사전을 만든다."""
        domains = {}
        for name in self._all_sp_names:
            step_idx = int(name.split("_")[1])
            extent = self._sp_extents.get(step_idx)
            domains[name] = [1, extent] if extent is not None else 1
        return domains

    def _build_budget_domains(self):
        """budget 유사 파라미터들의 초기 도메인 사전을 만든다."""
        return {
            spec['name']: [1, int(spec['limit'])]
            for spec in self._budget_specs
        }

    def _is_budget_name(self, name):
        """주어진 이름이 sampler 전용 budget 변수인지 반환한다."""
        return name in self._budget_spec_by_name

    def _get_budget_spec(self, name):
        """budget 변수 이름으로 sampler 스펙을 조회한다."""
        return self._budget_spec_by_name.get(name)

    def _enumerate_split_candidates(self, name, remaining):
        """남은 extent 안에서 split 변수 후보값들을 열거한다."""
        remaining = int(remaining)
        candidates = set(self.pm._divisors(remaining))
        if name in self.s._exception_split_names:
            candidates.add(int(self.hw['warp_size']))
        return sorted(value for value in candidates if 1 <= value <= remaining)

    def _get_dynamic_split_extent(self, step_idx, sym_map=None):
        """현재 sym_map 기준으로 split step의 동적 extent를 계산한다."""
        if sym_map is None:
            sym_map = self.s.sym_map
        expr = self.s._split_step_extents.get(step_idx)
        if expr is not None:
            val = eval_sym_extent(expr, sym_map)
            if val is not None:
                return int(val)
        return self._sp_extents.get(step_idx)

    def _get_group_remaining(self, step_idx, group_remaining, sym_map=None):
        """같은 split group에서 아직 남은 extent를 캐시와 함께 계산한다."""
        remaining = group_remaining.get(step_idx)
        if remaining is not None:
            return int(remaining)
        extent = self._get_dynamic_split_extent(step_idx, sym_map=sym_map)
        if extent is not None:
            group_remaining[step_idx] = int(extent)
        return extent

    def _build_dynamic_split_extents(self, sym_map=None):
        """모든 split step의 현재 동적 extent를 한 번에 계산한다."""
        if sym_map is None:
            sym_map = self.s.sym_map
        extents = {}
        for step_idx in self._sp_groups:
            extent = self._get_dynamic_split_extent(step_idx, sym_map=sym_map)
            if extent is not None:
                extents[step_idx] = int(extent)
        return extents

    def _get_split_extent_dependencies(self, step_idx):
        """split extent 식이 의존하는 sp_/ur_ 변수 집합을 추출한다."""
        expr = self.s._split_step_extents.get(step_idx)
        if expr is None:
            return set()
        expr_text = str(expr)
        return {
            name
            for name in re.findall(r"(?:sp|ur)_\d+(?:_\d+)?", expr_text)
            if name in self.s.sym_map
        }

    def _materialize_assignment_state(self, sym_map=None):
        """부분 할당을 실제 샘플링 상태처럼 전개해 params/domains/sym_map 스냅샷을 만든다."""
        requested = {}
        if sym_map is not None:
            requested = {name: int(value) for name, value in sym_map.items()}

        saved_sym_map = dict(self.s.sym_map)
        try:
            result = {}
            for name in self._all_sp_names:
                self.s.sym_map[name] = 1
            for name in self._ur_names:
                self.s.sym_map[name] = None

            domains = self._build_split_domains()
            domains.update(self._build_budget_domains())
            group_remaining = {}
            budget_remaining = {}
            effective_var_order = self.param_sampler._assign_initial_fixed_vars(
                self._var_order,
                domains,
                group_remaining,
                result,
            )

            for name, value in requested.items():
                if name in result and int(result[name]) != int(value):
                    raise ValueError(
                        f"Assignment {name}={value} conflicts with fixed value {result[name]}"
                    )

            for name in effective_var_order:
                if name not in requested:
                    continue

                value = int(requested[name])
                if self._is_budget_name(name):
                    result[name] = value
                    domains[name] = value
                    budget_remaining[name] = value
                    continue
                if name.startswith("ur_"):
                    if value not in self.pm.UNROLL_CANDIDATES:
                        raise ValueError(
                            f"{name}={value} is not a legal unroll candidate; "
                            f"candidates={self.pm.UNROLL_CANDIDATES}"
                        )
                    self.s.sym_map[name] = value
                    result[name] = value
                    continue

                step_idx = int(name.split("_")[1])
                extent = self._get_dynamic_split_extent(step_idx)

                if extent is None:
                    fixed_value = int(self.s.sym_map.get(name, 1))
                    if value != fixed_value:
                        raise ValueError(
                            f"{name} must remain fixed at {fixed_value}, got {value}"
                        )
                    result[name] = fixed_value
                    domains[name] = fixed_value
                    continue

                remaining = self._get_group_remaining(step_idx, group_remaining)
                candidates = self.param_sampler._get_split_candidates(
                    name,
                    domains,
                    group_remaining,
                    budget_remaining,
                    result,
                )

                if value not in candidates:
                    raise ValueError(
                        f"{name}={value} is not valid under the current assignment; "
                        f"candidates={candidates}"
                    )

                self.s.sym_map[name] = value
                result[name] = value
                domains[name] = value
                group_remaining[step_idx] = (remaining + value - 1) // value

                spec = self._budget_spec_by_factor.get(name)
                if spec is not None and spec['name'] in budget_remaining and value > 0:
                    budget_remaining[spec['name']] = int(budget_remaining[spec['name']]) // value

                constraint_indices = self._var_constraints.get(name, [])
                if constraint_indices:
                    self.domain_propagator.propagate_domain(name, domains)

            for name in self._ur_names:
                if name in result:
                    continue
                if name not in requested:
                    continue
                value = int(requested[name])
                if value not in self.pm.UNROLL_CANDIDATES:
                    raise ValueError(
                        f"{name}={value} is not a legal unroll candidate; "
                        f"candidates={self.pm.UNROLL_CANDIDATES}"
                    )
                self.s.sym_map[name] = value
                result[name] = value

            unknown = sorted(
                name
                for name in requested
                if name not in result and name not in self._ur_names and not self._is_budget_name(name)
            )
            if unknown:
                raise ValueError(f"Unknown or unapplied parameter names: {unknown}")

            return {
                'requested_params': dict(sorted(requested.items())),
                'params': dict(sorted(result.items())),
                'domains': self.domain_propagator.snapshot_domains(domains),
                'raw_domains': domains,
                'sym_map': dict(self.s.sym_map),
            }
        finally:
            self.s.sym_map = saved_sym_map

    def _build_observability_report(
        self,
        params,
        domains,
        include_constraints_text=False,
        include_vars=True,
        include_eval=True,
    ):
        """현재 할당과 도메인 상태를 관측용 리포트 구조로 정리한다."""
        domain_snapshot = self.domain_propagator.snapshot_domains(domains)
        analysis = self.domain_propagator.analyze_constraints_under_domains(domain_snapshot)
        constraints = {
            'text': None,
            'leftover': analysis['leftover_constraints'],
            'resolved_false': analysis['resolved_false_constraints'],
            'resolved_true_count': analysis['resolved_true_count'],
        }
        if include_constraints_text:
            constraints['text'] = self._get_constraints_with_assignment_str(
                params,
                include_vars=include_vars,
                include_eval=include_eval,
            )
        return {
            'assignment': {
                'params': dict(sorted(params.items())),
            },
            'domains': {
                'all': domain_snapshot,
                'fixed': analysis['fixed_values'],
                'remaining': analysis['remaining_domains'],
            },
            'constraints': constraints,
        }

    def _simplify_constraint_expr_text(self, constraint, record, sym_map):
        """제약 좌변 텍스트를 현재 sym_map 기준으로 치환·단순화한다."""
        return self._inspector.simplify_constraint_expr_text(constraint, record, sym_map)

    def _simplify_constraint_rhs_text(self, constraint, record, sym_map):
        """제약 우변 텍스트를 현재 sym_map 기준으로 치환·단순화한다."""
        return self._inspector.simplify_constraint_rhs_text(constraint, record, sym_map)

    def _format_expr(self, node, top_level=False):
        """표현식 노드를 inspector 포맷터를 통해 문자열로 바꾼다."""
        return self._inspector._format_expr(node, top_level=top_level)


    def _ensure_projected_gpu_constraints(self, kinds=None):
        """projected GPU 제약 번들을 필요 시 생성한다."""
        self.constraint_set._ensure_projected_gpu_constraints(kinds)

    def _get_var_order_phase_entries(self):
        """planner가 계산한 공개용 var-order phase entry 목록을 반환한다."""
        return self.var_order_planner.get_var_order_phase_entries()

    # ═══════════════════════════════════════════════════════════
    # 3) 제약 만족 파라미터 생성
    # ═══════════════════════════════════════════════════════════

    def next_unique_schedule(self, seen_state_fingerprints, rng=None):
        """이미 본 concrete state를 제외하고 다음 unique schedule을 반환한다."""
        return self.param_sampler.next_unique_schedule(
            seen_state_fingerprints,
            rng=rng,
        )

    def get_unique_search_stats(self):
        """현재 unique search 누적 통계를 반환한다."""
        return self.param_sampler.get_unique_search_stats()

    def params_to_state(self, params):
        """할당된 파라미터를 TVM auto_scheduler State로 변환해 반환한다 (concrete task 필요)."""
        if self._task is None:
            raise ValueError("params_to_state requires a concrete task context")

        normalized = self._normalize_concrete_params(params)
        if normalized is None:
            raise ValueError("params_to_state requires concrete integer sp_/ur_ values")

        from .concrete_gpu_verify import params_to_state_from_record, params_to_state_from_state
        if self._base_input is not None and self._base_result is not None:
            return params_to_state_from_record(
                self._task,
                self._base_input,
                self._base_result,
                normalized,
                split_extents=None,
            )
        split_extents = self._build_dynamic_split_extents(normalized)
        if self._base_state is not None:
            return params_to_state_from_state(
                self._task,
                self._base_state,
                normalized,
                split_extents=split_extents,
            )
        raise ValueError("params_to_state requires a base record or base state")


    def _get_constraint_records(self):
        """현재 constraint를 inspector record 목록으로 반환한다."""
        return self._inspector.get_constraint_records()

    def _get_constraints_str(self, include_vars=False, include_meta=False):
        """현재 constraint를 사람이 읽기 쉬운 문자열로 렌더링한다."""
        return self._inspector.get_constraints_str(
            include_vars=include_vars,
            include_meta=include_meta,
        )

    def _get_constraints_with_assignment_str(self, sym_map=None, include_vars=False, include_eval=True):
        """주어진 할당 아래에서 constraint를 치환해 문자열로 렌더링한다."""
        return self._inspector.get_constraints_with_assignment_str(
            sym_map=sym_map,
            include_vars=include_vars,
            include_eval=include_eval,
        )


#     # ------------------------------------------------------------------
#     # Deprecated
#     # ------------------------------------------------------------------

    def check_all_pruning(self, sym_map=None):
        """projected/심볼릭 pruning 제약만 검사해 위반 목록을 반환한다."""
        return self.constraint_set.check_all_pruning(sym_map)


    def check_all_final(self, sym_map=None):
        """가능한 concrete 검증까지 포함한 최종 위반 목록을 반환한다."""
        concrete_result = self._get_concrete_final_result(sym_map)
        return self._check_all_final_with_concrete_result(sym_map, concrete_result)


    def get_param_candidates(self, name, sym_map=None):
        """지정 변수에 대한 제약-만족 후보값과 현재 할당·도메인·제약 요약을 반환한다."""
        requested = {}
        if sym_map is not None:
            requested = dict(sym_map)
        requested.pop(name, None)

        assignment_state = self._materialize_assignment_state(requested)
        domains = assignment_state['raw_domains']

        if name in assignment_state['params']:
            candidates = [assignment_state['params'][name]]
        elif name.startswith("ur_"):
            candidates = list(self.pm.UNROLL_CANDIDATES)
        else:
            dom = domains.get(name)
            if dom is None:
                raise ValueError(f"Unknown parameter: {name}")
            if not isinstance(dom, list):
                candidates = [int(dom)]
            else:
                saved_sym_map = dict(self.s.sym_map)
                try:
                    self.s.sym_map = dict(assignment_state['sym_map'])
                    candidates = self.domain_propagator.candidate_values_for_domain(name, dom)
                    if candidates is None:
                        lo, hi = int(dom[0]), int(dom[1])
                        candidates = list(range(lo, hi + 1))
                    constraint_indices = self._var_constraints.get(name, [])
                    if constraint_indices:
                        candidates = self.domain_propagator.filter_by_constraints(
                            name,
                            candidates,
                            constraint_indices,
                            domains,
                        )
                finally:
                    self.s.sym_map = saved_sym_map

        report = self._build_observability_report(assignment_state['params'], domains)
        report['query'] = {
            'param_name': name,
            'requested_params': assignment_state['requested_params'],
        }
        report['candidates'] = list(candidates)
        return report

    def propagate_param_assignment(self, name, value, sym_map=None):
        """한 변수에 값을 할당한 뒤 도메인 전파를 수행하고 관측 리포트를 반환한다."""
        updated = {}
        if sym_map is not None:
            updated.update(sym_map)
        updated[name] = int(value)
        assignment_state = self._materialize_assignment_state(updated)
        report = self._build_observability_report(
            assignment_state['params'],
            assignment_state['raw_domains'],
        )
        report['query'] = {
            'param_name': name,
            'param_value': int(value),
            'requested_params': assignment_state['requested_params'],
        }
        return report

    def get_constraints_under_assignment(self, sym_map=None, include_vars=True, include_eval=True):
        """현재 할당 하에서 제약식 요약(텍스트·남은 제약 등)을 담은 관측 리포트를 반환한다."""
        assignment_state = self._materialize_assignment_state(sym_map)
        report = self._build_observability_report(
            assignment_state['params'],
            assignment_state['raw_domains'],
            include_constraints_text=True,
            include_vars=include_vars,
            include_eval=include_eval,
        )
        report['query'] = {
            'requested_params': assignment_state['requested_params'],
            'include_vars': bool(include_vars),
            'include_eval': bool(include_eval),
        }
        return report

    def get_full_var_order_entries(self):
        """플래너가 계산한 변수 순서와 phase 구간 정보를 반환한다."""
        phases = []
        param_order = []
        for phase_index, entry in enumerate(self._get_var_order_phase_entries()):
            phase_param_names = list(entry['param_names'])
            phase_start = len(param_order)
            param_order.extend(phase_param_names)
            phases.append({
                **entry,
                'phase_index': phase_index,
                'param_count': len(phase_param_names),
                'param_start': phase_start,
                'param_stop': len(param_order),
                'prefix_param_names': list(param_order),
            })
        return {
            'phase_count': len(phases),
            'param_order': param_order,
            'phases': phases,
        }

    def randomize_params(self, rng=None, max_retries=1):
        """제약을 만족하는 파라미터를 무작위로 한 번 샘플링해 반환한다."""
        return self.param_sampler.randomize_params(rng=rng, max_retries=max_retries)

    def reset_unique_search(self):
        """unique schedule 탐색 캐시를 초기화한다."""
        self.param_sampler.reset_unique_search()

    def randomize_params_prefix(self, stop_after_phase, rng=None, max_retries=1):
        """지정 phase까지 prefix만 샘플링한 관측 리포트(assignment, domains, constraints 등)를 반환한다."""
        return self.param_sampler.randomize_params_prefix(
            stop_after_phase,
            rng=rng,
            max_retries=max_retries,
        )


class _ScheduleGeneratorInspector:
    """ScheduleGenerator의 제약 포맷·관측용 내부 헬퍼."""

    def __init__(self, gen):
        """제너레이터를 받아 포맷/조회에 사용한다."""
        self.gen = gen

    def get_constraint_records(self):
        """현재 활성화된 제약식을 사람이 읽기 쉬운 dict 목록으로 반환."""
        g = self.gen
        scope_label_map = self._build_constraint_scope_label_map()
        records = []
        for idx, constraint in enumerate(g._constraints):
            op = "<=" if constraint['is_upper'] else ">="
            vars_in = sorted(constraint['vars'])
            block_scope = constraint.get('block_scope')
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
                # 'has_nonlinear': constraint['has_nonlinear'],
                'alias_entries': list(constraint.get('alias_entries', [])),
                'block_scope': block_scope,
                'block_scope_label': scope_label_map.get(block_scope),
            })
        return records

    def get_constraints_str(self, include_vars=False, include_meta=False):
        """constraint record를 scope 섹션까지 반영한 전체 문자열로 만든다."""
        records = self.get_constraint_records()
        if not any(record.get('block_scope') is not None for record in records):
            lines = []
            for record in records:
                lines.extend(
                    self._format_constraint_record_lines(
                        record,
                        include_aliases=True,
                        include_vars=include_vars,
                        include_meta=include_meta,
                    )
                )
            return "\n".join(lines)

        lines = []
        for section in self._group_constraint_records_by_scope(records):
            if lines:
                lines.append("")
            lines.append(f"[{section['label']}]")
            for record in section['records']:
                lines.extend(
                    self._format_constraint_record_lines(
                        record,
                        include_aliases=True,
                        include_vars=include_vars,
                        include_meta=include_meta,
                        indent="  ",
                    )
                )
        return "\n".join(lines)

    def get_constraints_with_assignment_str(self, sym_map=None, include_vars=False, include_eval=True):
        """constraint를 현재 할당 값으로 치환한 뒤 평가값까지 함께 문자열로 만든다."""
        g = self.gen
        if sym_map is None:
            sym_map = g.s.sym_map

        lines = []
        for constraint, record in zip(g._constraints, self.get_constraint_records()):
            instantiated_record = dict(record)
            instantiated_record['expr'] = self.simplify_constraint_expr_text(
                constraint, record, sym_map
            )
            instantiated_record['display_rhs'] = self.simplify_constraint_rhs_text(
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
                if lhs_val is None or rhs_val is None:
                    lines.append("  eval: (partial)")
                else:
                    lines.append(
                        f"  eval: {lhs_val} {instantiated_record['op']} {rhs_val}"
                    )
        return "\n".join(lines)

    def simplify_constraint_expr_text(self, constraint, record, sym_map):
        """constraint 좌변의 display text 또는 tree를 단순화된 문자열로 만든다."""
        display_text = constraint.get('display_text')
        if display_text is not None:
            return self._simplify_expr_text(display_text, sym_map)
        simplified = self._simplify_expr_node(constraint['tree'], sym_map)
        return self._format_tree(simplified)

    def simplify_constraint_rhs_text(self, constraint, record, sym_map):
        """constraint 우변의 display text를 단순화된 문자열로 만든다."""
        rhs = constraint.get('display_rhs')
        if rhs is None:
            rhs = record['display_rhs']
        if isinstance(rhs, int):
            return str(rhs)
        return self._simplify_expr_text(rhs, sym_map)


    def _format_alias_entry_lines(self, entry):
        """alias entry 하나를 출력용 여러 줄 문자열로 포맷한다."""
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
        """constraint 좌변을 display_text 우선 규칙으로 포맷한다."""
        display_text = constraint.get('display_text')
        if display_text:
            return self._format_display_text(display_text)
        return self._format_tree(constraint['tree'])

    def _format_constraint_record_lines(
        self,
        record,
        include_aliases=True,
        include_vars=False,
        include_meta=False,
        indent="",
    ):
        """constraint record 하나를 출력용 여러 줄 문자열로 바꾼다."""
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
        if include_meta:
            lines.append(f"{indent}  desc: {record['desc']}")
            # lines.append(f"{indent}  nonlinear: {record['has_nonlinear']}")
        if include_vars:
            vars_text = ", ".join(record['vars']) if record['vars'] else "(none)"
            lines.append(f"{indent}  vars: {vars_text}")
        return lines

    def _build_constraint_scope_label_map(self):
        """grid scope tuple을 출력용 section label로 매핑한다."""
        g = self.gen
        labels = {}
        next_index = 0
        for entry in g._get_var_order_phase_entries():
            scope = entry['grid_scope']
            if scope in labels:
                continue
            labels[scope] = f"grid_{next_index}: {g.constraint_set._format_block_scope(scope)}"
            next_index += 1
        for constraint in g._constraints:
            scope = constraint.get('block_scope')
            if scope is None or scope in labels:
                continue
            labels[scope] = f"grid_{next_index}: {g.constraint_set._format_block_scope(scope)}"
            next_index += 1
        return labels

    def _group_constraint_records_by_scope(self, records):
        """constraint record를 block scope별 섹션 순서로 묶는다."""
        grouped = {}
        order = []
        for record in records:
            scope = record.get('block_scope')
            if scope not in grouped:
                grouped[scope] = []
                order.append(scope)
            grouped[scope].append(record)

        sections = []
        for scope in order:
            label = grouped[scope][0].get('block_scope_label')
            if label is None:
                label = self.gen.constraint_set._format_block_scope(scope)
            sections.append({
                'scope': scope,
                'label': label,
                'records': grouped[scope],
            })
        return sections

    @staticmethod
    def _substitute_text(text, sym_map):
        """텍스트 식 안의 심볼 이름을 현재 값으로 치환한다."""
        substituted = str(text)
        for sym_name in sorted(sym_map.keys(), key=len, reverse=True):
            value = sym_map[sym_name]
            if value is None:
                continue
            substituted = substituted.replace(sym_name, str(value))
        return substituted

    def _simplify_expr_text(self, text, sym_map):
        """텍스트 식을 파싱해 가능한 범위에서 상수 접기까지 수행한다."""
        substituted = self._substitute_text(text, sym_map)
        normalized = " ".join(substituted.split())
        try:
            parsed = parse_expr_tree(normalized)
        except ValueError:
            return substituted
        simplified = self._simplify_expr_node(parsed, {})
        return self._format_expr(simplified, top_level=True)

    def _simplify_expr_node(self, node, sym_map):
        """ExprNode 트리를 재귀적으로 단순화한다."""
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
        """TVM PrimExpr 노드에 substitution과 analyzer simplify를 적용한다."""
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
            """단순화된 PrimExpr 안에 남은 자유변수를 다시 수집한다."""
            if isinstance(cur, tvm.tir.Var):
                var_map.setdefault(str(cur.name), cur)

        tvm.tir.stmt_functor.post_order_visit(simplified, visit)
        if not var_map:
            return ConstNode(int(simplified))
        return PrimExprNode(simplified)

    @staticmethod
    def _evaluate_display_rhs_value(constraint, sym_map):
        """constraint 우변의 표시값을 실제 정수로 평가한다."""
        rhs = constraint.get('display_rhs')
        if rhs is None:
            return constraint['rhs']
        if isinstance(rhs, int):
            return rhs
        return _ScheduleGeneratorInspector._evaluate_text_expr(rhs, sym_map)

    @staticmethod
    def _evaluate_display_lhs_value(constraint, sym_map):
        """constraint 좌변의 표시값을 실제 정수로 평가한다."""
        display_text = constraint.get('display_text')
        if display_text is None:
            vars_in = constraint['tree'].variables()
            if any(sym_map.get(name) is None for name in vars_in):
                return None
            try:
                return constraint['tree'].evaluate(sym_map)
            except Exception:
                return None
        return _ScheduleGeneratorInspector._evaluate_text_expr(display_text, sym_map)

    @staticmethod
    def _evaluate_text_expr(text, sym_map):
        """텍스트 식을 ExprNode 또는 SymExpr 경로로 평가한다."""
        normalized = " ".join(str(text).split())
        try:
            tree = parse_expr_tree(normalized)
        except ValueError:
            tree = None
        if tree is not None:
            vars_in = tree.variables()
            if any(sym_map.get(name) is None for name in vars_in):
                return None
            try:
                return tree.evaluate(sym_map)
            except Exception:
                return None
        evaluated = eval_sym_extent(SymExpr(normalized), sym_map)
        if isinstance(evaluated, str) and evaluated.startswith("EVAL_FAIL("):
            return None
        return evaluated

    def _format_tree(self, node):
        """표현식 트리를 case split까지 포함한 멀티라인 문자열로 포맷한다."""
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
        """긴 display text를 top-level 곱셈 기준으로 줄바꿈해 읽기 좋게 만든다."""
        parts = self._split_top_level(text, "*")
        if len(parts) <= 1:
            return text
        compact = " * ".join(parts)
        if len(compact) <= self.gen._FORMAT_WRAP_LIMIT:
            return compact
        return "\n".join([parts[0]] + [f"* {part}" for part in parts[1:]])

    def _split_top_level(self, text, sep):
        """괄호 깊이를 고려해 top-level 구분자 기준으로 문자열을 나눈다."""
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
        """ExprNode를 연산자 우선순위를 유지하며 문자열로 포맷한다."""
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
        """곱셈 체인을 평탄화해 operand 문자열 리스트로 바꾼다."""
        if isinstance(node, MulNode):
            return self._flatten_mul_parts(node.left) + self._flatten_mul_parts(node.right)
        if isinstance(node, ScaleMulNode):
            return self._flatten_mul_parts(node.child) + [str(node.scale)]
        return [self._format_mul_operand(node)]

    def _flatten_add_parts(self, node):
        """덧셈 체인을 평탄화해 operand 문자열 리스트로 바꾼다."""
        if isinstance(node, AddNode):
            return self._flatten_add_parts(node.left) + self._flatten_add_parts(node.right)
        return [self._format_expr(node)]

    def _format_mul_operand(self, node):
        """곱셈 문맥에서 하위 식을 필요한 괄호와 함께 포맷한다."""
        text = self._format_expr(node)
        if isinstance(node, (AddNode, SubNode, SumNode)):
            return f"({text})"
        return text

    def _format_div_operand(self, node, is_denominator):
        """나눗셈 문맥에서 분자/분모 operand를 안전하게 포맷한다."""
        text = self._format_expr(node)
        if isinstance(node, (AddNode, SubNode, SumNode)):
            return f"({text})"
        if is_denominator and isinstance(node, (MulNode, ScaleMulNode)):
            return f"({text})"
        return text

    def _format_joined(self, parts, separator, top_level=False):
        """연산자와 operand 조각들을 길이에 맞춰 한 줄 또는 여러 줄로 합친다."""
        compact = separator.join(parts)
        if not top_level or len(parts) <= 1 or len(compact) <= self.gen._FORMAT_WRAP_LIMIT:
            return compact
        lead = separator.strip()
        return "\n".join([parts[0]] + [f"{lead} {part}" for part in parts[1:]])

    def _format_call(self, name, args, top_level=False):
        """함수 호출 형태의 식을 길이에 맞춰 포맷한다."""
        compact = f"{name}(" + ", ".join(args) + ")"
        if not top_level or len(compact) <= self.gen._FORMAT_WRAP_LIMIT:
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
        """여러 줄 문자열 목록에 동일한 들여쓰기를 적용한다."""
        return [f"{prefix}{line}" for line in lines]

    @staticmethod
    def _normalize_expr_text(text):
        """표현식 문자열의 공백과 TVM 호출 표기를 정리한다."""
        text = text.replace("T.min(", "min(")
        text = text.replace("T.max(", "max(")
        text = re.sub(r",\s*", ", ", text)
        return text.strip()

    def _format_prim_expr(self, expr, top_level=False):
        """TVM PrimExpr를 사람이 읽기 쉬운 수식 문자열로 포맷한다."""
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
        """PrimExpr의 곱셈 체인을 평탄화한다."""
        import tvm

        if isinstance(expr, tvm.tir.Mul):
            return self._flatten_prim_mul(expr.a) + self._flatten_prim_mul(expr.b)
        return [self._format_prim_mul_operand(expr)]

    def _flatten_prim_add(self, expr):
        """PrimExpr의 덧셈 체인을 평탄화한다."""
        import tvm

        if isinstance(expr, tvm.tir.Add):
            return self._flatten_prim_add(expr.a) + self._flatten_prim_add(expr.b)
        return [self._format_prim_expr(expr)]

    def _extract_prim_ceil_chain(self, expr):
        """ceildiv가 중첩된 PrimExpr 체인을 풀어 바닥 식과 분모 목록을 분리한다."""
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
        """ceildiv 체인의 바닥 식을 문자열로 포맷한다."""
        import tvm

        if isinstance(expr, tvm.tir.Sub) and self._is_prim_int_one(expr.b):
            return self._format_prim_expr(expr.a)
        if isinstance(expr, tvm.tir.IntImm):
            return str(int(expr) + 1)
        return None

    @staticmethod
    def _is_prim_int_one(expr):
        """PrimExpr가 정수 상수 1인지 판별한다."""
        import tvm

        return isinstance(expr, tvm.tir.IntImm) and int(expr) == 1

    def _format_prim_mul_operand(self, expr):
        """PrimExpr 곱셈 문맥의 operand를 필요한 괄호와 함께 포맷한다."""
        import tvm

        text = self._format_prim_expr(expr)
        if isinstance(expr, (tvm.tir.Add, tvm.tir.Sub)):
            return f"({text})"
        return text

    def _format_prim_div_operand(self, expr, is_denominator):
        """PrimExpr 나눗셈 문맥의 operand를 필요한 괄호와 함께 포맷한다."""
        import tvm

        text = self._format_prim_expr(expr)
        if isinstance(expr, (tvm.tir.Add, tvm.tir.Sub)):
            return f"({text})"
        if is_denominator and isinstance(expr, (tvm.tir.Mul, tvm.tir.FloorDiv)):
            return f"({text})"
        return text
