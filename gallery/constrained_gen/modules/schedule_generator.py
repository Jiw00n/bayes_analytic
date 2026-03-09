"""
schedule_generator — ScheduleGenerator: HW 제약 기반 스케줄 파라미터 생성.

제약 목록:
  1) max_vectorize_bytes
  2) max_shared_memory
  3) max_threads
  4) max_vthread
  5) max_innermost_split
"""
from .sym_types import SymExpr, eval_sym_extent, builtins_min
from .param_manager import SymParamManager
from .expr_nodes import (
    ExprNode, ConstNode, VarNode, MulNode, AddNode, SubNode,
    MinNode, CeilDivNode, ScaleMulNode, SumNode,
    parse_expr_tree,
)


class ScheduleGenerator:
    """
    HW_PARAM 기반 제약식을 구축하고, 제약을 만족하는 파라미터를 생성하는 생성기.
    """

    DEFAULT_HW_PARAM = {
        'max_vector_bytes': 16,
        'max_shared_memory_per_block': 49152,
        'max_threads_per_block': 1024,
        'max_vthread_extent': 8,
        'max_innermost_split_factor': 64,
    }

    ALL_CONSTRAINT_KINDS = (
        'vectorize', 'shared_memory', 'max_threads',
        'vthread', 'innermost_split',
    )

    def __init__(self, sym_state, hw_param=None, enabled_constraints=None):
        self.s = sym_state
        self.hw = dict(self.DEFAULT_HW_PARAM)
        if hw_param is not None:
            self.hw.update(hw_param)
        self.pm = SymParamManager(sym_state)

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
        self._preprocess()

    # ═══════════════════════════════════════════════════════════
    # 1) 제약식 빌드
    # ═══════════════════════════════════════════════════════════

    def build_vectorize_constraints(self):
        limit = self.hw['max_vector_bytes']
        constraints = []
        for sid, iid, ext in self.s.get_vectorize_extents():
            dtype_bytes = self.s.stages[sid].dtype_bytes
            constraints.append({
                'stage_id': sid,
                'iter_id': iid,
                'sym_extent': ext,
                'dtype_bytes': dtype_bytes,
                'limit': limit,
                'desc': f"vectorize s{sid}.i{iid} ({self.s.stages[sid].op_name}): "
                        f"extent*{dtype_bytes} ≤ {limit}",
            })
        return constraints

    def check_vectorize(self, sym_map=None):
        if sym_map is None:
            sym_map = self.s.sym_map
        violations = []
        for c in self.build_vectorize_constraints():
            val = eval_sym_extent(c['sym_extent'], sym_map)
            if isinstance(val, int) and val * c['dtype_bytes'] > c['limit']:
                violations.append(f"{c['desc']}: actual={val}*{c['dtype_bytes']}={val*c['dtype_bytes']}")
        return violations

    def build_shared_memory_constraints(self):
        limit = self.hw['max_shared_memory_per_block']
        items = []
        for sid, op_name, ext in self.s.get_shared_memory_extents():
            dtype_bytes = self.s.stages[sid].dtype_bytes
            items.append({
                'stage_id': sid,
                'op_name': op_name,
                'sym_extent': ext,
                'dtype_bytes': dtype_bytes,
            })
        return {
            'items': items,
            'limit': limit,
            'desc': f"shared_memory: sum(extent*dtype_bytes) ≤ {limit}",
        }

    def check_shared_memory(self, sym_map=None):
        if sym_map is None:
            sym_map = self.s.sym_map
        c = self.build_shared_memory_constraints()
        total = 0
        parts = []
        for item in c['items']:
            val = eval_sym_extent(item['sym_extent'], sym_map)
            if isinstance(val, int):
                bytes_used = val * item['dtype_bytes']
                total += bytes_used
                parts.append(f"{item['op_name']}={val}*{item['dtype_bytes']}={bytes_used}")
        if total > c['limit']:
            return [f"{c['desc']}: actual={total} ({' + '.join(parts)})"]
        return []

    def build_max_threads_constraints(self):
        limit = self.hw['max_threads_per_block']
        constraints = []
        for sid, iid, ext in self.s.get_thread_extents():
            constraints.append({
                'stage_id': sid,
                'iter_id': iid,
                'sym_extent': ext,
                'limit': limit,
                'desc': f"max_threads s{sid}.i{iid} ({self.s.stages[sid].op_name}): "
                        f"extent ≤ {limit}",
            })
        return constraints

    def check_max_threads(self, sym_map=None):
        if sym_map is None:
            sym_map = self.s.sym_map
        violations = []
        for c in self.build_max_threads_constraints():
            val = eval_sym_extent(c['sym_extent'], sym_map)
            if isinstance(val, int) and val > c['limit']:
                violations.append(f"{c['desc']}: actual={val}")
        return violations

    def build_vthread_constraints(self):
        limit = self.hw['max_vthread_extent']
        constraints = []
        for sid, iid, ext in self.s.get_vthread_extents():
            constraints.append({
                'stage_id': sid,
                'iter_id': iid,
                'sym_extent': ext,
                'limit': limit,
                'desc': f"max_vthread s{sid}.i{iid} ({self.s.stages[sid].op_name}): "
                        f"extent ≤ {limit}",
            })
        return constraints

    def check_vthread(self, sym_map=None):
        if sym_map is None:
            sym_map = self.s.sym_map
        violations = []
        for c in self.build_vthread_constraints():
            val = eval_sym_extent(c['sym_extent'], sym_map)
            if isinstance(val, int) and val > c['limit']:
                violations.append(f"{c['desc']}: actual={val}")
        return violations

    def build_innermost_split_constraints(self):
        limit = self.hw['max_innermost_split_factor']
        sp_groups = self.pm._build_sp_groups()
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

    def check_innermost_split(self, sym_map=None):
        if sym_map is None:
            sym_map = self.s.sym_map
        violations = []
        for c in self.build_innermost_split_constraints():
            val = sym_map.get(c['sym_name'])
            if val is not None and isinstance(val, int) and val > c['limit']:
                violations.append(f"{c['desc']}: actual={val}")
        return violations

    def check_all(self, sym_map=None):
        violations = []
        if 'vectorize' in self._enabled:
            violations.extend(self.check_vectorize(sym_map))
        if 'shared_memory' in self._enabled:
            violations.extend(self.check_shared_memory(sym_map))
        if 'max_threads' in self._enabled:
            violations.extend(self.check_max_threads(sym_map))
        if 'vthread' in self._enabled:
            violations.extend(self.check_vthread(sym_map))
        if 'innermost_split' in self._enabled:
            violations.extend(self.check_innermost_split(sym_map))
        return violations

    # ═══════════════════════════════════════════════════════════
    # 2) 전처리
    # ═══════════════════════════════════════════════════════════

    def _preprocess(self):
        sp_groups = self.pm._build_sp_groups()
        sp_extents = self.pm._build_sp_extents(sp_groups)

        innermost_limit = self.hw['max_innermost_split_factor']
        innermost_names = set()
        if 'innermost_split' in self._enabled:
            for step_idx, names in sp_groups.items():
                innermost_names.add(names[-1])

        self._sp_groups = sp_groups
        self._sp_extents = sp_extents
        self._ur_names = [n for n in self.s.sym_map if n.startswith("ur_")]
        self._all_sp_names = []
        for step_idx in sorted(sp_groups.keys()):
            self._all_sp_names.extend(sp_groups[step_idx])

        self._innermost_names = innermost_names

        # 제약식 파싱
        self._constraints = []
        self._var_constraints = {}

        def _add_constraint(expr_tree, rhs, kind, desc, is_upper=True):
            idx = len(self._constraints)
            vars_in = expr_tree.variables()
            has_nonlinear = self._has_nonlinear(expr_tree)
            self._constraints.append({
                'tree': expr_tree,
                'rhs': rhs,
                'vars': vars_in,
                'kind': kind,
                'desc': desc,
                'is_upper': is_upper,
                'has_nonlinear': has_nonlinear,
            })
            for v in vars_in:
                self._var_constraints.setdefault(v, []).append(idx)

        # (a) vectorize
        if 'vectorize' in self._enabled:
            for c in self.build_vectorize_constraints():
                tree = parse_expr_tree(str(c['sym_extent']))
                if c['dtype_bytes'] != 1:
                    tree = ScaleMulNode(tree, c['dtype_bytes'])
                _add_constraint(tree, c['limit'], 'vectorize', c['desc'], is_upper=True)

        # (b) shared memory
        if 'shared_memory' in self._enabled:
            sm = self.build_shared_memory_constraints()
            if sm['items']:
                children = []
                for item in sm['items']:
                    tree = parse_expr_tree(str(item['sym_extent']))
                    if item['dtype_bytes'] != 1:
                        tree = ScaleMulNode(tree, item['dtype_bytes'])
                    children.append(tree)
                sum_tree = SumNode(children) if len(children) > 1 else children[0]
                _add_constraint(sum_tree, sm['limit'], 'shared_memory', sm['desc'], is_upper=True)

        # (c) max_threads
        if 'max_threads' in self._enabled:
            for c in self.build_max_threads_constraints():
                tree = parse_expr_tree(str(c['sym_extent']))
                _add_constraint(tree, c['limit'], 'max_threads', c['desc'], is_upper=True)

        # (d) vthread
        if 'vthread' in self._enabled:
            for c in self.build_vthread_constraints():
                tree = parse_expr_tree(str(c['sym_extent']))
                _add_constraint(tree, c['limit'], 'vthread', c['desc'], is_upper=True)

        # 변수 할당 순서
        self._compute_var_order()

    @staticmethod
    def _has_nonlinear(node):
        if isinstance(node, (MinNode, CeilDivNode)):
            return True
        if isinstance(node, (MulNode, AddNode, SubNode)):
            return ScheduleGenerator._has_nonlinear(node.left) or ScheduleGenerator._has_nonlinear(node.right)
        if isinstance(node, ScaleMulNode):
            return ScheduleGenerator._has_nonlinear(node.child)
        if isinstance(node, SumNode):
            return any(ScheduleGenerator._has_nonlinear(c) for c in node.children)
        return False

    def _compute_var_order(self):
        shared_vars = set()
        thread_vars = set()
        other_vars = set()

        for ci, c in enumerate(self._constraints):
            kind = c['kind']
            vs = c['vars']
            if kind == 'shared_memory':
                shared_vars.update(vs)
            elif kind == 'max_threads':
                thread_vars.update(vs)
            else:
                other_vars.update(vs)

        var_freq = {}
        for v in self._all_sp_names:
            var_freq[v] = len(self._var_constraints.get(v, []))

        group_priority = {}
        for step_idx, group in self._sp_groups.items():
            group_set = set(group)

            in_shared = bool(group_set & shared_vars)
            in_thread = bool(group_set & thread_vars)

            if in_shared:
                cat = 0
            elif in_thread:
                cat = 1
            else:
                cat = 2

            min_nonlinear = True
            total_freq = 0
            for v in group:
                total_freq += var_freq.get(v, 0)
                for ci in self._var_constraints.get(v, []):
                    if not self._constraints[ci]['has_nonlinear']:
                        min_nonlinear = False

            group_priority[step_idx] = (cat, min_nonlinear, -total_freq)

        sorted_steps = sorted(
            self._sp_groups.keys(),
            key=lambda si: group_priority.get(si, (3, True, 0))
        )

        self._var_order = []
        for step_idx in sorted_steps:
            self._var_order.extend(self._sp_groups[step_idx])

    # ═══════════════════════════════════════════════════════════
    # 3) 제약 만족 파라미터 생성
    # ═══════════════════════════════════════════════════════════

    def randomize_params(self, rng=None, max_retries=1):
        """모든 HW 제약을 만족하는 파라미터를 랜덤 생성.

        Args:
            rng: random.Random 인스턴스 또는 None
            max_retries: 사후 rejection 최대 재시도 횟수

        Returns:
            dict: {sym_name: value}

        Raises:
            RuntimeError: max_retries 초과 시
        """
        import random as _random
        if rng is None:
            rng = _random.Random()

        innermost_limit = self.hw['max_innermost_split_factor']

        violations = None
        for attempt in range(max_retries):
            result = {}
            for name in self._all_sp_names:
                self.s.sym_map[name] = 1

            domains = {}
            for name in self._all_sp_names:
                parts = name.split("_")
                step_idx = int(parts[1])
                ext = self._sp_extents.get(step_idx)
                if ext is not None:
                    domains[name] = [1, ext]
                else:
                    domains[name] = 1

            group_remaining = {}
            for step_idx, ext in self._sp_extents.items():
                group_remaining[step_idx] = ext

            ok = True
            for name in self._var_order:
                parts = name.split("_")
                step_idx = int(parts[1])
                length_idx = int(parts[2])

                extent = self._sp_extents.get(step_idx)
                if extent is None:
                    result[name] = self.s.sym_map[name]
                    domains[name] = self.s.sym_map[name]
                    continue

                remaining = group_remaining.get(step_idx, extent)
                candidates = self.pm._divisors(remaining)

                if name in self._innermost_names:
                    candidates = [c for c in candidates if c <= innermost_limit]

                dom = domains.get(name)
                if isinstance(dom, list):
                    if dom[1] < candidates[-1]:
                        candidates = [c for c in candidates if c <= dom[1]]
                    if dom[0] > candidates[0]:
                        candidates = [c for c in candidates if c >= dom[0]]

                constraint_indices = self._var_constraints.get(name, [])
                if constraint_indices:
                    candidates = self._filter_by_constraints(
                        name, candidates, constraint_indices, domains)

                if not candidates:
                    ok = False
                    break

                chosen = rng.choice(candidates)
                self.s.sym_map[name] = chosen
                result[name] = chosen
                domains[name] = chosen

                group_remaining[step_idx] = (remaining + chosen - 1) // chosen

                if constraint_indices:
                    self._propagate_domain(name, domains)

            if not ok:
                continue

            # unroll
            for name in self._ur_names:
                chosen = rng.choice(self.pm.UNROLL_CANDIDATES)
                self.s.sym_map[name] = chosen
                result[name] = chosen

            violations = self.check_all()
            if not violations:
                return result

        raise RuntimeError(
            f"Failed to find valid params after {max_retries} retries. "
            f"Last violations: {violations}")

    def _propagate_domain(self, assigned_name, domains):
        constraint_indices = self._var_constraints.get(assigned_name, [])
        sym_map = self.s.sym_map

        for ci in constraint_indices:
            c = self._constraints[ci]
            tree = c['tree']
            rhs = c['rhs']
            is_upper = c['is_upper']

            for other_var in c['vars']:
                if other_var == assigned_name:
                    continue
                dom = domains.get(other_var)
                if not isinstance(dom, list):
                    continue

                cur_lo, cur_hi = dom

                if is_upper:
                    old_val = sym_map.get(other_var, 1)
                    sym_map[other_var] = cur_hi
                    lhs_at_hi = tree.evaluate(sym_map)
                    if lhs_at_hi <= rhs:
                        sym_map[other_var] = old_val
                        continue

                    sym_map[other_var] = cur_lo
                    lhs_at_lo = tree.evaluate(sym_map)
                    if lhs_at_lo > rhs:
                        sym_map[other_var] = old_val
                        dom[1] = cur_lo
                        continue

                    lo, hi = cur_lo, cur_hi
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        sym_map[other_var] = mid
                        lhs_val = tree.evaluate(sym_map)
                        if lhs_val <= rhs:
                            lo = mid
                        else:
                            hi = mid - 1

                    sym_map[other_var] = old_val
                    if lo < cur_hi:
                        dom[1] = lo

                else:
                    old_val = sym_map.get(other_var, 1)
                    sym_map[other_var] = cur_lo
                    lhs_at_lo = tree.evaluate(sym_map)
                    if lhs_at_lo >= rhs:
                        sym_map[other_var] = old_val
                        continue

                    sym_map[other_var] = cur_hi
                    lhs_at_hi = tree.evaluate(sym_map)
                    if lhs_at_hi < rhs:
                        # other_var를 최대로 올려도 LHS가 RHS에 못 미침.
                        # 이는 다른 미확정 변수의 값에 따라 달라질 수 있으므로
                        # 도메인을 축소하지 않고 건너뜀.
                        sym_map[other_var] = old_val
                        continue

                    lo, hi = cur_lo, cur_hi
                    while lo < hi:
                        mid = (lo + hi) // 2
                        sym_map[other_var] = mid
                        lhs_val = tree.evaluate(sym_map)
                        if lhs_val >= rhs:
                            hi = mid
                        else:
                            lo = mid + 1

                    sym_map[other_var] = old_val
                    if lo > cur_lo:
                        dom[0] = lo

    def _filter_by_constraints(self, var_name, candidates, constraint_indices, domains):
        if not candidates:
            return candidates

        interval_domains = {}
        for v, d in domains.items():
            interval_domains[v] = d

        upper_constraints = []
        lower_constraints = []
        for ci in constraint_indices:
            c = self._constraints[ci]
            if c['is_upper']:
                upper_constraints.append(c)
            else:
                lower_constraints.append(c)

        max_valid_idx = len(candidates) - 1
        for c in upper_constraints:
            idx = self._bisect_upper(var_name, candidates, c, interval_domains)
            max_valid_idx = builtins_min(max_valid_idx, idx)

        min_valid_idx = 0
        for c in lower_constraints:
            idx = self._bisect_lower(var_name, candidates, c, interval_domains)
            min_valid_idx = max(min_valid_idx, idx)

        if min_valid_idx > max_valid_idx:
            return []

        return candidates[min_valid_idx:max_valid_idx + 1]

    def _bisect_upper(self, var_name, candidates, constraint, interval_domains):
        tree = constraint['tree']
        rhs = constraint['rhs']

        lo, hi = 0, len(candidates) - 1

        test_dom = dict(interval_domains)
        test_dom[var_name] = candidates[hi]
        lhs_min, _ = tree.interval(test_dom)
        if lhs_min <= rhs:
            return hi

        test_dom[var_name] = candidates[lo]
        lhs_min, _ = tree.interval(test_dom)
        if lhs_min > rhs:
            return -1

        while lo < hi:
            mid = (lo + hi + 1) // 2
            test_dom[var_name] = candidates[mid]
            lhs_min, _ = tree.interval(test_dom)
            if lhs_min <= rhs:
                lo = mid
            else:
                hi = mid - 1

        return lo

    def _bisect_lower(self, var_name, candidates, constraint, interval_domains):
        tree = constraint['tree']
        rhs = constraint['rhs']

        lo, hi = 0, len(candidates) - 1

        test_dom = dict(interval_domains)
        test_dom[var_name] = candidates[lo]
        _, lhs_max = tree.interval(test_dom)
        if lhs_max >= rhs:
            return lo

        test_dom[var_name] = candidates[hi]
        _, lhs_max = tree.interval(test_dom)
        if lhs_max < rhs:
            return hi + 1

        while lo < hi:
            mid = (lo + hi) // 2
            test_dom[var_name] = candidates[mid]
            _, lhs_max = tree.interval(test_dom)
            if lhs_max >= rhs:
                hi = mid
            else:
                lo = mid + 1

        return lo

    # ═══════════════════════════════════════════════════════════
    # 4) DFS 전수 열거
    # ═══════════════════════════════════════════════════════════

    def enumerate_all_params(self, max_results=100_000):
        """DFS로 모든 제약-만족 SP 파라미터 조합을 열거한다.

        unroll 변수는 SP 열거 후 카르테시안 곱으로 결합한다.

        Args:
            max_results: 최대 결과 수 (안전 장치)

        Returns:
            list[dict]: 각 원소는 {sp_X_Y: int, ..., ur_X: int, ...} 매핑
        """
        from itertools import product as itertools_product

        innermost_limit = self.hw['max_innermost_split_factor']
        sp_results = []

        def _dfs(var_idx, result, domains, group_remaining):
            if len(sp_results) >= max_results:
                return

            if var_idx == len(self._var_order):
                for name, val in result.items():
                    self.s.sym_map[name] = val
                violations = self.check_all({**result})
                if not violations:
                    sp_results.append(dict(result))
                return

            name = self._var_order[var_idx]
            parts = name.split("_")
            step_idx = int(parts[1])

            extent = self._sp_extents.get(step_idx)
            if extent is None:
                result[name] = self.s.sym_map.get(name, 1)
                domains[name] = result[name]
                _dfs(var_idx + 1, result, domains, group_remaining)
                del result[name]
                del domains[name]
                return

            remaining = group_remaining.get(step_idx, extent)
            candidates = self.pm._divisors(remaining)

            if name in self._innermost_names:
                candidates = [c for c in candidates if c <= innermost_limit]

            dom = domains.get(name)
            if isinstance(dom, list):
                if dom[1] < candidates[-1]:
                    candidates = [c for c in candidates if c <= dom[1]]
                if dom[0] > candidates[0]:
                    candidates = [c for c in candidates if c >= dom[0]]

            constraint_indices = self._var_constraints.get(name, [])
            if constraint_indices:
                candidates = self._filter_by_constraints(
                    name, candidates, constraint_indices, domains)

            if not candidates:
                return

            old_sym = self.s.sym_map.get(name, 1)
            old_remaining = group_remaining.get(step_idx, extent)

            for chosen in candidates:
                if len(sp_results) >= max_results:
                    return

                self.s.sym_map[name] = chosen
                result[name] = chosen

                saved_domains = {}
                for k, v in domains.items():
                    if isinstance(v, list):
                        saved_domains[k] = list(v)
                domains[name] = chosen

                group_remaining[step_idx] = (remaining + chosen - 1) // chosen

                if constraint_indices:
                    self._propagate_domain(name, domains)

                _dfs(var_idx + 1, result, domains, group_remaining)

                del result[name]
                self.s.sym_map[name] = old_sym
                group_remaining[step_idx] = old_remaining
                for k, saved_v in saved_domains.items():
                    domains[k] = saved_v
                domains.pop(name, None)

        # 초기 상태
        for name in self._all_sp_names:
            self.s.sym_map[name] = 1

        domains = {}
        for name in self._all_sp_names:
            parts = name.split("_")
            step_idx = int(parts[1])
            ext = self._sp_extents.get(step_idx)
            if ext is not None:
                domains[name] = [1, ext]
            else:
                domains[name] = 1

        group_remaining = {}
        for step_idx, ext in self._sp_extents.items():
            group_remaining[step_idx] = ext

        _dfs(0, {}, domains, group_remaining)

        # unroll 카르테시안 곱
        if not sp_results:
            return []

        ur_names = sorted(self._ur_names)
        if not ur_names:
            return sp_results

        ur_combos = list(itertools_product(
            *[self.pm.UNROLL_CANDIDATES for _ in ur_names]))

        all_results = []
        for sp in sp_results:
            for ur_vals in ur_combos:
                if len(all_results) >= max_results:
                    return all_results
                combined = dict(sp)
                for i, ur_name in enumerate(ur_names):
                    combined[ur_name] = ur_vals[i]
                all_results.append(combined)

        return all_results
