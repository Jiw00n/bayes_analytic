import random as _random
from itertools import product as itertools_product


class ParamSampler:
    def __init__(self, gen):
        """ScheduleGenerator를 받아 해당 제너레이터에 대한 파라미터 샘플링을 수행한다."""
        self.gen = gen
        self._unique_search_state = None

    @staticmethod
    def _copy_domains(domains):
        return {
            name: list(dom) if isinstance(dom, list) else int(dom)
            for name, dom in domains.items()
        }

    @staticmethod
    def _copy_group_remaining(group_remaining):
        return {step_idx: int(value) for step_idx, value in group_remaining.items()}

    def _restore_sym_map(self, saved_sym_map):
        self.gen.s.sym_map.clear()
        self.gen.s.sym_map.update(saved_sym_map)

    def reset_unique_search(self):
        """unique schedule 탐색 캐시를 초기화한다."""
        self._unique_search_state = None

    def get_unique_search_stats(self):
        """현재 unique search 누적 통계를 반환한다."""
        if self._unique_search_state is None:
            return {
                'emitted_unique': 0,
                'duplicates_skipped': 0,
                'exhausted_prefixes': 0,
            }
        return dict(self._unique_search_state['stats'])

    def _mark_prefix_exhausted(self, state, prefix_key):
        exhausted = state['exhausted_prefixes']
        if prefix_key in exhausted:
            return
        exhausted.add(prefix_key)
        state['stats']['exhausted_prefixes'] += 1

    @staticmethod
    def _prefix_assignment_key(search_order, result, depth):
        return tuple(
            (name, int(result[name]))
            for name in search_order[:depth]
        )

    def _ensure_unique_search_state(self):
        g = self.gen
        if self._unique_search_state is not None:
            return self._unique_search_state

        for name in g._ur_names:
            g.s.sym_map[name] = None

        base_result, base_domains, base_group_remaining, search_var_order = (
            self._initialize_unique_search_base_state(g._var_order)
        )
        search_order = list(search_var_order) + list(g._ur_names)
        self._unique_search_state = {
            'base_result': dict(base_result),
            'base_domains': self._copy_domains(base_domains),
            'base_group_remaining': self._copy_group_remaining(base_group_remaining),
            'base_sym_map': dict(g.s.sym_map),
            'search_order': search_order,
            'exhausted_prefixes': set(),
            'stats': {
                'emitted_unique': 0,
                'duplicates_skipped': 0,
                'exhausted_prefixes': 0,
            },
        }
        return self._unique_search_state

    # ------------------------------------------------------------------
    # Sampling-state initialization
    # ------------------------------------------------------------------

    def _build_split_domains(self):
        """모든 split 파라미터에 대한 초기 도메인 [1, extent] 딕셔너리를 만든다."""
        g = self.gen
        domains = {}
        for name in g._all_sp_names:
            step_idx = int(name.split("_")[1])
            extent = g._sp_extents.get(step_idx)
            domains[name] = [1, extent] if extent is not None else 1
        return domains

    def _initialize_unique_search_base_state(self, var_order):
        """unique search용 초기 상태를 만든다.

        retry sampler와 달리 singleton 도메인 변수를 미리 할당하지 않고,
        rollout 중 실제 해당 변수 차례가 왔을 때 할당한다.
        """
        g = self.gen
        result = {}
        for name in g._all_sp_names:
            g.s.sym_map[name] = 1

        domains = self._build_split_domains()
        group_remaining = {
            step_idx: extent
            for step_idx, extent in g._sp_extents.items()
        }
        return result, domains, group_remaining, list(var_order)

    # ------------------------------------------------------------------
    # Per-variable sampling
    # ------------------------------------------------------------------

    def _get_split_candidates(self, name, domains, group_remaining):
        g = self.gen
        innermost_limit = g.hw['max_innermost_split_factor']
        parts = name.split("_")
        step_idx = int(parts[1])
        extent = g._sp_extents.get(step_idx)

        if extent is None:
            return [int(g.s.sym_map.get(name, 1))]

        remaining = group_remaining.get(step_idx, extent)
        candidates = g.pm._divisors(remaining)

        if name in g._innermost_names:
            candidates = [c for c in candidates if c <= innermost_limit]

        dom = domains.get(name)
        if isinstance(dom, list):
            lo, hi = int(dom[0]), int(dom[1])
            candidates = [c for c in candidates if lo <= c <= hi]

        constraint_indices = g._var_constraints.get(name, [])
        if constraint_indices and candidates:
            candidates = g.domain_propagator.filter_by_constraints(
                name,
                candidates,
                constraint_indices,
                domains,
            )
        return candidates

    def _get_search_candidates(self, name, domains, group_remaining):
        g = self.gen
        if name.startswith("ur_"):
            return [int(value) for value in g.pm.UNROLL_CANDIDATES]
        return self._get_split_candidates(name, domains, group_remaining)

    def _apply_search_assignment(self, name, chosen, result, domains, group_remaining):
        g = self.gen
        chosen = int(chosen)
        g.s.sym_map[name] = chosen
        result[name] = chosen

        if name.startswith("ur_"):
            return

        parts = name.split("_")
        step_idx = int(parts[1])
        extent = g._sp_extents.get(step_idx)
        if extent is None:
            domains[name] = chosen
            return

        remaining = group_remaining.get(step_idx, extent)
        domains[name] = chosen
        group_remaining[step_idx] = (remaining + chosen - 1) // chosen

        constraint_indices = g._var_constraints.get(name, [])
        if constraint_indices:
            g.domain_propagator.propagate_domain(name, domains)

    # ------------------------------------------------------------------
    # Final validation and reporting
    # ------------------------------------------------------------------

    def _assign_unroll_vars(self, result, rng):
        """unroll 파라미터(ur_*)에 UNROLL_CANDIDATES 중 값을 무작위로 넣는다."""
        g = self.gen
        for name in g._ur_names:
            chosen = rng.choice(g.pm.UNROLL_CANDIDATES)
            g.s.sym_map[name] = chosen
            result[name] = chosen

    def _validate_sample(self, result, assign_unroll, require_full_validation):
        """샘플 result에 대해 hybrid 또는 exact 검증을 수행해 위반 목록을 반환한다."""
        g = self.gen
        if not require_full_validation:
            return []
        if assign_unroll or not g._ur_names:
            return g.check_all_hybrid(result)
        # return g.check_all_exact(result)

    def _propagate_exhausted_path(self, state, path_records):
        for record in reversed(path_records):
            prefix_key = record['prefix_key']
            if prefix_key in state['exhausted_prefixes']:
                continue
            child_prefixes = [
                prefix_key + ((record['name'], int(candidate)),)
                for candidate in record['candidates']
            ]
            if child_prefixes and all(
                child_prefix in state['exhausted_prefixes']
                for child_prefix in child_prefixes
            ):
                self._mark_prefix_exhausted(state, prefix_key)
                continue
            break

    def _random_unique_rollout(
        self,
        state,
        seen_state_fingerprints,
        rng,
    ):
        g = self.gen
        # breakpoint()
        search_order = state['search_order']
        root_prefix = tuple()
        if root_prefix in state['exhausted_prefixes']:
            return None

        self._restore_sym_map(state['base_sym_map'])
        result = dict(state['base_result'])
        domains = self._copy_domains(state['base_domains'])
        group_remaining = self._copy_group_remaining(state['base_group_remaining'])
        path_records = []

        for depth, name in enumerate(search_order):
            prefix_key = self._prefix_assignment_key(search_order, result, depth)
            if prefix_key in state['exhausted_prefixes']:
                return None

            candidates = self._get_search_candidates(name, domains, group_remaining)
            live_candidates = [
                int(candidate)
                for candidate in candidates
                if prefix_key + ((name, int(candidate)),) not in state['exhausted_prefixes']
            ]

            if not live_candidates:
                self._mark_prefix_exhausted(state, prefix_key)
                self._propagate_exhausted_path(state, path_records)
                return None
            


            # breakpoint()
            logits = [rng.uniform(0, 1) for _ in range(len(live_candidates))]

            chosen = max(zip(live_candidates, logits), key=lambda x: x[1])[0]

            path_records.append({
                'prefix_key': prefix_key,
                'name': name,
                'candidates': list(candidates),
            })
            self._apply_search_assignment(
                name,
                chosen,
                result,
                domains,
                group_remaining,
            )
        # breakpoint()
        leaf_prefix = self._prefix_assignment_key(search_order, result, len(search_order))
        self._mark_prefix_exhausted(state, leaf_prefix)
        violations = self._validate_sample(
            result,
            assign_unroll=True,
            require_full_validation=True,
        )
        if violations:
            self._propagate_exhausted_path(state, path_records)
            return None

        concrete_state = g.params_to_state(result)
        from .concrete_gpu_verify import concrete_state_fingerprint

        fingerprint = concrete_state_fingerprint(g._task, concrete_state)
        if fingerprint in seen_state_fingerprints:
            state['stats']['duplicates_skipped'] += 1
            self._propagate_exhausted_path(state, path_records)
            return None

        state['stats']['emitted_unique'] += 1
        return {
            'params': dict(result),
            'state': concrete_state,
            'fingerprint': fingerprint,
        }

    def next_unique_schedule(self, seen_state_fingerprints, rng=None):
        """이미 본 concrete state를 제외하고 다음 unique schedule을 찾아 반환한다."""
        g = self.gen
        if g._task is None:
            raise ValueError("next_unique_schedule requires a concrete task context")

        if rng is None:
            rng = _random.Random()

        state = self._ensure_unique_search_state()
        saved_sym_map = dict(g.s.sym_map)
        try:
            while tuple() not in state['exhausted_prefixes']:
                payload = self._random_unique_rollout(
                    state,
                    seen_state_fingerprints,
                    rng,
                )
                if payload is not None:
                    return payload
            return None
        finally:
            self._restore_sym_map(saved_sym_map)

    # ------------------------------------------------------------------
    # Deprecated
    # ------------------------------------------------------------------

    @staticmethod
    def _restore_sym_value(sym_map, name, old_value):
        if old_value is None:
            sym_map.pop(name, None)
            return
        sym_map[name] = old_value

    def _assign_initial_fixed_vars(self, var_order, domains, group_remaining, result):
        """이미 도메인이 1개 값으로 고정된 변수들을 먼저 할당하고, 미할당 변수 목록을 반환한다."""
        g = self.gen
        innermost_limit = g.hw['max_innermost_split_factor']
        progress = True

        while progress:
            progress = False
            for name in var_order:
                if name in result:
                    continue

                dom = domains.get(name)
                if isinstance(dom, list):
                    lo, hi = int(dom[0]), int(dom[1])
                    if lo != hi:
                        continue
                    fixed_value = lo
                else:
                    fixed_value = int(dom)

                parts = name.split("_")
                step_idx = int(parts[1])
                extent = g._sp_extents.get(step_idx)

                if extent is None:
                    g.s.sym_map[name] = fixed_value
                    domains[name] = fixed_value
                    result[name] = fixed_value
                    progress = True
                    continue

                pos = int(parts[2])
                group_names = g._sp_groups.get(step_idx, [])
                if any(prev_name not in result for prev_name in group_names[:pos]):
                    continue

                remaining = group_remaining.get(step_idx, extent)
                candidates = g.pm._divisors(remaining)
                if name in g._innermost_names:
                    candidates = [c for c in candidates if c <= innermost_limit]
                if fixed_value not in candidates:
                    continue

                g.s.sym_map[name] = fixed_value
                domains[name] = fixed_value
                result[name] = fixed_value
                group_remaining[step_idx] = (remaining + fixed_value - 1) // fixed_value

                constraint_indices = g._var_constraints.get(name, [])
                if constraint_indices:
                    g.domain_propagator.propagate_domain(name, domains)
                progress = True

        return [name for name in var_order if name not in result]

    def _initialize_split_sampling_state(self, var_order):
        """split 도메인·그룹 잔여량을 세팅하고 고정 변수를 할당한 뒤 (result, domains, group_remaining, 남은 변수 순서)를 반환한다."""
        g = self.gen
        result = {}
        for name in g._all_sp_names:
            g.s.sym_map[name] = 1

        domains = self._build_split_domains()
        group_remaining = {
            step_idx: extent
            for step_idx, extent in g._sp_extents.items()
        }
        effective_var_order = self._assign_initial_fixed_vars(
            var_order,
            domains,
            group_remaining,
            result,
        )
        return result, domains, group_remaining, effective_var_order

    def _sample_split_var(self, name, domains, group_remaining, rng, result):
        """한 split 변수에 대해 제약을 만족하는 후보 중 하나를 무작위로 골라 할당한다. 성공 여부를 반환."""
        g = self.gen
        innermost_limit = g.hw['max_innermost_split_factor']
        parts = name.split("_")
        step_idx = int(parts[1])
        extent = g._sp_extents.get(step_idx)

        if extent is None:
            result[name] = g.s.sym_map[name]
            domains[name] = g.s.sym_map[name]
            return True

        remaining = group_remaining.get(step_idx, extent)
        candidates = g.pm._divisors(remaining)

        if name in g._innermost_names:
            candidates = [c for c in candidates if c <= innermost_limit]

        dom = domains.get(name)
        if isinstance(dom, list):
            if dom[1] < candidates[-1]:
                candidates = [c for c in candidates if c <= dom[1]]
            if dom[0] > candidates[0]:
                candidates = [c for c in candidates if c >= dom[0]]

        constraint_indices = g._var_constraints.get(name, [])
        if constraint_indices:
            candidates = g.domain_propagator.filter_by_constraints(
                name,
                candidates,
                constraint_indices,
                domains,
            )

        if not candidates:
            return False

        chosen = rng.choice(candidates)
        g.s.sym_map[name] = chosen
        result[name] = chosen
        domains[name] = chosen
        group_remaining[step_idx] = (remaining + chosen - 1) // chosen

        if constraint_indices:
            g.domain_propagator.propagate_domain(name, domains)
        return True

    def _build_prefix_report(self, stop_after_phase, phases, prefix, params, domains):
        """prefix 구간 샘플링 결과를 관측 가능한 리포트 딕셔너리로 만든다."""
        g = self.gen
        report = g._build_observability_report(params, domains)
        report['query'] = {
            'requested_stop_after_phase': stop_after_phase,
        }
        report['phase_selection'] = {
            'resolved_phase_name': phases[-1]['phase_name'] if phases else None,
            'resolved_phase_family': phases[-1]['phase_family'] if phases else None,
            'resolved_phase_index': phases[-1]['phase_index'] if phases else None,
        }
        report['param_order'] = list(prefix)
        report['phases'] = phases
        return report

    def randomize_params_with_order(
        self,
        var_order,
        rng=None,
        max_retries=1,
        assign_unroll=True,
        require_full_validation=True,
        return_domains=False,
    ):
        """지정한 변수 순서대로 제약을 만족하는 파라미터를 무작위 샘플링한다. 실패 시 재시도."""
        import random as _random

        g = self.gen
        if rng is None:
            rng = _random.Random()

        violations = None
        for _ in range(max_retries):
            result, domains, group_remaining, effective_var_order = (
                self._initialize_split_sampling_state(var_order)
            )

            ok = True
            for name in effective_var_order:
                if not self._sample_split_var(name, domains, group_remaining, rng, result):
                    ok = False
                    break

            if not ok:
                continue

            if assign_unroll:
                self._assign_unroll_vars(result, rng)

            violations = self._validate_sample(
                result,
                assign_unroll=assign_unroll,
                require_full_validation=require_full_validation,
            )
            if not require_full_validation:
                if return_domains:
                    return result, g.domain_propagator.snapshot_domains(domains)
                return result

            if not violations:
                return result

        if require_full_validation:
            raise RuntimeError(
                f"Failed to find valid params after {max_retries} retries. "
                f"Last violations: {violations}"
            )
        raise RuntimeError(
            f"Failed to assign requested var prefix after {max_retries} retries."
        )

    def randomize_params(self, rng=None, max_retries=1):
        """전체 변수 순서로 파라미터를 한 번 무작위 샘플링해 할당 결과를 반환한다."""
        g = self.gen
        return self.randomize_params_with_order(
            g._var_order,
            rng=rng,
            max_retries=max_retries,
            assign_unroll=True,
            require_full_validation=True,
        )

    def randomize_params_prefix(self, stop_after_phase, rng=None, max_retries=1):
        """지정 phase까지의 prefix 변수만 샘플링하고 관측 리포트(assignment, domains 등)를 반환한다."""
        g = self.gen
        stop_idx = g.var_order_planner.resolve_var_order_stop_index(stop_after_phase)
        prefix = []
        phases = []
        for idx, entry in enumerate(g._get_var_order_phase_entries()):
            prefix.extend(entry['param_names'])
            phases.append({
                **entry,
                'phase_index': idx,
                'param_count': len(entry['param_names']),
                'param_start': len(prefix) - len(entry['param_names']),
                'param_stop': len(prefix),
                'prefix_param_names': list(prefix),
            })
            if idx == stop_idx:
                break
        params, domains = self.randomize_params_with_order(
            prefix,
            rng=rng,
            max_retries=max_retries,
            assign_unroll=False,
            require_full_validation=False,
            return_domains=True,
        )
        return self._build_prefix_report(stop_after_phase, phases, prefix, params, domains)

    def _enumerate_all_params(self, max_results=100_000):
        """제약을 만족하는 split 할당을 모두 열거한다 (최대 max_results개). 디버그/검증용."""
        g = self.gen
        innermost_limit = g.hw['max_innermost_split_factor']
        sp_results = []

        def _dfs(var_idx, result, domains, group_remaining):
            if len(sp_results) >= max_results:
                return

            if var_idx == len(g._var_order):
                for name, val in result.items():
                    g.s.sym_map[name] = val
                violations = g.check_all_exact(result)
                if not violations:
                    sp_results.append(dict(result))
                return

            name = g._var_order[var_idx]
            parts = name.split("_")
            step_idx = int(parts[1])

            extent = g._sp_extents.get(step_idx)
            if extent is None:
                result[name] = g.s.sym_map.get(name, 1)
                domains[name] = result[name]
                _dfs(var_idx + 1, result, domains, group_remaining)
                del result[name]
                del domains[name]
                return

            remaining = group_remaining.get(step_idx, extent)
            candidates = g.pm._divisors(remaining)

            if name in g._innermost_names:
                candidates = [c for c in candidates if c <= innermost_limit]

            dom = domains.get(name)
            if isinstance(dom, list):
                if dom[1] < candidates[-1]:
                    candidates = [c for c in candidates if c <= dom[1]]
                if dom[0] > candidates[0]:
                    candidates = [c for c in candidates if c >= dom[0]]

            constraint_indices = g._var_constraints.get(name, [])
            if constraint_indices:
                candidates = g.domain_propagator.filter_by_constraints(
                    name, candidates, constraint_indices, domains
                )

            if not candidates:
                return

            old_sym = g.s.sym_map.get(name, 1)
            old_remaining = group_remaining.get(step_idx, extent)

            for chosen in candidates:
                if len(sp_results) >= max_results:
                    return

                g.s.sym_map[name] = chosen
                result[name] = chosen

                saved_domains = {}
                for k, v in domains.items():
                    if isinstance(v, list):
                        saved_domains[k] = list(v)
                domains[name] = chosen

                group_remaining[step_idx] = (remaining + chosen - 1) // chosen

                if constraint_indices:
                    g.domain_propagator.propagate_domain(name, domains)

                _dfs(var_idx + 1, result, domains, group_remaining)

                del result[name]
                g.s.sym_map[name] = old_sym
                group_remaining[step_idx] = old_remaining
                for k, saved_v in saved_domains.items():
                    domains[k] = saved_v
                domains.pop(name, None)

        for name in g._all_sp_names:
            g.s.sym_map[name] = 1

        domains = {}
        for name in g._all_sp_names:
            parts = name.split("_")
            step_idx = int(parts[1])
            ext = g._sp_extents.get(step_idx)
            if ext is not None:
                domains[name] = [1, ext]
            else:
                domains[name] = 1

        group_remaining = {}
        for step_idx, ext in g._sp_extents.items():
            group_remaining[step_idx] = ext

        _dfs(0, {}, domains, group_remaining)

        if not sp_results:
            return []

        ur_names = sorted(g._ur_names)
        if not ur_names:
            return sp_results

        ur_combos = list(itertools_product(
            *[g.pm.UNROLL_CANDIDATES for _ in ur_names]
        ))

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
