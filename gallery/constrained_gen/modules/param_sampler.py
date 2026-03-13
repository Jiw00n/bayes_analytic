from itertools import product as itertools_product


class ParamSampler:
    def __init__(self, gen):
        self.gen = gen

    def _try_assign_initial_fixed_vars(self, var_order, domains, group_remaining, result):
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

    def _randomize_params_with_order(
        self,
        var_order,
        rng=None,
        max_retries=1,
        assign_unroll=True,
        require_full_validation=True,
        return_domains=False,
    ):
        import random as _random

        g = self.gen
        if rng is None:
            rng = _random.Random()

        innermost_limit = g.hw['max_innermost_split_factor']

        violations = None
        for _ in range(max_retries):
            result = {}
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

            effective_var_order = self._try_assign_initial_fixed_vars(
                var_order,
                domains,
                group_remaining,
                result,
            )

            ok = True
            for name in effective_var_order:
                parts = name.split("_")
                step_idx = int(parts[1])

                extent = g._sp_extents.get(step_idx)
                if extent is None:
                    result[name] = g.s.sym_map[name]
                    domains[name] = g.s.sym_map[name]
                    continue

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
                    ok = False
                    break

                chosen = rng.choice(candidates)
                g.s.sym_map[name] = chosen
                result[name] = chosen
                domains[name] = chosen

                group_remaining[step_idx] = (remaining + chosen - 1) // chosen

                if constraint_indices:
                    g.domain_propagator.propagate_domain(name, domains)

            if not ok:
                continue

            if assign_unroll:
                for name in g._ur_names:
                    chosen = rng.choice(g.pm.UNROLL_CANDIDATES)
                    g.s.sym_map[name] = chosen
                    result[name] = chosen

            if not require_full_validation:
                if return_domains:
                    return result, g.domain_propagator._snapshot_domains(domains)
                return result

            if assign_unroll or not g._ur_names:
                violations = g.check_all_hybrid(result)
            else:
                violations = g.check_all_exact(result)
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
        g = self.gen
        return self._randomize_params_with_order(
            g._var_order,
            rng=rng,
            max_retries=max_retries,
            assign_unroll=True,
            require_full_validation=True,
        )

    def randomize_params_prefix(self, stop_after_phase, rng=None, max_retries=1):
        g = self.gen
        stop_idx = g.var_order_planner._resolve_var_order_stop_index(stop_after_phase)
        prefix = []
        phases = []
        for idx, entry in enumerate(g.get_var_order_phase_entries()):
            prefix.extend(entry['vars'])
            phases.append(entry)
            if idx == stop_idx:
                break
        params, domains = self._randomize_params_with_order(
            prefix,
            rng=rng,
            max_retries=max_retries,
            assign_unroll=False,
            require_full_validation=False,
            return_domains=True,
        )

        analysis = g.domain_propagator.analyze_constraints_under_domains(domains)

        return {
            'stop_after_phase': stop_after_phase,
            'resolved_stop_phase_name': phases[-1]['name'] if phases else None,
            'resolved_stop_phase_family': phases[-1]['family'] if phases else None,
            'var_order': prefix,
            'params': params,
            'phases': phases,
            'domains': domains,
            'fixed_values': analysis['fixed_values'],
            'remaining_domains': analysis['remaining_domains'],
            'leftover_constraints': analysis['leftover_constraints'],
            'resolved_false_constraints': analysis['resolved_false_constraints'],
            'resolved_true_count': analysis['resolved_true_count'],
        }

    def enumerate_all_params(self, max_results=100_000):
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
