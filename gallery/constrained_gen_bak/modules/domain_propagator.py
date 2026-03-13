from itertools import product

from .expr_nodes import parse_expr_tree


class DomainPropagator:
    def __init__(self, gen):
        self.gen = gen

    @staticmethod
    def _snapshot_domains(domains):
        snapshot = {}
        for name, dom in domains.items():
            if isinstance(dom, list):
                snapshot[name] = [int(dom[0]), int(dom[-1])]
            else:
                snapshot[name] = int(dom)
        return snapshot

    @staticmethod
    def _fixed_and_remaining_from_domains(domains):
        fixed = {}
        remaining = {}
        for name, dom in domains.items():
            if isinstance(dom, list):
                lo, hi = int(dom[0]), int(dom[1])
                if lo == hi:
                    fixed[name] = lo
                else:
                    remaining[name] = [lo, hi]
            else:
                fixed[name] = int(dom)
        return fixed, remaining

    def analyze_constraints_under_domains(self, domains):
        g = self.gen
        domain_snapshot = self._snapshot_domains(domains)
        fixed_values, remaining_domains = self._fixed_and_remaining_from_domains(domain_snapshot)
        records = g.get_constraint_records()

        analysis = {
            'fixed_values': dict(sorted(fixed_values.items())),
            'remaining_domains': dict(sorted(remaining_domains.items())),
            'leftover_constraints': [],
            'resolved_false_constraints': [],
            'resolved_true_count': 0,
        }

        for constraint, record in zip(g._constraints, records):
            constraint_fixed = {
                name: fixed_values[name]
                for name in record['vars']
                if name in fixed_values
            }
            remaining_vars = [
                name for name in record['vars']
                if name not in constraint_fixed
            ]

            expr_text = " ".join(
                g._simplify_constraint_expr_text(constraint, record, constraint_fixed).split()
            )
            rhs_text = " ".join(
                str(g._simplify_constraint_rhs_text(constraint, record, constraint_fixed)).split()
            )
            constraint_text = f"{record['kind']}: {expr_text} {record['op']} {rhs_text}"

            remaining_domain_subset = {
                name: remaining_domains[name]
                for name in remaining_vars
                if name in remaining_domains
            }
            lhs_lo, lhs_hi = self._analyze_constraint_bounds(
                constraint,
                expr_text,
                constraint_fixed,
                remaining_domain_subset,
            )
            rhs = constraint['rhs']
            if constraint['is_upper']:
                always_true = lhs_hi <= rhs
                always_false = lhs_lo > rhs
            else:
                always_true = lhs_lo >= rhs
                always_false = lhs_hi < rhs

            if always_true:
                analysis['resolved_true_count'] += 1
                continue

            item = {
                'kind': record['kind'],
                'constraint': constraint_text,
            }
            if remaining_vars:
                item['vars'] = remaining_vars
                if remaining_domain_subset:
                    item['domains'] = remaining_domain_subset

            if always_false:
                analysis['resolved_false_constraints'].append(item)
            else:
                analysis['leftover_constraints'].append(item)

        return analysis

    def _analyze_constraint_bounds(self, constraint, expr_text, fixed_values, remaining_domains):
        if not remaining_domains:
            value = constraint['tree'].evaluate(fixed_values)
            return value, value

        enum_budget = 1
        for lo, hi in remaining_domains.values():
            enum_budget *= int(hi) - int(lo) + 1
            if enum_budget > 256:
                break

        if enum_budget <= 256:
            names = list(sorted(remaining_domains.keys()))
            ranges = [
                range(int(remaining_domains[name][0]), int(remaining_domains[name][1]) + 1)
                for name in names
            ]
            lo_val = None
            hi_val = None
            for values in product(*ranges):
                assignment = dict(fixed_values)
                assignment.update(zip(names, values))
                cur = constraint['tree'].evaluate(assignment)
                lo_val = cur if lo_val is None else min(lo_val, cur)
                hi_val = cur if hi_val is None else max(hi_val, cur)
            if lo_val is not None and hi_val is not None:
                return lo_val, hi_val

        try:
            parsed = parse_expr_tree(expr_text)
            return parsed.interval(remaining_domains)
        except ValueError:
            pass

        interval_domains = dict(fixed_values)
        interval_domains.update(remaining_domains)
        return constraint['tree'].interval(interval_domains)

    @staticmethod
    def _apply_upper_bound_to_domain(dom, hi_allowed):
        cur_lo, cur_hi = dom
        hi_allowed = int(hi_allowed)
        if hi_allowed < cur_lo:
            hi_allowed = cur_lo
        if hi_allowed < cur_hi:
            dom[1] = hi_allowed
            return True
        return False

    @staticmethod
    def _get_sym_value(sym_map, name):
        value = sym_map.get(name, 1)
        if value is None:
            return 1
        return int(value)

    def _propagate_product_form_upper(self, constraint, other_var, dom, sym_map):
        meta = constraint.get('product_form_meta')
        if not meta:
            return False

        factors = meta['factors']
        if other_var not in factors:
            return False

        scale = int(meta['scale'])
        if scale <= 0:
            return True

        base = scale
        skipped = False
        for factor_name in factors:
            if factor_name == other_var and not skipped:
                skipped = True
                continue
            base *= self._get_sym_value(sym_map, factor_name)
            if base > constraint['rhs']:
                break

        hi_allowed = constraint['rhs'] // base if base > 0 else dom[1]
        self._apply_upper_bound_to_domain(dom, hi_allowed)
        return True

    def _propagate_split_structure_upper(self, constraint, other_var, dom, sym_map):
        fast_path = constraint.get('fast_path')
        if not fast_path or fast_path.get('kind') != 'split_structure':
            return False

        extent = fast_path['extent']
        sym_name = fast_path['sym_name']
        dependency_names = fast_path['dependency_names']

        if other_var == sym_name:
            denom = 1
            for dep_name in dependency_names:
                denom *= self._get_sym_value(sym_map, dep_name)
                if denom >= extent:
                    denom = extent
                    break
            denom = min(denom, extent)
            hi_allowed = (extent + denom - 1) // denom
            self._apply_upper_bound_to_domain(dom, hi_allowed)
            return True

        if other_var not in dependency_names:
            return False

        split_value = self._get_sym_value(sym_map, sym_name)
        if split_value <= 1:
            return True

        numerator = (extent - 1) // (split_value - 1)
        rest = 1
        skipped = False
        for dep_name in dependency_names:
            if dep_name == other_var and not skipped:
                skipped = True
                continue
            rest *= self._get_sym_value(sym_map, dep_name)
            if rest > numerator:
                break

        hi_allowed = numerator // rest if rest > 0 else dom[1]
        self._apply_upper_bound_to_domain(dom, hi_allowed)
        return True

    def _propagate_upper_domain_fast(self, constraint, other_var, dom, sym_map):
        if self._propagate_split_structure_upper(constraint, other_var, dom, sym_map):
            return True
        if self._propagate_product_form_upper(constraint, other_var, dom, sym_map):
            return True
        return False

    def _candidate_values_for_domain(self, var_name, dom):
        g = self.gen
        if not var_name.startswith("sp_"):
            return None

        parts = var_name.split("_")
        if len(parts) != 3:
            return None

        step_idx = int(parts[1])
        pos = int(parts[2])
        extent = g._sp_extents.get(step_idx)
        group_names = g._sp_groups.get(step_idx)
        if extent is None or not group_names or pos >= len(group_names):
            return None

        remaining = extent
        for prev_name in group_names[:pos]:
            chosen = self._get_sym_value(g.s.sym_map, prev_name)
            remaining = (remaining + chosen - 1) // chosen

        candidates = g.pm._divisors(remaining)
        if var_name in g._innermost_names:
            candidates = [
                value
                for value in candidates
                if value <= g.hw['max_innermost_split_factor']
            ]

        lo, hi = int(dom[0]), int(dom[1])
        candidates = [value for value in candidates if lo <= value <= hi]
        return candidates or None

    def propagate_domain(self, assigned_name, domains):
        g = self.gen
        constraint_indices = g._var_constraints.get(assigned_name, [])
        sym_map = g.s.sym_map

        for ci in constraint_indices:
            c = g._constraints[ci]
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
                    if self._propagate_upper_domain_fast(c, other_var, dom, sym_map):
                        continue

                    candidates = self._candidate_values_for_domain(other_var, dom)
                    if candidates is not None:
                        old_val = sym_map.get(other_var, 1)
                        sym_map[other_var] = candidates[-1]
                        lhs_at_hi = tree.evaluate(sym_map)
                        if lhs_at_hi <= rhs:
                            sym_map[other_var] = old_val
                            continue

                        sym_map[other_var] = candidates[0]
                        lhs_at_lo = tree.evaluate(sym_map)
                        if lhs_at_lo > rhs:
                            sym_map[other_var] = old_val
                            dom[1] = candidates[0]
                            continue

                        lo_idx, hi_idx = 0, len(candidates) - 1
                        while lo_idx < hi_idx:
                            mid_idx = (lo_idx + hi_idx + 1) // 2
                            sym_map[other_var] = candidates[mid_idx]
                            lhs_val = tree.evaluate(sym_map)
                            if lhs_val <= rhs:
                                lo_idx = mid_idx
                            else:
                                hi_idx = mid_idx - 1

                        sym_map[other_var] = old_val
                        best = candidates[lo_idx]
                        if best < cur_hi:
                            dom[1] = best
                        continue

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

    def filter_by_constraints(self, var_name, candidates, constraint_indices, domains):
        g = self.gen
        if not candidates:
            return candidates

        interval_domains = {}
        for v, d in domains.items():
            interval_domains[v] = d

        upper_constraints = []
        lower_constraints = []
        for ci in constraint_indices:
            c = g._constraints[ci]
            if c['is_upper']:
                upper_constraints.append(c)
            else:
                lower_constraints.append(c)

        max_valid_idx = len(candidates) - 1
        for c in upper_constraints:
            idx = self._bisect_upper(var_name, candidates, c, interval_domains)
            max_valid_idx = min(max_valid_idx, idx)

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
