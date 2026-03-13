from .expr_nodes import AddNode, CeilDivNode, ConstNode, MaxNode, MinNode, MulNode, ScaleMulNode, SubNode, VarNode


class VarOrderPlanner:
    def __init__(self, gen):
        self.gen = gen

    def compute_var_order(self):
        g = self.gen
        legacy_order = self._compute_legacy_var_order()
        phase_entries = self._build_var_order_phase_entries()

        ordered = []
        seen = set()
        normalized_entries = []
        for entry in phase_entries:
            phase_vars = []
            for name in entry['vars']:
                if name in seen:
                    continue
                seen.add(name)
                ordered.append(name)
                phase_vars.append(name)
            normalized_entries.append({
                'name': entry['name'],
                'family': entry['family'],
                'label': entry['label'],
                'grid_scope': entry['grid_scope'],
                'grid_scope_label': entry['grid_scope_label'],
                'vars': phase_vars,
            })

        for name in legacy_order:
            if name in seen:
                continue
            seen.add(name)
            ordered.append(name)

        g._var_order_phase_entries = normalized_entries
        g._var_order = ordered

    def _build_var_order_phase_entries(self):
        g = self.gen
        max_thread_items = []
        if 'max_threads' in g._enabled:
            max_thread_items = g.constraint_set.build_max_threads_constraints()['items']

        max_vthread_items = []
        if 'max_vthread' in g._enabled:
            max_vthread_items = g.constraint_set.build_max_vthread_constraints()['items']

        vectorize_items = []
        if 'vectorize' in g._enabled:
            vectorize_items = g.constraint_set.build_vectorize_constraints()['items']

        shared_vars = set()
        if 'shared_memory' in g._enabled:
            shared_vars = set(g.constraint_set.build_shared_memory_constraints()['tree'].variables())

        (
            scope_infos,
            max_thread_items_by_scope,
            thread_axis_items_by_scope,
            vthread_items_by_scope,
        ) = self._build_grid_scope_infos(
            max_thread_items,
            max_vthread_items,
        )
        if not scope_infos:
            scope_infos = [{
                'grid_index': 0,
                'grid_scope': tuple(),
                'grid_scope_label': f"grid_0: {g.constraint_set._format_block_scope(tuple())}",
                'is_main_compute_anchor': True,
            }]
            max_thread_items_by_scope = {}
            thread_axis_items_by_scope = {}
            vthread_items_by_scope = {}

        thread_owned_vars = self._build_scope_owned_vars(scope_infos, thread_axis_items_by_scope)
        vthread_owned_vars = self._build_scope_owned_vars(scope_infos, vthread_items_by_scope)
        initial_domains = self._build_initial_domains()
        family_labels = dict(g.VAR_ORDER_PHASE_FAMILIES)
        phase_entries = []
        anchor_scope = next(
            (info['grid_scope'] for info in scope_infos if info.get('is_main_compute_anchor')),
            scope_infos[0]['grid_scope'],
        )

        vectorize_vars = []
        for item in vectorize_items:
            g.constraint_set._append_unique_vars(
                vectorize_vars,
                g.constraint_set._ordered_unique_tree_variables(item['tree']),
            )
        vectorize_var_set = set(vectorize_vars)
        shared_step_indices = self._collect_step_indices_for_vars(shared_vars)

        for scope_info in scope_infos:
            scope = scope_info['grid_scope']
            max_thread_scope_items = max_thread_items_by_scope.get(scope, [])
            max_vthread_scope_items = vthread_items_by_scope.get(scope, [])
            execution_items = max_thread_scope_items + max_vthread_scope_items
            execution_owned = []
            g.constraint_set._append_unique_vars(execution_owned, thread_owned_vars.get(scope, []))
            g.constraint_set._append_unique_vars(execution_owned, vthread_owned_vars.get(scope, []))
            execution_owned_set = set(execution_owned)

            thread_pure = self._collect_scoped_product_phase_vars(
                max_thread_scope_items,
                set(thread_owned_vars.get(scope, [])),
                want_scale=1,
            )
            vthread_pure = self._collect_scoped_product_phase_vars(
                max_vthread_scope_items,
                set(vthread_owned_vars.get(scope, [])),
                want_scale=1,
            )
            has_execution_pure_product = bool(thread_pure or vthread_pure)

            if has_execution_pure_product:
                phase_entries.append(
                    self._make_phase_entry(
                        scope_info,
                        'execution_max_threads_pure_product',
                        family_labels['execution_max_threads_pure_product'],
                        thread_pure,
                    )
                )
                phase_entries.append(
                    self._make_phase_entry(
                        scope_info,
                        'execution_max_vthread_pure_product',
                        family_labels['execution_max_vthread_pure_product'],
                        vthread_pure,
                    )
                )
                phase_entries.append(
                    self._make_phase_entry(
                        scope_info,
                        'execution_block_split_structure',
                        family_labels['execution_block_split_structure'],
                        self._collect_split_phase_vars_for_steps(
                            self._collect_step_indices_for_vars(execution_owned_set),
                            inner_first=True,
                        ),
                    )
                )
            else:
                phase_entries.append(
                    self._make_phase_entry(
                        scope_info,
                        'execution_non_product_direct_arm',
                        family_labels['execution_non_product_direct_arm'],
                        self._collect_non_product_phase_vars(
                            execution_items,
                            execution_owned_set,
                            initial_domains,
                            want_gate=False,
                        ),
                    )
                )
                phase_entries.append(
                    self._make_phase_entry(
                        scope_info,
                        'execution_non_product_gate_vars',
                        family_labels['execution_non_product_gate_vars'],
                        self._collect_non_product_phase_vars(
                            execution_items,
                            execution_owned_set,
                            initial_domains,
                            want_gate=True,
                        ),
                    )
                )

            if scope == anchor_scope:
                phase_entries.append(
                    self._make_phase_entry(
                        scope_info,
                        'memory_split_structure',
                        family_labels['memory_split_structure'],
                        self._collect_split_phase_vars_for_steps(
                            shared_step_indices,
                            inner_first=True,
                        ),
                    )
                )
                vector_scaled = self._collect_scoped_product_phase_vars(
                    vectorize_items,
                    vectorize_var_set,
                    want_scale='non_unit',
                )
                phase_entries.append(
                    self._make_phase_entry(
                        scope_info,
                        'instruction_scaled_product_upper_bound',
                        family_labels['instruction_scaled_product_upper_bound'],
                        vector_scaled,
                    )
                )
                vector_non_product = []
                for item in vectorize_items:
                    meta = g.constraint_set._extract_product_form_meta(item['tree'])
                    if meta is not None and int(meta['scale']) != 1:
                        continue
                    g.constraint_set._append_unique_vars(
                        vector_non_product,
                        [
                            name
                            for name in g.constraint_set._ordered_unique_tree_variables(item['tree'])
                            if name in vectorize_var_set and name not in vector_scaled
                        ],
                    )
                phase_entries.append(
                    self._make_phase_entry(
                        scope_info,
                        'instruction_non_product_min',
                        family_labels['instruction_non_product_min'],
                        vector_non_product,
                    )
                )

        return phase_entries

    def _build_grid_scope_infos(self, max_thread_items, max_vthread_items):
        g = self.gen
        max_thread_items_by_scope = {}
        thread_axis_items_by_scope = {}
        vthread_items_by_scope = {}

        total_scope_order = []
        seen_total_scopes = set()
        for item in max_thread_items:
            scope = item.get('block_scope', tuple())
            max_thread_items_by_scope.setdefault(scope, []).append(item)
            if item.get('axis_name') == 'threads per block':
                if scope not in seen_total_scopes:
                    seen_total_scopes.add(scope)
                    total_scope_order.append(scope)
                continue
            thread_axis_items_by_scope.setdefault(scope, []).append(item)

        for item in max_vthread_items:
            scope = item.get('block_scope', tuple())
            vthread_items_by_scope.setdefault(scope, []).append(item)

        anchor_scope = None
        vthread_scopes = set(vthread_items_by_scope.keys())
        for scope in total_scope_order:
            if scope in vthread_scopes:
                anchor_scope = scope
                break

        ordered_scopes = list(total_scope_order)
        if anchor_scope is not None:
            ordered_scopes = [anchor_scope] + [scope for scope in total_scope_order if scope != anchor_scope]

        scope_infos = []
        for idx, scope in enumerate(ordered_scopes):
            scope_infos.append({
                'grid_index': idx,
                'grid_scope': scope,
                'grid_scope_label': f"grid_{idx}: {g.constraint_set._format_block_scope(scope)}",
                'is_main_compute_anchor': scope == anchor_scope,
            })
        return scope_infos, max_thread_items_by_scope, thread_axis_items_by_scope, vthread_items_by_scope

    def _build_scope_owned_vars(self, scope_infos, items_by_scope):
        g = self.gen
        owned = {}
        for scope_info in scope_infos:
            scope = scope_info['grid_scope']
            ordered = []
            for item in items_by_scope.get(scope, []):
                g.constraint_set._append_unique_vars(
                    ordered,
                    g.constraint_set._ordered_unique_tree_variables(item['tree']),
                )
            owned[scope] = ordered
        return owned

    def _build_split_structure_phase_assignments(self, scope_infos, thread_owned_vars, vthread_owned_vars):
        g = self.gen
        assignments = {}
        for scope_info in scope_infos:
            scope = scope_info['grid_scope']
            assignments[(scope, 'split_structure_max_threads')] = []
            assignments[(scope, 'split_structure_max_vthread')] = []

        ownership = {}
        ownership_priority = {}
        for scope_info in scope_infos:
            scope = scope_info['grid_scope']
            for name in thread_owned_vars.get(scope, []):
                ownership.setdefault(name, (scope, 'split_structure_max_threads'))
                ownership_priority.setdefault(name, (scope_info['grid_index'], 0))
            for name in vthread_owned_vars.get(scope, []):
                ownership.setdefault(name, (scope, 'split_structure_max_vthread'))
                ownership_priority.setdefault(name, (scope_info['grid_index'], 1))

        for step_idx, names in sorted(g._sp_groups.items()):
            target = None
            target_priority = None
            for name in names:
                if name not in ownership:
                    continue
                cur_priority = ownership_priority[name]
                if target is None or cur_priority < target_priority:
                    target = ownership[name]
                    target_priority = cur_priority
            if target is None:
                continue
            assignments[target].append(step_idx)
        return assignments

    @staticmethod
    def _make_phase_entry(scope_info, family, family_label, vars_in_phase):
        return {
            'name': f"grid_{scope_info['grid_index']}__{family}",
            'family': family,
            'label': f"{family_label} [{scope_info['grid_scope_label']}]",
            'grid_scope': scope_info['grid_scope'],
            'grid_scope_label': scope_info['grid_scope_label'],
            'vars': list(vars_in_phase),
        }

    def _collect_scoped_product_phase_vars(self, items, candidate_var_names, want_scale):
        g = self.gen
        ordered = []
        if not candidate_var_names:
            return ordered
        for item in items:
            meta = g.constraint_set._extract_product_form_meta(item['tree'])
            if meta is None:
                continue
            scale = int(meta['scale'])
            if want_scale == 1 and scale != 1:
                continue
            if want_scale == 'non_unit' and scale == 1:
                continue
            filtered = []
            for name in meta['factors']:
                if name not in candidate_var_names or name in filtered:
                    continue
                filtered.append(name)
            if filtered:
                g.constraint_set._append_unique_vars(ordered, filtered)
        return ordered

    def _build_initial_domains(self):
        g = self.gen
        domains = {}
        for name in g._all_sp_names:
            step_idx = int(name.split("_")[1])
            extent = g._sp_extents.get(step_idx)
            if extent is None:
                domains[name] = 1
            else:
                domains[name] = [1, extent]
        return domains

    @classmethod
    def _quick_interval(cls, node, domains, cap=None):
        if isinstance(node, ConstNode):
            value = int(node.val)
            return value, value
        if isinstance(node, VarNode):
            dom = domains.get(node.name, 1)
            if isinstance(dom, list):
                return int(dom[0]), int(dom[1])
            value = int(dom)
            return value, value
        if isinstance(node, ScaleMulNode):
            lo, hi = cls._quick_interval(node.child, domains, cap=cap)
            scale = int(node.scale)
            return cls._clip_interval(lo * scale, hi * scale, cap)
        if isinstance(node, MulNode):
            left_lo, left_hi = cls._quick_interval(node.left, domains, cap=cap)
            right_lo, right_hi = cls._quick_interval(node.right, domains, cap=cap)
            return cls._clip_interval(left_lo * right_lo, left_hi * right_hi, cap)
        if isinstance(node, AddNode):
            left_lo, left_hi = cls._quick_interval(node.left, domains, cap=cap)
            right_lo, right_hi = cls._quick_interval(node.right, domains, cap=cap)
            return cls._clip_interval(left_lo + right_lo, left_hi + right_hi, cap)
        if isinstance(node, SubNode):
            left_lo, left_hi = cls._quick_interval(node.left, domains, cap=cap)
            right_lo, right_hi = cls._quick_interval(node.right, domains, cap=cap)
            return cls._clip_interval(left_lo - right_hi, left_hi - right_lo, cap)
        if isinstance(node, MinNode):
            left_lo, left_hi = cls._quick_interval(node.left, domains, cap=cap)
            right_lo, right_hi = cls._quick_interval(node.right, domains, cap=cap)
            return min(left_lo, right_lo), min(left_hi, right_hi)
        if isinstance(node, MaxNode):
            child_intervals = [cls._quick_interval(child, domains, cap=cap) for child in node.children]
            lo = max(interval[0] for interval in child_intervals)
            hi = max(interval[1] for interval in child_intervals)
            return cls._clip_interval(lo, hi, cap)
        if isinstance(node, CeilDivNode):
            left_lo, left_hi = cls._quick_interval(node.left, domains, cap=cap)
            right_lo, right_hi = cls._quick_interval(node.right, domains, cap=cap)
            denom_lo = max(int(right_lo), 1)
            denom_hi = max(int(right_hi), 1)
            lo = (int(left_lo) + denom_hi - 1) // denom_hi
            hi = (int(left_hi) + denom_lo - 1) // denom_lo
            return cls._clip_interval(lo, hi, cap)
        return node.interval(domains)

    @staticmethod
    def _clip_interval(lo, hi, cap):
        if cap is None:
            return lo, hi
        upper = int(cap)
        if lo > upper:
            lo = upper
        if hi > upper:
            hi = upper
        return lo, hi

    def _classify_non_product_item_vars(self, item, candidate_var_names, domains):
        g = self.gen
        tree = item['tree']
        if g.constraint_set._extract_product_form_meta(tree) is not None:
            return [], []
        if not isinstance(tree, MinNode):
            return [], []

        branches = [tree.left, tree.right]
        branch_infos = []
        for branch in branches:
            ordered_vars = [
                name for name in g.constraint_set._ordered_unique_tree_variables(branch)
                if name in candidate_var_names
            ]
            lower_bound, _ = self._quick_interval(branch, domains, cap=item['limit'] + 1)
            exotic_nodes, node_count = g.constraint_set._thread_tree_metrics(branch)
            branch_infos.append({
                'vars': ordered_vars,
                'lower_bound': lower_bound,
                'sort_key': (
                    len(ordered_vars),
                    exotic_nodes,
                    node_count,
                    len(repr(branch)),
                ),
            })

        if branch_infos[1]['lower_bound'] > item['limit']:
            return branch_infos[0]['vars'], []
        if branch_infos[0]['lower_bound'] > item['limit']:
            return branch_infos[1]['vars'], []

        direct_idx = 0
        if branch_infos[1]['sort_key'] < branch_infos[0]['sort_key']:
            direct_idx = 1
        gate_idx = 1 - direct_idx
        direct_vars = branch_infos[direct_idx]['vars']
        gate_vars = [name for name in branch_infos[gate_idx]['vars'] if name not in direct_vars]
        return direct_vars, gate_vars

    def _collect_non_product_phase_vars(self, items, candidate_var_names, domains, want_gate):
        g = self.gen
        ordered = []
        if not candidate_var_names:
            return ordered
        for item in items:
            direct_vars, gate_vars = self._classify_non_product_item_vars(
                item,
                candidate_var_names,
                domains,
            )
            phase_vars = gate_vars if want_gate else direct_vars
            if phase_vars:
                g.constraint_set._append_unique_vars(ordered, phase_vars)
        return ordered

    def _collect_step_indices_for_vars(self, var_names):
        g = self.gen
        target = set(var_names)
        step_indices = []
        for step_idx, names in sorted(g._sp_groups.items()):
            if any(name in target for name in names):
                step_indices.append(step_idx)
        return step_indices

    def _collect_split_phase_vars_for_steps(self, step_indices, inner_first=False):
        g = self.gen
        ordered = []
        for step_idx in step_indices:
            group = list(g._sp_groups.get(step_idx, []))
            group.sort(key=lambda name: int(name.split("_")[2]), reverse=inner_first)
            g.constraint_set._append_unique_vars(ordered, group)
        return ordered

    def _compute_legacy_var_order(self):
        g = self.gen
        shared_vars = set()
        thread_vars = set()
        other_vars = set()
        preferred_thread_vars = set(g._preferred_thread_vars)
        preferred_thread_rank = {
            name: idx for idx, name in enumerate(g._preferred_thread_vars)
        }

        for c in g._constraints:
            kind = c['kind']
            vs = c['vars']
            if kind == 'shared_memory':
                shared_vars.update(vs)
            elif kind in ('max_threads', 'max_vthread'):
                thread_vars.update(vs)
            else:
                other_vars.update(vs)

        var_freq = {}
        for v in g._all_sp_names:
            var_freq[v] = len(g._var_constraints.get(v, []))

        group_priority = {}
        for step_idx, group in g._sp_groups.items():
            group_set = set(group)

            in_preferred_thread = bool(group_set & preferred_thread_vars)
            in_shared = bool(group_set & shared_vars)
            in_thread = bool(group_set & thread_vars)

            if in_preferred_thread:
                cat = 0
            elif in_shared:
                cat = 1
            elif in_thread:
                cat = 2
            else:
                cat = 3

            min_nonlinear = True
            total_freq = 0
            preferred_hits = 0
            preferred_rank = 1 << 30
            for v in group:
                total_freq += var_freq.get(v, 0)
                if v in preferred_thread_vars:
                    preferred_hits += 1
                    preferred_rank = min(preferred_rank, preferred_thread_rank[v])
                for ci in g._var_constraints.get(v, []):
                    if not g._constraints[ci]['has_nonlinear']:
                        min_nonlinear = False

            group_priority[step_idx] = (
                cat,
                preferred_rank,
                min_nonlinear,
                -preferred_hits,
                -total_freq,
            )

        sorted_steps = sorted(
            g._sp_groups.keys(),
            key=lambda si: group_priority.get(si, (3, True, 0))
        )

        ordered = []
        for step_idx in sorted_steps:
            ordered_group = sorted(
                g._sp_groups[step_idx],
                key=lambda name: (
                    0 if name in preferred_thread_rank else 1,
                    preferred_thread_rank.get(name, 1 << 30),
                    int(name.split("_")[2]),
                ),
            )
            ordered.extend(ordered_group)
        return ordered

    def get_var_order_phase_entries(self):
        return [
            {
                'name': entry['name'],
                'family': entry['family'],
                'label': entry['label'],
                'grid_scope': entry['grid_scope'],
                'grid_scope_label': entry['grid_scope_label'],
                'vars': list(entry['vars']),
            }
            for entry in self.gen._var_order_phase_entries
        ]

    def _resolve_var_order_stop_index(self, stop_after_phase):
        g = self.gen
        exact_match = None
        last_family_match = None
        for idx, entry in enumerate(g._var_order_phase_entries):
            if entry['name'] == stop_after_phase:
                exact_match = idx
                break
            if entry['family'] == stop_after_phase:
                last_family_match = idx
        if exact_match is not None:
            return exact_match
        if last_family_match is not None:
            return last_family_match
        raise ValueError(f"Unknown var-order phase or family: {stop_after_phase}")

    def get_var_order_prefix(self, stop_after_phase):
        g = self.gen
        prefix = []
        stop_idx = self._resolve_var_order_stop_index(stop_after_phase)
        for idx, entry in enumerate(g._var_order_phase_entries):
            prefix.extend(entry['vars'])
            if idx == stop_idx:
                break
        return prefix
