from .expr_nodes import AddNode, CeilDivNode, ConstNode, MaxNode, MinNode, MulNode, ScaleMulNode, SubNode, VarNode


class VarOrderPlanner:
    def __init__(self, gen):
        """ScheduleGenerator를 받아 변수 할당 순서와 phase 구간을 계산한다."""
        self.gen = gen

    # ------------------------------------------------------------------
    # Public workflow entry and legacy fallback merge
    # ------------------------------------------------------------------

    def compute_var_order(self):
        """phase 우선 순서를 구한 뒤 legacy fallback을 붙여 제너레이터에 var_order를 저장한다."""
        g = self.gen
        phase_entries = self._build_var_order_phase_entries()
        normalized_entries, ordered = self._build_phase_first_order(phase_entries)

        all_params = self._order_param_names_by_step_index(
            list(g._all_sp_names) + list(g._ur_names)
        )
        missing = [name for name in all_params if name not in set(ordered)]
        if missing:
            normalized_entries.append({
                'name': 'grid_fallback__remaining_params',
                'family': 'fallback_remaining',
                'grid_scope': tuple(),
                'vars': missing,
            })
            ordered.extend(missing)

        g._var_order_phase_entries = normalized_entries
        g._var_order = ordered

    def _build_phase_first_order(self, phase_entries):
        """phase별 변수 목록을 앞에서부터 병합하며 중복 변수를 제거한다."""
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
            if not phase_vars:
                continue
            normalized_entries.append({
                'name': entry['name'],
                'family': entry['family'],
                'grid_scope': entry['grid_scope'],
                'vars': phase_vars,
            })
        return normalized_entries, ordered


    # ------------------------------------------------------------------
    # Phase-order construction
    # ------------------------------------------------------------------

    def _build_var_order_phase_entries(self):
        """execution, memory, instruction phase를 grid scope 단위로 조립한다."""
        g = self.gen
        # vectorize_items = []
        # if 'vectorize' in g._enabled:
        vectorize_items = g.constraint_set._build_vectorize_constraints()['items']

        # shared_vars = set()
        # if 'shared_memory' in g._enabled:
        shared_vars = set(g.constraint_set._build_shared_memory_constraints()['tree'].variables())

        scope_context = g.constraint_set._build_grid_scope_context()
        scope_infos = scope_context['scope_infos']
        max_thread_items_by_scope = scope_context['max_thread_items_by_scope']
        thread_axis_items_by_scope = scope_context['thread_axis_items_by_scope']
        vthread_items_by_scope = scope_context['vthread_items_by_scope']

        thread_owned_vars = self._build_scope_owned_vars(scope_infos, thread_axis_items_by_scope)
        vthread_owned_vars = self._build_scope_owned_vars(scope_infos, vthread_items_by_scope)
        initial_domains = self._build_initial_domains()
        family_labels = dict(g.VAR_ORDER_PHASE_FAMILIES)
        main_scope = scope_context['main_scope']

        vectorize_vars_by_scope = self._collect_constraint_vars_by_scope(
            scope_infos,
            vectorize_items,
            default_scope=main_scope,
        )
        budget_vars_by_scope_kind = self._collect_budget_vars_by_scope_kind(scope_infos)
        unroll_vars_by_scope = self._collect_unroll_vars_by_scope(
            scope_infos,
            default_scope=main_scope,
        )
        shared_step_indices = self._collect_step_indices_for_vars(shared_vars)
        execution_phase_entries_by_scope = {}
        memory_dependent_scopes = []
        remaining_non_main_scopes = []

        for scope_info in scope_infos:
            scope = scope_info['grid_scope']
            max_thread_scope_items = max_thread_items_by_scope.get(scope, [])
            max_vthread_scope_items = vthread_items_by_scope.get(scope, [])
            execution_items = (
                max_thread_scope_items
                + max_vthread_scope_items
            )
            execution_owned = []
            g.constraint_set._append_unique_vars(execution_owned, thread_owned_vars.get(scope, []))
            g.constraint_set._append_unique_vars(execution_owned, vthread_owned_vars.get(scope, []))
            execution_owned_set = set(execution_owned)
            execution_overlaps_memory = bool(shared_vars & execution_owned_set)

            thread_pure = self._collect_scoped_product_phase_vars(
                max_thread_scope_items,
                set(thread_owned_vars.get(scope, [])),
                want_scale=1,
                order_by_step_idx=True,
            )
            vthread_pure = self._collect_scoped_product_phase_vars(
                max_vthread_scope_items,
                set(vthread_owned_vars.get(scope, [])),
                want_scale=1,
                order_by_step_idx=True,
            )
            has_execution_pure_product = bool(thread_pure or vthread_pure)
            scope_phase_entries = []

            if has_execution_pure_product:
                scope_phase_entries.append(
                    self._make_phase_entry(
                        scope_info,
                        'execution_max_threads_pure_product',
                        family_labels['execution_max_threads_pure_product'],
                        list(budget_vars_by_scope_kind['thread'].get(scope, [])) + list(thread_pure),
                    )
                )
                scope_phase_entries.append(
                    self._make_phase_entry(
                        scope_info,
                        'execution_split_assignment',
                        family_labels['execution_split_assignment'],
                        self._collect_split_phase_vars_for_steps(
                            self._collect_step_indices_for_vars(execution_owned_set),
                            excluded_vars=vthread_pure,
                            inner_first=True,
                        ),
                    )
                )
                scope_phase_entries.append(
                    self._make_phase_entry(
                        scope_info,
                        'execution_max_vthread',
                        family_labels['execution_max_vthread'],
                        list(budget_vars_by_scope_kind['vthread'].get(scope, [])) + list(vthread_pure),
                    )
                )
            else:
                scope_phase_entries.append(
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
                scope_phase_entries.append(
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

            execution_phase_entries_by_scope[scope] = scope_phase_entries
            if scope == main_scope:
                continue
            if execution_overlaps_memory:
                memory_dependent_scopes.append(scope)
            else:
                remaining_non_main_scopes.append(scope)

        phase_entries = []
        main_scope_info = None
        for scope_info in scope_infos:
            if scope_info['grid_scope'] == main_scope:
                main_scope_info = scope_info
                break

        if main_scope_info is not None:
            phase_entries.extend(execution_phase_entries_by_scope.get(main_scope, []))

        for scope in memory_dependent_scopes:
            phase_entries.extend(execution_phase_entries_by_scope.get(scope, []))

        if main_scope_info is not None:
            phase_entries.append(
                self._make_phase_entry(
                    main_scope_info,
                    'memory_split_structure',
                    family_labels['memory_split_structure'],
                    self._collect_split_phase_vars_for_steps(
                        shared_step_indices,
                        inner_first=True,
                    ),
                )
            )

        for scope in remaining_non_main_scopes:
            phase_entries.extend(execution_phase_entries_by_scope.get(scope, []))

        # vectorize split의 extent는 prior split factor 할당에 의해 동적으로 결정되므로
        # 모든 execution/memory split이 끝난 뒤(= 가장 마지막) 할당한다. unroll은 그 뒤에 둔다.
        vectorize_phase_order = []
        unroll_phase_order = []
        ordered_scopes = []
        if main_scope_info is not None:
            ordered_scopes.append(main_scope)
        ordered_scopes.extend(memory_dependent_scopes)
        ordered_scopes.extend(remaining_non_main_scopes)
        seen_scopes = set()
        for scope in ordered_scopes:
            if scope in seen_scopes:
                continue
            seen_scopes.add(scope)
            vectorize_phase_order.append((scope, vectorize_vars_by_scope.get(scope, [])))
            unroll_phase_order.append((scope, unroll_vars_by_scope.get(scope, [])))

        for scope, vec_vars in vectorize_phase_order:
            if not vec_vars:
                continue
            phase_entries.append(
                self._make_phase_entry(
                    self._find_scope_info(scope_infos, scope),
                    'instruction_vectorize',
                    family_labels.get('instruction_vectorize', 'instruction_vectorize'),
                    list(vec_vars),
                )
            )

        for scope, unr_vars in unroll_phase_order:
            if not unr_vars:
                continue
            phase_entries.append(
                self._make_phase_entry(
                    self._find_scope_info(scope_infos, scope),
                    'instruction_unroll',
                    family_labels.get('instruction_unroll', 'instruction_unroll'),
                    list(unr_vars),
                )
            )

        return phase_entries

    # ------------------------------------------------------------------
    # Phase variable collection and legacy fallback heuristics
    # ------------------------------------------------------------------

    def _build_scope_owned_vars(self, scope_infos, items_by_scope):
        """scope별 constraint item에서 직접 소유한 변수를 순서대로 모은다."""
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

    @staticmethod
    def _find_scope_info(scope_infos, scope):
        """grid scope tuple에 대응하는 scope_info를 찾고, 없으면 기본 정보를 만든다."""
        for scope_info in scope_infos:
            if scope_info['grid_scope'] == scope:
                return scope_info
        return {
            'grid_index': 0,
            'grid_scope': scope,
        }

    @staticmethod
    def _make_phase_entry(scope_info, family, family_label, vars_in_phase):
        """내부 var-order phase entry 포맷을 만든다."""
        del family_label
        return {
            'name': f"grid_{scope_info['grid_index']}__{family}",
            'family': family,
            'grid_scope': scope_info['grid_scope'],
            'vars': list(vars_in_phase),
        }

    @staticmethod
    def _order_param_names_by_step_index(names):
        """split/unroll 이름을 step index 기준으로 정렬하고 중복을 제거한다."""
        ordered = []
        seen = set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            ordered.append(name)

        def order_key(name):
            parts = name.split("_")
            step_idx = int(parts[1]) if len(parts) > 1 else 1 << 30
            pos = int(parts[2]) if len(parts) > 2 else -1
            kind_order = 0 if name.startswith("sp_") else 1 if name.startswith("ur_") else 2
            return (step_idx, kind_order, pos, name)

        ordered.sort(key=order_key)
        return ordered

    def _collect_constraint_vars_by_scope(self, scope_infos, items, default_scope=None):
        """constraint item의 자유변수를 block scope별 instruction 후보로 모은다."""
        g = self.gen
        scope_set = {scope_info['grid_scope'] for scope_info in scope_infos}
        vars_by_scope = {scope: [] for scope in scope_set}
        for item in items:
            scope = item.get('block_scope', default_scope)
            if scope not in scope_set:
                scope = default_scope
            if scope not in vars_by_scope:
                vars_by_scope[scope] = []
            g.constraint_set._append_unique_vars(
                vars_by_scope[scope],
                g.constraint_set._ordered_unique_tree_variables(item['tree']),
            )
        return {
            scope: self._order_param_names_by_step_index(names)
            for scope, names in vars_by_scope.items()
        }

    def _collect_unroll_vars_by_scope(self, scope_infos, default_scope=None):
        """PragmaStep의 stage scope를 따라 unroll 변수를 grid별로 귀속시킨다."""
        g = self.gen
        scope_set = {scope_info['grid_scope'] for scope_info in scope_infos}
        vars_by_scope = {scope: [] for scope in scope_set}
        state = getattr(g.s, '_state', None)

        for name in g._ur_names:
            scope = default_scope
            step_idx = int(name.split("_")[1])
            if state is not None and step_idx < len(state.transform_steps):
                step = state.transform_steps[step_idx]
                if step.type_key.split(".")[-1] == "PragmaStep":
                    stage_id = int(step.stage_id)
                    scope = g.constraint_set._resolve_block_scope(
                        stage_id,
                        len(g.s.stages[stage_id].iters),
                    )
            if scope not in scope_set:
                scope = default_scope
            if scope not in vars_by_scope:
                vars_by_scope[scope] = []
            vars_by_scope[scope].append(name)

        return {
            scope: self._order_param_names_by_step_index(names)
            for scope, names in vars_by_scope.items()
        }

    def _collect_budget_vars_by_scope_kind(self, scope_infos):
        """budget spec을 scope별 thread/vthread 변수 이름으로 모은다."""
        scope_set = {scope_info['grid_scope'] for scope_info in scope_infos}
        vars_by_scope_kind = {
            'thread': {scope: [] for scope in scope_set},
            'vthread': {scope: [] for scope in scope_set},
        }
        default_scope = self.gen.constraint_set._build_grid_scope_context()['main_scope']
        for spec in getattr(self.gen, '_budget_specs', []):
            scope = spec.get('block_scope', default_scope)
            if scope not in scope_set:
                scope = default_scope
            vars_by_scope_kind[spec['budget_kind']].setdefault(scope, []).append(spec['name'])
        return vars_by_scope_kind

    @staticmethod
    def _merge_instruction_phase_vars(vectorize_vars, unroll_vars):
        """vectorize 변수 뒤에 unroll 변수를 붙여 instruction phase 순서를 만든다."""
        ordered = []
        seen = set()
        for name in list(vectorize_vars) + list(unroll_vars):
            if name in seen:
                continue
            seen.add(name)
            ordered.append(name)
        return ordered

    def _collect_scoped_product_phase_vars(
        self,
        items,
        candidate_var_names,
        want_scale,
        order_by_step_idx=False,
    ):
        """product-form 상한식에서 원하는 scale 조건의 변수만 골라 phase 후보를 만든다."""
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
            if order_by_step_idx:
                filtered.sort(
                    key=lambda name: (
                        int(name.split("_")[1]),
                        int(name.split("_")[2]),
                    )
                )
            if filtered:
                g.constraint_set._append_unique_vars(ordered, filtered)
        return ordered

    def _build_initial_domains(self):
        """split 변수의 초기 도메인 사전을 step extent 기반으로 구성한다."""
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
        """간단한 식 노드에 대해 빠른 구간 추정을 수행한다."""
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
        """추정 구간의 상한을 cap으로 잘라 interval 폭이 과도하게 커지지 않게 한다."""
        if cap is None:
            return lo, hi
        upper = int(cap)
        if lo > upper:
            lo = upper
        if hi > upper:
            hi = upper
        return lo, hi

    def _classify_non_product_item_vars(self, item, candidate_var_names, domains):
        """MinNode 기반 non-product 제약을 direct arm과 gate arm 변수로 나눈다."""
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
            branch_infos.append({
                'vars': ordered_vars,
                'lower_bound': lower_bound,
                'sort_key': (
                    len(ordered_vars),
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
        """non-product 제약들에서 direct 또는 gate 쪽 변수만 추려 phase를 구성한다."""
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
        """주어진 변수 집합이 포함된 split step index 목록을 찾는다."""
        g = self.gen
        target = set(var_names)
        step_indices = []
        for step_idx, names in sorted(g._sp_groups.items()):
            if any(name in target for name in names):
                step_indices.append(step_idx)
        return step_indices

    def _collect_split_phase_vars_for_steps(self, step_indices, excluded_vars=None, inner_first=False):
        """지정한 split step들에서 phase에 넣을 split 변수 순서를 만든다."""
        g = self.gen
        ordered = []
        excluded = set(excluded_vars or [])
        for step_idx in step_indices:
            group = list(g._sp_groups.get(step_idx, []))
            group.sort(key=lambda name: int(name.split("_")[2]), reverse=inner_first)
            for name in group:
                if name in excluded or name in ordered:
                    continue
                ordered.append(name)
        return ordered

    def get_var_order_phase_entries(self):
        """phase별 변수 목록·이름·family·grid_scope 등을 담은 진입점 목록을 반환한다."""
        def build_var_entry(name):
            """공개용 phase entry에 들어갈 파라미터 메타정보를 만든다."""
            entry = {
                'param_name': name,
                'param_kind': 'symbolic',
            }
            if name.startswith("sp_"):
                parts = name.split("_")
                step_idx = int(parts[1])
                pos = int(parts[2])
                group_vars = list(self.gen._sp_groups.get(step_idx, []))
                entry.update({
                    'param_kind': 'split',
                    'split_step_idx': step_idx,
                    'split_position': pos,
                    'split_extent': self.gen._sp_extents.get(step_idx),
                    'split_group_param_names': group_vars,
                    'collapsed_factor_param_names': group_vars[:pos],
                    'is_innermost': name in self.gen._innermost_names,
                })
            elif name.startswith("ur_"):
                entry.update({
                    'param_kind': 'unroll',
                    'unroll_step_idx': int(name.split("_")[1]),
                    'candidate_values': list(self.gen.pm.UNROLL_CANDIDATES),
                })
            elif name.startswith("thread_budget") or name.startswith("vthread_budget"):
                spec = self.gen._budget_spec_by_name.get(name, {})
                entry.update({
                    'param_kind': 'budget',
                    'budget_kind': spec.get('budget_kind'),
                    'budget_limit': spec.get('limit'),
                    'budget_factor_param_names': list(spec.get('factor_names', ())),
                })
            return entry

        return [
            {
                'phase_name': entry['name'],
                'phase_family': entry['family'],
                'grid_scope': entry['grid_scope'],
                'param_names': list(entry['vars']),
                'param_entries': [build_var_entry(name) for name in entry['vars']],
            }
            for entry in self.gen._var_order_phase_entries
        ]

    def resolve_var_order_stop_index(self, stop_after_phase):
        """prefix 샘플링 시 멈출 phase 이름/인덱스를 받아 phase 인덱스를 반환한다."""
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



    # ------------------------------------------------------------------
    # Deprecated
    # ------------------------------------------------------------------

    # @staticmethod
    # def _append_legacy_fallback_vars(ordered, seen, legacy_order):
    #     """legacy 순서에서 아직 등장하지 않은 변수만 뒤에 덧붙인다."""
    #     for name in legacy_order:
    #         if name in seen:
    #             continue
    #         seen.add(name)
    #         ordered.append(name)

    # def _compute_legacy_var_order(self):
    #     """예전 휴리스틱 기반 변수 순서를 계산해 비교나 fallback에 쓸 수 있게 한다."""
    #     g = self.gen
    #     shared_vars = set()
    #     thread_vars = set()
    #     other_vars = set()
    #     preferred_thread_vars = set(g._preferred_thread_vars)
    #     preferred_thread_rank = {
    #         name: idx for idx, name in enumerate(g._preferred_thread_vars)
    #     }

    #     for c in g._constraints:
    #         kind = c['kind']
    #         vs = c['vars']
    #         if kind == 'shared_memory':
    #             shared_vars.update(vs)
    #         elif kind in ('max_threads', 'max_vthread'):
    #             thread_vars.update(vs)
    #         else:
    #             other_vars.update(vs)

    #     var_freq = {}
    #     for v in g._all_sp_names:
    #         var_freq[v] = len(g._var_constraints.get(v, []))

    #     group_priority = {}
    #     for step_idx, group in g._sp_groups.items():
    #         group_set = set(group)

    #         in_preferred_thread = bool(group_set & preferred_thread_vars)
    #         in_shared = bool(group_set & shared_vars)
    #         in_thread = bool(group_set & thread_vars)

    #         if in_preferred_thread:
    #             cat = 0
    #         elif in_shared:
    #             cat = 1
    #         elif in_thread:
    #             cat = 2
    #         else:
    #             cat = 3

    #         min_nonlinear = True
    #         total_freq = 0
    #         preferred_hits = 0
    #         preferred_rank = 1 << 30
    #         for v in group:
    #             total_freq += var_freq.get(v, 0)
    #             if v in preferred_thread_vars:
    #                 preferred_hits += 1
    #                 preferred_rank = min(preferred_rank, preferred_thread_rank[v])
    #             for ci in g._var_constraints.get(v, []):
    #                 if not g._constraints[ci]['has_nonlinear']:
    #                     min_nonlinear = False

    #         group_priority[step_idx] = (
    #             cat,
    #             preferred_rank,
    #             min_nonlinear,
    #             -preferred_hits,
    #             -total_freq,
    #         )

    #     sorted_steps = sorted(
    #         g._sp_groups.keys(),
    #         key=lambda si: group_priority.get(si, (3, True, 0))
    #     )

    #     ordered = []
    #     for step_idx in sorted_steps:
    #         ordered_group = sorted(
    #             g._sp_groups[step_idx],
    #             key=lambda name: (
    #                 0 if name in preferred_thread_rank else 1,
    #                 preferred_thread_rank.get(name, 1 << 30),
    #                 int(name.split("_")[2]),
    #             ),
    #         )
    #         ordered.extend(ordered_group)
    #     return ordered
