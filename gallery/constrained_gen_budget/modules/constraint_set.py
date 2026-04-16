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
from .gpu_projection_constraints import (
    build_projected_gpu_context,
    build_projected_shared_memory_constraint_node,
    build_projected_vectorize_constraint_node,
)


class ConstraintSet:
    def __init__(self, gen):
        """ScheduleGenerator를 받아 제약 묶음·검사 로직을 담당한다."""
        # 제약 생성과 검사를 위한 제너레이터 참조를 보관한다.
        self.gen = gen

    # ------------------------------------------------------------------
    # Preprocess pipeline
    # ------------------------------------------------------------------

    def preprocess(self):
        """제약 묶음·변수별 제약 인덱스·변수 순서·phase 정보를 한 번에 계산해 제너레이터에 채운다."""
        # 전체 constraint 시스템과 변수 역인덱스를 초기화해 샘플링 준비를 끝낸다.
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
        g._projected_gpu = None
        g._projected_gpu_context = None
        g._grid_scope_context = None
        g._split_step_scope_map = None
        g._vectorize_constraint_bundle = None
        g._shared_memory_constraint_bundle = None
        g._max_threads_constraint_bundle = None
        g._max_threads_per_block_constraint_bundle = None
        g._max_vthread_constraint_bundle = None
        g._split_structure_constraint_bundle = None
        g._min_thread_extent_constraint_bundle = None
        g._budget_specs = []
        g._budget_spec_by_name = {}
        g._budget_spec_by_factor = {}
        g._all_budget_names = []
        g._vthread_clamped_sp_names = self._collect_clamped_sp_names_from_extents(self.gen.s.get_vthread_extents())
        g._thread_clamped_sp_names = self._collect_clamped_sp_names_from_extents(self.gen.s.get_thread_extents())

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
            block_scope=None,
        ):
            # 개별 제약을 공통 포맷으로 저장하고 변수별 역인덱스에도 등록한다.
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
            # has_nonlinear = self._has_nonlinear(expr_tree)
            product_form_meta = self._extract_product_form_meta(expr_tree) if is_upper else None
            # breakpoint()
            g._constraints.append({
                'tree': expr_tree,
                'rhs': rhs,
                'vars': vars_in,
                'kind': kind,
                'desc': desc,
                'is_upper': is_upper,
                # 'has_nonlinear': has_nonlinear,
                'display_text': display_text,
                'display_rhs': display_rhs,
                'alias_entries': list(alias_entries or []),
                'product_form_meta': product_form_meta,
                'fast_path': fast_path,
                'block_scope': block_scope,
            })
            if index_vars:
                for v in vars_in:
                    g._var_constraints.setdefault(v, []).append(idx)



        # breakpoint()
        if 'max_threads' in g._enabled:
            c = self._build_max_threads_constraints()
            for item in c['items']:
                _add_constraint(
                    item['tree'],
                    item['limit'],
                    'max_threads',
                    item['desc'],
                    is_upper=True,
                    display_text=str(item['sym_extent']),
                    alias_entries=item.get('alias_entries'),
                    block_scope=item.get('block_scope'),
                )

        if 'max_vthread' in g._enabled:
            c = self._build_max_vthread_constraints()
            for item in c['items']:
                _add_constraint(
                    item['tree'],
                    item['limit'],
                    'max_vthread',
                    item['desc'],
                    is_upper=True,
                    display_text=str(item['sym_extent']),
                    alias_entries=item.get('alias_entries'),
                    block_scope=item.get('block_scope'),
                )

        if 'max_threads_per_block' in g._enabled:
            c = self._build_max_threads_per_block_constraints()
            for item in c['items']:
                _add_constraint(
                    item['tree'],
                    item['limit'],
                    'max_threads_per_block',
                    item['desc'],
                    is_upper=True,
                    display_text=str(item['sym_extent']),
                    alias_entries=item.get('alias_entries'),
                    block_scope=item.get('block_scope'),
                )

        if 'min_thread_extent' in g._enabled:
            c = self._build_min_thread_extent_constraints()
            for item in c['items']:
                _add_constraint(
                    item['tree'],
                    item['limit'],
                    'min_thread_extent',
                    item['desc'],
                    is_upper=False,
                    display_text=str(item['sym_extent']),
                    alias_entries=item.get('alias_entries'),
                    block_scope=item.get('block_scope'),
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
                block_scope=sm.get('block_scope'),
            )

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
                    block_scope=item.get('block_scope', c.get('block_scope')),
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
                    block_scope=c.get('block_scope'),
                )

        budget_specs = self._build_budget_specs()
        g._budget_specs = budget_specs
        g._budget_spec_by_name = {
            spec['name']: spec for spec in budget_specs
        }
        g._budget_spec_by_factor = {}
        for spec in budget_specs:
            for factor_name in spec['factor_names']:
                g._budget_spec_by_factor[factor_name] = spec
        g._all_budget_names = [spec['name'] for spec in budget_specs]



        g.var_order_planner.compute_var_order()



    # ------------------------------------------------------------------
    # Constraint-family bundle builders
    # ------------------------------------------------------------------

    def _build_max_threads_constraints(self):
        # thread binding 축의 per-axis 상한 제약 아이템들을 만든다.
        g = self.gen
        if g._max_threads_constraint_bundle is not None:
            return g._max_threads_constraint_bundle
        g._max_threads_constraint_bundle = {
            'items': self._collect_thread_binding_axes()
        }
        return g._max_threads_constraint_bundle

    def _build_max_threads_per_block_constraints(self):
        # thread/vthread binding 축을 곱한 threads-per-block 제약 아이템들을 만든다.
        g = self.gen
        if g._max_threads_per_block_constraint_bundle is not None:
            return g._max_threads_per_block_constraint_bundle
        thread_items = self._collect_thread_binding_axes()
        vthread_items = self._collect_vthread_binding_axes()

        items = []
        for block_scope, scoped_items in self._group_binding_items_by_block_scope(
            thread_items + vthread_items
        ):
            total_item = self._build_thread_per_block_constraint_item(block_scope, scoped_items)
            if total_item is not None:
                items.append(total_item)
        g._max_threads_per_block_constraint_bundle = {
            'items': items
        }
        return g._max_threads_per_block_constraint_bundle


    def _build_max_vthread_constraints(self):
        # vthread 축의 extent 상한 제약 아이템 묶음을 구성한다.
        g = self.gen
        if g._max_vthread_constraint_bundle is not None:
            return g._max_vthread_constraint_bundle
        g._max_vthread_constraint_bundle = {
            'items': self._collect_vthread_binding_axes()
        }
        return g._max_vthread_constraint_bundle

    def _build_min_thread_extent_constraints(self):
        # MLT root의 thread 축에 대해 warp_size 이상 하한 제약을 만든다.
        g = self.gen
        if g._min_thread_extent_constraint_bundle is not None:
            return g._min_thread_extent_constraint_bundle

        items = []
        thread_items = self._build_max_threads_constraints()['items']
        for item in thread_items:
            meta = g.s._thread_extent_meta.get((item['stage_id'], item['iter_id']))
            if not meta or not meta.get('is_mlt_root_thread'):
                continue
            if meta.get('relax_min_thread_extent'):
                continue
            items.append({
                **item,
                'limit': g.hw['warp_size'],
                'desc': (
                    f"{item['axis_name']} extent "
                    f"{item['op_name']}:{item['iter_name']} ≥ {g.hw['warp_size']}"
                ),
            })

        g._min_thread_extent_constraint_bundle = {
            'items': items
        }
        return g._min_thread_extent_constraint_bundle


    def _build_vectorize_constraints(self):
        # projected vectorize 상한식을 제약 아이템 묶음으로 구성한다.
        g = self.gen
        # breakpoint()
        if g._vectorize_constraint_bundle is not None:
            return g._vectorize_constraint_bundle
        self._ensure_projected_gpu_constraints(('vectorize',))
        tree = g._projected_gpu['vector_node']
        limit = g.hw['max_vector_bytes']
        desc = "vectorize: runtime-projected selector upper bound ≤ max_vector_bytes"
        block_scope = g._grid_scope_context['main_scope']
        items = []
        for idx, child in enumerate(self._flatten_max_terms(tree)):
            items.append({
                'tree': child,
                'limit': limit,
                'desc': f"vectorize term {idx + 1}: runtime-projected selector upper bound ≤ max_vector_bytes",
                'block_scope': block_scope,
            })
        g._vectorize_constraint_bundle = {
            'tree': tree,
            'limit': limit,
            'desc': desc,
            'block_scope': block_scope,
            'items': items,
        }
        return g._vectorize_constraint_bundle


    def _build_shared_memory_constraints(self):
        # projected shared-memory 상한식을 단일 제약 묶음으로 구성한다.
        g = self.gen
        if g._shared_memory_constraint_bundle is not None:
            return g._shared_memory_constraint_bundle
        self._ensure_projected_gpu_constraints(('shared_memory',))
        block_scope = self._build_grid_scope_context()['main_scope']
        g._shared_memory_constraint_bundle = {
            'tree': g._projected_gpu['shared_node'],
            'limit': g.hw['max_shared_memory_per_block'],
            'desc': "shared_memory: runtime-projected shared bytes upper bound ≤ limit",
            'block_scope': block_scope,
        }
        return g._shared_memory_constraint_bundle

    def _build_grid_scope_context(self):
        # execution grid scope 정보와 main compute scope를 공용 컨텍스트로 계산한다.
        g = self.gen
        if g._grid_scope_context is not None:
            return g._grid_scope_context

        max_thread_items = []
        if 'max_threads' in g._enabled:
            max_thread_items = self._build_max_threads_constraints()['items']

        max_vthread_items = []
        if 'max_vthread' in g._enabled:
            max_vthread_items = self._build_max_vthread_constraints()['items']

        total_scope_order = []
        seen_total_scopes = set()
        for item in max_thread_items:
            scope = item.get('block_scope', tuple())
            if scope in seen_total_scopes:
                continue
            seen_total_scopes.add(scope)
            total_scope_order.append(scope)
        for item in max_vthread_items:
            scope = item.get('block_scope', tuple())
            if scope in seen_total_scopes:
                continue
            seen_total_scopes.add(scope)
            total_scope_order.append(scope)

        vthread_scopes = {
            item.get('block_scope', tuple())
            for item in max_vthread_items
        }

        main_scope = None
        for scope in total_scope_order:
            if scope in vthread_scopes:
                main_scope = scope
                break

        ordered_scopes = list(total_scope_order)
        if main_scope is not None:
            ordered_scopes = [main_scope] + [
                scope for scope in total_scope_order if scope != main_scope
            ]
        elif ordered_scopes:
            main_scope = ordered_scopes[0]

        scope_infos = []
        for idx, scope in enumerate(ordered_scopes):
            scope_infos.append({
                'grid_index': idx,
                'grid_scope': scope,
                'grid_scope_label': f"grid_{idx}: {self._format_block_scope(scope)}",
                'is_main_compute_scope': scope == main_scope,
            })

        if not scope_infos:
            scope_infos = [{
                'grid_index': 0,
                'grid_scope': tuple(),
                'grid_scope_label': f"grid_0: {self._format_block_scope(tuple())}",
                'is_main_compute_scope': True,
            }]
            main_scope = tuple()

        max_thread_items_by_scope = {}
        thread_axis_items_by_scope = {}
        vthread_items_by_scope = {}

        for item in max_thread_items:
            scope = item.get('block_scope', tuple())
            max_thread_items_by_scope.setdefault(scope, []).append(item)
            thread_axis_items_by_scope.setdefault(scope, []).append(item)

        for item in max_vthread_items:
            scope = item.get('block_scope', tuple())
            vthread_items_by_scope.setdefault(scope, []).append(item)

        g._grid_scope_context = {
            'scope_infos': scope_infos,
            'main_scope': main_scope,
            'max_thread_items_by_scope': max_thread_items_by_scope,
            'thread_axis_items_by_scope': thread_axis_items_by_scope,
            'vthread_items_by_scope': vthread_items_by_scope,
        }
        return g._grid_scope_context

    def _build_split_step_scope_map(self):
        # split_structure step들을 planner와 같은 grid-phase 귀속 순서로 scope에 매핑한다.
        g = self.gen
        if g._split_step_scope_map is not None:
            return g._split_step_scope_map

        scope_context = self._build_grid_scope_context()
        scope_infos = scope_context['scope_infos']
        main_scope = scope_context['main_scope']
        thread_axis_items_by_scope = scope_context['thread_axis_items_by_scope']
        vthread_items_by_scope = scope_context['vthread_items_by_scope']

        shared_vars = set()
        if 'shared_memory' in g._enabled:
            shared_vars = set(self._build_shared_memory_constraints()['tree'].variables())
        shared_step_indices = self._collect_step_indices_for_vars(shared_vars)

        step_scope_map = {}
        for scope_info in scope_infos:
            scope = scope_info['grid_scope']
            execution_owned = []
            for item in thread_axis_items_by_scope.get(scope, []):
                self._append_unique_vars(
                    execution_owned,
                    self._ordered_unique_tree_variables(item['tree']),
                )
            for item in vthread_items_by_scope.get(scope, []):
                self._append_unique_vars(
                    execution_owned,
                    self._ordered_unique_tree_variables(item['tree']),
                )
            for step_idx in self._collect_step_indices_for_vars(execution_owned):
                step_scope_map.setdefault(step_idx, scope)
            if scope == main_scope:
                for step_idx in shared_step_indices:
                    step_scope_map.setdefault(step_idx, scope)

        g._split_step_scope_map = step_scope_map
        return g._split_step_scope_map



    def _build_innermost_split_constraints(self):
        # 각 split step의 innermost factor 상한 제약 목록을 만든다.
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
        # SplitStep의 합법 factor 범위를 나타내는 split_structure 제약들을 만든다.
        g = self.gen
        if g._split_structure_constraint_bundle is not None:
            return g._split_structure_constraint_bundle
        constraints = []
        if g.s._state is None:
            g._split_structure_constraint_bundle = constraints
            return g._split_structure_constraint_bundle

        steps = g.s._state.transform_steps
        split_step_scope_map = self._build_split_step_scope_map()
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

                thread_clamped_names = g._thread_clamped_sp_names
                clamped_binding_names = (
                    g._vthread_clamped_sp_names | thread_clamped_names
                )
                if sym_name in thread_clamped_names:
                    denom_tree = self._build_uncapped_product_tree(dependency_names)
                    rhs_tree = CeilDivNode(ConstNode(extent), denom_tree)
                else:
                    denom_tree = self._build_split_bound_denominator(dependency_names, extent)
                if sym_name in clamped_binding_names and sym_name not in thread_clamped_names:
                    rhs_tree = self._build_binding_split_display_rhs(
                        names, extent, pos, inner_to_outer
                    )
                elif sym_name not in thread_clamped_names:
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
                    'block_scope': split_step_scope_map.get(step_idx),
                    'desc': (
                        f"split_structure {sym_name}: legal SplitStep upper bound "
                        f"derived from extent {extent}"
                    ),
                })

        g._split_structure_constraint_bundle = constraints
        # g._split_structure_constraint_bundle = self._dedupe_constraint_items(
        #     constraints, self._split_structure_item_key
        # )
        return g._split_structure_constraint_bundle




    # ------------------------------------------------------------------
    # 
    # ------------------------------------------------------------------

    def _build_thread_per_block_constraint_item(self, block_scope, scoped_items):
        # 한 block scope의 축들을 곱해 threads-per-block 제약 아이템을 만든다.
        g = self.gen
        canonical_items = self._canonicalize_block_scope_binding_items(scoped_items)
        if not canonical_items:
            return None

        thread_items = [item for item in canonical_items if item.get('annotation') in (6, 8, 10)]

        def _product_tree(items):
            tree = None
            for item in items:
                tree = item['tree'] if tree is None else MulNode(tree, item['tree'])
            return tree

        thread_tree = _product_tree(thread_items)
        if thread_tree is None:
            return None

        # verify_gpu_code checks max_threads_per_block against physical thread bindings
        # only; virtual threads are constrained separately by max_vthread.
        tree = thread_tree
        sym_extent = g._format_expr(thread_tree, top_level=True)
        factor_label = "threads"

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


    @staticmethod
    def _build_split_structure_fast_path(sym_name, dependency_names, extent):
        # split_structure 도메인 전파용 fast-path 메타데이터를 만든다.
        return {
            'kind': 'split_structure',
            'sym_name': sym_name,
            'dependency_names': tuple(dependency_names),
            'extent': int(extent),
        }

    def _build_binding_split_display_rhs(self, names, extent, pos, inner_to_outer):
        # thread/vthread clamp를 반영한 split_structure 표시용 우변 식을 만든다.
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

    def _build_budget_specs(self):
        # pure-product execution constraint에서 sampler용 budget 변수 스펙을 만든다.
        g = self.gen
        raw_specs = []
        for constraint in g._constraints:
            if constraint['kind'] not in ('max_threads', 'max_vthread'):
                continue
            meta = constraint.get('product_form_meta')
            if meta is None or int(meta.get('scale', 1)) != 1:
                continue
            factor_names = self._order_param_names_by_step_index(meta.get('factors', ()))
            if len(factor_names) <= 1:
                continue
            raw_specs.append({
                'budget_kind': 'thread' if constraint['kind'] == 'max_threads' else 'vthread',
                'constraint_kind': constraint['kind'],
                'factor_names': tuple(factor_names),
                'limit': int(constraint['rhs']),
                'block_scope': constraint.get('block_scope'),
            })

        budget_specs = []
        counters = {'thread': 0, 'vthread': 0}
        seen = set()
        for raw in raw_specs:
            key = (
                raw['budget_kind'],
                raw['factor_names'],
                raw['limit'],
                raw['block_scope'],
            )
            if key in seen:
                continue
            seen.add(key)
            budget_kind = raw['budget_kind']
            base_name = f"{budget_kind}_budget"
            name = base_name if counters[budget_kind] == 0 else f"{base_name}_{counters[budget_kind]}"
            counters[budget_kind] += 1
            budget_specs.append({
                **raw,
                'name': name,
            })
        return budget_specs


    def _collect_thread_binding_axes(self):
        # threadIdx 계열 annotation이 붙은 축을 anchor 기준 대표 아이템만 수집한다.
        g = self.gen
        axis_items = []
        for ann in (6, 8, 10):
            for sid, stage in enumerate(g.s.stages):
                if stage.compute_at == CA_INLINED:
                    continue
                for iid, it in enumerate(stage.iters):
                    if it.annotation != ann or it.extent is None:
                        continue
                    anchor_stage_id, anchor_iter_id = self._resolve_thread_axis_anchor(sid, iid, ann)
                    if anchor_stage_id != sid or anchor_iter_id != iid:
                        continue
                    axis_name = ANNOTATION_STR[ann]
                    limit = self._thread_axis_limit(ann)
                    axis_items.append({
                        'stage_id': sid,
                        'iter_id': iid,
                        'annotation': ann,
                        'block_scope': self._resolve_block_scope(sid, iid),
                        'op_name': stage.op_name,
                        'compute_at': stage.compute_at,
                        'iter_name': it.name,
                        'sym_min': it.min_value,
                        'sym_extent': it.extent,
                        'min_tree': parse_expr_tree(str(it.min_value)),
                        'tree': parse_expr_tree(str(it.extent)),
                        'axis_name': axis_name,
                        'anchor_stage_id': anchor_stage_id,
                        'anchor_iter_id': anchor_iter_id,
                        'limit': limit,
                        'desc': (
                            f"{axis_name} extent "
                            f"{stage.op_name}:{it.name} ≤ {limit}"
                        ),
                    })
            if not axis_items:
                continue

        return axis_items

    def _collect_vthread_binding_axes(self):
        # vthread annotation이 붙은 축을 anchor 기준 대표 아이템만 수집한다.
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
                if anchor_stage_id != sid or anchor_iter_id != iid:
                    continue
                block_scope = self._resolve_block_scope(sid, iid)

                axis_items.append({
                    'stage_id': sid,
                    'iter_id': iid,
                    'annotation': ann,
                    'block_scope': block_scope,
                    'axis_name': ANNOTATION_STR[ann],
                    'op_name': stage.op_name,
                    'compute_at': stage.compute_at,
                    'iter_name': it.name,
                    'sym_min': it.min_value,
                    'sym_extent': it.extent,
                    'min_tree': parse_expr_tree(str(it.min_value)),
                    'tree': parse_expr_tree(str(it.extent)),
                    'anchor_stage_id': anchor_stage_id,
                    'anchor_iter_id': anchor_iter_id,
                    'limit': g.hw['max_vthread_extent'],     # 수정하지 말 것.
                    'desc': (
                        f"{ANNOTATION_STR[ann]} extent "
                        f"{stage.op_name}:{it.name} ≤ {g.hw['max_vthread_extent']}"
                    ),
                })
        return axis_items


    def _resolve_thread_axis_anchor(self, stage_id, iter_id, ann):
        # compute_at 체인을 따라 같은 binding 축의 기준 anchor를 찾는다.
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
        # 주어진 축이 속한 blockIdx scope 경로를 역추적해 정규화한다.
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


    def _ensure_projected_gpu_context(self):
        # projected GPU 제약 생성에 필요한 공통 컨텍스트를 준비한다.
        g = self.gen
        if g._projected_gpu_context is not None:
            return
        g._projected_gpu_context = build_projected_gpu_context(g.s)

    def _ensure_projected_gpu_constraints(self, kinds=None):
        # 요청된 종류의 projected GPU 제약 노드를 lazy하게 생성한다.
        g = self.gen
        if kinds is None:
            kinds = ('vectorize', 'shared_memory')
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


    def _collect_clamped_sp_names_from_extents(self, extents):
        # binding extent들에서 min(...)로 clamp된 split 변수 이름들을 수집한다.
        g = self.gen
        names = set()
        for _, _, extent in extents:
            if extent is None:
                continue
            try:
                tree = parse_expr_tree(str(extent))
                # breakpoint()
            except ValueError:
                continue
            for factor in self._flatten_mul_nodes(tree):
                if isinstance(factor, MinNode) and isinstance(factor.left, VarNode):
                    names.add(factor.left.name)
        return names

    # ------------------------------------------------------------------
    # utils
    # ------------------------------------------------------------------

    @staticmethod
    def _binding_item_order_key(item):
        # binding 아이템을 annotation과 위치 기준으로 정렬하기 위한 키를 만든다.
        ann_order = {4: 0, 6: 1, 8: 2, 10: 3}
        return (
            ann_order.get(item.get('annotation'), 99),
            item['stage_id'],
            item['iter_id'],
        )

    @staticmethod
    def _group_binding_items_by_block_scope(items):
        # binding 아이템들을 block scope 단위로 순서를 유지하며 묶는다.
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
        # 같은 block scope 안에서는 축 이름별 첫 대표 아이템만 남긴다.
        chosen_by_axis = {}
        order = []
        for item in sorted(items, key=self._binding_item_order_key):
            key = item['axis_name']
            if key not in chosen_by_axis:
                chosen_by_axis[key] = item
                order.append(key)
        return [chosen_by_axis[key] for key in order]

    @staticmethod
    def _format_block_scope(block_scope):
        # block scope 튜플을 사람이 읽기 쉬운 문자열로 변환한다.
        if not block_scope:
            return "kernel root"
        return " > ".join(
            f"{axis}@s{stage_id}.i{iter_id}"
            for axis, stage_id, iter_id in block_scope
        )

    def _coerce_product_form_tree(self, node):
        # product-form 분석 전에 PrimExpr를 파싱하고 합법 표현으로 정규화한다.
        if isinstance(node, PrimExprNode):
            text = str(node).replace("T.min(", "min(").replace("T.max(", "max(")
            try:
                node = parse_expr_tree(text)
            except ValueError:
                return None
        return self._normalize_legal_product_tree(node)

    def _normalize_legal_product_tree(self, node):
        # clamp된 vthread split을 포함한 곱 식을 product-form 분석 가능하게 정규화한다.
        g = self.gen
        clamped_names = g._vthread_clamped_sp_names | getattr(g, '_thread_clamped_sp_names', set())
        if isinstance(node, MinNode):
            left = self._normalize_legal_product_tree(node.left)
            right = self._normalize_legal_product_tree(node.right)
            if isinstance(left, VarNode) and left.name in clamped_names:
                return left
            if isinstance(right, VarNode) and right.name in clamped_names:
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

    # def _extract_product_form_vars(self, node):
    #     # 정규화된 product-form 식에서 곱 인자 변수 목록만 추출한다.
    #     node = self._coerce_product_form_tree(node)
    #     if node is None:
    #         return None
    #     if isinstance(node, VarNode):
    #         return [node.name]
    #     if isinstance(node, ConstNode):
    #         return []
    #     if isinstance(node, ScaleMulNode):
    #         return self._extract_product_form_vars(node.child)
    #     if isinstance(node, MulNode):
    #         left = self._extract_product_form_vars(node.left)
    #         right = self._extract_product_form_vars(node.right)
    #         if left is None or right is None:
    #             return None
    #         return left + right
    #     return None

    def _extract_product_form_meta(self, node):
        # 정규화된 product-form 식에서 인자 목록과 스케일 계수를 함께 추출한다.
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
        # 식 트리를 순회하며 변수 등장 순서를 보존한 목록을 만든다.
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
        # 식 트리의 변수들을 첫 등장 순서대로 중복 없이 정리한다.
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
        # 대상 리스트 끝에 아직 없는 변수만 순서를 유지하며 추가한다.
        seen = set(target)
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            target.append(name)

    @staticmethod
    def _order_param_names_by_step_index(names):
        # split/unroll 스타일 이름을 step index 기준으로 정렬한다.
        ordered = []
        seen = set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            ordered.append(name)

        def order_key(name):
            parts = name.split("_")
            step_idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1 << 30
            pos = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else -1
            kind_order = 0 if name.startswith("sp_") else 1 if name.startswith("ur_") else 2
            return (step_idx, kind_order, pos, name)

        ordered.sort(key=order_key)
        return ordered

    def _thread_axis_limit(self, ann):
        # thread annotation 종류에 대응하는 하드웨어 상한값을 반환한다.
        g = self.gen
        if ann == 6:
            return g.hw['max_thread_x']
        if ann == 8:
            return g.hw['max_thread_y']
        if ann == 10:
            return g.hw['max_thread_z']
        return g.hw['max_threads_per_block']

    def _flatten_max_terms(self, node):
        # Max 트리를 평탄화해 각 항을 독립 제약 후보로 분리한다.
        if isinstance(node, MaxNode):
            terms = []
            for child in node.children:
                terms.extend(self._flatten_max_terms(child))
            return terms
        return [node]



    def _collect_step_indices_for_vars(self, var_names):
        # split 변수 집합이 속한 step_idx들을 원래 step 순서대로 모은다.
        g = self.gen
        wanted = set(var_names)
        step_indices = []
        for step_idx, names in sorted(g._sp_groups.items()):
            if wanted.intersection(names):
                step_indices.append(step_idx)
        return step_indices

    def _build_split_bound_denominator(self, names, extent):
        # split_structure 상한식의 분모에 들어갈 의존 변수 곱을 만든다.
        product_tree = self._build_uncapped_product_tree(names)
        return MinNode(product_tree, ConstNode(extent))

    def _build_uncapped_product_tree(self, names):
        # split factor들의 순수 곱(product) 트리를 만든다.
        product_tree = None
        for name in names:
            node = VarNode(name)
            product_tree = node if product_tree is None else MulNode(product_tree, node)
        if product_tree is None:
            return ConstNode(1)
        return product_tree


    def _flatten_mul_nodes(self, node):
        # 곱셈 트리를 평탄화해 각 factor 노드를 순서대로 꺼낸다.
        if isinstance(node, MulNode):
            return self._flatten_mul_nodes(node.left) + self._flatten_mul_nodes(node.right)
        return [node]


    # @staticmethod
    # def _dedupe_constraint_items(items, key_fn):
    #     # 키 함수 기준으로 중복 제약 아이템을 제거한다.
    #     deduped = []
    #     seen = set()
    #     for item in items:
    #         key = key_fn(item)
    #         if key in seen:
    #             continue
    #         seen.add(key)
    #         deduped.append(item)
    #     return deduped

    # @staticmethod
    # def _vectorize_item_key(item):
    #     # vectorize 제약 아이템의 중복 제거 키를 만든다.
    #     return ('vectorize', repr(item['tree']), item['limit'])

    # @staticmethod
    # def _thread_extent_item_key(item):
    #     # max_threads 제약 아이템의 중복 제거 키를 만든다.
    #     return (
    #         'max_threads',
    #         item['axis_name'],
    #         str(item['sym_extent']),
    #         item['limit'],
    #         item.get('block_scope', tuple()),
    #     )

    # @staticmethod
    # def _thread_block_item_key(item):
    #     # max_threads_per_block 제약 아이템의 중복 제거 키를 만든다.
    #     return (
    #         'max_threads_per_block',
    #         item['axis_name'],
    #         str(item['sym_extent']),
    #         item['limit'],
    #         item.get('block_scope', tuple()),
    #     )

    # @staticmethod
    # def _vthread_item_key(item):
    #     # max_vthread 제약 아이템의 중복 제거 키를 만든다.
    #     return (
    #         'max_vthread',
    #         item['axis_name'],
    #         str(item['sym_extent']),
    #         item['limit'],
    #         item.get('block_scope', tuple()),
    #     )

    # @staticmethod
    # def _min_thread_extent_item_key(item):
    #     # min_thread_extent 제약 아이템의 중복 제거 키를 만든다.
    #     return (
    #         'min_thread_extent',
    #         item['axis_name'],
    #         str(item['sym_extent']),
    #         item['limit'],
    #         item.get('block_scope', tuple()),
    #     )

    # @staticmethod
    # def _split_structure_item_key(item):
    #     # split_structure 제약 아이템의 중복 제거 키를 만든다.
    #     return ('split_structure', item['sym_name'], item['display_rhs'])






    # def _has_nonlinear(self, node):
    #     # 식 트리에 nonlinear 성격의 노드가 포함되는지 판정한다.
    #     if isinstance(node, (MinNode, CeilDivNode, PrimExprNode, CaseSplitNode, MaxNode)):
    #         return True
    #     if isinstance(node, (MulNode, AddNode, SubNode)):
    #         return self._has_nonlinear(node.left) or self._has_nonlinear(node.right)
    #     if isinstance(node, ScaleMulNode):
    #         return self._has_nonlinear(node.child)
    #     if isinstance(node, SumNode):
    #         return any(self._has_nonlinear(c) for c in node.children)
    #     return False




    # ------------------------------------------------------------------
    # Constraint-family checks
    # ------------------------------------------------------------------

    def check_all_pruning(self, sym_map=None):
        """projected/심볼릭 pruning 제약만 검사해 위반 문자열 목록을 반환한다."""
        # pruning 단계에서 쓰는 projected/구조 제약 위반을 한 번에 수집한다.
        g = self.gen
        violations = []
        if 'vectorize' in g._enabled:
            violations.extend(self._check_vectorize(sym_map))
        if 'shared_memory' in g._enabled:
            violations.extend(self._check_shared_memory(sym_map))
        if 'max_threads' in g._enabled:
            violations.extend(self._check_max_threads(sym_map))
        if 'max_threads_per_block' in g._enabled:
            violations.extend(self._check_max_threads_per_block(sym_map))
        if 'max_vthread' in g._enabled:
            violations.extend(self._check_max_vthread(sym_map))
        if 'min_thread_extent' in g._enabled:
            violations.extend(self._check_min_thread_extent(sym_map))
        if 'innermost_split' in g._enabled:
            violations.extend(self._check_innermost_split(sym_map))
        if 'split_structure' in g._enabled:
            violations.extend(self._check_split_structure(sym_map))
        return violations
        
    def _check_max_vthread(self, sym_map=None):
        # 현재 할당이 vthread extent 상한을 넘는지 검사한다.
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



    def _check_vectorize(self, sym_map=None):
        # 현재 할당이 projected vectorize 상한을 넘는지 검사한다.
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


    def _check_shared_memory(self, sym_map=None):
        # 현재 할당이 projected shared-memory 상한을 넘는지 검사한다.
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


    def _check_max_threads(self, sym_map=None):
        # 현재 할당이 per-axis thread 상한을 넘는지 검사한다.
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
                violations.append(f"{item['desc']}: actual={val}")
        return violations

    def _check_max_threads_per_block(self, sym_map=None):
        # 현재 할당이 threads-per-block 상한을 넘는지 검사한다.
        g = self.gen
        if sym_map is None:
            sym_map = g.s.sym_map
        c = self._build_max_threads_per_block_constraints()
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
                if params is not None:
                    if concrete_result is None:
                        concrete_result = g._get_concrete_final_result(params)
                    if concrete_result is not None and bool(concrete_result.get('ok')):
                        continue
                violations.append(f"{item['desc']}: actual={val}")
        return violations


    def _check_innermost_split(self, sym_map=None):
        # 현재 할당이 innermost split factor 상한을 넘는지 검사한다.
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
        # 현재 할당이 SplitStep의 구조적 상한을 위반하는지 검사한다.
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

    def _check_min_thread_extent(self, sym_map=None):
        # 현재 할당이 MLT root thread 최소 extent 하한을 위반하는지 검사한다.
        g = self.gen
        if sym_map is None:
            sym_map = g.s.sym_map
        c = self._build_min_thread_extent_constraints()
        if not c['items']:
            return []
        violations = []
        for item in c['items']:
            val = item['tree'].evaluate(sym_map)
            if not isinstance(val, int):
                violations.append(f"{item['desc']}: actual={item['axis_name']}={val}")
                continue
            if val < item['limit']:
                violations.append(f"{item['desc']}: actual={val}")
        return violations


    # ------------------------------------------------------------------
    # Deprecated
    # ------------------------------------------------------------------
