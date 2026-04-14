"""
symbolic_state — SymbolicState: task.compute_dag의 stage/iter 구조를 복사하고,
transform step 적용 시 split/unroll factor를 symbolic variable로 표현하는 순수 상태 객체.
"""
from collections import OrderedDict

from .sym_types import (
    SymExpr, SymIter, SymStage, ANNOTATION_STR,
    CA_ROOT, CA_INLINED, CA_ITER,
)


class SymbolicState:
    """
    Symbolic 버전의 auto_scheduler State.
    stages, sym_map, 내부 메타데이터를 보유하는 순수 상태 객체.
    transform step 적용은 TransformApplier, 파라미터 관리는 SymParamManager가 담당.
    """

    @staticmethod
    def _safe_int_extent(extent_expr):
        """TIR extent를 int로 변환. Sub/Add 등 심볼릭이면 simplify 후 재시도."""
        if extent_expr is None:
            return None
        try:
            return int(extent_expr)
        except TypeError:
            import tvm
            simplified = tvm.arith.Analyzer().simplify(extent_expr)
            return int(simplified)

    def __init__(self, compute_dag):
        """compute_dag의 op/stage 구조를 복사해 심볼릭 stages와 sym_map을 초기화한다."""
        self.stages = []
        self.sym_map = OrderedDict()
        self.compute_dag = compute_dag
        self._state = None  # TransformApplier.apply_steps에서 설정
        self._ca_saved_extents = {}  # {(stage_id, iter_id): SymExpr}
        self._ca_saved_mins = {}  # {(stage_id, iter_id): SymExpr}
        self._split_sym_products = {}  # {(stage_id, step_idx): SymExpr}
        self._split_step_extents = {}  # {step_idx: SymExpr} current extent before applying SplitStep
        self._cache_read_consumer = {}  # {cache_read_stage_id: consumer_stage_id}
        self._cache_read_stencil_info = {}  # {cr_stage_id: {cr_axis_idx: (pattern_kind, pattern_value, sp_name, rd_name)}}
        self._shared_fused_extents = {}  # {stage_id: SymExpr}
        self._exception_split_names = set()  # {sp_stepidx_lenidx}
        self._thread_extent_meta = {}  # {(stage_id, iter_id): {is_mlt_root_thread, relax_min_thread_extent}}

        for sid, op in enumerate(compute_dag.ops):
            if hasattr(op, 'axis'):
                dtype = str(op.output(0).dtype) if hasattr(op, 'output') else "float32"
                iters = []
                for axis in op.axis:
                    name = str(axis.var.name)
                    ext = self._safe_int_extent(axis.dom.extent) if axis.dom is not None else None
                    iters.append(SymIter(name, SymExpr(ext) if ext is not None else None,
                                         annotation=0, iter_kind=0,
                                         min_value=SymExpr(0)))
                for axis in op.reduce_axis:
                    name = str(axis.var.name)
                    ext = self._safe_int_extent(axis.dom.extent) if axis.dom is not None else None
                    iters.append(SymIter(name, SymExpr(ext) if ext is not None else None,
                                         annotation=0, iter_kind=1,
                                         min_value=SymExpr(0)))
                self.stages.append(SymStage(op.name, 'compute', iters, dtype=dtype))
            else:
                dtype = str(op.output(0).dtype) if hasattr(op, 'output') else "float32"
                self.stages.append(SymStage(op.name, 'placeholder', [], dtype=dtype))

    @staticmethod
    def _clone_symexpr(expr):
        if expr is None:
            return None
        return SymExpr(expr.val)

    def canonicalize_param_values(self):
        """Concrete init 값이 아닌 구조만 보존하도록 파라미터 기본값을 정규화한다."""
        for name in list(self.sym_map.keys()):
            if name.startswith("sp_"):
                self.sym_map[name] = 1
            elif name.startswith("ur_"):
                self.sym_map[name] = 0

    def clone(self):
        """현재 stages·sym_map·메타데이터를 복사한 새 SymbolicState를 반환한다."""
        cloned = SymbolicState(self.compute_dag)
        cloned.stages = [stage.clone() for stage in self.stages]
        cloned.sym_map = OrderedDict((name, value) for name, value in self.sym_map.items())
        cloned._state = self._state
        cloned._ca_saved_extents = {
            key: self._clone_symexpr(expr) for key, expr in self._ca_saved_extents.items()
        }
        cloned._ca_saved_mins = {
            key: self._clone_symexpr(expr) for key, expr in self._ca_saved_mins.items()
        }
        cloned._split_sym_products = {
            key: self._clone_symexpr(expr) for key, expr in self._split_sym_products.items()
        }
        cloned._split_step_extents = {
            step_idx: self._clone_symexpr(expr)
            for step_idx, expr in self._split_step_extents.items()
        }
        cloned._cache_read_consumer = dict(self._cache_read_consumer)
        cloned._cache_read_stencil_info = {
            sid: dict(info) for sid, info in self._cache_read_stencil_info.items()
        }
        cloned._shared_fused_extents = {
            sid: self._clone_symexpr(expr) for sid, expr in self._shared_fused_extents.items()
        }
        cloned._exception_split_names = set(self._exception_split_names)
        cloned._thread_extent_meta = {
            key: dict(meta) for key, meta in self._thread_extent_meta.items()
        }
        return cloned

    # ─── 내부 데이터 shift (CacheRead/CacheWrite stage 삽입 시) ───
    def _shift_ca_saved_extents(self, inserted_stage_id, offset=1):
        """stage 삽입 후 stage id 기반 메타데이터 key를 일괄 보정."""
        new_saved = {}
        for (sid, iid), expr in self._ca_saved_extents.items():
            new_sid = sid + offset if sid >= inserted_stage_id else sid
            new_saved[(new_sid, iid)] = expr
        self._ca_saved_extents = new_saved

        new_saved_mins = {}
        for (sid, iid), expr in self._ca_saved_mins.items():
            new_sid = sid + offset if sid >= inserted_stage_id else sid
            new_saved_mins[(new_sid, iid)] = expr
        self._ca_saved_mins = new_saved_mins

        new_split_prods = {}
        for (sid, step_idx), expr in self._split_sym_products.items():
            new_sid = sid + offset if sid >= inserted_stage_id else sid
            new_split_prods[(new_sid, step_idx)] = expr
        self._split_sym_products = new_split_prods

        new_cr_consumer = {}
        for cr_sid, consumer_sid in self._cache_read_consumer.items():
            new_cr = cr_sid + offset if cr_sid >= inserted_stage_id else cr_sid
            new_con = consumer_sid + offset if consumer_sid >= inserted_stage_id else consumer_sid
            new_cr_consumer[new_cr] = new_con
        self._cache_read_consumer = new_cr_consumer

        new_stencil = {}
        for cr_sid, info in self._cache_read_stencil_info.items():
            new_cr = cr_sid + offset if cr_sid >= inserted_stage_id else cr_sid
            new_stencil[new_cr] = info
        self._cache_read_stencil_info = new_stencil

        new_shared = {}
        for sid, ext in self._shared_fused_extents.items():
            new_sid = sid + offset if sid >= inserted_stage_id else sid
            new_shared[new_sid] = ext
        self._shared_fused_extents = new_shared

    # ─── 출력 ───
    def __str__(self):
        return self.to_str(delete_trivial_loop=False)

    def __repr__(self):
        return self.to_str(delete_trivial_loop=False)

    def to_str(self, delete_trivial_loop=True):
        lines = []
        placeholders = [s.op_name for s in self.stages if s.op_type == 'placeholder']
        lines.append("Placeholder: " + ", ".join(placeholders))
        for sid, stage in enumerate(self.stages):
            if stage.op_type == 'placeholder':
                continue
            if stage.compute_at == CA_ROOT:
                self._print_stage(lines, sid, 0, delete_trivial_loop)
        return "\n".join(lines)

    def _print_stage(self, lines, stage_id, base_indent, delete_trivial_loop):
        stage = self.stages[stage_id]
        if stage.auto_unroll_max_step is not None:
            lines.append(" " * base_indent + f"{stage.op_name} auto_unroll: {stage.auto_unroll_max_step}")
        if stage.storage_offset != 0:
            lines.append(" " * base_indent + f"{stage.op_name} storage_offset: {stage.storage_offset}")

        indent = 0
        for iid, it in enumerate(stage.iters):
            is_zero_min = (
                it.min_value is not None and it.min_value.is_concrete and it.min_value.val == 0
            )
            is_trivial = (
                it.extent is not None and it.extent.is_concrete and it.extent.val == 1 and is_zero_min
            )
            if not (delete_trivial_loop and is_trivial):
                ann = ANNOTATION_STR.get(it.annotation, "?")
                if it.extent is not None:
                    lines.append(
                        " " * (base_indent + indent)
                        + f"{ann} {it.name} ({it.min_value},{it.extent})"
                    )
                else:
                    lines.append(" " * (base_indent + indent) + f"{ann} {it.name} (None)")
                indent += 2

            for asid, astage in enumerate(self.stages):
                if (astage.compute_at == CA_ITER and
                    astage.attach_stage_id == stage_id and
                    astage.attach_iter_id == iid):
                    self._print_stage(lines, asid, base_indent + indent, delete_trivial_loop)

        lines.append(" " * (base_indent + indent) + f"{stage.op_name} = ...")

    # ─── Symbolic extent 조회 함수 ───
    def _collect_extents_by_annotation(self, ann_codes):
        """주어진 annotation 코드 집합에 해당하는 iter의 (stage_id, iter_id, SymExpr) 목록 반환."""
        results = []
        for sid, stage in enumerate(self.stages):
            if stage.compute_at == CA_INLINED:
                continue
            for iid, it in enumerate(stage.iters):
                if it.annotation in ann_codes:
                    results.append((sid, iid, it.extent))
        return results

    def get_vectorize_extents(self):
        return self._collect_extents_by_annotation({2})

    def get_thread_extents(self):
        return self._collect_extents_by_annotation({6, 8, 10})

    def get_vthread_extents(self):
        return self._collect_extents_by_annotation({4})

    def get_shared_memory_extents(self):
        results = []
        for sid, ext in sorted(self._shared_fused_extents.items()):
            results.append((sid, self.stages[sid].op_name, ext))
        return results
