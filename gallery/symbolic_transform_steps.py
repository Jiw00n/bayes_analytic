import os
import sys

project_root = "/root/work/tvm-ansor"
os.environ["TVM_HOME"] = f"{project_root}"
os.environ["TVM_LIBRARY_PATH"] = f"{project_root}/build-release"
if f"{project_root}/python" not in sys.path:
    sys.path.insert(0, f"{project_root}/python")

sys.path = [p for p in sys.path if not p.startswith(f"{project_root}/build")]
sys.path.append(f"{project_root}/build-release")
os.environ["LD_LIBRARY_PATH"] = f"{project_root}/build-release:" + os.environ.get("LD_LIBRARY_PATH", "")


import numpy as np
from util_manager import PathManager, get_network
import tvm
from tvm import auto_scheduler

TARGET = tvm.target.Target("cuda")

def get_tasks(mod, params, path_manager, verbose=True, get_pkl=True):
    if get_pkl:
        tasks, task_weights = path_manager.tasks_pkl_use()
    
    if get_pkl is False or tasks is None:
        print("Extract tasks...")
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, TARGET)
        if path_manager.tasks_pkl_check() is False:
            path_manager.tasks_pkl_save(tasks, task_weights)

    if verbose:
        for idx, task in enumerate(tasks):
            print("========== Task %d : %s  (workload key: %s) ==========" % (idx, task.desc, task.workload_key))
            print(task.compute_dag)
    
    print(f"Total tasks length : {len(tasks)}")
    return tasks, task_weights



from types import SimpleNamespace

args = SimpleNamespace(
    network="resnet_18",
    batch_size=1,
    dtype="float32",
    layout="NHWC",
    timenow=None,
    json=None
)

mod, params, input_shape, output_shape = get_network(args.network, args.batch_size, args.layout, dtype=args.dtype)
path_manager = PathManager(args.network, input_shape, args, None, json="/root/work/tvm-ansor/gallery/logs_json/tmp.json")
tasks, task_weights = get_tasks(None, params, path_manager, verbose=False, get_pkl=True)
tasks, task_weights = zip(*sorted(zip(tasks, task_weights), key=lambda x: x[0].desc))

search_policies = []
for idx, (task, weight) in enumerate(zip(tasks, task_weights)):
    print(f"T{idx} : {task.desc} ({weight})")
    search_policies.append(
        auto_scheduler.SketchPolicy(task, auto_scheduler.XGBModel())
    )



"""
SymbolicState: task.compute_dag의 stage/iter 구조를 그대로 복사하고,
transform step 적용 시 split/unroll factor를 symbolic variable로 표현하는 객체.

C++ loop_state.cc의 State/Stage/Iterator 구조를 Python으로 미러링합니다.

변수 네이밍:
  - SplitStep:             sp_{step_idx}_{length_idx}
  - FollowSplitStep:       src SplitStep의 sym_name 재사용
  - FollowFusedSplitStep:  src SplitStep들의 sym_name 곱
  - PragmaStep (unroll):   ur_{step_idx}

ComputeAt extent 복원 전략 (lazy):
  ComputeAt는 extent를 None으로만 설정.
  이후 Fuse/Split/FFSP에서 None extent를 만나면 그 시점(step_idx)에서
  ReplayStepsPartial + InferBound를 호출하여 concrete extent를 동적 복원.
  이렇게 해야 부모 stage의 split 결과가 반영된 정확한 extent를 얻을 수 있음.
"""
from collections import OrderedDict
import math

# ── Annotation 문자열 매핑 (C++ IteratorAnnotationString 동일) ──
ANNOTATION_STR = {
    0: "for",           # kNone
    1: "unroll",        # kUnroll
    2: "vectorize",     # kVectorize
    3: "parallel",      # kParallel
    4: "vthread",       # kVThread
    5: "blockIdx.x",    # kBlockX
    6: "threadIdx.x",   # kThreadX
    7: "blockIdx.y",    # kBlockY
    8: "threadIdx.y",   # kThreadY
    9: "blockIdx.z",    # kBlockZ
    10: "threadIdx.z",  # kThreadZ
    11: "tensorize",    # kTensorize
}

# ── ComputeAtKind ──
CA_ROOT    = 0  # kRoot
CA_INLINED = 1  # kInlined
CA_ITER    = 2  # kIter


class SymExpr:
    """
    Symbolic expression wrapper.
    실제 값(int)이면 그냥 int, symbolic이면 문자열.
    연산(ceil div, mul 등)을 문자열로 합성.
    """
    def __init__(self, val):
        self.val = val

    @property
    def is_concrete(self):
        return isinstance(self.val, int)

    def __repr__(self):
        return str(self.val) if self.val is not None else "None"

    def __str__(self):
        return str(self.val) if self.val is not None else "None"

    def __int__(self):
        if self.is_concrete:
            return self.val
        raise ValueError(f"Cannot convert symbolic '{self.val}' to int")

    @staticmethod
    def ceildiv(a, b):
        if isinstance(a, SymExpr): a = a.val
        if isinstance(b, SymExpr): b = b.val
        if isinstance(a, int) and isinstance(b, int):
            return SymExpr((a + b - 1) // b)
        return SymExpr(f"ceil({a}/({b}))")

    @staticmethod
    def _needs_parens_for_mul(s):
        """문자열 expression이 mul에서 사용될 때 괄호가 필요한지 판단.
        최외곽 레벨에서 + 또는 - 연산자가 있으면 True."""
        if not isinstance(s, str):
            return False
        depth = 0
        i = 0
        while i < len(s):
            c = s[i]
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            elif depth == 0 and c == '+':
                return True
            elif depth == 0 and c == '-' and i > 0 and s[i-1] == ' ':
                return True
            i += 1
        return False

    @staticmethod
    def mul(a, b):
        if isinstance(a, SymExpr): a = a.val
        if isinstance(b, SymExpr): b = b.val
        if isinstance(a, int) and isinstance(b, int):
            return SymExpr(a * b)
        if a == 1: return SymExpr(b)
        if b == 1: return SymExpr(a)
        # 연산자 우선순위 보호: 최외곽에 +/- 포함된 표현식은 괄호 추가
        a_str = str(a)
        b_str = str(b)
        if SymExpr._needs_parens_for_mul(a_str):
            a_str = f"({a_str})"
        if SymExpr._needs_parens_for_mul(b_str):
            b_str = f"({b_str})"
        return SymExpr(f"{a_str}*{b_str}")

    @staticmethod
    def product(items):
        result = SymExpr(1)
        for item in items:
            result = SymExpr.mul(result, item)
        return result


class SymIter:
    """Iterator (C++ Iterator 대응)"""
    def __init__(self, name, extent, annotation=0, iter_kind=0):
        self.name = name
        self.extent = extent         # SymExpr or None
        self.annotation = annotation # int
        self.iter_kind = iter_kind   # int

    def clone(self):
        return SymIter(self.name,
                       SymExpr(self.extent.val) if self.extent else None,
                       self.annotation, self.iter_kind)

    def __repr__(self):
        ann = ANNOTATION_STR.get(self.annotation, "?")
        if self.extent is not None:
            return f"{ann} {self.name} (0,{self.extent})"
        else:
            return f"{ann} {self.name} (None)"


class SymStage:
    """Stage (C++ Stage 대응)"""
    def __init__(self, op_name, op_type, iters, compute_at=CA_ROOT,
                 auto_unroll_max_step=None, storage_offset=0):
        self.op_name = op_name
        self.op_type = op_type
        self.iters = list(iters)
        self.compute_at = compute_at
        self.auto_unroll_max_step = auto_unroll_max_step
        self.storage_offset = storage_offset
        self.attach_stage_id = None
        self.attach_iter_id = None


class SymbolicState:
    """
    Symbolic 버전의 auto_scheduler State.
    task.compute_dag에서 초기화한 뒤, transform_steps를 순차 적용.
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
        self.stages = []
        self.sym_map = OrderedDict()
        self.compute_dag = compute_dag
        self._state = None  # apply_steps에서 설정
        self._ca_saved_extents = {}  # {(stage_id, iter_id): SymExpr} - CA로 소실된 symbolic extent 저장
        # CacheRead symbolic 지원용 데이터 구조
        self._split_sym_products = {}  # {(stage_id, step_idx): SymExpr} - split의 symbolic factor product
        self._cache_read_consumer = {}  # {cache_read_stage_id: consumer_stage_id}
        # Stencil info: {cr_stage_id: {cr_axis_idx: (stride, spatial_split_order, reduce_split_order)}}
        # stride>0 → extent = (spatial_prod - 1)*stride + reduce_prod
        # stride=0, spatial_split_order is not None → extent = spatial_split_prod (simple mapping)
        # stride=0, reduce_split_order is not None → extent = reduce_split_prod (simple mapping)
        self._cache_read_stencil_info = {}

        for sid, op in enumerate(compute_dag.ops):
            if hasattr(op, 'axis'):
                iters = []
                for axis in op.axis:
                    name = str(axis.var.name)
                    ext = self._safe_int_extent(axis.dom.extent) if axis.dom is not None else None
                    iters.append(SymIter(name, SymExpr(ext) if ext is not None else None,
                                         annotation=0, iter_kind=0))
                for axis in op.reduce_axis:
                    name = str(axis.var.name)
                    ext = self._safe_int_extent(axis.dom.extent) if axis.dom is not None else None
                    iters.append(SymIter(name, SymExpr(ext) if ext is not None else None,
                                         annotation=0, iter_kind=1))
                self.stages.append(SymStage(op.name, 'compute', iters))
            else:
                self.stages.append(SymStage(op.name, 'placeholder', []))

    def _shift_ca_saved_extents(self, inserted_stage_id):
        """CacheRead/CacheWrite로 stage가 삽입될 때, saved extents와 관련 데이터 key 업데이트."""
        new_saved = {}
        for (sid, iid), expr in self._ca_saved_extents.items():
            new_sid = sid + 1 if sid >= inserted_stage_id else sid
            new_saved[(new_sid, iid)] = expr
        self._ca_saved_extents = new_saved

        # _split_sym_products의 stage_id도 shift
        new_split_prods = {}
        for (sid, step_idx), expr in self._split_sym_products.items():
            new_sid = sid + 1 if sid >= inserted_stage_id else sid
            new_split_prods[(new_sid, step_idx)] = expr
        self._split_sym_products = new_split_prods

        # _cache_read_consumer의 stage_id도 shift
        new_cr_consumer = {}
        for cr_sid, consumer_sid in self._cache_read_consumer.items():
            new_cr = cr_sid + 1 if cr_sid >= inserted_stage_id else cr_sid
            new_con = consumer_sid + 1 if consumer_sid >= inserted_stage_id else consumer_sid
            new_cr_consumer[new_cr] = new_con
        self._cache_read_consumer = new_cr_consumer

        # _cache_read_stencil_info의 stage_id도 shift
        new_stencil = {}
        for cr_sid, info in self._cache_read_stencil_info.items():
            new_cr = cr_sid + 1 if cr_sid >= inserted_stage_id else cr_sid
            new_stencil[new_cr] = info
        self._cache_read_stencil_info = new_stencil

    # ─────────────────────────────────────────────
    #  출력 함수
    # ─────────────────────────────────────────────
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
            is_trivial = (it.extent is not None and it.extent.is_concrete and it.extent.val == 1)
            if not (delete_trivial_loop and is_trivial):
                ann = ANNOTATION_STR.get(it.annotation, "?")
                if it.extent is not None:
                    lines.append(" " * (base_indent + indent) + f"{ann} {it.name} (0,{it.extent})")
                else:
                    lines.append(" " * (base_indent + indent) + f"{ann} {it.name} (None)")
                indent += 2

            for asid, astage in enumerate(self.stages):
                if (astage.compute_at == CA_ITER and
                    astage.attach_stage_id == stage_id and
                    astage.attach_iter_id == iid):
                    self._print_stage(lines, asid, base_indent + indent, delete_trivial_loop)

        lines.append(" " * (base_indent + indent) + f"{stage.op_name} = ...")

    # ─────────────────────────────────────────────
    #  Transform Step 적용 (dispatcher)
    # ─────────────────────────────────────────────
    def apply_steps(self, state):
        self._state = state
        steps = state.transform_steps
        for i, step in enumerate(steps):
            tk = step.type_key.split(".")[-1]
            if tk == "AnnotationStep":
                self._apply_annotation(step)
            elif tk == "FuseStep":
                self._apply_fuse(step, i)
            elif tk == "PragmaStep":
                self._apply_pragma(step, i)
            elif tk == "ReorderStep":
                self._apply_reorder(step)
            elif tk == "SplitStep":
                self._apply_split(step, i)
            elif tk == "FollowSplitStep":
                self._apply_follow_split(step, steps, i)
            elif tk == "FollowFusedSplitStep":
                self._apply_follow_fused_split(step, steps, i)
            elif tk == "StorageAlignStep":
                self._apply_storage_align(step)
            elif tk == "ComputeAtStep":
                self._apply_compute_at(step)
            elif tk == "ComputeInlineStep":
                self._apply_compute_inline(step)
            elif tk == "ComputeRootStep":
                self._apply_compute_root(step)
            elif tk == "CacheReadStep":
                self._apply_cache_read(step, state, i)
            elif tk == "CacheWriteStep":
                self._apply_cache_write(step, state, i)
            else:
                print(f"  [WARN] Unhandled step type: {tk}")

        # 최종 InferBound: 혹시 남아있는 None extent 복원
        self._infer_bound_final(state)

    # ─────────────────────────────────────────────
    #  Lazy extent 복원 (CA stage에서 None extent를 만났을 때)
    # ─────────────────────────────────────────────
    def _restore_stage_extents_if_needed(self, stage_id, step_idx):
        """
        해당 stage에 None extent iter가 있으면,
        현재 step_idx 시점에서 partial InferBound로 concrete extent를 복원.
        CA로 소실된 symbolic extent가 있으면 InferBound와 매칭하여 symbolic 복원.
        CacheRead stage인 경우, consumer의 split products와 역매칭하여 symbolic 복원.
        Fuse/Split/FFSP 직전에 호출하여 정확한 extent를 확보.
        """
        stage = self.stages[stage_id]
        has_none = any(it.extent is None for it in stage.iters)
        if not has_none:
            return

        # 현재 step까지 replay + InferBound
        ps = tvm._ffi.get_global_func("auto_scheduler.ReplayStepsPartial")(
            self.compute_dag, self._state, step_idx)
        bounded = self.compute_dag.infer_bound_from_state(ps)

        if stage_id >= len(bounded.stages):
            return

        # CacheRead stage인 경우, consumer split products로 역매칭 준비
        cr_sym_candidates = None
        cr_stencil = None
        cr_ordered_splits = None
        if stage_id in self._cache_read_consumer:
            cr_sym_candidates = self._get_consumer_split_sym_products(stage_id)
            cr_stencil = self._cache_read_stencil_info.get(stage_id)
            if cr_stencil:
                # consumer splits를 step_idx 순서로 정렬하여 split_order → (sym_prod, eval) 매핑
                consumer_sid = self._cache_read_consumer[stage_id]
                ordered = [(si, prod) for (s, si), prod in self._split_sym_products.items()
                          if s == consumer_sid]
                ordered.sort(key=lambda x: x[0])
                cr_ordered_splits = [(SymExpr(prod.val), eval_sym_extent(prod, self.sym_map)) 
                                    for si, prod in ordered]

        real_stage = bounded.stages[stage_id]
        for iid in range(len(stage.iters)):
            if stage.iters[iid].extent is None and iid < len(real_stage.iters):
                real_it = real_stage.iters[iid]
                if real_it.range is not None:
                    real_ext = int(real_it.range.extent)
                    # 1) CA로 소실된 symbolic extent 복원 시도
                    saved = self._ca_saved_extents.get((stage_id, iid))
                    if saved is not None:
                        eval_val = eval_sym_extent(saved, self.sym_map)
                        if eval_val == real_ext:
                            stage.iters[iid].extent = saved
                            continue
                    # 2) CacheRead stage: consumer split products 역매칭
                    if cr_sym_candidates is not None:
                        matched_sym = self._match_cr_extent(real_ext, cr_sym_candidates)
                        if matched_sym is not None:
                            stage.iters[iid].extent = matched_sym
                            cr_sym_candidates.remove((real_ext, matched_sym))
                            continue
                        # 2b) Stencil 매칭: (sp_prod - 1) * stride + rd_prod
                        if cr_stencil and cr_ordered_splits and iid in cr_stencil:
                            stride, sp_order, rd_order = cr_stencil[iid]
                            if stride > 0 and sp_order is not None and rd_order is not None:
                                sp_sym, sp_eval = cr_ordered_splits[sp_order]
                                rd_sym, rd_eval = cr_ordered_splits[rd_order]
                                predicted = (sp_eval - 1) * stride + rd_eval
                                if predicted == real_ext:
                                    # symbolic: (sp_sym - 1) * stride + rd_sym
                                    stencil_expr = SymExpr(f"({sp_sym.val} - 1)*{stride} + {rd_sym.val}")
                                    stage.iters[iid].extent = stencil_expr
                                    continue
                    # 3) fallback: concrete
                    stage.iters[iid].extent = SymExpr(real_ext)

    def _get_consumer_split_sym_products(self, cache_read_stage_id):
        """
        CacheRead stage의 consumer에 적용된 split의 symbolic factor product 목록 반환.
        Returns: [(concrete_eval, SymExpr), ...] - eval 값과 symbolic expression 쌍
        """
        consumer_sid = self._cache_read_consumer.get(cache_read_stage_id)
        if consumer_sid is None:
            return None

        candidates = []
        for (sid, step_idx), sym_prod in self._split_sym_products.items():
            if sid == consumer_sid:
                eval_val = eval_sym_extent(sym_prod, self.sym_map)
                if isinstance(eval_val, int):
                    candidates.append((eval_val, SymExpr(sym_prod.val)))
        return candidates

    def _match_cr_extent(self, real_ext, candidates):
        """
        CacheRead axis의 concrete InferBound 값을 candidates에서 찾아 symbolic 반환.
        candidates: [(eval_val, SymExpr), ...]
        greedy: 첫 번째 매칭 반환.
        """
        for eval_val, sym_expr in candidates:
            if eval_val == real_ext:
                return sym_expr
        return None

    def _infer_bound_final(self, state):
        """최종 InferBound: 남아있는 None extent 복원.
        CA로 소실된 symbolic extent가 있으면 InferBound 결과와 매칭하여 복원.
        CacheRead stage인 경우, consumer split products와 역매칭."""
        from tvm.auto_scheduler.loop_state import StateObject
        state_obj = state if isinstance(state, StateObject) else state.state_object
        bounded = self.compute_dag.infer_bound_from_state(state_obj)

        for sid in range(len(self.stages)):
            sym_stage = self.stages[sid]
            if sid >= len(bounded.stages):
                continue
            real_stage = bounded.stages[sid]

            # CacheRead stage인 경우, consumer split products 역매칭 준비
            cr_sym_candidates = None
            cr_stencil = None
            cr_ordered_splits = None
            if sid in self._cache_read_consumer:
                cr_sym_candidates = self._get_consumer_split_sym_products(sid)
                cr_stencil = self._cache_read_stencil_info.get(sid)
                if cr_stencil:
                    consumer_sid = self._cache_read_consumer[sid]
                    ordered = [(si, prod) for (s, si), prod in self._split_sym_products.items()
                              if s == consumer_sid]
                    ordered.sort(key=lambda x: x[0])
                    cr_ordered_splits = [(SymExpr(prod.val), eval_sym_extent(prod, self.sym_map))
                                        for si, prod in ordered]

            for iid in range(len(sym_stage.iters)):
                sym_it = sym_stage.iters[iid]
                if sym_it.extent is None and iid < len(real_stage.iters):
                    real_it = real_stage.iters[iid]
                    if real_it.range is None:
                        continue
                    real_ext = int(real_it.range.extent)
                    # 1) CA로 소실된 symbolic extent 복원 시도
                    saved = self._ca_saved_extents.get((sid, iid))
                    if saved is not None:
                        eval_val = eval_sym_extent(saved, self.sym_map)
                        if eval_val == real_ext:
                            sym_it.extent = saved
                            continue
                        else:
                            sym_it.extent = SymExpr(real_ext)
                            continue
                    # 2) CacheRead stage: consumer split products 역매칭
                    if cr_sym_candidates is not None:
                        matched_sym = self._match_cr_extent(real_ext, cr_sym_candidates)
                        if matched_sym is not None:
                            sym_it.extent = matched_sym
                            cr_sym_candidates.remove((real_ext, matched_sym))
                            continue
                        # 2b) Stencil 매칭: (sp_prod - 1) * stride + rd_prod
                        if cr_stencil and cr_ordered_splits and iid in cr_stencil:
                            stride, sp_order, rd_order = cr_stencil[iid]
                            if stride > 0 and sp_order is not None and rd_order is not None:
                                sp_sym, sp_eval = cr_ordered_splits[sp_order]
                                rd_sym, rd_eval = cr_ordered_splits[rd_order]
                                predicted = (sp_eval - 1) * stride + rd_eval
                                if predicted == real_ext:
                                    stencil_expr = SymExpr(f"({sp_sym.val} - 1)*{stride} + {rd_sym.val}")
                                    sym_it.extent = stencil_expr
                                    continue
                    # 3) fallback: concrete
                    sym_it.extent = SymExpr(real_ext)

    # ─────────────────────────────────────────────
    #  AnnotationStep
    # ─────────────────────────────────────────────
    def _apply_annotation(self, step):
        self.stages[step.stage_id].iters[step.iter_id].annotation = int(step.annotation)

    # ─────────────────────────────────────────────
    #  FuseStep (lazy extent 복원 포함)
    # ─────────────────────────────────────────────
    def _apply_fuse(self, step, step_idx):
        sid = step.stage_id
        fused_ids = [int(x) for x in step.fused_ids]
        stage = self.stages[sid]

        if not fused_ids:
            new_it = SymIter("", SymExpr(1), annotation=0, iter_kind=3)
            stage.iters.insert(0, new_it)
            return

        # None extent가 있으면 이 시점에서 lazy 복원
        self._restore_stage_extents_if_needed(sid, step_idx)

        new_name = "@".join(stage.iters[fid].name for fid in fused_ids) + "@"

        new_extent = SymExpr(1)
        all_defined = True
        new_iter_kind = stage.iters[fused_ids[0]].iter_kind
        for fid in fused_ids:
            it = stage.iters[fid]
            if it.extent is not None:
                new_extent = SymExpr.mul(new_extent, it.extent)
            else:
                all_defined = False
            if it.iter_kind != new_iter_kind:
                new_iter_kind = 2

        new_it = SymIter(new_name,
                         new_extent if all_defined else None,
                         annotation=0, iter_kind=new_iter_kind)

        begin = fused_ids[0]
        end = fused_ids[-1]
        new_iters = stage.iters[:begin] + [new_it] + stage.iters[end + 1:]
        stage.iters = new_iters

        removed = len(fused_ids) - 1
        for other_stage in self.stages:
            if (other_stage.compute_at == CA_ITER and
                other_stage.attach_stage_id == sid):
                old_aid = other_stage.attach_iter_id
                if old_aid > end:
                    other_stage.attach_iter_id = old_aid - removed
                elif old_aid >= begin:
                    other_stage.attach_iter_id = begin

    # ─────────────────────────────────────────────
    #  PragmaStep — ur_{step_idx}
    # ─────────────────────────────────────────────
    def _apply_pragma(self, step, step_idx):
        sid = step.stage_id
        pragma_type = str(step.pragma_type)
        if pragma_type.startswith("auto_unroll_max_step"):
            parts = pragma_type.split("$")
            if len(parts) == 2:
                val = int(parts[1])
                sym_name = f"ur_{step_idx}"
                self.sym_map[sym_name] = val
                self.stages[sid].auto_unroll_max_step = SymExpr(sym_name)
        elif pragma_type == "debug_skip_region":
            self.stages[sid].compute_at = CA_ROOT
            self.stages[sid].attach_stage_id = None
            self.stages[sid].attach_iter_id = None

    # ─────────────────────────────────────────────
    #  ReorderStep
    # ─────────────────────────────────────────────
    def _apply_reorder(self, step):
        sid = step.stage_id
        after_ids = [int(x) for x in step.after_ids]
        stage = self.stages[sid]
        old_iters = stage.iters
        stage.iters = [old_iters[i] for i in after_ids]
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(after_ids)}
        for other_stage in self.stages:
            if (other_stage.compute_at == CA_ITER and
                other_stage.attach_stage_id == sid and
                other_stage.attach_iter_id in old_to_new):
                other_stage.attach_iter_id = old_to_new[other_stage.attach_iter_id]

    # ─────────────────────────────────────────────
    #  SplitStep — sp_{step_idx}_{length_idx}
    #  (lazy extent 복원 포함)
    # ─────────────────────────────────────────────
    def _apply_split(self, step, step_idx):
        sid = step.stage_id
        iid = step.iter_id
        lengths = list(step.lengths)
        inner_to_outer = bool(step.inner_to_outer)

        stage = self.stages[sid]

        # None extent가 있으면 lazy 복원
        self._restore_stage_extents_if_needed(sid, step_idx)

        orig_iter = stage.iters[iid]

        # tosplit_extent 결정: 현재 iter의 extent를 우선 사용.
        # CA stage에서는 step.extent가 부정확(CA 직후 InferBound 값)하므로,
        # lazy 복원된 iter extent(정확한 시점의 InferBound 값)를 사용해야 함.
        if orig_iter.extent is not None:
            tosplit_extent = orig_iter.extent  # symbolic/concrete 모두 가능
        elif step.extent is not None:
            tosplit_extent = SymExpr(int(step.extent))  # fallback
        else:
            tosplit_extent = None

        sym_lengths = []
        for li, length in enumerate(lengths):
            val = int(length) if length is not None else None
            sym_name = f"sp_{step_idx}_{li}"
            self.sym_map[sym_name] = val
            sym_lengths.append(SymExpr(sym_name))

        # Split factor의 symbolic product 저장 (CacheRead symbolic 복원용)
        sym_prod = SymExpr.product(sym_lengths)
        self._split_sym_products[(sid, step_idx)] = sym_prod

        outs = []
        if inner_to_outer:
            for i in range(len(lengths)):
                li = len(lengths) - i - 1
                name = f"{orig_iter.name}.{len(lengths) - i}"
                sym_ext = sym_lengths[li]
                outs.append(SymIter(name, sym_ext, annotation=0, iter_kind=orig_iter.iter_kind))
                if tosplit_extent is not None:
                    tosplit_extent = SymExpr.ceildiv(tosplit_extent, sym_ext)
                else:
                    tosplit_extent = None
            outs.append(SymIter(f"{orig_iter.name}.0", tosplit_extent,
                                annotation=0, iter_kind=orig_iter.iter_kind))
            outs = list(reversed(outs))
        else:
            for i in range(len(lengths)):
                name = f"{orig_iter.name}.{i}"
                sym_ext = sym_lengths[i]
                outs.append(SymIter(name, sym_ext, annotation=0, iter_kind=orig_iter.iter_kind))
                if tosplit_extent is not None:
                    tosplit_extent = SymExpr.ceildiv(tosplit_extent, sym_ext)
                else:
                    tosplit_extent = None
            outs.append(SymIter(f"{orig_iter.name}.{len(lengths)}", tosplit_extent,
                                annotation=0, iter_kind=orig_iter.iter_kind))

        new_iters = stage.iters[:iid] + outs + stage.iters[iid + 1:]
        stage.iters = new_iters

        shift = len(lengths)
        for other_stage in self.stages:
            if (other_stage.compute_at == CA_ITER and
                other_stage.attach_stage_id == sid and
                other_stage.attach_iter_id >= iid):
                other_stage.attach_iter_id += shift

    # ─────────────────────────────────────────────
    #  FollowSplitStep
    # ─────────────────────────────────────────────
    def _apply_follow_split(self, step, all_steps, step_idx):
        sid = step.stage_id
        iid = step.iter_id
        src_step_id = step.src_step_id
        n_split = int(step.n_split)

        src_step = all_steps[src_step_id]
        src_lengths = list(src_step.lengths)

        extracted = []
        for j in range(min(n_split - 1, len(src_lengths))):
            extracted.append(src_lengths[j])
        last = 1
        j_start = n_split - 1
        all_defined = True
        for j in range(j_start, len(src_lengths)):
            if src_lengths[j] is not None:
                last *= int(src_lengths[j])
            else:
                all_defined = False
                break
        extracted.append(last if all_defined else None)

        stage = self.stages[sid]

        # lazy extent 복원
        self._restore_stage_extents_if_needed(sid, step_idx)

        orig_iter = stage.iters[iid]
        orig_extent = None
        if orig_iter.extent is not None and orig_iter.extent.is_concrete:
            orig_extent = orig_iter.extent.val

        tosplit_extent = SymExpr(orig_extent) if orig_extent is not None else \
                         (orig_iter.extent if orig_iter.extent is not None else None)
        sym_lengths = []
        for li in range(len(extracted)):
            if li < n_split - 1:
                src_sym_name = f"sp_{src_step_id}_{li}"
            else:
                if j_start < len(src_lengths) and j_start == len(src_lengths) - 1:
                    src_sym_name = f"sp_{src_step_id}_{j_start}"
                else:
                    parts = [f"sp_{src_step_id}_{j}" for j in range(j_start, len(src_lengths))]
                    src_sym_name = "*".join(parts) if parts else str(extracted[li])
            sym_lengths.append(SymExpr(src_sym_name))

        outs = []
        for i in range(len(extracted)):
            li = len(extracted) - i - 1
            name = f"{orig_iter.name}.{len(extracted) - i}"
            sym_ext = sym_lengths[li]
            outs.append(SymIter(name, sym_ext, annotation=0, iter_kind=orig_iter.iter_kind))
            if tosplit_extent is not None:
                tosplit_extent = SymExpr.ceildiv(tosplit_extent, sym_ext)
            else:
                tosplit_extent = None
        outs.append(SymIter(f"{orig_iter.name}.0", tosplit_extent,
                            annotation=0, iter_kind=orig_iter.iter_kind))
        outs = list(reversed(outs))

        new_iters = stage.iters[:iid] + outs + stage.iters[iid + 1:]
        stage.iters = new_iters

        shift = len(extracted)
        for other_stage in self.stages:
            if (other_stage.compute_at == CA_ITER and
                other_stage.attach_stage_id == sid and
                other_stage.attach_iter_id >= iid):
                other_stage.attach_iter_id += shift

    # ─────────────────────────────────────────────
    #  FollowFusedSplitStep (lazy extent 복원 포함)
    # ─────────────────────────────────────────────
    def _apply_follow_fused_split(self, step, all_steps, step_idx):
        sid = step.stage_id
        iid = step.iter_id
        src_step_ids = [int(x) for x in step.src_step_ids]
        level = int(step.level)
        factor_or_nparts = bool(step.factor_or_nparts)

        stage = self.stages[sid]

        # lazy extent 복원
        self._restore_stage_extents_if_needed(sid, step_idx)

        orig_iter = stage.iters[iid]
        orig_extent = None
        if orig_iter.extent is not None and orig_iter.extent.is_concrete:
            orig_extent = orig_iter.extent.val

        tosplit_extent = SymExpr(orig_extent) if orig_extent is not None else \
                         (orig_iter.extent if orig_iter.extent is not None else None)

        src_sym_parts = []
        for sid_ref in src_step_ids:
            src_sym_parts.append(f"sp_{sid_ref}_{level}")
        fused_sym_expr = SymExpr("*".join(src_sym_parts) if len(src_sym_parts) > 1 else src_sym_parts[0])

        if factor_or_nparts:
            inner_ext = fused_sym_expr
            outer_ext = SymExpr.ceildiv(tosplit_extent, fused_sym_expr) if tosplit_extent else None
            outs = [
                SymIter(f"{orig_iter.name}.0", outer_ext, annotation=0, iter_kind=orig_iter.iter_kind),
                SymIter(f"{orig_iter.name}.1", inner_ext, annotation=0, iter_kind=orig_iter.iter_kind),
            ]
        else:
            outer_ext = fused_sym_expr
            inner_ext = SymExpr.ceildiv(tosplit_extent, fused_sym_expr) if tosplit_extent else None
            outs = [
                SymIter(f"{orig_iter.name}.0", outer_ext, annotation=0, iter_kind=orig_iter.iter_kind),
                SymIter(f"{orig_iter.name}.1", inner_ext, annotation=0, iter_kind=orig_iter.iter_kind),
            ]

        new_iters = stage.iters[:iid] + outs + stage.iters[iid + 1:]
        stage.iters = new_iters

        for other_stage in self.stages:
            if (other_stage.compute_at == CA_ITER and
                other_stage.attach_stage_id == sid and
                other_stage.attach_iter_id >= iid):
                other_stage.attach_iter_id += 1

    # ─────────────────────────────────────────────
    #  StorageAlignStep
    # ─────────────────────────────────────────────
    def _apply_storage_align(self, step):
        self.stages[step.stage_id].storage_offset = step.offset

    # ─────────────────────────────────────────────
    #  ComputeAtStep
    #  extent를 None으로만 설정 (lazy 복원: Fuse/Split에서 필요 시 복원)
    # ─────────────────────────────────────────────
    def _apply_compute_at(self, step):
        sid = step.stage_id
        target_sid = step.target_stage_id
        target_iid = step.target_iter_id
        stage = self.stages[sid]

        # symbolic extent를 저장한 뒤 None으로 리셋
        for iid, it in enumerate(stage.iters):
            if it.extent is not None and not it.extent.is_concrete:
                self._ca_saved_extents[(sid, iid)] = SymExpr(it.extent.val)
            it.extent = None

        stage.compute_at = CA_ITER
        stage.attach_stage_id = target_sid
        stage.attach_iter_id = target_iid

    # ─────────────────────────────────────────────
    #  ComputeInlineStep
    # ─────────────────────────────────────────────
    def _apply_compute_inline(self, step):
        stage = self.stages[step.stage_id]
        stage.compute_at = CA_INLINED
        stage.attach_stage_id = None
        stage.attach_iter_id = None

    # ─────────────────────────────────────────────
    #  ComputeRootStep
    # ─────────────────────────────────────────────
    def _apply_compute_root(self, step):
        sid = step.stage_id
        stage = self.stages[sid]
        for iid, it in enumerate(stage.iters):
            if it.extent is not None and not it.extent.is_concrete:
                self._ca_saved_extents[(sid, iid)] = SymExpr(it.extent.val)
            it.extent = None
        stage.compute_at = CA_ROOT
        stage.attach_stage_id = None
        stage.attach_iter_id = None

    # ─────────────────────────────────────────────
    #  CacheReadStep / CacheWriteStep
    # ─────────────────────────────────────────────
    def _apply_cache_read(self, step, state, step_idx):
        sid = step.stage_id
        added_stage_id = sid + 1

        ps_after = tvm._ffi.get_global_func("auto_scheduler.ReplayStepsPartial")(
            self.compute_dag, state, step_idx + 1)

        new_stage_real = ps_after.stages[added_stage_id]
        new_iters = []
        for it in new_stage_real.iters:
            ext = int(it.range.extent) if it.range is not None else None
            new_iters.append(SymIter(it.name, SymExpr(ext) if ext is not None else None,
                                      annotation=0, iter_kind=0))

        new_sym_stage = SymStage(new_stage_real.op.name, 'compute', new_iters)
        self.stages.insert(added_stage_id, new_sym_stage)

        # CA saved extents의 stage_id 업데이트
        self._shift_ca_saved_extents(added_stage_id)

        # CacheRead consumer 매핑 저장
        # reader_stage_ids는 step 적용 시점의 stage_id; 삽입 후 shift 필요
        reader_ids = [int(x) for x in step.reader_stage_ids]
        if reader_ids:
            consumer_sid = reader_ids[0]
            # CacheRead 삽입으로 consumer가 shift됨 (consumer >= added_stage_id인 경우)
            if consumer_sid >= added_stage_id:
                consumer_sid += 1
            self._cache_read_consumer[added_stage_id] = consumer_sid

            # Stencil 분석: consumer body에서 원본 tensor 접근 패턴 추출
            self._analyze_cache_read_stencil(added_stage_id, sid, consumer_sid)

        for other_stage in self.stages:
            if other_stage.compute_at == CA_ITER:
                if other_stage.attach_stage_id is not None and other_stage.attach_stage_id >= added_stage_id:
                    other_stage.attach_stage_id += 1

        for i in range(added_stage_id + 1, len(self.stages)):
            real_stage = ps_after.stages[i]
            self.stages[i].op_name = real_stage.op.name

    def _analyze_cache_read_stencil(self, cr_stage_id, orig_tensor_sid, consumer_sid):
        """
        Consumer op body에서 원본 tensor의 access pattern을 분석하여 stencil 정보 저장.
        
        pad_temp[nn, yy * stride + ry, xx * stride + rx, rc] 같은 패턴에서
        각 axis의 (stride, spatial_var_name, reduce_var_name) 정보를 추출하고,
        var_name → consumer axis 순서(= split 순서) 매핑을 통해
        {cr_axis_idx: (stride, spatial_split_order, reduce_split_order)} 형태로 저장.
        """
        from tvm import tir
        
        # 원본 tensor name (pad_temp.shared → pad_temp)
        cr_name = self.stages[cr_stage_id].op_name
        orig_name = cr_name.rsplit(".", 1)[0] if "." in cr_name else cr_name
        
        # Consumer op 찾기 (compute_dag.ops에서, CacheRead/CacheWrite 이전의 원본 ops)
        consumer_name = self.stages[consumer_sid].op_name
        # CacheWrite로 인해 이름에 .local이 붙을 수 있음 → 원본 이름으로 검색
        consumer_orig_name = consumer_name.rsplit(".", 1)[0] if "." in consumer_name else consumer_name
        
        consumer_op = None
        for op in self.compute_dag.ops:
            if op.name == consumer_orig_name or op.name == consumer_name:
                if hasattr(op, 'body'):
                    consumer_op = op
                    break
        
        if consumer_op is None or not hasattr(consumer_op, 'body') or len(consumer_op.body) == 0:
            return
        
        # Consumer body에서 원본 tensor의 ProducerLoad 찾기
        producer_loads = self._find_producer_loads(consumer_op.body[0], orig_name)
        if not producer_loads:
            return
        
        pl = producer_loads[0]  # 첫 번째 접근만 사용
        
        # Consumer axis var_name → split order 매핑
        spatial_vars = {str(ax.var.name): i for i, ax in enumerate(consumer_op.axis)}
        reduce_vars = {str(ax.var.name): len(consumer_op.axis) + i 
                      for i, ax in enumerate(consumer_op.reduce_axis)}
        
        stencil_info = {}
        for ax_idx, idx_expr in enumerate(pl.indices):
            result = self._analyze_index_expr(idx_expr, consumer_op.axis, consumer_op.reduce_axis)
            if result is None:
                continue
            stride, sp_name, rd_name = result
            sp_order = spatial_vars.get(sp_name) if sp_name else None
            rd_order = reduce_vars.get(rd_name) if rd_name else None
            stencil_info[ax_idx] = (stride, sp_order, rd_order)
        
        if stencil_info:
            self._cache_read_stencil_info[cr_stage_id] = stencil_info

    @staticmethod
    def _find_producer_loads(expr, tensor_name):
        """TIR expression에서 특정 tensor의 ProducerLoad를 모두 찾아 반환"""
        from tvm import tir
        results = []
        def visit(e):
            if isinstance(e, tir.ProducerLoad):
                if str(e.producer.name) == tensor_name:
                    results.append(e)
                return
            if isinstance(e, tir.Reduce):
                for s in e.source:
                    visit(s)
                return
            for attr in ['a', 'b']:
                if hasattr(e, attr):
                    visit(getattr(e, attr))
            if hasattr(e, 'args') and not isinstance(e, (tir.Var, tir.IntImm)):
                for arg in e.args:
                    visit(arg)
        visit(expr)
        return results

    @staticmethod
    def _analyze_index_expr(expr, spatial_axes, reduce_axes):
        """
        Index expression 분석: stride, spatial_var, reduce_var 반환.
        Returns: (stride, spatial_var_name, reduce_var_name) or None
        """
        from tvm import tir
        spatial_names = {str(v.var.name) for v in spatial_axes}
        reduce_names = {str(v.var.name) for v in reduce_axes}
        
        if isinstance(expr, tir.Var):
            name = str(expr.name)
            if name in spatial_names:
                return (0, name, None)
            elif name in reduce_names:
                return (0, None, name)
            return None
        
        if isinstance(expr, tir.Add):
            a, b = expr.a, expr.b
            def extract_mul_var(e):
                if isinstance(e, tir.Mul):
                    if isinstance(e.a, tir.Var) and isinstance(e.b, tir.IntImm):
                        return (str(e.a.name), int(e.b))
                    if isinstance(e.b, tir.Var) and isinstance(e.a, tir.IntImm):
                        return (str(e.b.name), int(e.a))
                elif isinstance(e, tir.Var):
                    return (str(e.name), 1)
                return None
            
            mul_info = extract_mul_var(a)
            if mul_info and isinstance(b, tir.Var):
                sp_name, stride = mul_info
                rd_name = str(b.name)
                if sp_name in spatial_names and rd_name in reduce_names:
                    return (stride, sp_name, rd_name)
            
            mul_info = extract_mul_var(b)
            if mul_info and isinstance(a, tir.Var):
                sp_name, stride = mul_info
                rd_name = str(a.name)
                if sp_name in spatial_names and rd_name in reduce_names:
                    return (stride, sp_name, rd_name)
            
            if isinstance(a, tir.Var) and isinstance(b, tir.Var):
                a_name, b_name = str(a.name), str(b.name)
                if a_name in spatial_names and b_name in reduce_names:
                    return (1, a_name, b_name)
                if b_name in spatial_names and a_name in reduce_names:
                    return (1, b_name, a_name)
        
        return None

    def _apply_cache_write(self, step, state, step_idx):
        sid = step.stage_id
        ps_after = tvm._ffi.get_global_func("auto_scheduler.ReplayStepsPartial")(
            self.compute_dag, state, step_idx + 1)

        new_stage_real = ps_after.stages[sid]
        new_iters = []
        for it in new_stage_real.iters:
            ext = int(it.range.extent) if it.range is not None else None
            new_iters.append(SymIter(it.name, SymExpr(ext) if ext is not None else None,
                                      annotation=0, iter_kind=0))

        new_sym_stage = SymStage(new_stage_real.op.name, 'compute', new_iters)
        self.stages.insert(sid, new_sym_stage)

        # CA saved extents의 stage_id 업데이트
        self._shift_ca_saved_extents(sid)

        for other_stage in self.stages:
            if other_stage.compute_at == CA_ITER:
                if other_stage.attach_stage_id is not None and other_stage.attach_stage_id >= sid:
                    other_stage.attach_stage_id += 1

        for i in range(len(self.stages)):
            if i < len(ps_after.stages):
                real_stage = ps_after.stages[i]
                self.stages[i].op_name = real_stage.op.name


# ─────────────────────────────────────────────────────────────
#  검증 유틸리티
# ─────────────────────────────────────────────────────────────

def eval_sym_extent(expr, sym_map):
    """SymExpr의 문자열을 sym_map으로 치환하여 eval로 계산"""
    if expr is None:
        return None
    s_val = str(expr)
    if s_val == "None":
        return None
    try:
        return int(s_val)
    except ValueError:
        pass
    evaluated = s_val
    evaluated = evaluated.replace("ceil(", "math.ceil(")
    for sym_name in sorted(sym_map.keys(), key=len, reverse=True):
        if sym_map[sym_name] is not None:
            evaluated = evaluated.replace(sym_name, str(sym_map[sym_name]))
    try:
        return int(eval(evaluated))
    except Exception:
        return f"EVAL_FAIL({s_val}→{evaluated})"


def verify_symbolic_state(task, state, verbose=False):
    """
    SymbolicState를 생성·적용한 뒤 실제 InferBound된 state와 비교.
    Returns: (ok: bool, summary: str)
    """
    sym_state = SymbolicState(task.compute_dag)
    sym_state.apply_steps(state)
    bounded = task.compute_dag.infer_bound_from_state(state)

    stage_mismatch = []
    name_mm = 0
    ann_mm = 0
    ext_mm = 0
    ext_total = 0
    details = []

    n_stages = min(len(bounded.stages), len(sym_state.stages))
    if len(bounded.stages) != len(sym_state.stages):
        details.append(f"Stage count: real={len(bounded.stages)} sym={len(sym_state.stages)}")

    for sid in range(n_stages):
        rs = bounded.stages[sid]
        ss = sym_state.stages[sid]
        if len(rs.iters) != len(ss.iters):
            stage_mismatch.append((sid, len(rs.iters), len(ss.iters)))
            continue
        for iid in range(len(rs.iters)):
            ri, si = rs.iters[iid], ss.iters[iid]
            if str(ri.name) != si.name:
                name_mm += 1
                if verbose and name_mm <= 5:
                    details.append(f"  NAME s{sid}.i{iid}: real='{ri.name}' sym='{si.name}'")
            if int(ri.annotation) != si.annotation:
                ann_mm += 1
                if verbose and ann_mm <= 5:
                    details.append(f"  ANN  s{sid}.i{iid}: real={int(ri.annotation)} sym={si.annotation}")
            re_ext = int(ri.range.extent) if ri.range is not None else None
            se_ext = eval_sym_extent(si.extent, sym_state.sym_map)
            ext_total += 1
            if re_ext != se_ext:
                ext_mm += 1
                if verbose and ext_mm <= 5:
                    details.append(f"  EXT  s{sid}.i{iid}('{si.name}'): real={re_ext} sym={si.extent}→eval={se_ext}")

    ok = (len(stage_mismatch) == 0 and name_mm == 0 and ann_mm == 0 and ext_mm == 0
          and len(bounded.stages) == len(sym_state.stages))
    parts = []
    if stage_mismatch:
        parts.append(f"iter_count_mm={len(stage_mismatch)}")
    if name_mm:
        parts.append(f"name_mm={name_mm}")
    if ann_mm:
        parts.append(f"ann_mm={ann_mm}")
    if ext_mm:
        parts.append(f"ext_mm={ext_mm}/{ext_total}")
    summary = "PASS" if ok else "FAIL(" + ", ".join(parts) + ")"
    if verbose and details:
        summary += "\n" + "\n".join(details)
    return ok, summary, sym_state


# ═══════════════════════════════════════════════════════════════
# 종합 검증: 모든 task × 여러 state에서 SymbolicState 검증
# ═══════════════════════════════════════════════════════════════
import time


total_pass = 0
total_fail = 0
fail_details = []

t0 = time.time()

for task_idx in range(len(tasks)):
    print(f"Testing Task {task_idx+1:2d}/{len(tasks)}...")
    task = tasks[task_idx]
    policy = search_policies[task_idx]

    try:
        init_states = policy.sample_initial_population()
        states = policy.evolutionary_search(init_states, 200)
    except Exception as e:
        print(f"T{task_idx:2d} [{task.desc[:50]:50s}]  ⚠️  sample failed: {e}")
        continue

    task_pass = 0
    task_fail = 0
    task_fail_msgs = []

    for si in range(len(states)):
        state = states[si]
        try:
            ok, summary, sym_state = verify_symbolic_state(task, state, verbose=True)
        except Exception as e:
            ok = False
            summary = f"EXCEPTION: {e}"

        if ok:
            task_pass += 1
        else:
            task_fail += 1
            task_fail_msgs.append(f"    state[{si}]: {summary}")

    total_pass += task_pass
    total_fail += task_fail

    status = "✅" if task_fail == 0 else "❌"
    print(f"T{task_idx:2d} {status} {task_pass}/{len(states)} pass  [{task.desc[:60]}]")
    if task_fail_msgs:
        for msg in task_fail_msgs:
            print(msg)
        fail_details.extend(task_fail_msgs)

elapsed = time.time() - t0

print()
print("=" * 70)
print(f"Total: {total_pass + total_fail} tests, "
      f"{total_pass} passed, {total_fail} failed  "
      f"({elapsed:.1f}s)")
if total_fail == 0:
    print("✅✅✅ ALL TESTS PASSED! ✅✅✅")
else:
    print(f"❌ {total_fail} failures")