"""
symbolic_state_bridge — symbolic-state construction and parameter bookkeeping.
"""
from .sym_types import eval_sym_extent
from .symbolic_state import SymbolicState
from .structural_sketch import build_canonical_state
from .transform_applier import TransformApplier


ANN_VTHREAD = 4
ANN_BLOCK_X = 5
ANN_THREAD_X = 6
ANN_THREAD_Y = 8
ANN_THREAD_Z = 10
THREAD_ANNOS = {ANN_THREAD_X, ANN_THREAD_Y, ANN_THREAD_Z}


class SymParamManager:
    """
    SymbolicState의 sym_map 파라미터를 조회·검증·수정하는 매니저.
    """

    UNROLL_CANDIDATES = [0, 16, 64, 512, 1024]

    def __init__(self, sym_state):
        """SymbolicState를 받아 파라미터 조회·검증용으로 보관한다."""
        self.s = sym_state

    def _build_sp_groups(self):
        """SP 그룹 구성: {step_idx : [sym_name, ...]} (length_idx 순 정렬)."""
        sp_groups = {}
        for name in self.s.sym_map:
            if name.startswith("sp_"):
                parts = name.split("_")
                step_idx = int(parts[1])
                sp_groups.setdefault(step_idx, []).append(name)
        for step_idx in sp_groups:
            sp_groups[step_idx].sort(key=lambda n: int(n.split("_")[2]))
        return sp_groups

    def _build_sp_extents(self, sp_groups):
        """SP {step_idx : extent} 매핑 (SplitStep의 원본 extent)."""
        sp_extents = {}
        if self.s._state is not None:
            steps = self.s._state.transform_steps
            for step_idx in sp_groups:
                if step_idx < len(steps):
                    step = steps[step_idx]
                    tk = step.type_key.split(".")[-1]
                    if tk == "SplitStep" and step.extent is not None:
                        sp_extents[step_idx] = int(step.extent)
        return sp_extents

    @staticmethod
    def _divisors(n):
        """양의 정수 n의 약수 목록을 오름차순으로 반환."""
        if n <= 0:
            return [1]
        divs = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return sorted(divs)


def _step_type_name(step):
    """TVM transform step 객체에서 마지막 type 이름만 추출한다."""
    return step.type_key.split(".")[-1]


def _is_stage_number_changing_step(step):
    """이 step이 이후 stage id를 밀어내는 종류인지 판별한다."""
    return _step_type_name(step) in {"CacheReadStep", "CacheWriteStep", "RfactorStep"}


def _get_target_stage_id_in_state(state, step_idx):
    """중간 step의 stage_id를 최종 state 기준 stage_id로 보정한다."""
    stage_inc = 0
    base_stage_id = int(state.transform_steps[step_idx].stage_id)
    for i in range(step_idx + 1, len(state.transform_steps)):
        step = state.transform_steps[i]
        if _is_stage_number_changing_step(step):
            if int(step.stage_id) <= base_stage_id + stage_inc:
                stage_inc += 1
    return base_stage_id + stage_inc


def _get_task_warp_size(task, default=32):
    """task hardware params에서 warp size를 읽고 없으면 기본값을 쓴다."""
    hardware_params = getattr(task, "hardware_params", None)
    if hardware_params is None:
        return int(default)
    try:
        warp_size = getattr(hardware_params, "warp_size")
    except AttributeError:
        return int(default)
    if warp_size is None:
        return int(default)
    return int(warp_size)


def _compute_total_space_extent(stage):
    """stage의 공간축 extent 곱을 계산해 전체 공간 크기를 구한다."""
    op = getattr(stage, "op", None)
    axes = getattr(op, "axis", None)
    if axes is None:
        return None
    total = 1
    for axis in axes:
        dom = getattr(axis, "dom", None)
        if dom is None or dom.extent is None:
            return None
        total *= int(dom.extent)
    return int(total)


def _mark_mlt_root_thread_meta(meta, history_state, final_state, step_idx, warp_size):
    """MLT root thread 축에 최소 thread extent 완화 여부를 메타로 기록한다."""
    final_stage_id = _get_target_stage_id_in_state(history_state, step_idx)
    if final_stage_id >= len(final_state.stages):
        return

    total_space_extent = _compute_total_space_extent(final_state.stages[final_stage_id])
    relax = (
        total_space_extent is not None and total_space_extent <= int(warp_size) * 2
    )
    final_stage = final_state.stages[final_stage_id]
    for iter_id, it in enumerate(final_stage.iters):
        if int(it.annotation) not in THREAD_ANNOS:
            continue
        meta[(final_stage_id, iter_id)] = {
            "is_mlt_root_thread": True,
            "relax_min_thread_extent": bool(relax),
        }


def _collect_gpu_split_meta(task, state):
    """GPU 전용 split 예외와 thread-extent 메타정보를 transform step에서 수집한다."""
    warp_size = _get_task_warp_size(task)
    exception_split_names = set()
    thread_extent_meta = {}
    steps = state.transform_steps
    final_state = state
    if len(getattr(state, "stages", [])) == 0:
        try:
            final_state = task.compute_dag.infer_bound_from_state(state)
        except Exception:  # pylint: disable=broad-except
            final_state = state

    for i, step in enumerate(steps):
        step_type = _step_type_name(step)

        if step_type == "SplitStep" and i + 1 < len(steps):
            next_step = steps[i + 1]
            if (
                _step_type_name(next_step) == "AnnotationStep"
                and int(next_step.stage_id) == int(step.stage_id)
                and int(next_step.iter_id) == 1
                and int(next_step.annotation) == ANN_THREAD_X
            ):
                exception_split_names.add(f"sp_{i}_0")

        if step_type == "FuseStep" and i + 3 < len(steps):
            sp_step = steps[i + 1]
            block_step = steps[i + 2]
            thread_step = steps[i + 3]
            if (
                _step_type_name(sp_step) == "SplitStep"
                and _step_type_name(block_step) == "AnnotationStep"
                and _step_type_name(thread_step) == "AnnotationStep"
                and int(sp_step.stage_id) == int(step.stage_id)
                and int(block_step.stage_id) == int(step.stage_id)
                and int(thread_step.stage_id) == int(step.stage_id)
                and int(block_step.iter_id) == 0
                and int(block_step.annotation) == ANN_BLOCK_X
                and int(thread_step.iter_id) == 1
                and int(thread_step.annotation) == ANN_THREAD_X
            ):
                exception_split_names.add(f"sp_{i + 1}_0")

        if step_type == "FuseStep" and i + 5 < len(steps):
            block_step = steps[i + 1]
            vthread_fuse = steps[i + 2]
            vthread_step = steps[i + 3]
            thread_fuse = steps[i + 4]
            thread_step = steps[i + 5]
            if (
                _step_type_name(block_step) == "AnnotationStep"
                and _step_type_name(vthread_fuse) == "FuseStep"
                and _step_type_name(vthread_step) == "AnnotationStep"
                and _step_type_name(thread_fuse) == "FuseStep"
                and _step_type_name(thread_step) == "AnnotationStep"
                and int(block_step.stage_id) == int(step.stage_id)
                and int(vthread_fuse.stage_id) == int(step.stage_id)
                and int(vthread_step.stage_id) == int(step.stage_id)
                and int(thread_fuse.stage_id) == int(step.stage_id)
                and int(thread_step.stage_id) == int(step.stage_id)
                and int(block_step.annotation) == ANN_BLOCK_X
                and int(vthread_step.annotation) == ANN_VTHREAD
                and int(thread_step.annotation) == ANN_THREAD_X
            ):
                _mark_mlt_root_thread_meta(
                    thread_extent_meta, state, final_state, i + 5, warp_size
                )

    return {
        "exception_split_names": exception_split_names,
        "thread_extent_meta": thread_extent_meta,
    }


def build_symbolic_state(task, state):
    """task와 state로부터 SymbolicState를 만들고 transform steps를 적용한 뒤 반환한다."""
    if task is None or not hasattr(task, "compute_dag"):
        raise ValueError("build_symbolic_state now requires a task, not only compute_dag")

    gpu_meta = _collect_gpu_split_meta(task, state)
    sym = SymbolicState(task.compute_dag)
    sym._exception_split_names = set(gpu_meta["exception_split_names"])
    sym._thread_extent_meta = {
        key: dict(value) for key, value in gpu_meta["thread_extent_meta"].items()
    }
    applier = TransformApplier(sym)
    applier.apply_steps(state)
    return sym


def verify_symbolic_state(task, state, sym_state, verbose=False):
    """Compare sym_state extents against TVM infer_bound on the concrete state.

    Substitutes the state's concrete sp_*/ur_* values into sym_state.sym_map,
    evaluates each iter's symbolic extent, and compares with the real extent
    from task.compute_dag.infer_bound_from_state(state).

    Returns (ok, summary). Restores the original sym_map before returning.
    """
    saved_sym_map = dict(sym_state.sym_map)
    try:
        for step_idx, step in enumerate(state.transform_steps):
            tk = step.type_key.split(".")[-1]
            if tk == "SplitStep":
                for li, length in enumerate(step.lengths):
                    sym_name = f"sp_{step_idx}_{li}"
                    if sym_name in sym_state.sym_map:
                        sym_state.sym_map[sym_name] = (
                            int(length) if length is not None else None
                        )
            elif tk == "PragmaStep":
                pragma_type = str(step.pragma_type)
                if "auto_unroll_max_step$" in pragma_type:
                    val = int(pragma_type.split("$")[1])
                    sym_name = f"ur_{step_idx}"
                    if sym_name in sym_state.sym_map:
                        sym_state.sym_map[sym_name] = val

        bounded = task.compute_dag.infer_bound_from_state(state)

        stage_mismatch = []
        name_mm = 0
        ann_mm = 0
        ext_mm = 0
        ext_tight = 0
        ext_total = 0
        details = []

        n_stages = min(len(bounded.stages), len(sym_state.stages))
        if len(bounded.stages) != len(sym_state.stages):
            details.append(
                f"Stage count: real={len(bounded.stages)} sym={len(sym_state.stages)}"
            )

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
                        details.append(
                            f"  NAME s{sid}.i{iid}: real='{ri.name}' sym='{si.name}'"
                        )
                if int(ri.annotation) != si.annotation:
                    ann_mm += 1
                    if verbose and ann_mm <= 5:
                        details.append(
                            f"  ANN  s{sid}.i{iid}: real={int(ri.annotation)} sym={si.annotation}"
                        )
                re_ext = int(ri.range.extent) if ri.range is not None else None
                se_ext = eval_sym_extent(si.extent, sym_state.sym_map)
                ext_total += 1
                if re_ext is not None and isinstance(se_ext, int):
                    if se_ext > re_ext:
                        ext_mm += 1
                        if verbose and ext_mm <= 5:
                            details.append(
                                f"  EXT> s{sid}.i{iid}('{si.name}'): real={re_ext} "
                                f"sym={si.extent}→eval={se_ext}"
                            )
                    elif se_ext < re_ext:
                        ext_tight += 1
                        if verbose and ext_tight <= 5:
                            details.append(
                                f"  EXT< s{sid}.i{iid}('{si.name}'): real={re_ext} "
                                f"sym={si.extent}→eval={se_ext} (tight)"
                            )
                elif re_ext != se_ext:
                    ext_mm += 1
                    if verbose and ext_mm <= 5:
                        details.append(
                            f"  EXT  s{sid}.i{iid}('{si.name}'): real={re_ext} "
                            f"sym={si.extent}→eval={se_ext}"
                        )
    finally:
        sym_state.sym_map.clear()
        sym_state.sym_map.update(saved_sym_map)

    ok = (
        len(stage_mismatch) == 0
        and name_mm == 0
        and ann_mm == 0
        and ext_mm == 0
        and len(bounded.stages) == len(sym_state.stages)
    )
    parts = []
    if stage_mismatch:
        parts.append(f"iter_count_mm={len(stage_mismatch)}")
    if name_mm:
        parts.append(f"name_mm={name_mm}")
    if ann_mm:
        parts.append(f"ann_mm={ann_mm}")
    if ext_mm:
        parts.append(f"ext_mm={ext_mm}/{ext_total}")
    if ext_tight:
        parts.append(f"ext_tight={ext_tight}/{ext_total}")
    summary = "PASS" if ok else "FAIL(" + ", ".join(parts) + ")"
    if ok and ext_tight:
        summary = "PASS(" + ", ".join(parts) + ")"
    if verbose and details:
        summary += "\n" + "\n".join(details)
    return ok, summary


# ------------------------------------------------------------------
# Deprecated
# ------------------------------------------------------------------


# def verify_symbolic_state(task, state, sym_state, verbose=False):
#     """state의 구체 파라미터를 sym_map에 넣고 InferBound 결과와 심볼 상태를 비교해 (ok, summary)를 반환한다."""
#     saved_sym_map = dict(sym_state.sym_map)
#     for step_idx, step in enumerate(state.transform_steps):
#         tk = step.type_key.split(".")[-1]
#         if tk == "SplitStep":
#             for li, length in enumerate(step.lengths):
#                 sym_name = f"sp_{step_idx}_{li}"
#                 if sym_name in sym_state.sym_map:
#                     sym_state.sym_map[sym_name] = int(length) if length is not None else None
#         elif tk == "PragmaStep":
#             pragma_type = str(step.pragma_type)
#             if "auto_unroll_max_step$" in pragma_type:
#                 val = int(pragma_type.split("$")[1])
#                 sym_name = f"ur_{step_idx}"
#                 if sym_name in sym_state.sym_map:
#                     sym_state.sym_map[sym_name] = val

#     bounded = task.compute_dag.infer_bound_from_state(state)

#     stage_mismatch = []
#     name_mm = 0
#     ann_mm = 0
#     ext_mm = 0
#     ext_tight = 0
#     ext_total = 0
#     details = []

#     n_stages = min(len(bounded.stages), len(sym_state.stages))
#     if len(bounded.stages) != len(sym_state.stages):
#         details.append(f"Stage count: real={len(bounded.stages)} sym={len(sym_state.stages)}")

#     for sid in range(n_stages):
#         rs = bounded.stages[sid]
#         ss = sym_state.stages[sid]
#         if len(rs.iters) != len(ss.iters):
#             stage_mismatch.append((sid, len(rs.iters), len(ss.iters)))
#             continue
#         for iid in range(len(rs.iters)):
#             ri, si = rs.iters[iid], ss.iters[iid]
#             if str(ri.name) != si.name:
#                 name_mm += 1
#                 if verbose and name_mm <= 5:
#                     details.append(f"  NAME s{sid}.i{iid}: real='{ri.name}' sym='{si.name}'")
#             if int(ri.annotation) != si.annotation:
#                 ann_mm += 1
#                 if verbose and ann_mm <= 5:
#                     details.append(f"  ANN  s{sid}.i{iid}: real={int(ri.annotation)} sym={si.annotation}")
#             re_ext = int(ri.range.extent) if ri.range is not None else None
#             se_ext = eval_sym_extent(si.extent, sym_state.sym_map)
#             ext_total += 1
#             if re_ext is not None and se_ext is not None:
#                 if se_ext > re_ext:
#                     ext_mm += 1
#                     if verbose and ext_mm <= 5:
#                         details.append(f"  EXT> s{sid}.i{iid}('{si.name}'): real={re_ext} sym={si.extent}→eval={se_ext}")
#                 elif se_ext < re_ext:
#                     ext_tight += 1
#                     if verbose and ext_tight <= 5:
#                         details.append(f"  EXT< s{sid}.i{iid}('{si.name}'): real={re_ext} sym={si.extent}→eval={se_ext} (tight)")
#             elif re_ext != se_ext:
#                 ext_mm += 1
#                 if verbose and ext_mm <= 5:
#                     details.append(f"  EXT  s{sid}.i{iid}('{si.name}'): real={re_ext} sym={si.extent}→eval={se_ext}")

#     sym_state.sym_map = saved_sym_map

#     ok = (len(stage_mismatch) == 0 and name_mm == 0 and ann_mm == 0 and ext_mm == 0
#           and len(bounded.stages) == len(sym_state.stages))
#     parts = []
#     if stage_mismatch:
#         parts.append(f"iter_count_mm={len(stage_mismatch)}")
#     if name_mm:
#         parts.append(f"name_mm={name_mm}")
#     if ann_mm:
#         parts.append(f"ann_mm={ann_mm}")
#     if ext_mm:
#         parts.append(f"ext_mm={ext_mm}/{ext_total}")
#     if ext_tight:
#         parts.append(f"ext_tight={ext_tight}/{ext_total}")
#     summary = "PASS" if ok else "FAIL(" + ", ".join(parts) + ")"
#     if ok and ext_tight:
#         summary = "PASS(" + ", ".join(parts) + ")"
#     if verbose and details:
#         summary += "\n" + "\n".join(details)
#     return ok, summary
