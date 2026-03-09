"""
param_manager — SymParamManager (파라미터 조회/검증), build_symbolic_state, verify_symbolic_state.
"""
from .sym_types import SymExpr, eval_sym_extent
from .symbolic_state import SymbolicState
from .transform_applier import TransformApplier


class SymParamManager:
    """
    SymbolicState의 sym_map 파라미터를 조회·검증·수정하는 매니저.
    """

    UNROLL_CANDIDATES = [0, 16, 64, 512, 1024]

    def __init__(self, sym_state):
        self.s = sym_state

    def _build_sp_groups(self):
        """SP 그룹 구성: step_idx → [sym_name, ...] (length_idx 순 정렬)."""
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
        """SP step_idx → extent 매핑 (SplitStep의 원본 extent)."""
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


def build_symbolic_state(compute_dag, state):
    """compute_dag + state로부터 SymbolicState를 생성하고 steps를 적용.
    Returns: SymbolicState (apply 완료)
    """
    sym = SymbolicState(compute_dag)
    applier = TransformApplier(sym)
    applier.apply_steps(state)
    return sym


def verify_symbolic_state(task, state, sym_state, verbose=False):
    """
    다른 state의 구체적 파라미터를 sym_map에 반영한 뒤,
    InferBound 결과와 SymbolicState 구조를 비교한다.

    Returns: (ok: bool, summary: str)
    """
    saved_sym_map = dict(sym_state.sym_map)
    for step_idx, step in enumerate(state.transform_steps):
        tk = step.type_key.split(".")[-1]
        if tk == "SplitStep":
            for li, length in enumerate(step.lengths):
                sym_name = f"sp_{step_idx}_{li}"
                if sym_name in sym_state.sym_map:
                    sym_state.sym_map[sym_name] = int(length) if length is not None else None
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
    ext_mm = 0       # sym > real (under-estimate, real error)
    ext_tight = 0    # sym < real (tighter bound, OK — e.g. min(split_factor, orig_extent))
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
            if re_ext is not None and se_ext is not None:
                if se_ext > re_ext:
                    # sym overestimates — real error
                    ext_mm += 1
                    if verbose and ext_mm <= 5:
                        details.append(f"  EXT> s{sid}.i{iid}('{si.name}'): real={re_ext} sym={si.extent}→eval={se_ext}")
                elif se_ext < re_ext:
                    # sym is tighter (e.g. min(split_factor, orig_extent) < split_factor)
                    # This is acceptable — TVM InferBound doesn't clamp split inner extents.
                    ext_tight += 1
                    if verbose and ext_tight <= 5:
                        details.append(f"  EXT< s{sid}.i{iid}('{si.name}'): real={re_ext} sym={si.extent}→eval={se_ext} (tight)")
            elif re_ext != se_ext:
                ext_mm += 1
                if verbose and ext_mm <= 5:
                    details.append(f"  EXT  s{sid}.i{iid}('{si.name}'): real={re_ext} sym={si.extent}→eval={se_ext}")

    # sym_map 원복
    sym_state.sym_map = saved_sym_map

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
    if ext_tight:
        parts.append(f"ext_tight={ext_tight}/{ext_total}")
    summary = "PASS" if ok else "FAIL(" + ", ".join(parts) + ")"
    if ok and ext_tight:
        summary = "PASS(" + ", ".join(parts) + ")"
    if verbose and details:
        summary += "\n" + "\n".join(details)
    return ok, summary
