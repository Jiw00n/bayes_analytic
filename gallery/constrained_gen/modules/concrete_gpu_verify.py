"""Concrete TVM lowering, state reconstruction, and GPU verification utilities."""
import hashlib
import json
import tvm
from tvm import auto_scheduler, tir
from tvm.auto_scheduler.measure_record import load_record_from_string, dump_record_to_string


# ─── ScheduleToModule + GPU passes ───
_s2m = tvm.get_global_func("driver.schedule_to_module")
_verify_gpu_code = tvm.get_global_func("tir.analysis.verify_gpu_code")
_verify_gpu_code_errors = tvm.get_global_func("tir.analysis.verify_gpu_code_errors")

GPU_PASSES = tvm.transform.Sequential(
    [
        tir.transform.InjectPrefetch(),
        tir.transform.StorageFlatten(64, False),
        tir.transform.NarrowDataType(32),
        tir.transform.Simplify(),
        tir.transform.VectorizeLoop(True),
        tir.transform.InjectVirtualThread(),
        tir.transform.StorageRewrite(),
        tir.transform.Simplify(),
    ]
)

GPU_VERIFY_CONSTRAINTS = {
    "max_shared_memory_per_block": 49152,
    "max_local_memory_per_block": 2**31 - 1,
    "max_threads_per_block": 1024,
    "max_thread_x": 1024,
    "max_thread_y": 1024,
    "max_thread_z": 64,
    "max_vthread": 8,
    "max_vector_bytes": 16,
}


def lower_with_gpu_passes(task, state):
    """State → TE schedule → IRModule → GPU pass pipeline 적용."""
    sch, tensors = task.compute_dag.apply_steps_from_state(state)
    mod = _s2m(sch, tensors, "main", {})
    return GPU_PASSES(mod)


def verify_gpu_func_errors(func, constraints=None):
    """PrimFunc 하나에 대해 verify_gpu_code의 상세 violation 문자열을 반환."""
    if constraints is None:
        constraints = GPU_VERIFY_CONSTRAINTS
    return [str(err) for err in _verify_gpu_code_errors(func, constraints)]


def verify_gpu_module_errors(mod, constraints=None):
    """Lowered IRModule 전체에 대해 verify_gpu_code violation 문자열을 반환."""
    if constraints is None:
        constraints = GPU_VERIFY_CONSTRAINTS
    errors = []
    for gv, func in mod.functions.items():
        if not isinstance(func, tvm.tir.PrimFunc):
            continue
        func_errors = verify_gpu_func_errors(func, constraints)
        if len(func_errors) <= 1:
            errors.extend(func_errors)
            continue
        prefix = f"{gv.name_hint}: "
        errors.extend(
            [msg if msg.startswith(prefix) else f"{prefix}{msg}" for msg in func_errors]
        )
    return errors


def _patch_record_steps(record, params, split_extents=None):
    """Measure record JSON의 step payload에 params를 반영한다."""
    steps = record["i"][1][1]

    for name, val in params.items():
        if name.startswith("sp_"):
            parts = name.split("_")
            step_idx = int(parts[1])
            length_idx = int(parts[2])
            s = steps[step_idx]
            if s[0] == "SP" and length_idx < len(s[4]):
                s[4][length_idx] = int(val)
        elif name.startswith("ur_"):
            parts = name.split("_")
            step_idx = int(parts[1])
            s = steps[step_idx]
            if s[0] == "PR":
                s[3] = f"auto_unroll_max_step${int(val)}"

    if split_extents:
        for step_idx, extent in split_extents.items():
            if 0 <= int(step_idx) < len(steps):
                s = steps[int(step_idx)]
                if s[0] == "SP":
                    s[3] = int(extent)


def _params_to_state_from_measure_record(base_inp, base_res, params, split_extents=None):
    """기존 MeasureInput/MeasureResult record를 patch해 새 State를 복원한다."""
    record_str = dump_record_to_string(base_inp, base_res)
    record = json.loads(record_str)
    _patch_record_steps(record, params, split_extents=split_extents)
    patched_str = json.dumps(record)
    new_inp, _ = load_record_from_string(patched_str)
    return new_inp.state


def params_to_state_from_state(task, base_state, params, split_extents=None):
    """Raw concrete State에 params를 적용한 새 auto_scheduler State를 반환."""
    base_inp = auto_scheduler.MeasureInput(task, base_state)
    base_res = auto_scheduler.MeasureResult(
        [tvm.tir.FloatImm("float32", 1.0)],
        0,
        "",
        0.0,
        0,
    )
    return _params_to_state_from_measure_record(
        base_inp,
        base_res,
        params,
        split_extents=split_extents,
    )


def build_state_record_steps_payload(task, state):
    """Concrete State를 measure-record step payload JSON 문자열로 정규화해 반환한다."""
    measure_input = auto_scheduler.MeasureInput(task, state)
    measure_result = auto_scheduler.MeasureResult(
        [tvm.tir.FloatImm("float32", 1.0)],
        0,
        "",
        0.0,
        0,
    )
    record = json.loads(dump_record_to_string(measure_input, measure_result))
    return json.dumps(record["i"][1][1], sort_keys=True, separators=(",", ":"))


def concrete_state_fingerprint(task, state):
    """Concrete State의 transform-step payload를 기반으로 안정 해시를 반환한다."""
    payload = build_state_record_steps_payload(task, state)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------
# Deprecated
# ------------------------------------------------------------------


# def verify_gpu_module(mod, constraints=None):
#     """Lowered IRModule에 대해 tir.analysis.verify_gpu_code로 GPU 제약 확인."""
#     if constraints is None:
#         constraints = GPU_VERIFY_CONSTRAINTS
#     for _, f in mod.functions.items():
#         if isinstance(f, tvm.tir.PrimFunc):
#             if not _verify_gpu_code(f, constraints):
#                 return False
#     return True


# def params_to_state_from_record(task, base_inp, base_res, params, split_extents=None):
#     """Record 기반 sketch에 params를 적용한 새 auto_scheduler State를 반환."""
#     del task
#     return _params_to_state_from_measure_record(
#         base_inp,
#         base_res,
#         params,
#         split_extents=split_extents,
#     )
