"""
tvm_verify — TVM API를 통한 스케줄 유효성 검증 유틸리티.

lower_with_gpu_passes, verify_gpu_module, params_to_state 등.
"""
import json
import tvm
from tvm import tir
from tvm.auto_scheduler.measure_record import load_record_from_string, dump_record_to_string


# ─── ScheduleToModule + GPU passes ───
_s2m = tvm.get_global_func("driver.schedule_to_module")

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


def verify_gpu_module(mod, constraints=None):
    """Lowered IRModule에 대해 tir.analysis.verify_gpu_code로 GPU 제약 확인."""
    if constraints is None:
        constraints = GPU_VERIFY_CONSTRAINTS
    verify_fn = tvm.get_global_func("tir.analysis.verify_gpu_code")
    for _, f in mod.functions.items():
        if isinstance(f, tvm.tir.PrimFunc):
            if not verify_fn(f, constraints):
                return False
    return True


def params_to_state(task, base_inp, base_res, params):
    """ScheduleGenerator가 생성한 params를 적용한 새 auto_scheduler State를 반환."""
    record_str = dump_record_to_string(base_inp, base_res)
    record = json.loads(record_str)
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

    patched_str = json.dumps(record)
    new_inp, _ = load_record_from_string(patched_str)
    return new_inp.state
