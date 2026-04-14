"""Load record line 377 from the JSON and verify its SP 2 0 96 value against ground truth."""
import sys
sys.path.insert(0, "/root/work/tvm-ansor/gallery")

import tvm
from tvm import auto_scheduler
from common import load_and_register_tasks

import os
JSON_PATH = os.environ.get("JSON_PATH",
    "/root/work/tvm-ansor/gallery/415_([e7c984cba151d5c7c1e081f0b1910087,[1,112,112,32],[3,3,32,1],[1,1,1,32],[1,112,112,32]],cuda).json")
TARGET_EXT = int(os.environ.get("TARGET_EXT", "96"))
TARGET_LEN = int(os.environ.get("TARGET_LEN", "1"))

def main():
    tasks = load_and_register_tasks()
    task = tasks[415]

    reader = auto_scheduler.RecordReader(JSON_PATH)
    records = list(reader)
    print(f"[diag] loaded {len(records)} records")

    # find a record whose step 31 matches SP 2 0 96 [1] 1
    target = None
    for ri, (minp, _mres) in enumerate(records):
        steps = minp.state.transform_steps
        for step in steps:
            if (hasattr(step, "extent") and hasattr(step, "lengths")
                    and hasattr(step, "inner_to_outer")
                    and int(step.stage_id) == 2 and int(step.iter_id) == 0):
                try:
                    ev = int(step.extent)
                except Exception:
                    ev = None
                lengths = [int(l) if l is not None else None for l in step.lengths]
                if ev == TARGET_EXT and lengths == [TARGET_LEN]:
                    target = (ri, minp)
                    break
        if target is not None:
            break

    if target is None:
        print("[diag] no matching record"); return
    ri, minp = target
    print(f"[diag] using record #{ri}")

    state_obj = minp.state.state_object if hasattr(minp.state, "state_object") else minp.state

    replay_fn = tvm._ffi.get_global_func("auto_scheduler.ReplayStepsPartial")

    # Find the index of the stage 2 iter 0 SP step
    sp_idx = None
    for i, step in enumerate(state_obj.transform_steps):
        if (hasattr(step, "extent") and hasattr(step, "lengths")
                and hasattr(step, "inner_to_outer")
                and int(step.stage_id) == 2 and int(step.iter_id) == 0):
            sp_idx = i; break
    print(f"[diag] SP 2 0 is at step index {sp_idx}")

    # Prefix replay + InferBound (ground truth for live extent)
    prefix = replay_fn(task.compute_dag, state_obj, sp_idx)
    probed = task.compute_dag.infer_bound_from_state(prefix)
    probed_obj = probed.state_object if hasattr(probed, "state_object") else probed
    it = probed_obj.stages[2].iters[0]
    r = it.range
    live = int(r.extent) if r is not None else None
    print(f"[diag] JSON stored extent = {TARGET_EXT}")
    print(f"[diag] live extent via prefix-replay InferBound = {live}")

    # Full pretty-print
    from tvm.auto_scheduler.loop_state import State as PyState
    state_py = PyState(state_obj, task.compute_dag)
    print("\n=== full schedule ===")
    print(str(state_py))


if __name__ == "__main__":
    main()
