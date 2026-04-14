"""Diagnostic: compare raw sketch-time SplitStep.extent vs FixSplitExtentsInState output."""
import sys
sys.path.insert(0, "/root/work/tvm-ansor/gallery")

import tvm
from tvm import auto_scheduler
from common import load_and_register_tasks

_fix_split_extents = tvm._ffi.get_global_func("auto_scheduler.FixSplitExtentsInState")


def splitsteps(state_obj, label):
    print(f"\n=== {label} ===")
    for i, step in enumerate(state_obj.transform_steps):
        tk = getattr(step, "_type_key", None) or step.type_key if hasattr(step, "type_key") else None
        # fall back: try attribute-based detection
        has_extent = hasattr(step, "extent")
        has_lengths = hasattr(step, "lengths")
        has_inner_to_outer = hasattr(step, "inner_to_outer")
        if not (has_extent and has_lengths and has_inner_to_outer):
            continue
        ext = getattr(step, "extent", None)
        try:
            ext_val = int(ext) if ext is not None else None
        except Exception:
            ext_val = str(ext)
        lengths = [int(l) if l is not None else None for l in step.lengths]
        print(f"  step[{i:3d}] stage={int(step.stage_id)} iter={int(step.iter_id)} "
              f"extent={ext_val} lengths={lengths}")


def main():
    tasks = load_and_register_tasks()
    task = tasks[415]

    policy = auto_scheduler.SketchPolicy(
        task, auto_scheduler.XGBModel(),
        params={'evolutionary_search_num_iters': 1,
                'evolutionary_search_population': 256},
        verbose=0,
    )
    states = policy.sample_initial_population()
    print(f"[diag] got {len(states)} init population states")

    # Find a sample whose FFI output gives thousand-scale extent for stage 2 iter 0
    replay_fn = tvm._ffi.get_global_func("auto_scheduler.ReplayStepsPartial")
    picked = None
    for idx, s_obj in enumerate(states):
        fx = _fix_split_extents(task.compute_dag, s_obj)
        for step in fx.transform_steps:
            if (hasattr(step, "extent") and hasattr(step, "lengths")
                    and hasattr(step, "inner_to_outer")
                    and int(step.stage_id) == 2 and int(step.iter_id) == 0):
                try:
                    ev = int(step.extent)
                except Exception:
                    ev = None
                if ev is not None and ev >= 1000:
                    picked = (idx, s_obj, fx, ev)
                    break
        if picked is not None:
            break

    if picked is None:
        print("\n[diag] no thousand-scale FFI extent found in this batch; "
              "trying larger pop may help. Using states[0] anyway.")
        picked = (0, states[0], _fix_split_extents(task.compute_dag, states[0]), None)

    idx, state_obj, fixed, ffi_ext = picked
    print(f"\n[diag] picked sample index {idx}, ffi stage=2 iter=0 extent={ffi_ext}")

    splitsteps(state_obj, "raw init population (sketch-time frozen)")
    splitsteps(fixed, "after FixSplitExtentsInState")

    # GROUND-TRUTH CHECK 1: replay step prefix (0..30) and InferBound at that point
    from tvm.auto_scheduler.loop_state import State as PyState
    replay_fn = tvm._ffi.get_global_func("auto_scheduler.ReplayStepsPartial")
    prefix_state = replay_fn(task.compute_dag, state_obj, 31)  # apply steps 0..30
    probed = task.compute_dag.infer_bound_from_state(prefix_state)
    probed_obj = probed.state_object if hasattr(probed, "state_object") else probed
    it = probed_obj.stages[2].iters[0]
    r = it.range
    print("\n=== GROUND TRUTH via prefix-replay + InferBound ===")
    print(f"  stage=2 iter=0 name={it.name} range.extent="
          f"{int(r.extent) if r is not None else None}")

    # GROUND-TRUTH CHECK 2: lower the full init-population state and print loop structure
    print("\n=== full schedule pretty print (shows real loop extents) ===")
    state_py = PyState(state_obj, task.compute_dag)
    print(str(state_py))


if __name__ == "__main__":
    main()
