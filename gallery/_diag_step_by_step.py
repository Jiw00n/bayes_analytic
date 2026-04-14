"""Step-by-step verification of Ansor sketch + init population behavior.

Stages:
  1. GenerateSketches only — verify deterministic, no coop-fetch SP, root-tile SP have NullOpt lengths
  2. + init_fill_tile_size (target_rule=1) — verify lengths get filled, stored extents unchanged, still no coop-fetch SP
  3. + init_thread_bind (target_rule=12) — verify coop-fetch SP appears, stored extents vary per sample
"""
import sys, os
sys.path.insert(0, "/root/work/tvm-ansor/gallery")

import tvm
from tvm import auto_scheduler
from common import load_and_register_tasks


def split_step_summary(state_obj):
    """Return list of (idx, stage_id, iter_id, extent, lengths) for SplitSteps."""
    out = []
    for i, step in enumerate(state_obj.transform_steps):
        if not (hasattr(step, "extent") and hasattr(step, "lengths")
                and hasattr(step, "inner_to_outer")):
            continue
        try:
            ext = int(step.extent) if step.extent is not None else None
        except Exception:
            ext = str(step.extent)
        lengths = []
        for l in step.lengths:
            try:
                lengths.append(int(l) if l is not None else None)
            except Exception:
                lengths.append(str(l))
        out.append((i, int(step.stage_id), int(step.iter_id), ext, lengths))
    return out


def has_fuse_at(state_obj, stage_id):
    for step in state_obj.transform_steps:
        if step.__class__.__name__ == "FuseStep":
            pass
        # Use type-key-style detection:
        cls_name = type(step).__name__
    # Check via attribute
    for step in state_obj.transform_steps:
        if hasattr(step, "fused_ids"):
            try:
                if int(step.stage_id) == stage_id:
                    return True
            except Exception:
                pass
    return False


def main():
    tasks = load_and_register_tasks()
    task = tasks[415]

    print("=" * 70)
    print("STEP 1 — GenerateSketches (deterministic)")
    print("=" * 70)

    policy = auto_scheduler.SketchPolicy(
        task, auto_scheduler.XGBModel(),
        params={'evolutionary_search_num_iters': 1,
                'evolutionary_search_population': 8},
        verbose=0,
    )
    gen_fn = tvm._ffi.get_global_func("auto_scheduler.SketchPolicyGenerateSketches")
    sketches = gen_fn(policy)
    print(f"\n[step1] generated {len(sketches)} sketches")

    for si, sk in enumerate(sketches):
        sk_obj = sk.state_object if hasattr(sk, "state_object") else sk
        print(f"\n  --- sketch[{si}] SplitSteps ---")
        for entry in split_step_summary(sk_obj):
            print(f"    step[{entry[0]:3d}] stage={entry[1]} iter={entry[2]} "
                  f"extent={entry[3]} lengths={entry[4]}")
        # also count fuse steps for reference
        n_fuse = sum(1 for s in sk_obj.transform_steps if hasattr(s, "fused_ids"))
        print(f"    (fuse steps so far: {n_fuse})")

    print("\n" + "=" * 70)
    print("STEP 2 — + init_fill_tile_size only (target_rule=1)")
    print("=" * 70)

    rule_tgt_fn = tvm._ffi.get_global_func("auto_scheduler.SketchPolicySampleInitPop_Rule_tgt")
    print("\n[step2] sampling 3 states with target_rule=1 (fill_tile_size only)")
    for sample_i in range(3):
        states = rule_tgt_fn(policy, 1)
        s_obj = states[0].state_object if hasattr(states[0], "state_object") else states[0]
        print(f"\n  --- sample[{sample_i}] SplitSteps ---")
        for entry in split_step_summary(s_obj):
            print(f"    step[{entry[0]:3d}] stage={entry[1]} iter={entry[2]} "
                  f"extent={entry[3]} lengths={entry[4]}")

    print("\n" + "=" * 70)
    print("STEP 3 — + init_thread_bind (target_rule=12)")
    print("=" * 70)

    print("\n[step3] sampling 3 states with target_rule=12 (fill_tile_size + thread_bind)")
    fix_fn = tvm._ffi.get_global_func("auto_scheduler.FixSplitExtentsInState")
    for sample_i in range(3):
        states = rule_tgt_fn(policy, 12)
        s_obj = states[0].state_object if hasattr(states[0], "state_object") else states[0]
        print(f"\n  --- sample[{sample_i}] BEFORE FFI (post thread_bind) ---")
        before = split_step_summary(s_obj)
        for entry in before:
            print(f"    step[{entry[0]:3d}] stage={entry[1]} iter={entry[2]} "
                  f"extent={entry[3]} lengths={entry[4]}")

        fixed = fix_fn(task.compute_dag, s_obj)
        print(f"  --- sample[{sample_i}] AFTER  FixSplitExtentsInState ---")
        after = split_step_summary(fixed)
        for entry in after:
            print(f"    step[{entry[0]:3d}] stage={entry[1]} iter={entry[2]} "
                  f"extent={entry[3]} lengths={entry[4]}")

        # Show only diffs
        diffs = []
        for b, a in zip(before, after):
            if b[3] != a[3]:
                diffs.append((b, a))
        if diffs:
            print(f"  --- sample[{sample_i}] DIFFS (extent before -> after) ---")
            for b, a in diffs:
                print(f"    step[{b[0]:3d}] stage={b[1]} iter={b[2]}: "
                      f"{b[3]} -> {a[3]}  (lengths={b[4]})")


if __name__ == "__main__":
    main()
