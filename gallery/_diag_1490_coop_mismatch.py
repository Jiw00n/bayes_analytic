"""Check task 1490 measured records for coop-fetch stored/live extent mismatch.

For each record:
  - Find coop-fetch SplitSteps (lengths has 1 element, after FU step)
  - Compare stored extent vs live extent (prefix-replay + InferBound)
  - Check if prod(lengths) divides live extent
  - Report mismatches
"""
import sys, os, json
sys.path.insert(0, "/root/work/tvm-ansor/gallery")

import tvm
from tvm import auto_scheduler
from common import load_and_register_tasks

JSON_PATH = "/root/work/tvm-ansor/gallery/constrained_gen/data/measured_ansor/1490_([3eda1939e30b947e921f5e1814346365,[1,56,56,128],[6,6,32,128],[1,56,56,32]],cuda).json"

def main():
    tasks = load_and_register_tasks()
    task = tasks[1490]

    reader = auto_scheduler.RecordReader(JSON_PATH)
    records = list(reader)
    print(f"[diag] loaded {len(records)} records for task 1490")

    replay_fn = tvm._ffi.get_global_func("auto_scheduler.ReplayStepsPartial")

    n_coop = 0
    n_stored_eq_live = 0
    n_stored_ne_live = 0
    n_len_divides_live = 0
    n_len_not_divides_live = 0
    bad_examples = []

    for ri, (minp, _mres) in enumerate(records):
        state_obj = (minp.state.state_object
                     if hasattr(minp.state, "state_object") else minp.state)
        steps = list(state_obj.transform_steps)

        for si, step in enumerate(steps):
            if not (hasattr(step, "extent") and hasattr(step, "lengths")
                    and hasattr(step, "inner_to_outer")):
                continue
            lengths = []
            for l in step.lengths:
                try:
                    lengths.append(int(l) if l is not None else None)
                except Exception:
                    lengths.append(None)
            if None in lengths:
                continue
            # coop-fetch: single-element lengths, preceded by FU step
            if len(lengths) != 1:
                continue
            # Verify preceded by FU on same stage (optional heuristic)
            stage_id = int(step.stage_id)

            try:
                stored = int(step.extent) if step.extent is not None else None
            except Exception:
                stored = None
            if stored is None:
                continue

            n_coop += 1
            length_val = lengths[0]

            # Get live extent via prefix-replay + InferBound
            try:
                prefix = replay_fn(task.compute_dag, state_obj, si)
                probed = task.compute_dag.infer_bound_from_state(prefix)
                probed_obj = (probed.state_object
                              if hasattr(probed, "state_object") else probed)
                it = probed_obj.stages[stage_id].iters[int(step.iter_id)]
                r = it.range
                live = int(r.extent) if r is not None else None
            except Exception as e:
                live = None

            if live is None:
                continue

            if live == stored:
                n_stored_eq_live += 1
            else:
                n_stored_ne_live += 1

            if length_val > 0 and live % length_val == 0:
                n_len_divides_live += 1
            else:
                n_len_not_divides_live += 1
                bad_examples.append({
                    "record": ri,
                    "step_idx": si,
                    "stage": stage_id,
                    "iter": int(step.iter_id),
                    "stored": stored,
                    "live": live,
                    "length": length_val,
                    "live_mod_len": live % length_val if length_val > 0 else "N/A",
                })

        if ri % 100 == 0:
            print(f"  ... processed {ri+1}/{len(records)} records, "
                  f"coop={n_coop}, mismatch={n_len_not_divides_live}")

    print(f"\n=== RESULTS (task 1490, {len(records)} records) ===")
    print(f"  coop-fetch splits total     : {n_coop}")
    print(f"  stored == live               : {n_stored_eq_live}/{n_coop}")
    print(f"  stored != live               : {n_stored_ne_live}/{n_coop}")
    print(f"  length divides live           : {n_len_divides_live}/{n_coop}")
    print(f"  length NOT divides live       : {n_len_not_divides_live}/{n_coop}")

    if bad_examples:
        print(f"\n=== BAD EXAMPLES (length does NOT divide live) ===")
        for ex in bad_examples[:20]:
            print(f"  rec={ex['record']:4d} step={ex['step_idx']:3d} "
                  f"stage={ex['stage']:2d} iter={ex['iter']} "
                  f"stored={ex['stored']:6d} live={ex['live']:6d} "
                  f"len={ex['length']:3d} live%len={ex['live_mod_len']}")
    else:
        print("\n  (no bad examples — all lengths divide live)")


if __name__ == "__main__":
    main()
