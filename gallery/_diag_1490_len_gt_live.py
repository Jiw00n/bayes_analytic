"""Check task 1490: cases where coop-fetch length > live extent."""
import sys, os
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
    print(f"[diag] loaded {len(records)} records")

    replay_fn = tvm._ffi.get_global_func("auto_scheduler.ReplayStepsPartial")

    n_coop = 0
    n_len_gt_live = 0
    n_len_eq_live = 0
    n_len_lt_live = 0
    examples_gt = []

    for ri, (minp, _mres) in enumerate(records):
        state_obj = (minp.state.state_object
                     if hasattr(minp.state, "state_object") else minp.state)

        for si, step in enumerate(state_obj.transform_steps):
            if not (hasattr(step, "extent") and hasattr(step, "lengths")
                    and hasattr(step, "inner_to_outer")):
                continue
            lengths = []
            for l in step.lengths:
                try:
                    lengths.append(int(l) if l is not None else None)
                except Exception:
                    lengths.append(None)
            if None in lengths or len(lengths) != 1:
                continue

            stage_id = int(step.stage_id)
            try:
                stored = int(step.extent) if step.extent is not None else None
            except Exception:
                stored = None
            if stored is None:
                continue

            n_coop += 1
            length_val = lengths[0]

            try:
                prefix = replay_fn(task.compute_dag, state_obj, si)
                probed = task.compute_dag.infer_bound_from_state(prefix)
                probed_obj = (probed.state_object
                              if hasattr(probed, "state_object") else probed)
                it = probed_obj.stages[stage_id].iters[int(step.iter_id)]
                r = it.range
                live = int(r.extent) if r is not None else None
            except Exception:
                live = None

            if live is None:
                continue

            if length_val > live:
                n_len_gt_live += 1
                if len(examples_gt) < 30:
                    examples_gt.append({
                        "rec": ri, "step": si, "stage": stage_id,
                        "stored": stored, "live": live, "length": length_val,
                        "waste": length_val - live,
                    })
            elif length_val == live:
                n_len_eq_live += 1
            else:
                n_len_lt_live += 1

        if ri % 500 == 0:
            print(f"  ... {ri+1}/{len(records)}, coop={n_coop}, len>live={n_len_gt_live}")

    print(f"\n=== RESULTS (task 1490, {len(records)} records) ===")
    print(f"  coop-fetch total  : {n_coop}")
    print(f"  length > live     : {n_len_gt_live}/{n_coop} ({100*n_len_gt_live/max(n_coop,1):.2f}%)")
    print(f"  length == live    : {n_len_eq_live}/{n_coop}")
    print(f"  length < live     : {n_len_lt_live}/{n_coop}")

    if examples_gt:
        print(f"\n=== EXAMPLES: length > live (factor larger than actual extent) ===")
        for ex in examples_gt:
            print(f"  rec={ex['rec']:4d} step={ex['step']:3d} stage={ex['stage']:2d} "
                  f"stored={ex['stored']:6d} live={ex['live']:6d} "
                  f"len={ex['length']:4d} waste={ex['waste']}")
    else:
        print("\n  (no cases of length > live)")


if __name__ == "__main__":
    main()
