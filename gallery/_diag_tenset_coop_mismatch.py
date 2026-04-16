"""Check tenset dataset for coop-fetch stored/live extent mismatch + length>live cases."""
import sys, os
sys.path.insert(0, "/root/work/tvm-ansor/gallery")

import tvm
from tvm import auto_scheduler
from common import load_and_register_tasks

# Tenset T4 dataset: same shape (Winograd 56,56,128) as tvm-ansor task 1490
JSON_PATH = "/root/work/tenset/dataset/measure_records_tenset/t4/([4d356f3b296aca3c7fb0ada084b3c2f7,1,56,56,128,6,6,32,128,1,56,56,32],cuda).json"
TENSET_NETWORK_INFO = "/root/work/tenset/dataset/network_info_tenset"
TARGET_HASH = "4d356f3b296aca3c7fb0ada084b3c2f7"


def main():
    # Load tvm-ansor tasks (task 1490 has same winograd shape)
    tasks = load_and_register_tasks()
    task = tasks[1490]
    print(f"[diag] using tvm-ansor task 1490 compute_dag for same-shape winograd")
    print(f"[diag] workload_key: {task.workload_key[:120]}")

    # Register same tensors under tenset workload key so RecordReader can reconstruct states
    import json as _json
    tenset_key = _json.dumps([TARGET_HASH, 1, 56, 56, 128, 6, 6, 32, 128, 1, 56, 56, 32])
    auto_scheduler.workload_registry.register_workload_tensors(
        tenset_key, task.compute_dag.tensors)
    print(f"[diag] registered tenset key: {tenset_key[:80]}")

    # Patch tenset JSON: rename old target fields to current TVM format
    import tempfile
    patched_path = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False).name
    fixes = 0
    with open(JSON_PATH, 'r') as fin, open(patched_path, 'w') as fout:
        for line in fin:
            new_line = line.replace("shared_memory_per_block", "max_shared_memory_per_block")
            # but we introduced double-max_: undo accidental double replace
            new_line = new_line.replace("max_max_shared_memory_per_block", "max_shared_memory_per_block")
            if new_line != line:
                fixes += 1
            fout.write(new_line)
    print(f"[diag] patched {fixes} lines -> {patched_path}")

    reader = auto_scheduler.RecordReader(patched_path)
    records = list(reader)
    print(f"[diag] loaded {len(records)} records")

    replay_fn = tvm._ffi.get_global_func("auto_scheduler.ReplayStepsPartial")

    n_coop = 0
    n_stored_eq_live = 0
    n_stored_ne_live = 0
    n_len_div_live = 0
    n_len_not_div_live = 0
    n_len_gt_live = 0
    examples_bad_div = []
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

            if live == stored:
                n_stored_eq_live += 1
            else:
                n_stored_ne_live += 1

            if length_val > 0 and live % length_val == 0:
                n_len_div_live += 1
            else:
                n_len_not_div_live += 1
                if len(examples_bad_div) < 15:
                    examples_bad_div.append({
                        "rec": ri, "step": si, "stage": stage_id,
                        "stored": stored, "live": live, "length": length_val,
                        "mod": live % length_val,
                    })

            if length_val > live:
                n_len_gt_live += 1
                if len(examples_gt) < 15:
                    examples_gt.append({
                        "rec": ri, "step": si, "stage": stage_id,
                        "stored": stored, "live": live, "length": length_val,
                    })

        if ri % 500 == 0:
            print(f"  ... {ri+1}/{len(records)}, coop={n_coop}, "
                  f"bad_div={n_len_not_div_live}, len>live={n_len_gt_live}")

    print(f"\n=== TENSET T4 RESULTS ({len(records)} records) ===")
    print(f"  coop-fetch total    : {n_coop}")
    print(f"  stored == live       : {n_stored_eq_live}/{n_coop}")
    print(f"  stored != live       : {n_stored_ne_live}/{n_coop}")
    print(f"  length divides live   : {n_len_div_live}/{n_coop}")
    print(f"  length NOT div live   : {n_len_not_div_live}/{n_coop} "
          f"({100*n_len_not_div_live/max(n_coop,1):.2f}%)")
    print(f"  length > live         : {n_len_gt_live}/{n_coop} "
          f"({100*n_len_gt_live/max(n_coop,1):.2f}%)")

    if examples_bad_div:
        print(f"\n=== length ∤ live examples ===")
        for ex in examples_bad_div:
            print(f"  rec={ex['rec']:4d} step={ex['step']:3d} stage={ex['stage']:2d} "
                  f"stored={ex['stored']:6d} live={ex['live']:6d} "
                  f"len={ex['length']:4d} live%len={ex['mod']}")

    if examples_gt:
        print(f"\n=== length > live examples ===")
        for ex in examples_gt:
            print(f"  rec={ex['rec']:4d} step={ex['step']:3d} stage={ex['stage']:2d} "
                  f"stored={ex['stored']:6d} live={ex['live']:6d} "
                  f"len={ex['length']:4d}")


if __name__ == "__main__":
    main()
