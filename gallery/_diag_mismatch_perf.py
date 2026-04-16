"""Analyze performance of coop-fetch mismatch cases in both datasets."""
import sys
sys.path.insert(0, "/root/work/tvm-ansor/gallery")

import tvm
from tvm import auto_scheduler
from common import load_and_register_tasks

TVM_JSON = "/root/work/tvm-ansor/gallery/constrained_gen/data/measured_ansor/1490_([3eda1939e30b947e921f5e1814346365,[1,56,56,128],[6,6,32,128],[1,56,56,32]],cuda).json"
TENSET_JSON = "/root/work/tenset/dataset/measure_records_tenset/t4/([4d356f3b296aca3c7fb0ada084b3c2f7,1,56,56,128,6,6,32,128,1,56,56,32],cuda).json"


def record_time_ms(mres):
    """Return mean time in ms, or +inf if failed."""
    try:
        costs = [float(c) for c in mres.costs]
        if not costs:
            return float("inf")
        avg = sum(costs) / len(costs)
        if avg >= 1e9:  # 1e10 = invalid
            return float("inf")
        return avg * 1000.0  # ms
    except Exception:
        return float("inf")


def analyze(task, json_path, label, patch_target=False):
    print(f"\n{'='*70}\n== {label} ({json_path[:80]}...)\n{'='*70}")
    path = json_path
    if patch_target:
        import tempfile
        path = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False).name
        with open(json_path) as fin, open(path, 'w') as fout:
            for line in fin:
                s = line.replace("shared_memory_per_block", "max_shared_memory_per_block")
                s = s.replace("max_max_shared_memory_per_block", "max_shared_memory_per_block")
                fout.write(s)

    replay_fn = tvm._ffi.get_global_func("auto_scheduler.ReplayStepsPartial")

    reader = auto_scheduler.RecordReader(path)
    records = list(reader)
    print(f"[loaded {len(records)} records]")

    # Compute all times
    all_times = [record_time_ms(mres) for (_, mres) in records]
    valid_times = sorted(t for t in all_times if t != float("inf"))
    best_time = valid_times[0] if valid_times else None
    median_time = valid_times[len(valid_times)//2] if valid_times else None
    worst_valid = valid_times[-1] if valid_times else None
    print(f"[valid measurements: {len(valid_times)}/{len(records)}]")
    if best_time:
        print(f"[perf range: best={best_time:.4f}ms, median={median_time:.4f}ms, worst={worst_valid:.4f}ms]")

    # Find bad cases and their times
    bad_div_records = []   # (ri, time_ms, example-dict)
    bad_gt_records = []

    for ri, (minp, mres) in enumerate(records):
        state_obj = (minp.state.state_object
                     if hasattr(minp.state, "state_object") else minp.state)
        t_ms = all_times[ri]

        found_bad_div_in_this_rec = False
        found_bad_gt_in_this_rec = False
        div_detail = None
        gt_detail = None

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

            try:
                stored = int(step.extent)
            except Exception:
                continue
            stage_id = int(step.stage_id)

            try:
                prefix = replay_fn(task.compute_dag, state_obj, si)
                probed = task.compute_dag.infer_bound_from_state(prefix)
                probed_obj = (probed.state_object
                              if hasattr(probed, "state_object") else probed)
                it = probed_obj.stages[stage_id].iters[int(step.iter_id)]
                live = int(it.range.extent) if it.range is not None else None
            except Exception:
                live = None
            if live is None:
                continue

            length_val = lengths[0]
            if length_val > 0 and live % length_val != 0:
                if not found_bad_div_in_this_rec:
                    found_bad_div_in_this_rec = True
                    div_detail = dict(stage=stage_id, stored=stored, live=live, length=length_val)
            if length_val > live:
                if not found_bad_gt_in_this_rec:
                    found_bad_gt_in_this_rec = True
                    gt_detail = dict(stage=stage_id, stored=stored, live=live, length=length_val)

        if found_bad_div_in_this_rec:
            bad_div_records.append((ri, t_ms, div_detail))
        if found_bad_gt_in_this_rec:
            bad_gt_records.append((ri, t_ms, gt_detail))

        if ri % 1000 == 0:
            print(f"  ... {ri}/{len(records)}")

    print(f"\n-- records with length ∤ live: {len(bad_div_records)} --")
    if bad_div_records:
        times = sorted(r[1] for r in bad_div_records if r[1] != float("inf"))
        invalid = sum(1 for r in bad_div_records if r[1] == float("inf"))
        print(f"    valid: {len(times)}, invalid/failed: {invalid}")
        if times:
            print(f"    time range: best={times[0]:.4f}ms median={times[len(times)//2]:.4f}ms worst={times[-1]:.4f}ms")

        # Rank against overall
        if best_time:
            best_in_bad = min((r[1] for r in bad_div_records if r[1] != float("inf")), default=None)
            if best_in_bad:
                rank = sum(1 for t in valid_times if t < best_in_bad)
                pct = 100 * rank / len(valid_times)
                print(f"    best bad-div case: {best_in_bad:.4f}ms "
                      f"(rank {rank+1}/{len(valid_times)}, top {pct:.1f}%)")

        # Top 5 fastest bad cases
        bad_sorted = sorted(bad_div_records, key=lambda x: x[1])
        print(f"    top 5 fastest bad-div records:")
        for (ri, t, det) in bad_sorted[:5]:
            t_str = f"{t:.4f}ms" if t != float("inf") else "FAILED"
            print(f"      rec={ri:4d} time={t_str:12s} stage={det['stage']} "
                  f"stored={det['stored']} live={det['live']} len={det['length']}")

    print(f"\n-- records with length > live: {len(bad_gt_records)} --")
    for (ri, t, det) in bad_gt_records:
        t_str = f"{t:.4f}ms" if t != float("inf") else "FAILED"
        print(f"    rec={ri:4d} time={t_str:12s} stage={det['stage']} "
              f"stored={det['stored']} live={det['live']} len={det['length']}")


def main():
    tasks = load_and_register_tasks()
    task = tasks[1490]

    # Register tenset workload alias
    import json as _json
    tenset_key = _json.dumps(["4d356f3b296aca3c7fb0ada084b3c2f7",
                              1, 56, 56, 128, 6, 6, 32, 128, 1, 56, 56, 32])
    auto_scheduler.workload_registry.register_workload_tensors(
        tenset_key, task.compute_dag.tensors)

    analyze(task, TVM_JSON, "TVM-ANSOR 1490", patch_target=False)
    analyze(task, TENSET_JSON, "TENSET T4", patch_target=True)


if __name__ == "__main__":
    main()
