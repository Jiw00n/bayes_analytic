"""Verify claimed Ansor invariants against real records:

  H1: prod(lengths) == stored_extent   for every SplitStep in JSON
  H2: stored_extent usually != live extent (after full-replay InferBound)
  H3: prod(lengths) often does NOT divide live extent (boundary-check cases)
"""
import sys, os
sys.path.insert(0, "/root/work/tvm-ansor/gallery")

import tvm
from tvm import auto_scheduler
from common import load_and_register_tasks

JSON_PATH = os.environ.get(
    "JSON_PATH",
    "/root/work/tvm-ansor/gallery/constrained_gen/data/measured_family_ansor/415_([e7c984cba151d5c7c1e081f0b1910087,[1,112,112,32],[3,3,32,1],[1,1,1,32],[1,112,112,32]],cuda).json",
)
MAX_RECORDS = int(os.environ.get("MAX_RECORDS", "50"))

def prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p

def main():
    tasks = load_and_register_tasks()
    task = tasks[415]

    reader = auto_scheduler.RecordReader(JSON_PATH)
    records = list(reader)[:MAX_RECORDS]
    print(f"[diag] loaded {len(records)} records")

    replay_fn = tvm._ffi.get_global_func("auto_scheduler.ReplayStepsPartial")

    from collections import defaultdict
    # per stage-category: "tile" (stage 3 etc.) or "coop_fetch" (stage 2,4 with lengths.size()==1)
    stats = defaultdict(lambda: {
        "n": 0, "prod_divides_stored": 0, "stored_eq_live": 0,
        "stored_ne_live": 0, "prod_divides_live": 0, "prod_not_divides_live": 0,
        "nontrivial_lengths": 0,
    })
    examples_bad_live = []

    for ri, (minp, _mres) in enumerate(records):
        state_obj = (minp.state.state_object
                     if hasattr(minp.state, "state_object") else minp.state)
        for si, step in enumerate(state_obj.transform_steps):
            if not (hasattr(step, "extent") and hasattr(step, "lengths")
                    and hasattr(step, "inner_to_outer")):
                continue
            try:
                stored = int(step.extent) if step.extent is not None else None
            except Exception:
                stored = None
            if stored is None:
                continue
            lengths = [int(l) for l in step.lengths if l is not None]
            if len(lengths) != len(list(step.lengths)):
                continue
            p = prod(lengths)

            # categorize
            cat = "coop_fetch" if len(lengths) == 1 else "tile"
            st = stats[cat]
            st["n"] += 1
            if p > 1:
                st["nontrivial_lengths"] += 1
            if stored > 0 and stored % p == 0:
                st["prod_divides_stored"] += 1

            try:
                prefix = replay_fn(task.compute_dag, state_obj, si)
                probed = task.compute_dag.infer_bound_from_state(prefix)
                probed_obj = (probed.state_object
                              if hasattr(probed, "state_object") else probed)
                it = probed_obj.stages[int(step.stage_id)].iters[int(step.iter_id)]
                r = it.range
                live = int(r.extent) if r is not None else None
            except Exception:
                live = None

            if live is None:
                continue
            if live == stored:
                st["stored_eq_live"] += 1
            else:
                st["stored_ne_live"] += 1

            if p > 0 and live % p == 0:
                st["prod_divides_live"] += 1
            else:
                st["prod_not_divides_live"] += 1
                if len(examples_bad_live) < 10:
                    examples_bad_live.append(
                        (ri, si, int(step.stage_id), int(step.iter_id),
                         stored, lengths, live))

    print("\n=== RESULTS (per category) ===")
    for cat, st in stats.items():
        n = st["n"]
        print(f"\n[{cat}]  n={n}")
        print(f"  prod(lengths) divides stored : {st['prod_divides_stored']}/{n}")
        print(f"  stored == live                : {st['stored_eq_live']}/{n}")
        print(f"  stored != live (stale)        : {st['stored_ne_live']}/{n}")
        print(f"  prod(lengths) divides live    : {st['prod_divides_live']}/{n}")
        print(f"  prod(lengths) NOT div live    : {st['prod_not_divides_live']}/{n}")
        print(f"  nontrivial lengths (p>1)      : {st['nontrivial_lengths']}/{n}")

    print("\nBAD: prod(lengths) does not divide live extent")
    print("(rec, step, stage, iter, stored, lengths, live):")
    for e in examples_bad_live:
        print(f"  {e}")


if __name__ == "__main__":
    main()
