# 2603132324 Exact GPU Case Stats Cache

- Agent: `integrator`
- Date: `2026-03-13 23:24`
- Status: `accepted`
- Decision Topic: `process-local cache for exact GPU case stats extraction`

## Inputs Considered

- Relevant artifacts:
  - [2603132305_optimizer_exact_no_cache_hotspot_investigation.md](/root/work/tvm-ansor/docs_agents/handoffs/2603132305_optimizer_exact_no_cache_hotspot_investigation.md)
  - [2603132310_integrator_exact_gpu_parallel_extract.md](/root/work/tvm-ansor/docs_agents/handoffs/2603132310_integrator_exact_gpu_parallel_extract.md)

## Files Checked

- `gallery/constrained_gen/modules/exact_gpu_constraints.py`: `_EXTRACT_ALL_GPU_CASE_STATS`, `build_exact_constraint_nodes`
- `gallery/constrained_gen/modules/constraint_set.py`: `_ensure_exact_gpu_constraints`
- `gallery/constrained_gen/validate.py`: narrow validate path

## Decision

- Chosen direction:
  - Add a process-local in-memory cache for `_EXTRACT_ALL_GPU_CASE_STATS(...)` results.
  - Key the cache by the textual `pre_func` IR and the normalized selector-case table.
  - Reuse cached case stats when a later `ScheduleGenerator` rebuilds exact constraints for the same symbolic pre-vectorize TIR and the same vector-case list.
- Rejected alternatives:
  - disk/file cache
  - cross-process persistence
  - caching whole `ScheduleGenerator` or whole exact-node dicts
- Why:
  - The expensive work is repeated exact case extraction.
  - Reusing only the extracted case stats is the narrowest cache that avoids recomputation while preserving the current exact-node construction flow.
  - `tvm.ir.structural_hash` / `tvm.ir.structural_equal` did not match repeated builds here even when the printed TIR was identical, so the cache key uses stable textual IR instead.

## Impact

- Files changed:
  - `gallery/constrained_gen/modules/exact_gpu_constraints.py`
- Validation run in main session:
  - `python -m py_compile gallery/constrained_gen/modules/exact_gpu_constraints.py gallery/constrained_gen/modules/constraint_set.py gallery/constrained_gen/validate.py`
  - `python gallery/constrained_gen/validate.py --task-index 29`
  - `python gallery/constrained_gen/validate.py --task-index 383`
  - one-process repeated exact init timing on task `383`
- Measured outcome:
  - task `29`: `~2.36s`, pass
  - task `383`: `~9.08s`, pass
  - task `383` exact init in one process:
    - first build: `~6.07s`
    - second build with same concrete sketch: `~0.14s`
    - repeated same-process speedup: `~43.4x`

## Remaining Uncertainty

- This cache is process-local only. Fresh processes still pay the cold exact-init cost once.
- Narrow validation only. A broader shard may still be worth running if repeated same-process rebuilds are common in the next workflow.

## Next Owner

- Recommended owner: `validator`
- Recommended next step:
  - leave a role-pure narrow validation note for the same commands and cold/warm measurement before broader rollout
