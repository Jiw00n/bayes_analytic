# Shared Projected Pre-Func Split Follow-up

- Agent: `integrator`
- Date: `2026-03-12`
- Status: `completed`
- Decision Topic: `post-patch status after narrow validator evidence`

## What Changed

- `gallery/constrained_gen/modules/exact_gpu_constraints.py`
  - added `build_projected_shared_memory_constraint_node(...)`
  - the new builder walks `projection_context["pre_func"]`, sums `shared` `Allocate` extents times element bytes, and projects the total over runtime domains
- `gallery/constrained_gen/modules/constraint_set.py`
  - `_ensure_projected_gpu_constraints()` now builds `shared_memory` projected constraints directly from the projected context
  - `shared_memory` projected construction no longer forces `_ensure_exact_gpu_constraints()`

## Files And Functions Checked

- `gallery/constrained_gen/modules/exact_gpu_constraints.py`
  - `build_projected_gpu_context`
  - `build_projected_shared_memory_constraint_node`
  - `build_projected_constraint_nodes`
- `gallery/constrained_gen/modules/constraint_set.py`
  - `build_shared_memory_constraints`
  - `_ensure_projected_gpu_constraints`
  - `_ensure_exact_gpu_constraints`
- `src/auto_scheduler/exact_gpu_constraints.cc`
  - `LowerSymbolicPreVectorize`
  - `BuildGpuCaseStats`

## Artifacts

- Performance / behavior:
  - `/tmp/projected_gpu_full_validation/optimizer/shared_projected_split_measurements_20260312.json`
  - `/tmp/projected_gpu_full_validation/optimizer/shared_old_vs_new_projected_20260312.json`
- Validator smoke:
  - `/tmp/projected_gpu_full_validation/validator/shared_projected_regression_smoke_20260312.json`
  - `/tmp/projected_gpu_full_validation/validator/shared_split_validate_projected_smoke_20260312.json`
  - `/tmp/projected_gpu_full_validation/validator/shared_split_validate_exact_smoke_20260312.json`
  - `/tmp/projected_gpu_full_validation/validator/shared_split_validate_exact_smoke_mismatches_20260312.json`
- Notes:
  - `docs_agents/handoffs/2026-03-12-validator-shared-projected-split-smoke.md`
  - `docs_agents/handoffs/2026-03-12-integrator-shared-projected-pre-func-split.md`

## Concrete Outcome

- Hard shared-heavy sketches no longer build exact GPU constraints when only projected `shared_memory` is requested.
- On sampled hard sketches `2` and `3`, projected shared construction dropped from the previous exact-table path to sub-second behavior:
  - sketch `2`, `shared_memory` projected build about `697 ms`
  - sketch `3`, `shared_memory` projected build about `759 ms`
- Narrow generation smoke passed:
  - `randomize_success=2`
  - `concrete_invalid=0`
- Narrow exact-vs-final smoke still shows one `shared_memory` false reject on `sketch_index=2`.
- That false reject is likely pre-existing rather than introduced by this patch:
  - sampled `new_eval` and old exact-derived `old_eval` match on sketches `0,1,2,3`
  - on the representative mismatch sketch `2`, both old and new projected shared values are `76800`

## Remaining Uncertainty

- Reviewer sign-off is still pending in this session.
- The validator shard is narrow and does not establish broad-rate behavior.
- The remaining `shared_memory` false reject still needs separate root-cause work if we want to reduce pruning conservatism.

## Next Recommended Owner

- Recommended owner: `reviewer`
- Recommended next step:
  - confirm whether the current narrow evidence is sufficient to accept this patch as non-regressive
  - if accepted, route the remaining `runtime_projection_upper_bound_insufficient` false reject to `specialist`
