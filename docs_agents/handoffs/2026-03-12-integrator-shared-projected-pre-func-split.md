# Shared Projected Pre-Func Split

- Agent: `integrator`
- Date: `2026-03-12`
- Status: `accepted`
- Decision Topic: `derive shared_memory projected constraints from pre-vectorize shared allocations`

## Inputs Considered

- Reviewer note:
  - `docs_agents/handoffs/2026-03-12-reviewer-shared-projected-split-smoke-review.md`
- Specialist note: none
- Relevant artifacts:
  - `/tmp/projected_gpu_full_validation/optimizer/shared_projected_split_measurements_20260312.json`
  - `/tmp/projected_gpu_full_validation/validator/shared_projected_regression_smoke_20260312.json`
  - `/tmp/projected_gpu_full_validation/validator/shared_split_validate_projected_smoke_20260312.json`
  - `/tmp/projected_gpu_full_validation/validator/shared_split_validate_exact_smoke_20260312.json`
  - `/tmp/projected_gpu_full_validation/validator/shared_split_validate_exact_smoke_mismatches_20260312.json`

## Files Checked

- `gallery/constrained_gen/modules/exact_gpu_constraints.py`: `build_projected_gpu_context()`, `build_exact_constraint_nodes()`, `build_projected_constraint_nodes()`
- `gallery/constrained_gen/modules/constraint_set.py`: `build_shared_memory_constraints()`, `_ensure_projected_gpu_constraints()`, `_ensure_exact_gpu_constraints()`
- `gallery/constrained_gen/modules/symbolic_state.py`: `get_shared_memory_extents()`
- `gallery/constrained_gen/modules/transform_applier.py`: `_apply_fuse()`, `_apply_split()`
- `src/auto_scheduler/exact_gpu_constraints.cc`: `BuildGpuCaseStats()`, `ExtractAllGpuCaseStats()`, `LowerSymbolicPreVectorize()`

## Decision

- Chosen direction:
  - Added a direct projected shared-memory builder that walks `projection_context["pre_func"]`, sums `shared` `Allocate` extents times element bytes, and projects the total over runtime domains.
  - Changed `_ensure_projected_gpu_constraints()` so `shared_memory` projected constraints no longer force `_ensure_exact_gpu_constraints()`.
- Rejected alternatives:
  - Reusing `SymbolicState._shared_fused_extents` as the pruning bound source.
  - Building shared projected bounds from current shared stage iter products.
- Why:
  - `pre_func` already contains symbolic shared `Allocate` extents with runtime variables, so it is the closest semantics-preserving projected source available without enumerating exact vector selector cases.
  - `SymbolicState._shared_fused_extents` and current shared stage iter products both undercount on hard batch-matmul sketches and therefore are not safe projected upper bounds.

## Impact

- Files likely to change:
  - `gallery/constrained_gen/modules/exact_gpu_constraints.py`
  - `gallery/constrained_gen/modules/constraint_set.py`
- Validation needed after change:
  - representative shared projected regression
  - narrow generation smoke
  - narrow exact-vs-final smoke
  - reviewer follow-up still recommended because `shared_split_validate_exact_smoke` shows one existing shared-memory false reject on `sketch_index=2`

## Next Owner

- Recommended owner: `integrator`
- Recommended next step:
  - keep the remaining shared-memory false reject as a separate correctness item and re-profile the next bottleneck before choosing the next optimization target
