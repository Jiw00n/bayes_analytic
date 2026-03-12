# Exact Concrete Fast Path

- Agent: `integrator`
- Date: `2026-03-12`
- Status: `in_progress`

## What Changed

- [exact_gpu_constraints.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/exact_gpu_constraints.py)
  - `build_exact_constraint_nodes(...)` now records per-kind `case_expr_vars`
- [constraint_set.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/constraint_set.py)
  - `_evaluate_exact_upper_bounds(...)` now uses a concrete-evaluable fast path when sampled params fully assign all variables referenced by exact case expressions
  - added `_can_evaluate_exact_cases_concretely(...)`
  - added `_evaluate_exact_upper_bounds_concretely(...)`

## Files And Functions Checked

- [constraint_set.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/constraint_set.py)
  - `check_all_exact`
  - `_evaluate_exact_upper_bounds`
  - `_exact_upper_bound_from_interval`
- [expr_nodes.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/expr_nodes.py)
  - `CaseSplitNode._augment_domains`
  - `CaseSplitNode.feasible_case_values`
  - `CaseSplitNode.interval_with_feasible_cases`
  - `PrimExprNode.evaluate`
- [exact_gpu_constraints.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/exact_gpu_constraints.py)
  - `build_exact_constraint_nodes`

## Concrete Outcome So Far

- On hard sketches `2` and `3`, local timing checks dropped one-shot `randomize_params()` latency to about:
  - sketch `2`: `~0.042 s`
  - sketch `3`: `~0.061 s`
- The new path returns concrete exact maxima in cases where the legacy interval path returned the sentinel-derived `None`.
- Narrow generation smoke on sketches `2` and `3` passed:
  - `40/40` randomize successes
  - `0` concrete invalid
- Narrow exact-vs-concrete validation on sketches `2` and `3` reproduced one pre-existing projected shared-memory false reject on sketch `2`:
  - pruning rejected
  - final checker accepted
  - concrete checker accepted
  - this mismatch is on projected `shared_memory`, not on the new exact fast path

## Artifacts

- Local generation smoke:
  - `/tmp/projected_gpu_full_validation/validator/local_exact_fast_path_gen_smoke_20260312/summary.json`
  - `/tmp/projected_gpu_full_validation/validator/local_exact_fast_path_gen_smoke_20260312/invalid.jsonl`
- Local exact-vs-concrete smoke:
  - `/tmp/projected_gpu_full_validation/validator/local_exact_fast_path_exact_smoke_20260312/summary.json`
  - `/tmp/projected_gpu_full_validation/validator/local_exact_fast_path_exact_smoke_20260312/mismatch.jsonl`

## Remaining Uncertainty

- Full all-sketch generation validation is still running in sharded validator sessions.
- Reviewer sign-off is still pending.
- The projected shared-memory false reject on sketch `2` remains as separate correctness debt.

## Next Recommended Owner

- Recommended owner: `validator`
- Recommended next step: finish the full 2000-attempt-per-sketch sharded generation sweep and hand raw artifacts to `reviewer`.
