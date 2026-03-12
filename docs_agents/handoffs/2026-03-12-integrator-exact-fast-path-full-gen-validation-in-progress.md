# Exact Fast Path And Full Generation Validation In Progress

- Agent: `integrator`
- Date: `2026-03-12`
- Status: `in_progress`

## What Changed

- Updated [exact_gpu_constraints.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/exact_gpu_constraints.py) so exact case construction records per-kind case-expression variable sets in `case_expr_vars`.
- Updated [constraint_set.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/constraint_set.py) so `_evaluate_exact_upper_bounds()` takes a concrete-evaluable fast path:
  - keep the shared feasible-case scan
  - when case expressions depend only on assigned symbolic params, evaluate exact case expressions directly with `evaluate()` instead of going through `interval()`
  - keep the old interval-based fallback for non-concrete paths

## Files And Functions Checked

- [exact_gpu_constraints.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/exact_gpu_constraints.py)
  - `build_exact_constraint_nodes`
- [constraint_set.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/constraint_set.py)
  - `_evaluate_exact_upper_bounds`
  - `_can_evaluate_exact_cases_concretely`
  - `_evaluate_exact_upper_bounds_concretely`
- [validate_projected_gpu_generation.py](/root/work/tvm-ansor/gallery/constrained_gen/validate_projected_gpu_generation.py)
- [validate_exact_gpu_constraints.py](/root/work/tvm-ansor/gallery/constrained_gen/validate_exact_gpu_constraints.py)

## Narrow Validation Outcome

- Local generation smoke on hard sketches `2` and `3`:
  - `40/40` randomize successes
  - `0` concrete invalids
  - artifacts:
    - `/tmp/projected_gpu_full_validation/validator/local_exact_fast_path_gen_smoke_20260312/summary.json`
    - `/tmp/projected_gpu_full_validation/validator/local_exact_fast_path_gen_smoke_20260312/invalid.jsonl`
- Local exact-vs-concrete smoke on sketches `2` and `3` still reproduces one pre-existing `shared_memory` projected false reject on sketch `2`; this is unrelated to the new exact fast path.
  - artifacts:
    - `/tmp/projected_gpu_full_validation/validator/local_exact_fast_path_exact_smoke_20260312/summary.json`
    - `/tmp/projected_gpu_full_validation/validator/local_exact_fast_path_exact_smoke_20260312/mismatch.jsonl`

## Performance Outcome

- On representative hard sketches, a single `randomize_params()` call dropped from the earlier `~0.23s to ~0.31s` range to about:
  - sketch `2`: `~0.04s`
  - sketch `3`: `~0.06s`

## Full Validation Status

- Full generation validation is running under a long-lived supervisor shell session using 12 shards of 76 sketches each.
- Root artifact base:
  - `/tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/`
- Per-shard run logs:
  - `/tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_00/run.log`
  - ...
  - `/tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_11/run.log`
- Current early progress snapshot when this note was written:
  - `shard_00`: reached `[15/76]` with `invalid=0`, `rand_fail=0`
  - `shard_01`: reached `[15/76]` with `invalid=0`, `rand_fail=0`
  - `shard_08`: reached `[5/76]` with `invalid=0`, `rand_fail=0`
  - other shards were still processing their first five sketches, which is expected because some first-sketch workloads take several minutes per 2000-attempt block

## Remaining Uncertainty

- The full 912-sketch x 2000-attempt sweep is still running and has not finished yet.
- Final aggregate invalid/randomize-fail counts are therefore not available in this note.

## Next Recommended Owner

- Recommended owner: `integrator`
- Recommended next step:
  - continue polling the supervisor session
  - once all shard summaries are present, merge them into a single aggregate report
  - then hand the artifact set to `reviewer`
