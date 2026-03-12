# Projected/Exact Split, Lazy Exact, And Validation Dedupe

- Agent: `integrator`
- Date: `2026-03-12`
- Status: `completed`

## What Changed

- Split projected and exact GPU-constraint cache/build paths in:
  - [constraint_set.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/constraint_set.py)
  - [schedule_generator.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/schedule_generator.py)
  - [exact_gpu_constraints.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/exact_gpu_constraints.py)
- Added lightweight projected GPU context and vectorize-only projected builder so vectorize pruning no longer forces exact case-table construction during init.
- Made exact GPU construction lazy relative to vectorize-only projected usage.
- Removed duplicate concrete verify in [validate_projected_gpu_generation.py](/root/work/tvm-ansor/gallery/constrained_gen/validate_projected_gpu_generation.py) by reusing `gen.get_concrete_final_result(params)` when available.
- Removed the redundant second `get_concrete_final_result()` path inside `check_all_hybrid()` by sharing the already-fetched concrete result with final checking logic.

## Files And Functions Checked

- [exact_gpu_constraints.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/exact_gpu_constraints.py)
  - `build_projected_gpu_context`
  - `build_projected_vectorize_constraint_node`
  - `build_exact_constraint_nodes`
  - `build_projected_constraint_nodes`
- [constraint_set.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/constraint_set.py)
  - `build_vectorize_constraints`
  - `build_shared_memory_constraints`
  - `_ensure_exact_gpu_constraints`
  - `_ensure_projected_gpu_constraints`
- [schedule_generator.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/schedule_generator.py)
  - `check_all_hybrid`
  - `check_all_final`
  - `_ensure_projected_gpu_constraints`
- [projected_gpu_validation.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/projected_gpu_validation.py)
  - projected diagnostics helpers
- [validate_projected_gpu_generation.py](/root/work/tvm-ansor/gallery/constrained_gen/validate_projected_gpu_generation.py)
  - randomize/verification loop

## Validation And Artifacts

- Smoke-tested projected generation validation:
  - `python gallery/constrained_gen/validate_projected_gpu_generation.py --all-sketches --limit 1 --attempts-per-sketch 1 --max-retries 1 --summary-path /tmp/projected_gpu_full_validation/validator/post_patch_validate_projected_smoke.json`
- Compared `check_all_exact()` against the legacy composition of individual exact checks on a hard sketch sample.
- Measured subset init timings and lazy exact behavior on `sketch_index=2`.

## Concrete Outcome

- Hard-sketch init timings after the split:
  - `all`: about `20.86 s`
  - `shared_only`: about `20.69 s`
  - `vectorize_only`: about `246 ms`
  - `no_exact_gpu_kinds`: about `22.7 ms`
- `vectorize_only` no longer builds exact GPU constraints during init; exact is built only when `check_vectorize_exact()` is actually called.
- `collect_gpu_projection_diagnostics()` still works after the cache split.
- `validate_projected_gpu_generation.py` now reuses the cached concrete final result instead of lowering/verifying the same params again in the outer loop.

## Remaining Uncertainty

- This split is partial, not complete:
  - `shared_memory` projected pruning still depends on exact case-table construction
  - so `all` and `shared_only` init remain dominated by exact case generation
- A fuller projected/exact split still needs a projected shared-memory path that does not depend on enumerating exact selector cases.

## Next Recommended Owner

- Recommended owner: `validator`
- Recommended next step: run narrow exact-vs-concrete and projected-generation shards on the patched Python path, then decide whether to attack shared-memory projected pruning next.
