## Summary

Removed exact-symbolic fallback from the pruning-side constraint checks used by the active generation path.

## Files Changed

- `gallery/constrained_gen/modules/constraint_set.py`

## What Changed

Removed exact fallback from these pruning checks:

- `_check_vectorize()`
- `_check_shared_memory()`
- `_check_max_threads()`
- `_check_max_vthread()`

These functions now:

- evaluate the projected/pruning constraint
- if concrete params are available, allow a concrete-final `ok` result to suppress the pruning violation
- otherwise report the projected/pruning violation directly

They no longer:

- call `_ensure_exact_gpu_constraints()`
- compare against `g._exact_gpu[...]`
- use exact-symbolic upper bounds as a fallback to suppress or reinterpret pruning violations

## Why

The active generation path should not depend on exact-symbolic checking:

- `randomize_params()` already accepts through the hybrid/concrete-final path
- `generate_programs.py` no longer runs an explicit exact post-check
- keeping exact fallback inside pruning checks still left exact-symbolic in the active generation pipeline

## Verification

- `python -m py_compile gallery/constrained_gen/modules/constraint_set.py gallery/constrained_gen/generate_programs.py`
- `python gallery/constrained_gen/generate_programs.py --task-index 0`
- `python gallery/constrained_gen/validate.py --task-index 0`

Observed results:

- generation smoke passed
- validation smoke passed

## Remaining Uncertainty

- `validate.py` still reports `exact` because it is a diagnostics/validation harness, not the active generation entrypoint.
- `check_all_exact()` and exact-node construction remain in the codebase for explicit exact diagnostics and validation.

## Next Recommended Owner

- `integrator` if you want to remove exact from validation reporting or make exact strictly diagnostics-only at the API layer.
