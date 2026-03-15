## Summary

Removed exact-symbolic post-checking from the active `generate_programs.py` path so generation acceptance is aligned with the current sampler/hybrid concrete-final workflow.

## Files Changed

- `gallery/constrained_gen/generate_programs.py`

## What Changed

- Deleted the explicit post-sampling calls to:
  - `gen.check_all_pruning(params)`
  - `gen.check_all_exact(params)`
- Kept the explicit `gen.check_all_final(params)` recheck before converting params to concrete state and saving the record.

## Why

- Current active sampling already accepts candidates through `randomize_params()` -> `check_all_hybrid()` in the `from_task_state(...)` path.
- In that path, hybrid acceptance is concrete-final driven, not exact-symbolic driven.
- The old `generate_programs.py` code then reintroduced exact-symbolic as a stronger post-check gate, which caused concrete-valid schedules to be rejected.

## Verification

- `python -m py_compile gallery/constrained_gen/generate_programs.py`
- `python gallery/constrained_gen/generate_programs.py --task-index 0`

Smoke result:

- `selected_tasks=1`
- `successes=1`
- `failures=0`

## Remaining Uncertainty

- This change removes the explicit exact gate from `generate_programs.py`, but exact-symbolic fallback logic still exists inside some pruning checks in `constraint_set.py`.
- If the goal is to remove exact-symbolic from the entire active generation path, those fallback paths need a separate decision.

## Next Recommended Owner

- `integrator` if you want to remove exact fallback from pruning checks as well.
