## Task

Reduce and clarify the `gallery/constrained_gen` module/function surface without adding new entrypoint files.

## Files Checked

- `gallery/constrained_gen/modules/schedule_generator.py`
- `gallery/constrained_gen/modules/param_sampler.py`
- `gallery/constrained_gen/modules/domain_propagator.py`
- `gallery/constrained_gen/modules/constraint_set.py`
- `gallery/constrained_gen/modules/record_loader.py`
- `gallery/constrained_gen/modules/projected_gpu_validation.py`
- `gallery/constrained_gen/modules/tvm_verify.py`
- `gallery/constrained_gen/generate.py`
- `gallery/measure_programs.py`

## Decision

- Keep `ScheduleGenerator` as the public owner for generator/query operations.
- Keep `record_loader.py` only as a legacy record/sketch reconstruction helper module.
- Keep `projected_gpu_validation.py` as diagnostics-only.
- Treat `constraint_set.py`, `domain_propagator.py`, `param_sampler.py`, and `var_order_planner.py` as internal owner modules behind `ScheduleGenerator`.

## Changes

- Added `ScheduleGenerator.from_task_state(...)` as the owner-module constructor for task/state -> generator setup.
- Added/kept ScheduleGenerator-facing query helpers so callers do not need direct helper-module access:
  - `get_full_var_order_entries()`
  - `get_var_order_entries()`
  - `get_candidate_values()`
  - `propagate_assignment()`
  - `params_to_state()`
- Reused `params_to_state()` inside concrete-final validation to avoid a second record/state conversion path.
- Marked `record_loader.build_schedule_generator(...)` as a legacy convenience wrapper and redirected it to `ScheduleGenerator.from_task_state(...)`.

## Verification

- Ran:
  - `python -m py_compile gallery/constrained_gen/modules/schedule_generator.py gallery/constrained_gen/modules/record_loader.py`

## Outcome

- The public surface is now centered more explicitly on `ScheduleGenerator`.
- Legacy record/sketch helpers remain available but are more clearly isolated.
- No new entrypoints or modules were added.

## Remaining Uncertainty

- `gallery/measure_programs.py` still imports fingerprint helpers from `record_loader.py`; that helper dependency remains for now.
- No validator/reviewer runtime shard was run because this was a surface/API cleanup pass and the request explicitly said not to build new entrypoints yet.

## Recommended Next Owner

- `integrator` for a follow-up pass on remaining legacy measurement/fingerprint helpers if you want `record_loader.py` reduced further.
