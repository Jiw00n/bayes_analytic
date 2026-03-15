## Task

Reduce and clarify the active module/function surface in `gallery/constrained_gen` without adding the future entrypoint files yet.

## Files Checked

- `gallery/constrained_gen/modules/schedule_generator.py`
- `gallery/constrained_gen/modules/var_order_planner.py`
- `gallery/constrained_gen/modules/record_loader.py`
- `gallery/constrained_gen/modules/__init__.py`
- `gallery/constrained_gen/generate.py`
- `gallery/constrained_gen/modules/domain_propagator.py`
- `gallery/constrained_gen/modules/param_sampler.py`
- `gallery/constrained_gen/modules/tvm_verify.py`

## Changes

- Consolidated the future-facing public API onto `ScheduleGenerator`:
  - added `from_task_state(...)`
  - added assignment/domain helpers:
    - `get_assignment_state(...)`
    - `get_param_candidates(...)`
    - `propagate_param_assignment(...)`
    - `get_constraints_under_assignment(...)`
  - added concrete reconstruction entry:
    - `params_to_state(...)`
  - added var-order aliases:
    - `get_var_order_entries()`
    - `get_full_var_order_entries()`
    - compatibility aliases `get_candidate_values()` and `propagate_assignment()`
- Extended `VarOrderPlanner.get_var_order_phase_entries()` payloads with `var_entries`, including split metadata and `collapsed_factors`.
- Pointed `record_loader.build_schedule_generator(...)` at `ScheduleGenerator.from_task_state(...)` so the record helper is a wrapper instead of a second construction owner.
- Reduced the package surface in `modules/__init__.py` to `ScheduleGenerator` only.
- Switched `gallery/constrained_gen/generate.py` to construct generators via `ScheduleGenerator.from_task_state(...)`.

## Verification

- Ran:
  - `python -m py_compile gallery/constrained_gen/modules/schedule_generator.py gallery/constrained_gen/modules/var_order_planner.py gallery/constrained_gen/modules/record_loader.py gallery/constrained_gen/modules/__init__.py gallery/constrained_gen/generate.py`

## Outcome

- Active public construction and parameter-inspection paths now live on `ScheduleGenerator`.
- `record_loader.py` stays available for legacy/diagnostic record ingestion, but no longer owns generator construction.
- The package export surface is smaller and closer to the intended future shape.

## Remaining Uncertainty

- `ScheduleGenerator` still exposes the older per-constraint forwarding block (`build_*`, `check_*_exact`, etc.). Those methods now look more clearly like the next internalization target.
- The future top-level entrypoints (`generate_programs.py`, `validate.py`) are still absent in the current worktree.
- No validator/reviewer pass was run because this was an API/surface refactor with no intended checker-semantic change.

## Recommended Next Owner

- `integrator` for the next reduction pass on the remaining per-constraint checker forwards in `schedule_generator.py`.
