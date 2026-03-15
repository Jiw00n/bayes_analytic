## Task

Clarify the active public surface of `gallery/constrained_gen`, keep the future API centered on `ScheduleGenerator`, and isolate legacy/diagnostic helper paths without adding new entrypoint files.

## Files Checked

- `gallery/constrained_gen/modules/schedule_generator.py`
- `gallery/constrained_gen/modules/record_loader.py`
- `gallery/constrained_gen/modules/projected_gpu_validation.py`
- `gallery/constrained_gen/modules/param_sampler.py`
- `gallery/measure_programs.py`

## Keep / Internalize / Legacy Split

- Keep public:
  - `gallery/constrained_gen/modules/__init__.py`: `build_symbolic_state`, `ScheduleGenerator`
  - `ScheduleGenerator` public assignment/state API in `gallery/constrained_gen/modules/schedule_generator.py`
  - top-level program scripts only (`gallery/constrained_gen/generate.py`, `gallery/measure_programs.py`)
- Internalize behind `ScheduleGenerator`:
  - `constraint_set.py`
  - `var_order_planner.py`
  - `domain_propagator.py`
  - `param_sampler.py`
  - `tvm_verify.py`
- Legacy / diagnostic:
  - `record_loader.py` for record-based recovery/grouping helpers
  - `projected_gpu_validation.py` for mismatch triage only

## Changes

- Removed the shadow copy of the assignment/public-helper block at the end of `schedule_generator.py`, leaving one active implementation for:
  - `get_param_candidates(...)`
  - `propagate_param_assignment(...)`
  - `get_constraints_under_assignment(...)`
  - `params_to_state(...)`
- Kept the clearer `ScheduleGenerator` facade and added alias-style surface methods for the target future API:
  - `get_full_var_order_entries()` at `schedule_generator.py:1219`
  - `get_candidate_values()` at `schedule_generator.py:1238`
  - `propagate_assignment()` at `schedule_generator.py:1241`
- Routed `get_concrete_final_result()` through `ScheduleGenerator.params_to_state()` at `schedule_generator.py:230-263` so params-to-state now has one owner on the facade path.
- Confirmed/retained legacy and diagnostics labeling:
  - `record_loader.py:1-5`
  - `projected_gpu_validation.py:1`

## Verification

- Ran:
  - `python -m py_compile gallery/constrained_gen/modules/schedule_generator.py gallery/constrained_gen/modules/param_sampler.py gallery/constrained_gen/modules/record_loader.py gallery/constrained_gen/modules/projected_gpu_validation.py`

## Outcome

- `ScheduleGenerator` now has one visible assignment/inspection/state-conversion surface instead of two competing copies with different signatures.
- The future public API can live on `ScheduleGenerator` without exposing `domain_propagator`, `param_sampler`, or `tvm_verify` directly.
- Record-based loading and projected-GPU differential helpers are explicitly isolated as legacy/diagnostic paths.

## Remaining Uncertainty

- `schedule_generator.py` still exposes many per-constraint forwarding methods (`build_*`, `check_*`) that could be reduced further in a later pass if callers remain absent.
- `gallery/measure_programs.py` still imports fingerprint helpers from `record_loader.py`; that dependency remains until those pure sketch helpers get a non-legacy owner.

## Next Recommended Owner

- `integrator` for an optional second pass that trims the per-constraint forwarding methods or relocates the sketch fingerprint helpers out of `record_loader.py`.
