## Summary

- Audited `gallery/constrained_gen/generate_programs.py` active path into `modules/`.
- Grouped clear non-active public/legacy helpers under per-file `Deprecated` sections where the active call path does not depend on them.
- Kept low-level shared internals in place when they are mixed with active logic and a pure reordering would be high-risk or ambiguous.

## Files Checked

- `gallery/constrained_gen/generate_programs.py`
- `gallery/constrained_gen/modules/task_paths.py`
- `gallery/constrained_gen/modules/schedule_generator.py`
- `gallery/constrained_gen/modules/param_sampler.py`
- `gallery/constrained_gen/modules/constraint_set.py`
- `gallery/constrained_gen/modules/var_order_planner.py`
- `gallery/constrained_gen/modules/concrete_gpu_verify.py`
- `gallery/constrained_gen/modules/gpu_projection_constraints.py`
- `gallery/constrained_gen/modules/gpu_case_constraints.py`
- `gallery/constrained_gen/modules/gpu_projection_diagnostics.py`
- `gallery/constrained_gen/modules/legacy_record_sketch_io.py`
- `gallery/constrained_gen/modules/domain_propagator.py`
- `gallery/constrained_gen/modules/symbolic_state_bridge.py`

## Active-Path Basis

- `generate_programs.py` uses:
  - `task_paths.load_and_register_tasks`
  - `task_paths.get_to_measure_gen_filename`
  - `ScheduleGenerator.from_task_state`
  - `ScheduleGenerator.next_unique_schedule`
  - `ScheduleGenerator.get_unique_search_stats`
  - `ScheduleGenerator.params_to_state`
  - concrete fingerprinting / lowering helpers from `concrete_gpu_verify.py`
- Exact-check, observability, prefix-report, legacy record I/O, and diagnostics helpers are outside the current active path.

## Changes

- Added or extended `Deprecated` sections in:
  - `task_paths.py`
  - `schedule_generator.py`
  - `param_sampler.py`
  - `constraint_set.py`
  - `var_order_planner.py`
  - `concrete_gpu_verify.py`
  - `gpu_projection_constraints.py`
  - `gpu_case_constraints.py`
  - `gpu_projection_diagnostics.py`
  - `legacy_record_sketch_io.py`
- In `schedule_generator.py`, moved clear non-active public APIs for pruning/exact/observability/random-prefix sampling below `params_to_state()` under `Deprecated`.
- In `param_sampler.py`, marked the legacy retry/prefix sampling block as deprecated; unique active-generation search remains above it.

## Validation

- `python -m py_compile` on the edited module set
- Narrow smoke:
  - built task 0 generator from `generate_concrete_sketches(task)[0]`
  - `gen.next_unique_schedule(set())` returned a payload
  - `gen.get_unique_search_stats()` remained valid

## Remaining Uncertainty

- I did not reorder every low-level internal helper in files like `domain_propagator.py`, `expr_nodes.py`, or `transform_applier.py`; many of those helpers are intertwined with active logic and were not cleanly separable as “deprecated” without large-risk code motion.
- The deprecated grouping is therefore “clear active-surface exclusion” rather than a whole-repo dead-code proof.

## Next Owner

- `validator` if you want a broader `generate_programs.py` regression run after this reordering.
