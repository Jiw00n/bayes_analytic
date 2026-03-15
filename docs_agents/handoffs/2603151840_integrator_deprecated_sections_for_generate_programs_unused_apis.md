## Summary

- Reorganized `generate_programs.py`-inactive APIs under per-file `Deprecated` sections, focusing on the active generation path and the public/legacy helpers outside that path.
- The largest moves were in `modules/param_sampler.py` and `modules/schedule_generator.py`, where validation/prefix/random-retry APIs were pushed below the active unique-generation APIs.

## Files Checked

- `gallery/constrained_gen/generate_programs.py`
- `gallery/constrained_gen/modules/task_paths.py`
- `gallery/constrained_gen/modules/schedule_generator.py`
- `gallery/constrained_gen/modules/param_sampler.py`
- `gallery/constrained_gen/modules/concrete_gpu_verify.py`
- `gallery/constrained_gen/modules/symbolic_state_bridge.py`
- `gallery/constrained_gen/modules/gpu_case_constraints.py`
- `gallery/constrained_gen/modules/gpu_projection_diagnostics.py`
- `gallery/constrained_gen/modules/legacy_record_sketch_io.py`

## Active-path basis

- `generate_programs.py` active path was traced as:
  - `load_and_register_tasks()`
  - `get_to_measure_gen_filename()`
  - `ScheduleGenerator.from_task_state(...)`
  - `ScheduleGenerator.next_unique_schedule(...)`
  - `ScheduleGenerator.get_unique_search_stats()`
  - `ScheduleGenerator.params_to_state(...)`
  - concrete final lowering / GPU verify helpers

## Concrete changes

- `modules/param_sampler.py`
  - kept unique-search APIs at the top
  - moved retry sampler / prefix report / exhaustive enumeration helpers under `Deprecated`
- `modules/schedule_generator.py`
  - kept active generation helpers near the top
  - moved observability / validation / random-retry APIs under `Deprecated`
- Current file contents of the smaller legacy/diagnostic modules were also verified against the same active-path criterion.

## Validation

- `python -m py_compile` on:
  - `gallery/constrained_gen/modules/task_paths.py`
  - `gallery/constrained_gen/modules/symbolic_state_bridge.py`
  - `gallery/constrained_gen/modules/concrete_gpu_verify.py`
  - `gallery/constrained_gen/modules/gpu_case_constraints.py`
  - `gallery/constrained_gen/modules/gpu_projection_diagnostics.py`
  - `gallery/constrained_gen/modules/legacy_record_sketch_io.py`
  - `gallery/constrained_gen/modules/param_sampler.py`
  - `gallery/constrained_gen/modules/schedule_generator.py`
- `python gallery/constrained_gen/generate_programs.py --task-index 0 --records-per-task 1`
  - summary: `selected=1 ok=1 skipped=1 exhausted=0 failed=0`

## Remaining uncertainty

- This classification is intentionally scoped to the current `generate_programs.py` active generation path, not to all other tooling under `gallery/constrained_gen/`.
- Internal helpers that still participate in preprocess, symbolic-state construction, propagation, or concrete final lowering were left in place even if they are not called directly from `generate_programs.py`.

## Next owner

- `validator` only if a broader multi-task generation smoke is needed after the reordering.
