## Summary

- Added the first research/debug entrypoint at `gallery/constrained_gen/validate.py`.
- Used the existing active concrete-sketch workflow already present in `gallery/constrained_gen/generate.py`:
  - `load_and_register_tasks()`
  - `SketchPolicy(...).generate_concrete_sketches()`
  - `ScheduleGenerator.from_task_state(...)`
- Kept the CLI minimal with exactly one selector:
  - `--task-index`
  - `--workload-key`
  - `--all`

## Files Changed

- `gallery/constrained_gen/validate.py`

## Entry Point Behavior

- selects tasks via `load_and_register_tasks()`
- generates concrete sketches through `SketchPolicy(..., params={"sample_init_no_invalid": 1}).generate_concrete_sketches()`
- picks the first available concrete sketch per selected task
- attempts the intended observability path:
  - `get_full_var_order_entries()`
  - `randomize_params_prefix(...)`
  - `randomize_params()`
  - `check_all_pruning(...)`
  - `check_all_exact(...)`
  - `check_all_final(...)`
- prints concise per-task summaries on success
- prints structured JSON failure payloads instead of ad hoc strings when a stage fails

## Verification

- `python -m py_compile gallery/constrained_gen/validate.py`
- smoke run:
  - `python gallery/constrained_gen/validate.py --task-index 0`

## Workflow Assumption

- The active sketch-construction workflow is not notebook-only.
- The current Python entrypoint path is the same one already visible in `gallery/constrained_gen/generate.py`.

## Current Blocker

The smoke run hits an existing upstream core regression during generator construction:

- stage: `construct_schedule_generator`
- error: `AttributeError: 'ScheduleGenerator' object has no attribute '_format_expr'`

Observed from:

- `ScheduleGenerator.from_task_state(...)`
- `ConstraintSet.preprocess()`
- `ConstraintSet._build_max_threads_constraints()`
- `ConstraintSet._build_thread_per_block_constraint_item(...)`

The failure payload from `validate.py --task-index 0` was:

```json
{
  "error": "AttributeError: 'ScheduleGenerator' object has no attribute '_format_expr'",
  "selected_sketch_index": 0,
  "sketch_count": 1,
  "stage": "construct_schedule_generator",
  "task_desc": "vm_mod_fused_nn_adaptive_avg_pool3d",
  "task_index": 0,
  "workload_key": "[\"1aa729c96f4afc0cf6bf84dff07364c6\", [1, 18, 9, 1, 512], [1, 1, 1, 1, 512]]"
}
```

## Next Recommended Owner

- `integrator` for the core fix in `gallery/constrained_gen/modules/schedule_generator.py` / `gallery/constrained_gen/modules/constraint_set.py`, since the new entrypoint is blocked by an existing facade/helper regression rather than by missing workflow logic.
