## Summary

- Fixed the core regression that blocked `gallery/constrained_gen/validate.py`.
- Kept the fix narrow by restoring a private formatting bridge on `ScheduleGenerator` rather than widening `ConstraintSet`'s surface.
- Reran the requested compile check and the `--task-index 0` validate smoke successfully.

## Files Changed

- `gallery/constrained_gen/modules/schedule_generator.py`

## Exact Root Cause

`ConstraintSet` still calls `g._format_expr(...)` while building:

- `_build_thread_per_block_constraint_item(...)`
- `_build_split_structure_constraints(...)`

That helper used to live on `ScheduleGenerator`, but after the formatting ownership moved into the internal inspector helper, the facade bridge for `_format_expr(...)` was no longer present.

Result:

- `ScheduleGenerator.from_task_state(...)`
- `ConstraintSet.preprocess()`
- `ConstraintSet._build_max_threads_constraints()`
- `ConstraintSet._build_thread_per_block_constraint_item(...)`

failed with:

- `AttributeError: 'ScheduleGenerator' object has no attribute '_format_expr'`

## Fix

Restored a narrow internal bridge on `ScheduleGenerator`:

- `_format_expr(self, node, top_level=False)`

which now delegates to:

- `self._inspector._format_expr(...)`

This keeps the existing internal ownership boundary intact:

- `ConstraintSet` still talks to the facade object it owns
- formatting still lives under the internal inspector helper

## Verification

Requested compile check:

- `python -m py_compile gallery/constrained_gen/modules/schedule_generator.py gallery/constrained_gen/modules/constraint_set.py gallery/constrained_gen/validate.py`

Requested smoke:

- `python gallery/constrained_gen/validate.py --task-index 0`

Smoke output:

```text
[task 0] vm_mod_fused_nn_adaptive_avg_pool3d sketches=1 selected_sketch=0 phase_count=7
  prefix phase=grid_0__execution_non_product_direct_arm params=0 remaining_domains=2 leftover_constraints=0 resolved_false=0
  sampled_params=3 pruning=0 exact=0 final=0
validation_ok selected_tasks=1
```

## Outcome

- The first research/debug entrypoint is restored for the narrow smoke path.
- No checker or sampling semantics were intentionally changed.

## Next Recommended Owner

- `integrator` if you want to continue with broader `validate.py` output shaping or multi-task smoke coverage.
