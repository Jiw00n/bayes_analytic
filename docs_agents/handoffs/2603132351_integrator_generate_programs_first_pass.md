## Summary

- Implemented the first pass of `gallery/constrained_gen/generate_programs.py`.
- Kept the CLI minimal with exactly one selector:
  - `--task-index`
  - `--workload-key`
- Reused the active workflow already present in the repo:
  - `SketchPolicy(...).generate_concrete_sketches()`
  - `ScheduleGenerator.from_task_state(...)`
  - `ScheduleGenerator.randomize_params()`
  - `ScheduleGenerator.params_to_state(...)`
  - `auto_scheduler.save_records(...)`

## Files Changed

- `gallery/constrained_gen/generate_programs.py`

## Behavior

- For each selected task, the entrypoint:
  - generates concrete sketches in order
  - tries sketches sequentially
  - stops after the first sketch that produces a clean checker pass and a saved record
- The generated state must pass:
  - `check_all_pruning(...)`
  - `check_all_exact(...)`
  - `check_all_final(...)`
- Failures are emitted as structured JSON with stable fields:
  - `task_index`
  - `workload_key`
  - `stage`
  - `sketch_index`
  - `error`
- The final summary prints:
  - selected task count
  - success count
  - failure count
  - failure stage histogram when failures occur

## Output Path Contract

- Records are appended to the constrained-gen output path from `modules/common.py`:
  - `get_to_measure_gen_filename(task)`
- Current path pattern:
  - `/root/work/tvm-ansor/gallery/dataset/to_measure_gen_programs/{clean_name((task.workload_key, target.kind))}.json`

This uses the existing `auto_scheduler.save_records(...)` path rather than inventing a second record-writing flow.

## Verification

- `python -m py_compile gallery/constrained_gen/generate_programs.py`
- `python gallery/constrained_gen/generate_programs.py --task-index 0`

Observed smoke output:

```text
[task 0] start vm_mod_fused_nn_adaptive_avg_pool3d
[task 0] OK sketch=0 params=3 saved=/root/work/tvm-ansor/gallery/dataset/to_measure_gen_programs/([1aa729c96f4afc0cf6bf84dff07364c6,[1,18,9,1,512],[1,1,1,1,512]],cuda).json
generate_programs_summary selected_tasks=1 successes=1 failures=0
```

## Remaining Uncertainty

- `auto_scheduler.save_records(...)` appends to the per-task JSON file, so repeated runs for the same selected task will append additional generated records.
- This first pass intentionally stops after the first valid saved record per selected task; it does not yet search for multiple valid outputs or attempt any broader generation strategy.

## Next Recommended Owner

- `integrator` for any later pass that broadens generation count, adds `--all`, or coordinates generation output deduplication policy.
