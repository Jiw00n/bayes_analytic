## Summary

- Updated `gallery/constrained_gen/generate_programs.py` for the second pass.
- Added exactly one selector among:
  - `--task-index`
  - `--workload-key`
  - `--all`
- Added optional `--workers` while keeping task-level parallelism only.

## Files Changed

- `gallery/constrained_gen/generate_programs.py`

## Worker Model

- `--workers` does **not** share generator state across tasks.
- The parent process selects tasks, then launches one child `generate_programs.py --task-index ...` subprocess per task up to the worker limit.
- Each child runs the normal single-task flow:
  - concrete sketch generation
  - `ScheduleGenerator.from_task_state(...)`
  - parameter sampling
  - checker gates
  - `params_to_state(...)`
  - `auto_scheduler.save_records(...)`
- The parent aggregates child results and prints one final summary.

This replaced two rejected implementations encountered during verification:

- thread-level parallel TVM work failed with `parallel_for` reentrancy in exact checking
- `ProcessPoolExecutor` failed in this sandbox with `PermissionError: [Errno 13]` from `SemLock`

The current subprocess-per-task model avoids both issues while preserving task-level parallelism only.

## Behavior Changes

- `--all` now selects the full registered task set
- `--workers` defaults to `1`
- when `--workers > 1`, task execution runs concurrently through child subprocesses
- structured JSON failures are preserved
- concise per-task progress lines are preserved
- final summary remains:
  - selected task count
  - success count
  - failure count
  - failure stage histogram when failures occur

## Verification

- `python -m py_compile gallery/constrained_gen/generate_programs.py`
- `python gallery/constrained_gen/generate_programs.py --task-index 0`

Observed CLI smoke output:

```text
[task 0] start vm_mod_fused_nn_adaptive_avg_pool3d
[task 0] OK sketch=0 params=3 saved=/root/work/tvm-ansor/gallery/dataset/to_measure_gen_programs/([1aa729c96f4afc0cf6bf84dff07364c6,[1,18,9,1,512],[1,1,1,1,512]],cuda).json
generate_programs_summary selected_tasks=1 successes=1 failures=0
```

Small multi-task worker smoke:

- There are `849` registered tasks and `0` duplicate workload keys, so there is no cheap CLI multi-task smoke via `--workload-key`.
- Running `--all` would be materially broader than a narrow verification shard.
- Instead, I ran a narrow internal two-task smoke through `_run_selected_tasks(..., workers=2)` to verify the worker path without expanding the public CLI surface.

Observed worker smoke output:

```text
[task 0] start vm_mod_fused_nn_adaptive_avg_pool3d
[task 0] OK sketch=0 params=3 saved=/root/work/tvm-ansor/gallery/dataset/to_measure_gen_programs/([1aa729c96f4afc0cf6bf84dff07364c6,[1,18,9,1,512],[1,1,1,1,512]],cuda).json
[task 1] start vm_mod_fused_nn_adaptive_avg_pool2d_2
[task 1] OK sketch=0 params=3 saved=/root/work/tvm-ansor/gallery/dataset/to_measure_gen_programs/([25252ef28760d56401943904a46661f3,[1,16,16,480],[1,1,1,480]],cuda).json
{'selected_tasks': 2, 'successes': 2, 'failures': 0, 'stages': ['ok', 'ok']}
```

## Remaining Uncertainty

- `--all --workers N` has not been run end-to-end in this pass because that would launch generation across all 849 tasks.
- The child-process model relies on invoking the same script with `--task-index` and a hidden internal result-emission flag; this is intentionally internal and not part of the public CLI contract.

## Next Recommended Owner

- `integrator` if the next pass needs broader generation policy, output dedupe policy, or a dedicated cheap multi-task validation mode.
