# generate_programs hot-path profile

## Goal

Profile why `python gallery/constrained_gen/generate_programs.py --all --workers 16 --records-per-task 4000`
appears to stall after task 83.

## Files checked

- `gallery/constrained_gen/generate_programs.py`
- `gallery/constrained_gen/modules/schedule_generator.py`
- `gallery/constrained_gen/modules/param_sampler.py`
- `gallery/constrained_gen/modules/task_paths.py`

## Representative tasks

- `83` `vm_mod_fused_nn_avg_pool2d_4`
- `84` `vm_mod_fused_nn_conv2d_add_add_clip_divide_multiply_6`
- `85` `vm_mod_fused_nn_conv2d_add_clip_9`
- `86` `vm_mod_fused_nn_conv2d_add_add_clip_divide_multiply_4`

## Measured breakdown

Single-sketch, single-process, warm-loop generation timing:

- task 83
  - `generate_concrete_sketches`: `0.0010 s`
  - `construct_schedule_generator`: `0.1962 s`
  - one `randomize_params`: `0.00679 s`
  - one `check_all_final`: `~0.00001 s`
  - one `params_to_state`: `0.00020 s`
  - 20-record repeat avg: `0.00676 s / record`

- task 84
  - `generate_concrete_sketches`: `0.0067 s`
  - `construct_schedule_generator`: `0.8857 s`
  - one `randomize_params`: `0.1111 s`
  - one `check_all_final`: `0.00012 s`
  - one `params_to_state`: `0.00029 s`
  - 20-record repeat avg: `0.1093 s / record`

- task 85
  - `construct_schedule_generator`: `0.9835 s`
  - 20-record repeat avg: `0.1105 s / record`

- task 86
  - `construct_schedule_generator`: `0.8846 s`
  - 20-record repeat avg: `0.1114 s / record`

Batch save cost for task 84 with 200 records:

- generation loop total: `21.2343 s`
- generation loop avg: `0.10617 s / record`
- `save_records_batch`: `0.00548 s`
- save cost per record: `0.000027 s`

## Interpretation

The slowdown after task 83 is real, but it is not a hang and not an exact-symbolic bottleneck.

- Task 83 is easy: about `0.0068 s / record`
- Tasks 84-86 are much heavier: about `0.11 s / record`
- That is about `16x` slower per record than task 83

Projected runtime for `--records-per-task 4000`:

- task 83: about `27 s`
- task 84/85/86: about `440 s` (`~7.3 min`) per task, plus about `0.9-1.0 s` generator construction

So once the run reaches the heavier conv-heavy region after task 83, progress naturally becomes much slower.

## Why it looks worse than it is

`generate_programs.py` makes this hard to see:

- workers run as subprocesses with `stdout=PIPE`, so parent output is buffered until each worker finishes
- with large `records_per_task`, a worker can run for minutes before the parent prints anything
- records are saved only after the full batch for that task is collected

## Additional checks

- output path collisions are not the cause
  - checked all tasks
  - duplicate generated output paths: `0`
- direct spot run:
  - `python gallery/constrained_gen/generate_programs.py --task-index 83 --records-per-task 5`
  - succeeds quickly

## Outcome

The primary bottleneck for large-batch generation is:

- repeated `randomize_params()` on heavier tasks
- secondarily `ScheduleGenerator.from_task_state(...)` construction

It is **not**:

- `check_all_final()`
- `params_to_state()`
- `save_records_batch()`
- output path collisions

## Next recommendation

- If the goal is usability/observability:
  - stream worker progress to parent
  - flush/save in chunks
  - print periodic per-task progress/ETA
- If the goal is throughput:
  - optimize `randomize_params()` hot path on heavy tasks
  - or reduce `records_per_task` / shard by workload difficulty first
