## Summary

Ran a replay-coverage shard against current constrained-gen using ground-truth auto-scheduler records from:

- `gallery/logs_json/resnet_18/resnet_18-B1.json`
- `gallery/logs_json/(resnet_50,[(1,224,224,3)])/*.json`

This validation used the current task registry, mapped by `workload_key`, and **did not** use exact-symbolic as an acceptance criterion.

## Replay Method

For each matched record:

1. map `record.task.workload_key` to the current task from `load_and_register_tasks()`
2. extract params from `state.transform_steps`
   - `SplitStep.lengths` -> `sp_*`
   - `auto_unroll_max_step$...` pragma -> `ur_*`
3. build `ScheduleGenerator.from_task_state(current_task, record_state)`
4. run:
   - `check_all_pruning(params)`
   - `check_all_final(params)`
5. run `params_to_state(params)` and compare reconstructed state to original state at record-string level

## Shard Size

Two deterministic replay shards were run:

- quick shard: `per_workload_cap = 5`
- broader shard: `per_workload_cap = 20`

## Results

### ResNet-18 log

- total workloads in log: `24`
- workloads present in current task registry: `10`
- total records in log: `2408`
- records whose workload exists in current task registry: `1192`
- replayed records in quick shard: `50`
- replayed records in broader shard: `200`

Outcomes on both shards:

- constructor failures: `0`
- pruning false rejects: `0`
- final false rejects: `0`
- roundtrip mismatches: `0`

### ResNet-50 log directory

- total workloads in log set: `29`
- workloads present in current task registry: `12`
- total records in log set: `54219`
- records whose workload exists in current task registry: `22240`
- replayed records in quick shard: `60`
- replayed records in broader shard: `240`

Outcomes on both shards:

- constructor failures: `0`
- pruning false rejects: `0`
- final false rejects: `0`
- roundtrip mismatches: `0`

## Interpretation

- For the workloads that currently exist in `load_and_register_tasks()`, the replayed shard showed no evidence of:
  - symbolic-state construction failure
  - pruning false reject
  - concrete-final false reject
  - `params_to_state` roundtrip drift
- The main limitation is **task-registry overlap**, not replay correctness:
  - current task registry covers only part of the ground-truth logs

## Remaining Uncertainty

- This note covers deterministic shards, not the full overlapping record set.

## Next Recommended Owner

- `validator` or `integrator` if you want a broader replay sweep over the overlapping workload subset only.
