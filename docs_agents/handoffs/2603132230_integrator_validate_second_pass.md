## Summary

- Updated `gallery/constrained_gen/validate.py` from a single-task smoke script into the main narrow validation harness for the cleaned-up workflow.
- Kept the CLI unchanged with exactly one selector among:
  - `--task-index`
  - `--workload-key`
  - `--all`
- Added multi-task continuation and a research-triage final summary.

## Files Changed

- `gallery/constrained_gen/validate.py`

## Behavior Changes

- per-task execution now returns structured internal results instead of only booleans
- `--all` already continues across tasks and now ends with:
  - selected task count
  - success count
  - failure count
  - failure stage histogram
- zero-sketch tasks are now classified clearly as:
  - `stage = "zero_sketches"`
- structured JSON failure payloads are preserved
- concise task-status lines are printed for both success and failure cases

## Verification

- `python -m py_compile gallery/constrained_gen/validate.py`
- `python gallery/constrained_gen/validate.py --task-index 0`

Observed smoke output:

```text
[task 0] OK vm_mod_fused_nn_adaptive_avg_pool3d sketches=1 selected_sketch=0 phase_count=7
  prefix phase=grid_0__execution_non_product_direct_arm params=0 remaining_domains=2 leftover_constraints=0 resolved_false=0
  sampled_params=3 pruning=0 exact=0 final=0
validation_summary selected_tasks=1 successes=1 failures=0
validation_ok
```

## Why `--all` Was Not Run In Verification

- `validate.py` still has no built-in cheap subset mode for `--all`
- `--all` would force concrete-sketch generation for every registered task
- that is materially broader than the requested narrow smoke verification

So verification stopped at the requested single-task smoke rather than performing a broad multi-task run.

## Example Final Summary

Success case:

- `validation_summary selected_tasks=1 successes=1 failures=0`

Failure case shape:

- `validation_summary selected_tasks=N successes=M failures=K`
- followed by:
  - `failure_stage_histogram`
  - one `stage=count` line per failing stage

## Next Recommended Owner

- `integrator` if you want a dedicated cheap-subset mode for `--all` in a later entrypoint pass.
