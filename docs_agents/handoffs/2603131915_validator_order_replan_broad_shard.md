# 2603131915 Order Replan Broad Shard

- Agent: `validator`
- Date: `2026-03-13 19:15`
- Status: `completed`
- Run Type: `generation_validation`

## Scope

- Goal: validate the updated var-order phases in `gallery/constrained_gen/modules/var_order_planner.py` against concrete GPU verification on a broader raw-state shard.
- Shard or indices: first `96` tasks from `load_and_register_tasks()`, up to `2` raw concrete states per task, `2` generation attempts per state.
- Reason for this run: broaden coverage beyond the earlier 2-task smoke while still keeping the run small enough to finish quickly.

## Files Checked

- `gallery/constrained_gen/modules/projected_gpu_validation.py`: confirmed current task/state -> `ScheduleGenerator` raw-state path
- `gallery/constrained_gen/modules/schedule_generator.py`: exercised current phase entries and `randomize_params(max_retries=1)`
- `gallery/constrained_gen/modules/tvm_verify.py`: exercised `params_to_state_from_state()`, `lower_with_gpu_passes()`, `verify_gpu_module_errors()`

## Commands Run

```bash
source /root/work/venv/bin/activate
export TVM_HOME=/root/work/tvm-ansor
export PYTHONPATH=$TVM_HOME/python
export TVM_LIBRARY_PATH=$TVM_HOME/build-release
python - <<'PY'
# validator shard:
# - load first 96 tasks
# - generate up to 2 raw concrete states/task
# - build symbolic state
# - create ScheduleGenerator(task=..., base_state=...)
# - run randomize_params(max_retries=1)
# - lower concretely and verify GPU constraints
# - write summary.json, details.jsonl, run.log
PY
```

## Artifacts

- Raw output dir: `/tmp/projected_gpu_full_validation/validator/order_replan_broad_shard_260313/`
- Summary file: `/tmp/projected_gpu_full_validation/validator/order_replan_broad_shard_260313/summary.json`
- Detailed file: `/tmp/projected_gpu_full_validation/validator/order_replan_broad_shard_260313/details.jsonl`
- Run log: `/tmp/projected_gpu_full_validation/validator/order_replan_broad_shard_260313/run.log`

## Result Summary

- Sketches processed: `129` raw concrete states across `96` tasks
- Main counts:
  - `attempt_rows=258`
  - `concrete_invalid_count=0`
  - `exception_count=0`
  - `execution_pure_product_states=28`
  - `execution_non_product_states=108`
  - `execution_both_states=7`
  - `execution_neither_states=0`
- Reproducer confirmed: `yes` for both execution families being encountered in the shard

## Key Findings

- The new phase order was exercised across both execution patterns:
  - pure-product execution states were observed (`28`)
  - non-product execution states were observed (`108`)
  - some states contained both execution phase types across scopes (`7`)
- All sampled generations in this broader shard lowered and passed concrete GPU verification.

## Uncertainty

- Missing evidence: this is still a shard, not a full repeated raw-state sweep across all tasks/states.
- Known limitations of this run:
  - capped to the first `96` tasks
  - capped to `2` raw states per task
  - capped to `2` generation attempts per state

## Next Owner

- Recommended owner: `reviewer`
- Recommended next step: review whether this broader shard is sufficient evidence for the new ordering semantics, or request a wider sweep.
