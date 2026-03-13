# Raw-State Narrow Validation Shard

- Owner: `validator`

## What was run

- Executed a narrow raw-state validation shard against the new universal generator path using a Python one-off command from `gallery/constrained_gen`.
- For each selected task:
  - constructed `SketchPolicy(task, RandomModel())`
  - called `generate_concrete_sketches()`
  - for each returned raw `State`, built a generator via `build_schedule_generator_from_state(task, state)`
  - ran `randomize_params(rng=..., max_retries=1)`
  - reconstructed a concrete state with `params_to_state_from_state(...)`
  - lowered with GPU passes and ran `verify_gpu_module_errors(...)`
- Configuration:
  - task indices: `0, 68, 124, 149, 293, 320`
  - rounds per task: `4`
  - attempts per raw state: `3`

## Files/functions checked

- `gallery/constrained_gen/modules/projected_gpu_validation.py`
  - `build_schedule_generator_from_state`
- `gallery/constrained_gen/modules/schedule_generator.py`
  - `has_concrete_final_context`
  - `get_concrete_final_result`
- `gallery/constrained_gen/modules/tvm_verify.py`
  - `params_to_state_from_state`
  - `lower_with_gpu_passes`
  - `verify_gpu_module_errors`
- `python/tvm/auto_scheduler/search_policy.py`
  - `generate_concrete_sketches`

## Artifacts

- Summary:
  - `/tmp/projected_gpu_full_validation/validator/raw_state_narrow_2603131112/summary.json`
- Per-attempt details:
  - `/tmp/projected_gpu_full_validation/validator/raw_state_narrow_2603131112/details.jsonl`

## Concrete outcome

- Overall:
  - tasks: `6`
  - raw states tested: `28`
  - randomize success: `84`
  - randomize fail: `0`
  - concrete invalid: `0`
  - elapsed: `35.64s`
- Covered the previously problematic task:
  - `task_idx=124`
  - `task_desc=vm_mod_fused_nn_conv2d_add_clip_6`
  - raw states tested: `4`
  - randomize success: `12`
  - concrete invalid: `0`
- Additional task coverage:
  - `task_idx=0` `vm_mod_fused_nn_adaptive_avg_pool3d`
  - `task_idx=68` `vm_mod_fused_nn_dense_add`
  - `task_idx=149` `vm_mod_fused_nn_batch_matmul_1`
  - `task_idx=293` `vm_mod_fused_nn_batch_matmul_2`
  - `task_idx=320` `vm_mod_fused_nn_conv2d_add_add_3`

## Remaining uncertainty

- This is a narrow shard, not a full sweep across all tasks and all random raw states.
- It validates the new raw-state concrete-final path on several varied tasks, including the previously problematic case, but does not yet establish global coverage.

## Next recommended owner

- `reviewer` to assess whether this evidence is sufficient or whether a broader raw-state shard is needed.
