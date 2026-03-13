# Repeated Raw-State Validation Shard

- Owner: `validator`
- Validation mode: `single-session validation only`

## What was run

- Repeated raw-state validation for the universal generator path that uses:
  - `SketchPolicy.generate_concrete_sketches()`
  - `ScheduleGenerator(sym, task=task, base_state=state)`
  - `randomize_params(rng=..., max_retries=1)`
  - `params_to_state_from_state(...)`
  - `lower_with_gpu_passes(...)`
  - `verify_gpu_module_errors(...)`
- Selected representative tasks:
  - `task_idx=68` `vm_mod_fused_nn_dense_add`
  - `task_idx=124` `vm_mod_fused_nn_conv2d_add_clip_6`
  - `task_idx=149` `vm_mod_fused_nn_batch_matmul_1`
  - `task_idx=293` `vm_mod_fused_nn_batch_matmul_2`
  - `task_idx=320` `vm_mod_fused_nn_conv2d_add_add_3`
- Per task:
  - `10` repeated `generate_concrete_sketches()` calls
  - all returned raw states validated
  - `3` sampling/verification attempts per raw state

## Files/functions checked

- `python/tvm/auto_scheduler/search_policy.py`
- `src/auto_scheduler/search_policy/sketch_policy.cc`
- `gallery/constrained_gen/modules/schedule_generator.py`
- `gallery/constrained_gen/modules/tvm_verify.py`
- `gallery/constrained_gen/modules/param_manager.py`
- `gallery/constrained_gen/modules/record_loader.py`

## Artifacts

- Summary:
  - `/tmp/projected_gpu_full_validation/validator/raw_state_repeated_2603131120/summary.json`
- Detailed attempt log:
  - `/tmp/projected_gpu_full_validation/validator/raw_state_repeated_2603131120/details.jsonl`

## Outcome

- Repeated calls did not expose failures.
- Aggregate:
  - `total_raw_states_seen=60`
  - `total_attempts=180`
  - `randomize_fail=0`
  - `concrete_invalid=0`
  - `exceptions=0`
- Per-task diversity summary:
  - `task_idx=68`: `20` raw states across repeats, `2` unique step signatures, `2` unique sketch fingerprints
  - `task_idx=124`: `10` raw states across repeats, `1` unique step signature, `1` unique sketch fingerprint
  - `task_idx=149`: `10` raw states across repeats, `1` unique step signature, `1` unique sketch fingerprint
  - `task_idx=293`: `10` raw states across repeats, `1` unique step signature, `1` unique sketch fingerprint
  - `task_idx=320`: `10` raw states across repeats, `1` unique step signature, `1` unique sketch fingerprint
- Notable runtime:
  - full shard wall-clock `~38.2s`
  - slowest task was `task_idx=124` at `~14.1s`

## Remaining uncertainty

- This shard covered repeated-call variability on a narrow but representative task set.
- Some tasks showed only one unique returned sketch/fingerprint across `10` repeats, so broader coverage may still be useful if we want to characterize raw-state diversity more exhaustively.

## Next recommended owner

- `reviewer` to assess whether this repeated raw-state evidence is sufficient.
