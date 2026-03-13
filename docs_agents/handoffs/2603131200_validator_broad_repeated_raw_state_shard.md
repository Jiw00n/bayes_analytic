# Broad Repeated Raw-State Validation Shard

- Owner: `validator`

## What was run

- Broad repeated raw-state shard after the structural representative state refactor.
- Selected `26` tasks across representative families:
  - `conv2d`
  - `dense`
  - `batch_matmul`
  - `pool`
  - `misc` fused ops
- For each selected task:
  - reused one `SketchPolicy`
  - called `generate_concrete_sketches()` `6` times
  - validated every returned raw state
  - ran `2` `randomize_params(max_retries=1)` attempts per raw state
  - reconstructed with `params_to_state_from_state(...)`
  - lowered with `lower_with_gpu_passes(...)`
  - checked `verify_gpu_module_errors(...)`

## Files/functions checked

- `gallery/constrained_gen/modules/param_manager.py`
- `gallery/constrained_gen/modules/projected_gpu_validation.py`
- `gallery/constrained_gen/modules/schedule_generator.py`
- `gallery/constrained_gen/modules/param_sampler.py`
- `gallery/constrained_gen/modules/tvm_verify.py`

## Artifacts

- Summary:
  - `/tmp/projected_gpu_full_validation/validator/raw_state_repeated_broad_2603131159/summary.json`
- Detailed per-attempt rows:
  - `/tmp/projected_gpu_full_validation/validator/raw_state_repeated_broad_2603131159/details.jsonl`

## Outcome

- `selected_task_count=26`
- `repeat_calls=156`
- `raw_states_seen=222`
- `randomize_success=444`
- `concrete_valid=444`
- `concrete_invalid=0`
- `exceptions=0`
- `elapsed_sec=99.444`

- No repeated raw-state failures were found in this broad shard.
- Diversity was observed and recorded per task via:
  - unique sketch fingerprint count
  - unique step signature count

## Remaining uncertainty

- This is a broad shard, not a full-task sweep.
- A reviewer should confirm whether this evidence is sufficient or whether a still wider raw-state shard is warranted.

## Next recommended owner

- `reviewer`
