# Raw State Concrete-Final Support

- Owner: `integrator`
- Validation mode: `single-session validation only`

## What changed

- Added raw-state patching support in [tvm_verify.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/tvm_verify.py).
  - Extracted record-step patching into `_patch_record_steps(...)`.
  - Added `params_to_state_from_record(...)`.
  - Added `params_to_state_from_state(...)` using `MeasureInput(task, base_state)` plus a dummy `MeasureResult`.
- Extended [schedule_generator.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/schedule_generator.py) so `ScheduleGenerator(..., task=..., base_state=state)` counts as concrete-final context.
  - `has_concrete_final_context()` now accepts either record context or raw `base_state`.
  - `get_concrete_final_result()` now dispatches to record-based or raw-state-based state reconstruction.
- Extended [projected_gpu_validation.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/projected_gpu_validation.py).
  - `build_schedule_generator(...)` now passes `base_state` when no record is present.
  - Added `build_schedule_generator_from_state(task, state)`.

## Files/functions checked

- `gallery/constrained_gen/modules/tvm_verify.py`
- `gallery/constrained_gen/modules/schedule_generator.py`
- `gallery/constrained_gen/modules/projected_gpu_validation.py`
- `python/tvm/auto_scheduler/search_policy.py`
- `src/auto_scheduler/search_policy/sketch_policy.cc`

## Why this was needed

- `SketchPolicy.generate_concrete_sketches()` returns random concrete `State`s, not measure records.
- The old generator only had a record-based concrete-final path, so raw concrete states fell back to `check_all_exact()`.
- Some raw sketches hit exact case explosion there and appeared to hang.

## Outcome

- Raw concrete sketches can now use concrete-final validation without measure records.
- Reproduced raw-state generation on `task_idx=124` (`vm_mod_fused_nn_conv2d_add_clip_6`) returned quickly instead of stalling on exact case-table build.
- Existing record-based example generation still passed unchanged.

## Artifacts

- Raw-state validation summary:
  - `/tmp/projected_gpu_full_validation/integrator/raw_state_concrete_final_2603131106/summary.json`
- Record-based example output:
  - `/tmp/projected_gpu_full_validation/example_generate_single_sketch/sketch_2_generated.json`
  - `/tmp/projected_gpu_full_validation/example_generate_single_sketch/sketch_2_summary.json`

## Remaining uncertainty

- This turn validated the raw-state path on a targeted reproducer and confirmed record-path regression was not observed in the example flow.
- A broader validator-owned shard for `generate_concrete_sketches()` raw states has not been run yet.

## Next recommended owner

- `validator` for a narrow raw-state generation shard across several tasks/sketches returned by `generate_concrete_sketches()`.
