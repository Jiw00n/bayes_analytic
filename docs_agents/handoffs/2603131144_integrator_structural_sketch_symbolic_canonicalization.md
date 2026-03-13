# Structural Sketch Symbolic Canonicalization

- Owner: `integrator`
- Validation mode: `single-session validation only`

## What changed

- Patched [transform_applier.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/transform_applier.py) in two narrow spots that were leaking concrete init-population values into `SymbolicState` recovery for the same structural sketch.
  - `_recover_iter_extent(...)` no longer returns concrete `1` for `compute_at` stages before attempting symbolic recovery.
  - `_get_safe_saved_extent(...)` now preserves saved symbolic extents instead of filtering them by the current concrete `real_ext`.
  - `_match_compute_at_inner_extent(...)` now requires base iterator-name agreement and takes `iter_id`, so it cannot attach an unrelated inner symbolic extent to another loop just because the current concrete extent happened to match.

## Root cause

- `build_symbolic_state()` itself is simple: [param_manager.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/param_manager.py) just creates `SymbolicState` and runs `TransformApplier.apply_steps(state)`.
- The non-canonical behavior came from [transform_applier.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/transform_applier.py) recovery helpers for stages whose extents were cleared by `ComputeAtStep` / `CacheReadStep`.
- The concrete-dependent paths were:
  - early `real_ext == 1` short-circuit in `_recover_iter_extent(...)`
  - `saved_eval <= real_ext` gate in `_get_safe_saved_extent(...)`
  - name-agnostic `eval(extent) == real_ext` matching in `_match_compute_at_inner_extent(...)`
- These let the same structural sketch reconstruct different symbolic extents depending on whether a random init-population choice happened to evaluate some symbolic loop to `1` or to the same concrete extent as an unrelated loop.

## Outcome

- Repeated `generate_concrete_sketches()` runs now produced one `SymbolicState` signature per structural sketch fingerprint on the two confirmed repro tasks:
  - `task_idx=68` `vm_mod_fused_nn_dense_add`
  - `task_idx=124` `vm_mod_fused_nn_conv2d_add_clip_6`
- `ScheduleGenerator` signatures remained one-per-fingerprint as before.
- Local smoke also kept raw-state sampling clean on those tasks.

## Artifacts

- Local canonicalization + sampling smoke:
  - `/tmp/projected_gpu_full_validation/integrator/canonicalization_patch_smoke_260313/summary.json`
  - `/tmp/projected_gpu_full_validation/integrator/canonicalization_patch_smoke_260313/details.jsonl`

## Remaining uncertainty

- This turn only did local reproducer validation after the patch.
- A validator-owned confirmation shard was requested but was not yet available at note time.
- The patch intentionally narrows compute-at recovery matching; broader coverage across more tasks should still be checked.

## Next recommended owner

- `validator` for a narrow repeated raw-state shard confirming unique symbolic signatures per sketch fingerprint and clean sampling on the patched tasks.
