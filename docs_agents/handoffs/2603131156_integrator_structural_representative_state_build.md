# Structural Representative State Build

- Owner: `integrator`
- Validation mode: `single-session validation only`

## What changed

- Removed the structural-sketch template cache approach from [param_manager.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/param_manager.py).
- Added [structural_sketch.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/structural_sketch.py) with:
  - `build_canonical_param_values(state, split_value=1, unroll_value=0)`
  - `build_canonical_state(task, state, split_value=1, unroll_value=0)`
- `build_symbolic_state(...)` now requires `task` and builds the symbolic state from a canonical representative concrete state for the structural sketch, instead of from the incoming random concrete state.
- Updated constrained-gen call sites to pass `task` into `build_symbolic_state(...)`.

## Why this is more fundamental

- The previous variability came from building symbolic states directly from arbitrary concrete init-population states.
- `TransformApplier` uses replay/infer-bound paths that can depend on those concrete values.
- The new flow changes the input model:
  - raw concrete state
  - structural sketch
  - canonical representative state
  - symbolic state
- This removes dependence on the original random concrete values at symbolic-build time.

## Files/functions checked

- `gallery/constrained_gen/modules/structural_sketch.py`
- `gallery/constrained_gen/modules/param_manager.py`
- `gallery/constrained_gen/modules/projected_gpu_validation.py`
- `gallery/constrained_gen/generate.py`
- `gallery/constrained_gen/example_generate_single_sketch.py`
- `gallery/constrained_gen/profile_schedule_generator_timing.py`
- `gallery/constrained_gen/audit_non_pruning_correctness.py`

## Validation

- Signature stability after the refactor:
  - `task_idx=68`
    - both structural sketch fingerprints had `num_sym_signatures=1`, `num_gen_signatures=1`
  - `task_idx=124`
    - the structural sketch fingerprint had `num_sym_signatures=1`, `num_gen_signatures=1`
- Raw-state smoke still passed:
  - `task_idx=68`: `validated_states=10`
  - `task_idx=124`: `validated_states=5`
- Existing record-based example still passed:
  - `/tmp/projected_gpu_full_validation/example_generate_single_sketch/sketch_2_summary.json`

## Artifacts

- Prior signature/result bundle:
  - `/tmp/projected_gpu_full_validation/integrator/structural_sketch_canonicalization_260313/summary.json`

## Remaining uncertainty

- `TransformApplier` still contains concrete-value-aware replay helpers internally.
- The important change is that `build_symbolic_state(...)` no longer feeds arbitrary random concrete states into those helpers.
- A broader validator-owned repeated raw-state shard would still be useful.

## Next recommended owner

- `validator` for a wider repeated raw-state shard if broader confidence is needed.
