# Structural Sketch Canonicalization

- Owner: `integrator`
- Validation mode: `single-session validation only`

## What changed

- Added cloning support for symbolic objects:
  - [sym_types.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/sym_types.py) `SymStage.clone()`
  - [symbolic_state.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/symbolic_state.py) `SymbolicState.clone()`
- Added structural-sketch template caching in [param_manager.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/param_manager.py) `build_symbolic_state(...)`.
  - Cache key: `(id(compute_dag), state_sketch_fingerprint(state))`
  - Cached object is a cloned symbolic-state template
  - Later builds for the same structural sketch return cloned copies of the same template
- Canonicalized symbolic parameter defaults in [symbolic_state.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/symbolic_state.py) with `canonicalize_param_values()`.
  - `sp_* -> 1`
  - `ur_* -> 0`

## Why this was needed

- Repeated `generate_concrete_sketches()` calls could produce concrete states with the same structural sketch fingerprint.
- Before this patch, `build_symbolic_state()` could still yield different symbolic-state objects for those equivalent sketches.
- `ScheduleGenerator` signatures were already matching in the observed repros, but the underlying `SymbolicState` representation was not canonical.

## Outcome

- Same structural sketch now reuses the same symbolic-state template within the process.
- Repeated raw-state checks on representative tasks now show one symbolic-state signature and one generator signature per structural sketch fingerprint.

## Files/functions checked

- `gallery/constrained_gen/modules/param_manager.py`
- `gallery/constrained_gen/modules/symbolic_state.py`
- `gallery/constrained_gen/modules/sym_types.py`
- `gallery/constrained_gen/modules/transform_applier.py`
- `src/auto_scheduler/search_policy/sketch_policy.cc`

## Validation

- Signature stability artifact:
  - `/tmp/projected_gpu_full_validation/integrator/structural_sketch_canonicalization_260313/summary.json`
- Narrow repeated check summary:
  - `task_idx=68`: two structural sketches, each with `num_sym_signatures=1`, `num_gen_signatures=1`
  - `task_idx=124`: one structural sketch, `num_sym_signatures=1`, `num_gen_signatures=1`
- Raw-state smoke after patch:
  - `task_idx=68`: `validated_states=10`
  - `task_idx=124`: `validated_states=5`
- Existing record-based example still passed:
  - `/tmp/projected_gpu_full_validation/example_generate_single_sketch/sketch_2_summary.json`

## Remaining uncertainty

- This patch enforces canonical symbolic-state reuse by structural-sketch template caching.
- It does not fully eliminate the underlying concrete-dependent reconstruction heuristics inside `TransformApplier`; it bypasses their variability at the `build_symbolic_state(...)` boundary.

## Next recommended owner

- `validator` if we want a wider repeated raw-state shard across more tasks.
- `specialist` only if we want to remove the remaining concrete-dependent variability inside `TransformApplier` itself rather than relying on canonical template reuse.
