## Task

Clarify the current `gallery/constrained_gen` surface before adding new entrypoints, and trim only dead thin wrappers that had no code callers.

## Files Checked

- `gallery/constrained_gen/generate.py`
- `gallery/constrained_gen/modules/__init__.py`
- `gallery/constrained_gen/modules/param_manager.py`
- `gallery/constrained_gen/modules/projected_gpu_validation.py`
- `gallery/constrained_gen/modules/record_loader.py`
- `gallery/constrained_gen/modules/schedule_generator.py`
- `gallery/constrained_gen/modules/tvm_verify.py`
- `gallery/measure_programs.py`

## Concrete Caller Findings

- Current package-level public surface is already minimal in `modules/__init__.py`:
  - `build_symbolic_state`
  - `ScheduleGenerator`
- Active top-level constrained-gen script in the current tree is only `gallery/constrained_gen/generate.py`.
- Cross-repo non-package callers that still keep helper modules visible:
  - `gallery/measure_programs.py` imports `modules.common.{TO_MEASURE_GEN_PROGRAM_FOLDER, load_and_register_tasks, get_measure_record_filename, get_to_measure_gen_filename}`
  - `gallery/measure_programs.py` imports `modules.record_loader.{state_sketch_fingerprint, sketch_fingerprint_hash}`
- Concrete reconstruction helpers still used by active constrained-gen code:
  - `schedule_generator.py` uses `tvm_verify.{lower_with_gpu_passes, params_to_state_from_record, params_to_state_from_state, verify_gpu_module_errors}`
  - `structural_sketch.py` uses `tvm_verify.params_to_state_from_state`
- Diagnostics module usage remains internal/diagnostic:
  - `projected_gpu_validation.py` calls `ScheduleGenerator` formatting/final-check helpers but had no code callers outside itself in the current tree.

## Changes

- Marked `record_loader.py` as a legacy record/sketch helper module instead of part of the active generator core.
- Marked `projected_gpu_validation.py` as diagnostics-only.
- Removed dead thin wrappers with no code callers:
  - `record_loader.load_tasks_by_workload()`
  - `tvm_verify.params_to_state()`
  - `tvm_verify.params_to_lowered_gpu_module()`

## Verification

- Ran syntax check:
  - `python -m py_compile gallery/constrained_gen/modules/record_loader.py gallery/constrained_gen/modules/tvm_verify.py gallery/constrained_gen/modules/projected_gpu_validation.py`
- Searched the repo for removed symbols outside docs/notebooks:
  - no remaining code callers found for `load_tasks_by_workload`
  - no remaining code callers found for `params_to_state(...)`
  - no remaining code callers found for `params_to_lowered_gpu_module(...)`

## Outcome

- The active surface is now easier to describe:
  - package surface: `build_symbolic_state`, `ScheduleGenerator`
  - active script surface: `generate.py`
  - still-visible helper surface only where another repo script currently imports it (`common.py`, record fingerprint helpers)
- Record ingestion and projected-GPU mismatch logic are now more clearly separated as legacy/diagnostic paths rather than peer public APIs.

## Remaining Uncertainty

- `ScheduleGenerator` still exposes many constraint-family forwarding methods with no current code callers. Those are the largest remaining surface area inside the active core.
- No runtime validation shard was run because this pass only trimmed dead wrappers and clarified ownership.

## Recommended Next Owner

- `integrator` for the next reduction step inside `ScheduleGenerator`: consolidate the remaining per-constraint forwarding methods into the smaller future public API.
