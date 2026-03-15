## Task

Consolidate record/sketch ingestion helpers into a single module now that the active workflow is direct `generate_concrete_sketches() -> build_symbolic_state()` rather than record grouping.

## Files Checked

- `gallery/constrained_gen/modules/record_loader.py`
- `gallery/constrained_gen/modules/projected_gpu_validation.py`
- `gallery/constrained_gen/modules/param_manager.py`
- `gallery/constrained_gen/modules/schedule_generator.py`
- `gallery/constrained_gen/order_test.ipynb`

## Changes

- Moved record/sketch ingestion helpers into `gallery/constrained_gen/modules/record_loader.py`:
  - `load_tasks_by_workload`
  - `load_sketch_lines`
  - `load_sketch_record`
  - `build_schedule_generator`
  - `build_schedule_generator_from_state`
- Reduced `gallery/constrained_gen/modules/projected_gpu_validation.py` to diagnostics plus imports from `record_loader.py`.

## Verification

- Ran syntax check:
  - `python -m py_compile gallery/constrained_gen/modules/record_loader.py gallery/constrained_gen/modules/projected_gpu_validation.py`

## Outcome

- The `all_sketches.json` / recovered-record / state-to-generator helper path now has one owning module: `record_loader.py`.
- `projected_gpu_validation.py` no longer owns loading/building helpers and is closer to a pure diagnostics module.

## Remaining Uncertainty

- No runtime validation was run.
- Dead-code removal for the record/grouping path is still pending; this change only centralizes ownership.

## Recommended Next Owner

- `integrator` for the next cleanup step: remove unused exports/imports and decide whether `record_loader.py` itself should be reduced to a legacy module or retired from the active workflow.
