## Summary

- Renamed the internal `ScheduleGenerator` explainability owner from `_ScheduleGeneratorExplainability` to `_ScheduleGeneratorInspector`.
- Removed the dead `record_loader.build_schedule_generator(...)` wrapper so `record_loader.py` stays focused on legacy record/sketch I/O and sketch fingerprint helpers.
- Sharpened `common.py` ownership around path/task-loading helpers, with the remaining Relay/torch/string helpers clearly marked as legacy shared utilities.

## Files Checked

- `gallery/constrained_gen/modules/schedule_generator.py`
- `gallery/constrained_gen/modules/record_loader.py`
- `gallery/constrained_gen/modules/common.py`
- `gallery/constrained_gen/MODULE_AND_WORKFLOW_REFERENCE.md`
- `docs_agents/CODEX_WORKING_CONTEXT.md`

## Concrete Changes

- `schedule_generator.py`
  - renamed the internal helper class to `_ScheduleGeneratorInspector`
  - renamed the facade field from `self._explain` to `self._inspector`
  - updated all bridge/helper call sites and static self-references
- `record_loader.py`
  - tightened the module docstring around legacy record/sketch I/O ownership
  - kept sketch fingerprint helpers plus legacy sketch-dump loading helpers
  - removed `build_schedule_generator(...)` because no current Python callers remained
- `common.py`
  - added a module docstring that distinguishes active path/task-loading ownership from legacy shared utilities
  - introduced `_register_task_workloads(tasks)` to deduplicate TVM workload registration
  - clarified sections so path/task-loading helpers are separated from misc legacy helpers

## Artifacts

- No external artifact directory for this cleanup-only pass.

## Outcome

- Active workflow ownership is easier to scan:
  - `ScheduleGenerator` keeps the facade path
  - `record_loader.py` is explicit legacy record/sketch I/O
  - `common.py` reads as path/task-loading first, legacy utility tail second

## Remaining Uncertainty

- `record_loader.py` still exposes sketch fingerprint helpers because `gallery/measure_programs.py` imports them directly.
- `common.py` still contains a few non-core helpers (`convert_to_nhwc`, `dtype2torch`, `str2bool`) because they have adjacent script callers or no clear replacement owner yet.

## Next Recommended Owner

- `integrator` if another cleanup pass wants to relocate the remaining sketch fingerprint helpers or trim the legacy utility tail further.
