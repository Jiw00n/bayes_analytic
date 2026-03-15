## Task

Reduce redundant package exports and thin wrapper surface in `gallery/constrained_gen`, while keeping the current `generate_concrete_sketches() -> build_symbolic_state()` workflow intact.

## Files Checked

- `gallery/constrained_gen/modules/__init__.py`
- `gallery/constrained_gen/modules/record_loader.py`
- `gallery/constrained_gen/modules/projected_gpu_validation.py`
- `gallery/constrained_gen/modules/schedule_generator.py`
- `gallery/constrained_gen/MODULE_AND_WORKFLOW_REFERENCE.md`
- `gallery/constrained_gen/MODULE_AND_WORKFLOW_REFERENCE.ko.md`
- `docs_agents/CODEX_WORKING_CONTEXT.md`

## Changes

- Kept `modules/__init__.py` as a minimal package surface:
  - `build_symbolic_state`
  - `ScheduleGenerator`
- Removed unused `ScheduleGenerator` forwarding/alias methods with no current code callers:
  - `check_all`
  - `_compute_var_order`
  - `_resolve_var_order_stop_index`
  - `_randomize_params_with_order`
  - `_snapshot_domains`
  - `_fixed_and_remaining_from_domains`
  - `analyze_constraints_under_domains`
  - `_analyze_constraint_bounds`
  - `_apply_upper_bound_to_domain`
  - `_get_sym_value`
  - `_propagate_domain`
  - `_filter_by_constraints`
- Removed `record_loader.build_schedule_generator_from_state()` as an unused thin wrapper.
- Stopped `projected_gpu_validation.py` from re-exporting record/sketch loader and builder helpers; it now stays on the diagnostics side.
- Refreshed the module/workflow reference docs and working-context note so they reflect the reduced surface.

## Verification

- Ran syntax check:
  - `python -m py_compile gallery/constrained_gen/modules/__init__.py gallery/constrained_gen/modules/record_loader.py gallery/constrained_gen/modules/projected_gpu_validation.py gallery/constrained_gen/modules/schedule_generator.py`
- Searched for removed wrapper names in the current code tree to confirm no remaining code callers.

## Outcome

- Package-level imports are now intentionally small.
- Record/sketch reconstruction lives in `record_loader.py`.
- `projected_gpu_validation.py` no longer acts as a second import surface for loader/builder helpers.
- `ScheduleGenerator` lost a batch of unused forwarding methods and is a smaller facade.

## Remaining Uncertainty

- Some current project/agent docs outside the touched files still mention older validation entrypoints and `projected_gpu_validation.py` ownership patterns.
- No runtime validation shard was run; this was a surface-cleanup pass only.

## Recommended Next Owner

- `integrator` for a follow-up pass on the remaining constraint-checker forwarding methods in `schedule_generator.py` if you want the facade reduced even further.
