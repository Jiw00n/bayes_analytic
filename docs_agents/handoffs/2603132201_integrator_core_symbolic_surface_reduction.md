## Summary

- Reduced the visible `ConstraintSet` surface so the main readable entry is `preprocess()`, `check_all_pruning()`, and `check_all_exact()`.
- Split `VarOrderPlanner.compute_var_order()` into explicit phase-first ordering and legacy-fallback merge stages.
- Split `ParamSampler` into clearer internal stages for initialization, per-variable sampling, final validation, and prefix-report assembly.

## Files Changed

- `gallery/constrained_gen/modules/constraint_set.py`
- `gallery/constrained_gen/modules/var_order_planner.py`
- `gallery/constrained_gen/modules/param_sampler.py`
- `gallery/constrained_gen/MODULE_AND_WORKFLOW_REFERENCE.md`
- `docs_agents/CODEX_WORKING_CONTEXT.md`

## ConstraintSet Surface

Moved to private-by-convention:

- `_build_vectorize_constraints`
- `_build_shared_memory_constraints`
- `_build_max_threads_constraints`
- `_build_max_vthread_constraints`
- `_build_innermost_split_constraints`
- `_build_split_structure_constraints`
- `_check_vectorize`
- `_check_vectorize_exact`
- `_check_shared_memory`
- `_check_shared_memory_exact`
- `_check_max_threads`
- `_check_max_threads_exact`
- `_check_max_vthread`
- `_check_max_vthread_exact`
- `_check_innermost_split`
- `_check_split_structure`

Kept as direct-call compatibility shims because a current caller still needs them:

- `check_innermost_split`
- `check_split_structure`

## VarOrderPlanner Structure

- `compute_var_order()` now reads as:
  - build phase entries
  - normalize phase-first order
  - append legacy fallback vars
- removed the unused `get_var_order_prefix(...)`
- removed the unused `_build_split_structure_phase_assignments(...)`

## ParamSampler Structure

New internal stage helpers:

- `_assign_initial_fixed_vars(...)`
- `_build_split_domains(...)`
- `_initialize_split_sampling_state(...)`
- `_sample_split_var(...)`
- `_assign_unroll_vars(...)`
- `_validate_sample(...)`
- `_build_prefix_report(...)`

Additional cleanup:

- kept `_try_assign_initial_fixed_vars(...)` as a compatibility alias for the current `ScheduleGenerator` caller
- renamed `enumerate_all_params(...)` to `_enumerate_all_params(...)` because it has no active caller

## Verification

- `python -m py_compile gallery/constrained_gen/modules/constraint_set.py gallery/constrained_gen/modules/var_order_planner.py gallery/constrained_gen/modules/param_sampler.py`

## Outcome

- No checker or sampling semantics were intentionally changed.
- The active-path files now read more clearly as staged workflows rather than large mixed method inventories.

## Next Recommended Owner

- `integrator` for any further same-file cleanup passes on `domain_propagator.py` or the remaining cross-module helper touchpoints.
