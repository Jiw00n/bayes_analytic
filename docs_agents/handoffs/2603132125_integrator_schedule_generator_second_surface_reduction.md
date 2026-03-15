## Task

Second surface-reduction pass for `gallery/constrained_gen`, focused on shrinking `ScheduleGenerator` to workflow-level public methods.

## Files Changed

- `gallery/constrained_gen/modules/schedule_generator.py`
- `gallery/constrained_gen/modules/constraint_set.py`
- `gallery/constrained_gen/modules/domain_propagator.py`
- `gallery/constrained_gen/modules/param_sampler.py`
- `gallery/constrained_gen/modules/projected_gpu_validation.py`
- `gallery/constrained_gen/MODULE_AND_WORKFLOW_REFERENCE.md`
- `docs_agents/CODEX_WORKING_CONTEXT.md`

## Removed Public Methods

Removed per-constraint forwarding methods from `ScheduleGenerator`:

- `build_vectorize_constraints`
- `check_vectorize`
- `check_vectorize_exact`
- `build_shared_memory_constraints`
- `check_shared_memory`
- `check_shared_memory_exact`
- `build_max_threads_constraints`
- `check_max_threads`
- `check_max_threads_exact`
- `build_max_vthread_constraints`
- `check_max_vthread`
- `check_max_vthread_exact`
- `build_innermost_split_constraints`
- `build_split_structure_constraints`
- `check_innermost_split`
- `check_split_structure`

Removed extra no-caller aliases / surface methods:

- `get_var_order_entries`
- `get_var_order_prefix`
- `get_candidate_values`
- `propagate_assignment`
- `enumerate_all_params`

Internalized helper methods:

- `has_concrete_final_context` -> `_has_concrete_final_context`
- `get_concrete_final_result` -> `_get_concrete_final_result`
- `get_constraint_records` -> `_get_constraint_records`
- `get_constraints_str` -> `_get_constraints_str`
- `get_constraints_with_assignment_str` -> `_get_constraints_with_assignment_str`
- `get_structural_highlights_str` -> `_get_structural_highlights_str`
- `get_raw_exact_constraints_str` -> `_get_raw_exact_constraints_str`
- `get_var_order_phase_entries` -> `_get_var_order_phase_entries`
- `get_assignment_state` -> `_get_assignment_state`

## Internal Call-Site Updates

- `ScheduleGenerator._check_all_final_with_concrete_result(...)` now calls `ConstraintSet` directly for innermost/split-structure checks.
- `constraint_set.py` now uses `g._get_concrete_final_result(...)`.
- `domain_propagator.py` now uses `g._get_constraint_records()`.
- `param_sampler.py` now uses `g._get_var_order_phase_entries()`.
- `projected_gpu_validation.py` now uses the underscored helper names.

## Resulting Intended Public Surface

- `from_task_state`
- `get_full_var_order_entries`
- `get_param_candidates`
- `propagate_param_assignment`
- `get_constraints_under_assignment`
- `randomize_params`
- `randomize_params_prefix`
- `params_to_state`
- `check_all_pruning`
- `check_all_exact`
- `check_all_hybrid`
- `check_all_final`

## Verification

- Ran:
  - `python -m py_compile gallery/constrained_gen/modules/schedule_generator.py gallery/constrained_gen/modules/constraint_set.py gallery/constrained_gen/modules/domain_propagator.py gallery/constrained_gen/modules/param_sampler.py gallery/constrained_gen/modules/projected_gpu_validation.py gallery/constrained_gen/generate.py`

## Documentation Status

- Updated:
  - `gallery/constrained_gen/MODULE_AND_WORKFLOW_REFERENCE.md`
  - `docs_agents/CODEX_WORKING_CONTEXT.md`
- Missing from current worktree:
  - `gallery/constrained_gen/MODULE_AND_WORKFLOW_REFERENCE.ko.md`

## Remaining Uncertainty

- `MODULE_AND_WORKFLOW_REFERENCE.ko.md` was referenced in the task but is not present in the current worktree, so no Korean translation refresh was possible in this pass.

## Recommended Next Owner

- `validator` if you want a narrow post-refactor regression pass over projected diagnostics and generator prefix sampling.
