## Summary

- Normalized the research/debug-facing structured outputs for the cleaned-up `ScheduleGenerator` facade.
- Reduced `ScheduleGenerator` assignment-report clutter by consolidating shared nested report assembly behind `_materialize_assignment_state(...)` and `_build_observability_report(...)`.
- Aligned phase payloads between `get_full_var_order_entries()` and `randomize_params_prefix()` so they use the same phase field names.

## Files Changed

- `gallery/constrained_gen/modules/schedule_generator.py`
- `gallery/constrained_gen/modules/param_sampler.py`
- `gallery/constrained_gen/modules/var_order_planner.py`
- `gallery/constrained_gen/modules/domain_propagator.py`
- `gallery/constrained_gen/MODULE_AND_WORKFLOW_REFERENCE.md`
- `docs_agents/CODEX_WORKING_CONTEXT.md`

## Normalized Research-Facing Schemas

### `get_full_var_order_entries()`

Returns:

- `phase_count`
- `param_order`
- `phases`

Each phase entry now uses:

- `phase_name`
- `phase_family`
- `phase_label`
- `phase_index`
- `grid_scope`
- `grid_scope_label`
- `param_names`
- `param_entries`
- `param_count`
- `param_start`
- `param_stop`
- `prefix_param_names`

Each param entry now uses:

- `param_name`
- `param_kind`
- split-specific:
  - `split_step_idx`
  - `split_position`
  - `split_extent`
  - `split_group_param_names`
  - `collapsed_factor_param_names`
  - `is_innermost`
- unroll-specific:
  - `unroll_step_idx`
  - `candidate_values`

### Assignment-style reports

Used by:

- `get_param_candidates(...)`
- `propagate_param_assignment(...)`
- `get_constraints_under_assignment(...)`
- `randomize_params_prefix(...)`

Shared nested fields:

- `assignment`
  - `{params}`
- `domains`
  - `{all, fixed, remaining}`
- `constraints`
  - `{text, leftover, resolved_false, resolved_true_count}`

Method-specific fields:

- `get_param_candidates(...)`
  - `query = {param_name, requested_params}`
  - `candidates`
- `propagate_param_assignment(...)`
  - `query = {param_name, param_value, requested_params}`
- `get_constraints_under_assignment(...)`
  - `query = {requested_params, include_vars, include_eval}`
- `randomize_params_prefix(...)`
  - `query = {requested_stop_after_phase}`
  - `phase_selection = {resolved_phase_name, resolved_phase_family, resolved_phase_index}`
  - `param_order`
  - `phases`

### Constraint-analysis items

The nested `constraints.leftover` and `constraints.resolved_false` items now use:

- `constraint_kind`
- `constraint_text`
- `param_names` when unresolved params remain
- `domains` when unresolved param domains remain

## Verification

- `python -m py_compile gallery/constrained_gen/modules/schedule_generator.py gallery/constrained_gen/modules/param_sampler.py gallery/constrained_gen/modules/var_order_planner.py gallery/constrained_gen/modules/domain_propagator.py`

## Outcome

- No propagation, checking, or sampling semantics were intentionally changed.
- The cleaned-up core modules now expose one clearer structured observability contract for research/debug workflows.

## Next Recommended Owner

- `integrator` for any final pass that wants to trim or standardize the remaining diagnostics-only string helpers under `ScheduleGeneratorInspector`.
