## Summary

- Removed the remaining temporary compatibility shims that were only bridging `ScheduleGenerator` to newer private-by-convention methods.
- Updated `ScheduleGenerator` to call the intended internal methods directly.
- Refactored `DomainPropagator` for readability so it now reads in staged sections: domain views, constraint analysis, upper-bound propagation, and candidate filtering.

## Files Changed

- `gallery/constrained_gen/modules/schedule_generator.py`
- `gallery/constrained_gen/modules/param_sampler.py`
- `gallery/constrained_gen/modules/constraint_set.py`
- `gallery/constrained_gen/modules/domain_propagator.py`
- `gallery/constrained_gen/MODULE_AND_WORKFLOW_REFERENCE.md`
- `docs_agents/CODEX_WORKING_CONTEXT.md`

## Removed Shims

- `ConstraintSet.check_innermost_split(...)`
- `ConstraintSet.check_split_structure(...)`
- `ParamSampler._try_assign_initial_fixed_vars(...)`

## Remaining Intentional Internal Helpers

- `ConstraintSet._check_innermost_split(...)`
- `ConstraintSet._check_split_structure(...)`
- `ParamSampler._assign_initial_fixed_vars(...)`
- `DomainPropagator.propagate_domain(...)`
- `DomainPropagator.filter_by_constraints(...)`
- `DomainPropagator._candidate_values_for_domain(...)`

## DomainPropagator Structure

- domain snapshots and state views:
  - `_snapshot_domains(...)`
  - `_fixed_and_remaining_from_domains(...)`
- constraint analysis:
  - `analyze_constraints_under_domains(...)`
  - `_analyze_constraint_record(...)`
  - `_analyze_constraint_bounds(...)`
  - `_enumerate_constraint_bounds(...)`
  - `_interval_bounds_from_expr_text(...)`
- upper-bound propagation:
  - `_propagate_constraint_to_var(...)`
  - `_propagate_upper_constraint_to_var(...)`
  - `_tighten_upper_domain_from_candidates(...)`
  - `_tighten_upper_domain_by_interval(...)`
  - `_propagate_lower_constraint_to_var(...)`
- candidate filtering / bisection:
  - `_candidate_values_for_domain(...)`
  - `filter_by_constraints(...)`
  - `_partition_constraints(...)`
  - `_bisect_upper(...)`
  - `_bisect_lower(...)`

## Verification

- `python -m py_compile gallery/constrained_gen/modules/schedule_generator.py gallery/constrained_gen/modules/param_sampler.py gallery/constrained_gen/modules/constraint_set.py gallery/constrained_gen/modules/domain_propagator.py`

## Outcome

- No propagation or checker semantics were intentionally changed.
- The read path through propagation and partial-domain analysis is easier to scan without introducing any new module layer.

## Next Recommended Owner

- `integrator` for any final same-file cleanup pass on the remaining direct internal helper touchpoints in `ScheduleGenerator`.
