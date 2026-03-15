## Summary

- Performed a narrow private-surface cleanup pass in the constrained-gen helper owners.
- Kept the `ScheduleGenerator` facade surface intentionally small.
- Renamed only underscore-prefixed methods that are already acting like semantically important internal workflow stages across module boundaries.

## Files Changed

- `gallery/constrained_gen/modules/domain_propagator.py`
- `gallery/constrained_gen/modules/param_sampler.py`
- `gallery/constrained_gen/modules/var_order_planner.py`
- `gallery/constrained_gen/modules/schedule_generator.py`

## Names Promoted To Non-Underscore

- `DomainPropagator._snapshot_domains(...)` -> `DomainPropagator.snapshot_domains(...)`
  - rationale: this is a stable state-view helper used outside the class's deepest implementation path
- `DomainPropagator._candidate_values_for_domain(...)` -> `DomainPropagator.candidate_values_for_domain(...)`
  - rationale: this is a meaningful domain-expansion stage referenced by other owners, not just a hidden local detail
- `ParamSampler._randomize_params_with_order(...)` -> `ParamSampler.randomize_params_with_order(...)`
  - rationale: this is the core ordered-sampling workflow stage behind both full sampling and prefix sampling
- `VarOrderPlanner._resolve_var_order_stop_index(...)` -> `VarOrderPlanner.resolve_var_order_stop_index(...)`
  - rationale: this is the named phase-boundary resolution step used by prefix sampling, so the non-underscore name better matches its role

## Names Intentionally Kept Private

- entrypoint-script helpers in:
  - `generate_programs.py`
  - `measure_programs.py`
  - `validate.py`
  - rationale: they are file-local CLI glue and not reusable ownership boundaries
- `ScheduleGenerator` underscore helpers such as:
  - `_materialize_assignment_state(...)`
  - `_build_observability_report(...)`
  - `_get_var_order_phase_entries(...)`
  - rationale: keeping these private preserves the facade's deliberately reduced visible API
- deep constraint/checker implementation helpers in:
  - `constraint_set.py`
  - `domain_propagator.py`
  - `param_sampler.py`
  - `var_order_planner.py`
  - rationale: these remain real implementation detail rather than stable stage-owner names

## Verification

- `python -m py_compile gallery/constrained_gen/modules/schedule_generator.py gallery/constrained_gen/modules/param_sampler.py gallery/constrained_gen/modules/var_order_planner.py gallery/constrained_gen/modules/domain_propagator.py gallery/constrained_gen/generate_programs.py gallery/constrained_gen/measure_programs.py gallery/constrained_gen/validate.py gallery/constrained_gen/modules/constraint_set.py`

Outcome:

- compile passed
- no stale references remained for:
  - `_snapshot_domains`
  - `_candidate_values_for_domain`
  - `_randomize_params_with_order`
  - `_resolve_var_order_stop_index`

## Next Recommended Owner

- `integrator` if a later pass wants to normalize private naming inside `ScheduleGenerator` itself without reopening its public facade surface.
