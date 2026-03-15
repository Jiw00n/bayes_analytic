## Task

Refactor `gallery/constrained_gen/modules/schedule_generator.py` to reduce utility-like clutter while keeping `ScheduleGenerator` as the public facade.

## Files Changed

- `gallery/constrained_gen/modules/schedule_generator.py`
- `gallery/constrained_gen/MODULE_AND_WORKFLOW_REFERENCE.md`
- `docs_agents/CODEX_WORKING_CONTEXT.md`

## What Stayed In `ScheduleGenerator`

Workflow/public facade methods stayed in the main class body:

- `from_task_state(...)`
- `check_all_pruning(...)`
- `check_all_exact(...)`
- `check_all_hybrid(...)`
- `check_all_final(...)`
- `get_full_var_order_entries(...)`
- `get_param_candidates(...)`
- `propagate_param_assignment(...)`
- `get_constraints_under_assignment(...)`
- `randomize_params(...)`
- `randomize_params_prefix(...)`
- `params_to_state(...)`

Stateful workflow-adjacent helpers also stayed:

- concrete-final cache/context helpers
- assignment/domain-state construction helpers
- exact/projected constraint lazy-init helpers

## What Moved / Internalized

Added one same-module internal helper owner:

- `_ScheduleGeneratorExplainability`

Moved the formatting/simplification machinery into that helper:

- constraint record rendering
- assignment-reflected constraint formatting
- expression simplification
- expression evaluation for display
- raw exact-constraint rendering
- all `_format_*`, `_simplify_*`, `_evaluate_*`, and `_flatten_*` implementation bodies

Kept only thin bridge methods on `ScheduleGenerator` for current internal callers:

- `_get_constraint_records(...)`
- `_get_constraints_str(...)`
- `_get_constraints_with_assignment_str(...)`
- `_simplify_constraint_expr_text(...)`
- `_simplify_constraint_rhs_text(...)`
- `_get_raw_exact_constraints_str(...)`

## Outcome

- The top of `ScheduleGenerator` is now much more clearly the workflow path.
- Explainability/formatting code remains in the same file and under one owner, so there is no new public API and no helper-module sprawl.
- Checker and sampling semantics were not intentionally changed.

## Verification

- Ran:
  - `python -m py_compile gallery/constrained_gen/modules/schedule_generator.py gallery/constrained_gen/modules/domain_propagator.py gallery/constrained_gen/modules/projected_gpu_validation.py gallery/constrained_gen/modules/constraint_set.py gallery/constrained_gen/modules/param_sampler.py`

## Recommended Next Owner

- `validator` for a narrow regression check if you want confirmation that projected diagnostics and prefix explainability outputs remain stable.
