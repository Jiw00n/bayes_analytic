## Summary

- Changed unique-schedule search initialization so it no longer pre-assigns singleton-domain split vars before building `state["search_order"]`.
- Unique search now keeps the full `g._var_order` and assigns fixed-domain vars only when rollout reaches that variable.

## Files Checked

- `gallery/constrained_gen/modules/param_sampler.py`
- `gallery/constrained_gen/modules/schedule_generator.py`
- `gallery/constrained_gen/generate_programs.py`
- `docs_agents/CODEX_WORKING_CONTEXT.md`

## Change

- Added `ParamSampler._initialize_unique_search_base_state(...)`.
- Updated `ParamSampler._ensure_unique_search_state()` to use that helper instead of `_initialize_split_sampling_state(...)`.
- Left retry sampling paths (`randomize_params_with_order`) unchanged.

## Validation

- `python -m py_compile /root/work/tvm-ansor/gallery/constrained_gen/modules/param_sampler.py`
- Narrow smoke from `gallery/constrained_gen`:
  - built task 0 generator
  - confirmed `state["search_order"][:len(gen._var_order)] == list(gen._var_order)`
  - confirmed first `next_unique_schedule(set())` payload includes all split vars (`missing_split_vars = 0`)

## Remaining Uncertainty

- This only changes the unique active-generation path. Legacy retry sampler and assignment-materialization paths still pre-assign singleton-domain vars.

## Next Owner

- `validator` if a broader active-generation regression check is needed.
