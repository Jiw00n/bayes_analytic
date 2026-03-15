# doc sync for current active generation status

## What changed

Updated active docs so they match the current code after:

- projected/exact owner split
- removal of exact gating from the active `generate_programs.py` path
- current `check_all_hybrid()` semantics

## Files updated

- `docs_agents/CODEX_WORKING_CONTEXT.md`
- `gallery/constrained_gen/MODULE_AND_WORKFLOW_REFERENCE.md`

## Main corrections

- `check_all_hybrid()` is now described correctly:
  - concrete-final first when concrete context exists
  - exact only as fallback when concrete context is unavailable
- `generate_programs.py` is now described correctly:
  - active runtime samples params, checks final acceptance, converts params to concrete state, and saves
  - it does not gate normal generation on `check_all_exact()`
- module ownership is now described correctly:
  - `gpu_projection_constraints.py` owns projected pruning helpers
  - `gpu_case_constraints.py` owns exact case tables and post-vectorize exact lowering
- module reference top-level entrypoint section now names:
  - `validate.py`
  - `generate_programs.py`
  - `measure_programs.py`
  - with `generate.py` explicitly described as exploratory

## Outcome

The active docs now match the current normal generation path:

- projected pruning + concrete final validation for active generation
- exact retained for validation and diagnostics

## Remaining uncertainty

- `MODULE_AND_WORKFLOW_REFERENCE.md` is large and historical; the most important semantic mismatches were corrected, but future cleanup could still compress older narrative sections.

## Next recommended owner

- none required
