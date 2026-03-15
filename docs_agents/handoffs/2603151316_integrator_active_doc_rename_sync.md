## Summary

Synced active constrained-gen documentation to the current renamed module and entrypoint layout.

## Files Changed

- `AGENTS.md`
- `docs_agents/CODEX_WORKING_CONTEXT.md`
- `docs_agents/HANDOFF_WORKFLOW.md`

## What Changed

- Replaced stale validation entrypoint references:
  - `validate_exact_gpu_constraints.py` -> `validate.py`
  - `validate_projected_gpu_generation.py` -> `generate_programs.py`
- Added `measure_programs.py` where current execution flow now includes measurement entrypoints.
- Replaced stale module references:
  - `tvm_verify.py` -> `concrete_gpu_verify.py`
  - `exact_gpu_constraints.py` -> `gpu_case_constraints.py`
- Updated validator/validation workflow text so active docs describe the current execution harnesses instead of removed scripts.
- Fixed the validation-required file list in `AGENTS.md` and `HANDOFF_WORKFLOW.md` so it points at the current execution entrypoints.

## Checks

- Searched active docs for stale names:
  - `validate_exact_gpu_constraints.py`
  - `validate_projected_gpu_generation.py`
  - `tvm_verify.py`
  - `exact_gpu_constraints.py`
- Result: no remaining matches in the three active docs above.

## Remaining Uncertainty

- Historical handoff notes and any user-deleted prompt files may still contain old names by design; this pass only updated active documentation.

## Next Recommended Owner

- `integrator` if you want a second pass on broader naming consistency outside the active docs.
