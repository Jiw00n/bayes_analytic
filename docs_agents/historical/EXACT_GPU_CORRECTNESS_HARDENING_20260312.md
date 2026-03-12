# Exact GPU Correctness Hardening 2026-03-12

## Historical Status

This file is a historical record for the March 12, 2026 correctness-hardening pass.

- Do not use this file as the startup document for new sessions.
- Use `AGENTS.md` and `docs_agents/CODEX_WORKING_CONTEXT.md` for the current workflow.
- Treat all statements here as date-bound and verify against current code before acting.

## Scope

This document records the work completed after the earlier schedule-generator refactor handoff.

Primary goals for this pass:

1. stop using pruning-only acceptance for full sampled assignments
2. align final checker semantics with concrete lowered TIR verification
3. investigate and remove non-pruning exact mismatches
4. leave reproducible validation artifacts and a final correctness summary

This pass did not attempt to solve projected pruning false rejects. That issue remains separate.

## Code Areas Touched

The work was concentrated in these files:

- `gallery/constrained_gen/modules/schedule_generator.py`
- `gallery/constrained_gen/modules/constraint_set.py`
- `gallery/constrained_gen/modules/exact_gpu_constraints.py`
- `gallery/constrained_gen/modules/projected_gpu_validation.py`
- `gallery/constrained_gen/modules/tvm_verify.py`
- `gallery/constrained_gen/modules/expr_nodes.py`
- `gallery/constrained_gen/modules/param_sampler.py`
- `src/auto_scheduler/exact_gpu_constraints.cc`
- `src/tir/analysis/verify_gpu_code.cc`
- `gallery/constrained_gen/audit_non_pruning_correctness.py`
- `gallery/constrained_gen/refresh_all_sketches_non_pruning_validation.py`

## Main Changes

### 1. Hybrid full-assignment acceptance

Full sampled assignments no longer stop at pruning-only acceptance.

- `ParamSampler._randomize_params_with_order()` was switched to hybrid acceptance.
- `ScheduleGenerator.check_all_hybrid()` now prefers concrete final validation when concrete context exists.
- `check_all()` remained a pruning alias for compatibility, but no longer determines full sampled acceptance.

Practical effect:

- generated schedules are no longer rejected solely because projected pruning is conservative at the final accept step

### 2. Final checker semantics aligned to concrete lowered TIR

The final checker was aligned with the actual lowered GPU verifier path.

- `ScheduleGenerator.get_concrete_final_result()` and `check_all_final()` became the source of truth for final semantics
- the old symbolic thread-binding interpretation no longer controls final acceptance

Practical effect:

- historical `symbolic_thread_binding_semantics_mismatch` no longer appears as a final-checker mismatch

### 3. Exact symbolic checks separated from projected pruning

Explicit exact-check APIs were added and then hardened.

- `check_all_exact()` became a distinct path instead of implicitly reusing pruning semantics
- exact checks for vectorize, shared memory, max vthread, and later max threads were added explicitly
- exact unknowns are now handled more defensively instead of being treated as hard rejects

Practical effect:

- exact symbolic checks are now materially closer to the lowered TIR semantics and are no longer the same as projected pruning checks

### 4. Exact case-stat extraction hardened

The exact GPU case extractor was strengthened on the C++ side.

- runtime-variable upper bounds are folded before returning case stats
- `pos_inf` or sentinel-style results are no longer allowed to flow through as if they were meaningful exact values
- vector-byte exact stats now use the extracted post-vectorize value directly instead of reapplying an extra conservative `selector_extent * scalar_bytes` bound

Practical effect:

- the earlier `custom_exact_lowering_mismatch` bucket collapsed substantially
- representative exact false rejects stopped showing the old `2^60`-style sentinel contamination

### 5. Exact max-threads semantics moved onto post-vectorize TIR case stats

The last remaining non-pruning issue was exact `max_threads`.

Previous state:

- `check_all_exact()` still reused the symbolic `check_max_threads()` path
- this overestimated some `vthread * threadIdx.x` combinations and produced exact-only false rejects

Fix:

- `src/auto_scheduler/exact_gpu_constraints.cc` now extracts `threads_per_block` from post-vectorize TIR
- runtime variables are eliminated by upper-bounding before case stats are returned
- Python exact nodes now include `max_threads_node`
- `check_all_exact()` now uses `check_max_threads_exact()` instead of the symbolic pruning check

Practical effect:

- the last historical exact-only false rejects were removed

### 6. Validation and diagnosis tooling

New or extended tooling was used to make the work reproducible.

- `audit_non_pruning_correctness.py`
- `refresh_all_sketches_non_pruning_validation.py`
- exact-lowering differential collection in `modules/projected_gpu_validation.py`
- detailed verifier error extraction through `modules/tvm_verify.py` and `src/tir/analysis/verify_gpu_code.cc`

## Issue Evolution

The work addressed the historical buckets from the earlier handoff.

### Historical buckets from the earlier phase

- `custom_exact_lowering_mismatch`
- `symbolic_thread_binding_semantics_mismatch`
- `runtime_projection_upper_bound_insufficient`

### Current interpretation after this pass

- `custom_exact_lowering_mismatch`
  - largely resolved as a correctness issue
  - the dominant problems turned out to be exact-case-stat extraction and exact-model conservatism, not concrete lowering disagreement
- `symbolic_thread_binding_semantics_mismatch`
  - removed from final semantics first
  - later removed from exact `max_threads` semantics by moving exact checks onto post-vectorize TIR case stats
- `runtime_projection_upper_bound_insufficient`
  - still relevant for projected pruning false rejects
  - intentionally left out of scope for this pass

## Representative Validation Artifacts

### Lowering differential artifacts

- `/tmp/projected_gpu_full_validation/lowering-diff/index_118_exact_lowering_diff.json`
- `/tmp/projected_gpu_full_validation/lowering-diff/index_156_exact_lowering_diff_min.json`

These were used to show that the suspected exact mismatches were not concrete verifier mismatches.

### Historical candidate recheck

This recheck targeted the earlier `custom_exact_lowering_mismatch` candidates.

- `/tmp/projected_gpu_full_validation/historical_custom_exact_candidate_indices.txt`
- `/tmp/projected_gpu_full_validation/historical_custom_exact_candidates_recheck.txt`
- `/tmp/projected_gpu_full_validation/historical_custom_exact_candidates_recheck_summary.json`
- `/tmp/projected_gpu_full_validation/historical_custom_exact_candidates_recheck_summary.md`

Final outcome:

- 58 candidate sections checked
- 58 fully clean
- 0 exact false rejects
- 0 hybrid false rejects
- 0 final false rejects
- 0 concrete reverification mismatches

### Full recorded-valid exact audit

This is the broadest validation artifact from this pass.

- `/tmp/projected_gpu_full_validation/full_exact_recorded_audit_20260312/merged_summary.json`
- `/tmp/projected_gpu_full_validation/full_exact_recorded_audit_20260312/merged_mismatches.jsonl`

Final merged summary:

- `checked = 912`
- `concrete_ok = 912`
- `concrete_bad = 0`
- `exact_mismatch = 0`
- `hybrid_mismatch = 0`
- `final_mismatch = 0`
- `exact_false_reject = 0`
- `hybrid_false_reject = 0`
- `final_false_reject = 0`

`merged_mismatches.jsonl` is empty.

## Validation Notes

Important caveat:

- the 912-wide merged summary is `mode = recorded_valid_exact_audit`
- this means it is a full recorded-valid audit of exact, hybrid, and final checker behavior against the recorded-valid oracle
- it is not a full concrete re-lowering sweep for all 912 sketches

Why the conclusion is still strong:

- the historically problematic non-pruning candidate set was rechecked with concrete reverification and came back clean
- the broad 912-sketch recorded-valid audit also came back clean

Taken together, these were sufficient to conclude that non-pruning correctness issues were no longer visible in the current code

## Current Conclusion

As of this pass:

- pruning-only acceptance for full assignments was removed
- final checker semantics were aligned with concrete lowered TIR verification
- exact vectorize mismatches were removed
- exact shared and max-vthread sentinel contamination was neutralized
- exact max-threads semantics were moved onto post-vectorize TIR case stats
- historical non-pruning mismatch candidates rechecked clean
- 912-sketch recorded-valid exact audit clean

## Remaining Open Problem

The remaining known correctness issue is outside the scope of this pass:

- projected pruning false rejects

This includes conservative projected upper bounds such as:

- runtime-projected shared-memory upper bounds
- projected vectorize upper bounds
- projected thread-binding upper bounds used during prefix pruning

These can still reject candidates that are valid under exact or concrete final semantics.

## Recommended Next Focus

If work resumes after this historical point, the next recommended topic is:

1. projected pruning false rejects

Suggested entry points:

- `gallery/constrained_gen/modules/constraint_set.py`
- `gallery/constrained_gen/modules/domain_propagator.py`
- `gallery/constrained_gen/modules/projected_gpu_validation.py`
- `gallery/constrained_gen/validate_projected_gpu_generation.py`
- `gallery/constrained_gen/refresh_all_sketches_prefix_through_split_structure.py`

