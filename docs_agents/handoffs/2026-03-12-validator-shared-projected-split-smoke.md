# Shared Projected Split Smoke

- Agent: `validator`
- Date: `2026-03-12`
- Status: `completed`
- Run Type: `other`

## Scope

- Goal:
  - check that the shared projected split does not change representative shared projected values after exact build
  - check that narrow generation still produces concrete-valid schedules
  - check whether narrow exact-vs-final behavior shows any new mismatch
- Shard or indices:
  - shared projected regression: sketch indices `0,1,2,3`
  - generation validation: `start=0`, `limit=2`
  - exact validation: `start=0`, `limit=4`
- Reason for this run:
  - post-patch smoke after deriving `shared_memory` projected constraints from `pre_func` shared allocations

## Files Checked

- `gallery/constrained_gen/modules/exact_gpu_constraints.py`: `build_projected_gpu_context()`, `build_exact_constraint_nodes()`
- `gallery/constrained_gen/modules/constraint_set.py`: `_ensure_projected_gpu_constraints()`
- `gallery/constrained_gen/validate_projected_gpu_generation.py`: generation smoke entry point
- `gallery/constrained_gen/validate_exact_gpu_constraints.py`: exact-vs-final smoke entry point

## Commands Run

```bash
source /root/work/venv/bin/activate
export TVM_HOME=/root/work/tvm-ansor
export PYTHONPATH=$TVM_HOME/python
export TVM_LIBRARY_PATH=$TVM_HOME/build-release
python - <<'PY'
# wrote /tmp/projected_gpu_full_validation/validator/shared_projected_regression_smoke_20260312.json
PY
python gallery/constrained_gen/validate_projected_gpu_generation.py --all-sketches --start 0 --limit 2 --attempts-per-sketch 1 --max-retries 1 --summary-path /tmp/projected_gpu_full_validation/validator/shared_split_validate_projected_smoke_20260312.json
python gallery/constrained_gen/validate_exact_gpu_constraints.py --start 0 --limit 4 --summary-path /tmp/projected_gpu_full_validation/validator/shared_split_validate_exact_smoke_20260312.json --mismatch-path /tmp/projected_gpu_full_validation/validator/shared_split_validate_exact_smoke_mismatches_20260312.json
```

## Artifacts

- Raw output dir: `/tmp/projected_gpu_full_validation/validator/`
- Summary file:
  - `/tmp/projected_gpu_full_validation/validator/shared_projected_regression_smoke_20260312.json`
  - `/tmp/projected_gpu_full_validation/validator/shared_split_validate_projected_smoke_20260312.json`
  - `/tmp/projected_gpu_full_validation/validator/shared_split_validate_exact_smoke_20260312.json`
- Detailed file:
  - `/tmp/projected_gpu_full_validation/validator/shared_split_validate_exact_smoke_mismatches_20260312.json`

## Result Summary

- Sketches processed:
  - shared projected regression: `4`
  - generation validation: `2`
  - exact validation: `4`
- Main counts:
  - shared projected regression: `all_projected_stable=true`
  - generation validation: `randomize_success=2`, `concrete_invalid=0`
  - exact validation: `mismatch_total=1`, `false_reject=1`, `false_accept=0`, `final_checker_mismatch=0`
- Reproducer confirmed: `partial`

## Key Findings

- Finding 1:
  - On sketch indices `0,1,2,3`, `shared_memory` projected values were identical before and after forcing exact build, and `_exact_gpu` was still unset before the explicit exact call.
- Finding 2:
  - Narrow generation smoke passed with no concrete-invalid schedules on the first two sketches.
- Finding 3:
  - Narrow exact validation still shows one shared-memory false reject on `sketch_index=2` (`vm_mod_fused_nn_batch_matmul_3`), with projected shared bytes `76800` over the `49152` limit while both final and concrete checks pass. The reported root cause remains `runtime_projection_upper_bound_insufficient`.

## Uncertainty

- Missing evidence:
  - no broad shard was run, so this does not prove absence of regressions outside the sampled indices
- Known limitations of this run:
  - the remaining false reject appears consistent with the existing projected shared-memory conservatism, but this smoke run alone does not prove it is pre-existing across all shards
  - single-session validation only: the intended separate validator subagent did not surface file-based output during this turn, so the main session wrote the raw artifacts directly

## Next Owner

- Recommended owner: `reviewer`
- Recommended next step:
  - review the smoke artifacts and decide whether the remaining `shared_memory` false reject is sufficient evidence for specialist follow-up or can be treated as known existing conservatism
