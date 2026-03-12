# Shared Projected Split Smoke Review

- Agent: `reviewer`
- Date: `2026-03-12`
- Status: `approved`
- Review Target: `2026-03-12-validator-shared-projected-split-smoke.md`

## Reviewed Inputs

- Validator note:
  - `docs_agents/handoffs/2026-03-12-validator-shared-projected-split-smoke.md`
  - `docs_agents/handoffs/2026-03-12-validator-shared-split-generation-smoke.md`
- Validator artifacts:
  - `/tmp/projected_gpu_full_validation/validator/shared_projected_regression_smoke_20260312.json`
  - `/tmp/projected_gpu_full_validation/validator/shared_split_validate_projected_smoke_20260312.json`
  - `/tmp/projected_gpu_full_validation/validator/shared_split_validate_exact_smoke_20260312.json`
  - `/tmp/projected_gpu_full_validation/validator/shared_split_validate_exact_smoke_mismatches_20260312.json`
  - `/tmp/projected_gpu_full_validation/validator/shared_split_smoke_20260312/summary.json`

## Review Checks

- Sanity checks performed:
  - confirmed the shared projected regression artifact shows `all_projected_stable=true` across representative sketch indices `0,1,2,3`
  - confirmed both generation smoke summaries report `concrete_invalid=0`
  - confirmed the exact smoke mismatch is isolated to `sketch_index=2` and is classified as `runtime_projection_upper_bound_insufficient`
- Scripts or summaries used:
  - direct inspection of the validator summary and mismatch JSON files listed above

## Findings

- Confirmed issue kinds:
  - no new concrete-generation failure was established by the smoke shards
  - one projected shared-memory false reject remains on `sketch_index=2`
- Representative examples:
  - `sketch_index=2`, `vm_mod_fused_nn_batch_matmul_3`: projected shared bytes `76800` exceed the `49152` limit while `final_ok=true` and `concrete_ok=true`
  - representative shared projected regression stayed stable before vs after explicit exact build on sketch indices `0,1,2,3`
- Missing or weak evidence:
  - no broad shard was run, so this review cannot prove full absence of regressions outside the smoke sample

## Escalation Decision

- Evidence sufficient for specialist escalation: `no`
- If no, what validator should add:
  - only if broader confidence is needed, run a slightly wider shared-heavy exact-vs-final shard to confirm the remaining false reject rate is unchanged from the pre-patch baseline
- If yes, what specialist should investigate first:
  - n/a

## Next Owner

- Recommended owner: `integrator`
- Recommended next step:
  - treat the shared projected split as acceptable for the initialization bottleneck fix, while keeping the existing `shared_memory` runtime-projection false reject on the backlog as a separate correctness task
