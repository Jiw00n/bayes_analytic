# 2603132351 Exact Cold Path Evidence Review

- Agent: `reviewer`
- Date: `2026-03-13 23:51`
- Status: `approved`
- Review Target: `recent exact cold-path optimization evidence for src/auto_scheduler/exact_gpu_constraints.cc`

## Reviewed Inputs

- Validator note:
  - validator-produced evidence was provided to review in the parent session as:
    - representative shard pass on tasks `0, 29, 68, 101, 126, 383`
    - performance change on task `383`: cold exact init `~6.1s -> ~0.45s`, validate total `~9.0s -> ~3.35s`
- Validator artifacts:
  - `/tmp/projected_gpu_full_validation/reviewer/exact_cold_path_review/summary.txt`
- Additional notes reviewed:
  - [2603132333_specialist_exact_cold_path_pass_assessment.md](/root/work/tvm-ansor/docs_agents/handoffs/2603132333_specialist_exact_cold_path_pass_assessment.md)
  - [2603132342_integrator_exact_gpu_stats_pipeline_lightening.md](/root/work/tvm-ansor/docs_agents/handoffs/2603132342_integrator_exact_gpu_stats_pipeline_lightening.md)

## Review Checks

- Sanity checks performed:
  - read current [exact_gpu_constraints.cc](/root/work/tvm-ansor/src/auto_scheduler/exact_gpu_constraints.cc) around `MakePostVectorizePipeline`, `MakePostVectorizeStatsPipeline`, `BuildGpuCaseStats`, and `ExtractAllGpuCaseStats`
  - verified current code keeps the full pipeline for single-case lowering and uses the lighter pipeline only for all-case stats extraction
  - verified specialist rationale that `InjectVirtualThread()` is still retained
- Scripts or summaries used:
  - current code inspection only
  - specialist and integrator handoff summaries listed above

## Findings

- Confirmed issue kinds:
  - confirmed performance hotspot was in a specialist-owned exact lowering path
  - confirmed implementation is narrow: it removes pass-level `Simplify()` only from the all-case cold stats pipeline, not from the general single-case lowering path
- Likely root-cause ownership: `specialist-owned`
- Representative examples:
  - task `383` cold exact init improved from about `6.1s` to about `0.45s`
  - task `383` validate total improved from about `9.0s` to about `3.35s`
  - representative shard `0, 29, 68, 101, 126, 383` all passed
- Missing or weak evidence:
  - no broader shard beyond the representative set
  - validator evidence was not provided to review as a standalone detailed markdown note path in this session, only as reviewed run results and the integrator note

## Escalation Decision

- Evidence sufficient for specialist escalation: `yes`
- If no, what validator should add:
  - not applicable
- If yes, what specialist should investigate first:
  - not applicable for this patch; specialist analysis has already been completed and matches the landed direction

## Acceptance Decision

- Evidence sufficient for patch acceptance: `yes, for now`
- Residual risk:
  - broader-task semantic coverage is still limited
  - this should be treated as accepted for the current narrow optimization step, not as proof that all exact-lowering cases are unaffected
- Another validator pass required before acceptance: `no`
- Another validator pass recommended before broader rollout: `yes`

## Next Owner

- Recommended owner: `integrator`
- Recommended next step:
  - keep the patch
  - if the team wants higher confidence before wider rollout, run a `10-20` task broader representative shard and then stop unless a mismatch appears
