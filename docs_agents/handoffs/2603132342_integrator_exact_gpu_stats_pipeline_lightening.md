# 2603132342 Exact GPU Stats Pipeline Lightening

- Agent: `integrator`
- Date: `2026-03-13 23:42`
- Status: `accepted`
- Decision Topic: `further cold exact lowering reduction after parallel extract + in-memory cache`

## Inputs Considered

- Prior hotspot notes:
  - [2603132305_optimizer_exact_no_cache_hotspot_investigation.md](/root/work/tvm-ansor/docs_agents/handoffs/2603132305_optimizer_exact_no_cache_hotspot_investigation.md)
  - [2603132310_integrator_exact_gpu_parallel_extract.md](/root/work/tvm-ansor/docs_agents/handoffs/2603132310_integrator_exact_gpu_parallel_extract.md)
  - [2603132324_integrator_exact_gpu_case_stats_cache.md](/root/work/tvm-ansor/docs_agents/handoffs/2603132324_integrator_exact_gpu_case_stats_cache.md)
  - [2603132326_integrator_remaining_exact_perf_headroom.md](/root/work/tvm-ansor/docs_agents/handoffs/2603132326_integrator_remaining_exact_perf_headroom.md)

## Files Checked

- `src/auto_scheduler/exact_gpu_constraints.cc`
  - `MakePostVectorizePipeline`
  - `ExtractAllGpuCaseStats`
  - `BuildGpuCaseStats`
  - `GpuCaseStatsCollector`
- `gallery/constrained_gen/validate.py`

## Decision

- Chosen direction:
  - Introduce a lighter per-case stats pipeline for `ExtractAllGpuCaseStats(...)` that omits the global `tir::transform::Simplify()` pass.
  - Keep the full pipeline for the single-case lowering entrypoint.
- Why:
  - `BuildGpuCaseStats(...)` already performs local `Analyzer.Simplify(...)` on the collected shared/vthread/thread totals.
  - `GpuCaseStatsCollector` consumes post-vectorize IR shape, attr-based thread/vthread extents, shared allocates, and vectorized dtypes; the narrow validate path remained correct without the full IR-level simplify pass.
  - This keeps the change confined to the cold all-case extraction hotspot.

## Patch Summary

- `src/auto_scheduler/exact_gpu_constraints.cc`
  - added `MakePostVectorizeStatsPipeline()`
  - `ExtractAllGpuCaseStats(...)` now uses the lighter stats pipeline
  - `LowerSymbolicPostVectorize(...)` continues to use the original full pipeline

## Validation

- Build:
  - `ninja -C build-release tvm`
- Python syntax:
  - `python -m py_compile gallery/constrained_gen/modules/exact_gpu_constraints.py gallery/constrained_gen/modules/constraint_set.py gallery/constrained_gen/validate.py`
- Narrow validate:
  - `python gallery/constrained_gen/validate.py --task-index 29`
  - `python gallery/constrained_gen/validate.py --task-index 383`
- Representative validate shard:
  - task `0`
  - task `29`
  - task `68`
  - task `101`
  - task `126`
  - task `383`

## Measured Outcome

- Before this change, after the earlier parallel extract + cache work:
  - task `383` cold exact init: about `6.1s`
  - task `383` validate total: about `9.0s`
- After this change:
  - task `383` cold exact init: about `0.45s`
  - task `383` validate total: about `3.35s`
- Representative validate shard results:
  - task `0`: pass, about `2.38s`
  - task `29`: pass, about `2.38s`
  - task `68`: pass, about `2.37s`
  - task `101`: pass, about `2.35s`
  - task `126`: pass, about `2.41s`
  - task `383`: pass, about `3.35s`

## Remaining Risk

- This is still representative narrow validation, not a broad sweep.
- A role-pure validator/reviewer pass is still desirable before treating the lighter stats pipeline as fully signed off.
- A pending specialist investigation did not complete within this session, so this decision is based on current code inspection plus measured narrow validation.

## Next Owner

- Recommended owner: `validator`
- Recommended next step:
  - rerun the representative task shard as validator evidence
  - if clean, have reviewer judge whether the current evidence is sufficient
