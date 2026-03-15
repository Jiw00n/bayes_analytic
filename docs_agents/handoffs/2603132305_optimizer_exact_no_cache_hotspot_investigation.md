# 2603132305 Exact No-Cache Hotspot Investigation

- Agent: `optimizer`
- Date: `2026-03-13 23:05`
- Status: `completed`
- Optimization Topic: `task 383 exact constraint build hotspot without caching`

## Scope

- Goal: reconfirm the exact-build hotspot for `vm_mod_fused_nn_batch_matmul` and identify the highest-payoff no-caching optimization directions.
- Target metric: first-call latency of exact constraint initialization used by `build_exact_constraint_nodes(...)`.
- Why this hotspot was selected: current `validate.py` exact path is dominated by first exact build on task 383.

## Files Checked

- `gallery/constrained_gen/modules/exact_gpu_constraints.py`: `_enumerate_selector_value_tuples()`, `build_projected_gpu_context()`, `build_exact_constraint_nodes()`
- `src/auto_scheduler/exact_gpu_constraints.cc`: `MakePostVectorizePipeline()`, `LowerSymbolicPostVectorizeWithPipeline()`, `BuildGpuCaseStats()`, `ExtractAllGpuCaseStats()`
- `gallery/constrained_gen/modules/constraint_set.py`: `_ensure_exact_gpu_constraints()`

## Measurement Setup

- Commands or scripts run:
  - task 383 focused Python timing shards under the repo env
  - exact-path microbench calling `_EXTRACT_ALL_GPU_CASE_STATS(...)`
  - one-case timing comparing `_LOWER_SYMBOLIC_POST_VECTORIZE(...)` vs `_EXTRACT_GPU_CASE_STATS(...)`
- Input shard or workload:
  - task index `383`
  - task desc `vm_mod_fused_nn_batch_matmul`
  - first concrete sketch from `SketchPolicy(...).generate_concrete_sketches()`
- Baseline artifact path:
  - `/tmp/projected_gpu_full_validation/optimizer/exact_hotspot_20260313/task383_reconfirm.json`
  - `/tmp/projected_gpu_full_validation/optimizer/exact_hotspot_20260313/task383_deep_profile.json`

## Bottleneck Finding

- Confirmed hotspot:
  - `build_exact_constraint_nodes()` in `exact_gpu_constraints.py` spends almost all of its time in `_EXTRACT_ALL_GPU_CASE_STATS(...)`.
  - `ExtractAllGpuCaseStats(...)` in C++ spends that time in `LowerSymbolicPostVectorizeWithPipeline(...)` for each vector case.
- Evidence:
  - reconfirmed timing:
    - `build_projected_gpu_context`: `0.220s`
    - selector case enumeration: `0.006s`
    - `extract_all_gpu_case_stats`: `20.286s`
  - task 383 has:
    - `selector_count = 2`
    - `vector_case_count = 256`
    - `runtime_domain_count = 32`
  - scaling is almost perfectly linear in case count:
    - `1` case: `0.079s`
    - `4` cases: `0.316s`
    - `16` cases: `1.262s`
    - `64` cases: `5.067s`
    - `128` cases: `10.149s`
    - `256` cases: `20.329s`
    - steady-state cost is about `79 ms/case`
  - one-case breakdown:
    - `lower_only`: `77.95 ms` avg
    - `extract_one_case`: `79.08 ms` avg
    - conclusion: `BuildGpuCaseStats(...)` is cheap; per-case post-vectorize lowering dominates
  - selector-case tightening is not the first win for task 383:
    - exact case map size is `256`
    - `feasible_case_count_under_sampled_params` is also `256`
    - this task does not show obvious over-enumeration that later collapses away

## Change Summary

- What changed:
  - no code changes
  - new timing artifacts only
- What was intentionally not changed:
  - no caching
  - no exact-check semantics changes
  - no entrypoint changes

## Results

- Before:
  - first exact-build cost for task 383 is about `20.3s` inside `_EXTRACT_ALL_GPU_CASE_STATS(...)`
- After:
  - investigation only; no patch landed
- Artifact paths:
  - `/tmp/projected_gpu_full_validation/optimizer/exact_hotspot_20260313/task383_reconfirm.json`
  - `/tmp/projected_gpu_full_validation/optimizer/exact_hotspot_20260313/task383_deep_profile.json`

## No-Caching Optimization Candidates

1. Highest payoff: replace per-case post-vectorize lowering with one symbolic post-vectorize lowering plus case-specific evaluation.
   - Current structure lowers `256` concrete vector cases independently in `ExtractAllGpuCaseStats(...)`.
   - A symbolic selector-aware post-vectorize path would change the cost shape from `O(num_cases * lowering)` to roughly `O(lowering) + O(num_cases * cheap_eval)`.
   - Expected impact for task 383:
     - current exact extractor: about `20.3s`
     - likely target range: about `0.5s` to `2.0s`
   - Risk:
     - semantics-sensitive
     - requires specialist/integrator ownership because it changes how exact stats are derived, not just how they are timed

2. Next best narrow C++ optimization: make `ExtractAllGpuCaseStats(...)` use a lighter post-vectorize path than the full `VectorizeLoop -> InjectVirtualThread -> StorageRewrite -> Simplify` pipeline for every case.
   - Current code constructs an `IRModule`, runs the full sequential pass pipeline, then collects cheap stats.
   - The one-case timing shows almost the whole `79 ms` is in this lowering path.
   - Expected impact for task 383:
     - if the heavy path can be cut by about half, `20.3s -> 8s to 12s`
     - if only trailing cleanup like `Simplify()` is removable, upside is smaller, likely `10%` to `30%`
   - Risk:
     - medium
     - still semantics-sensitive because shared-memory and thread-bound stats depend on post-vectorize IR shape

3. Low priority for task 383: tighter selector enumeration in Python.
   - `_enumerate_selector_value_tuples()` is already cheap (`0.006s`)
   - this task keeps all `256` cases feasible even under sampled params
   - expected impact on task 383 is negligible unless later evidence shows many impossible tuples on other workloads

## Correctness Risk

- What might regress:
  - exact symbolic upper bounds for vectorize/shared-memory/max-vthread/max-threads
  - parity between exact symbolic stats and concrete lowered behavior
- Required follow-up validation:
  - validator should rerun `validate.py` narrow shards on task 383 and at least one conv2d-like task after any landed patch
  - reviewer should compare exact/final behavior before treating a performance patch as safe

## Next Owner

- Recommended owner: `specialist`
- Recommended next step:
  - inspect whether a symbolic selector-aware post-vectorize analysis is feasible in `src/auto_scheduler/exact_gpu_constraints.cc`
  - if that is too risky, prototype the lighter-weight no-cache C++ extractor path and measure task 383 before/after
