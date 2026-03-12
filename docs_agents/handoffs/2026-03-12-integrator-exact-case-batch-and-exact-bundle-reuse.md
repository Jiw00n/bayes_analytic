# Exact Case Batch And Bundle Reuse

- Agent: `integrator`
- Date: `2026-03-12`
- Status: `completed`

## What Changed

- Added batched exact GPU case extraction in [exact_gpu_constraints.cc](/root/work/tvm-ansor/src/auto_scheduler/exact_gpu_constraints.cc):
  - new global FFI: `constrained_gen.extract_all_gpu_case_stats`
  - fused per-case TIR collector for shared bytes, max vthread, max threads, runtime domains, and max vector bytes
  - reused one post-vectorize pass pipeline object inside the batch loop
- Updated [exact_gpu_constraints.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/exact_gpu_constraints.py) `build_exact_constraint_nodes()` to call the batch FFI instead of `extract_gpu_case_stats` once per selector case.
- Added `CaseSplitNode.feasible_case_values()` and `CaseSplitNode.interval_with_feasible_cases()` in [expr_nodes.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/expr_nodes.py).
- Updated [constraint_set.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/constraint_set.py) `check_all_exact()` to compute feasible cases once and reuse them across `vectorize/shared_memory/max_threads/max_vthread` exact bounds.

## Files And Functions Checked

- [exact_gpu_constraints.cc](/root/work/tvm-ansor/src/auto_scheduler/exact_gpu_constraints.cc)
  - `LowerSymbolicPostVectorize`
  - `ExtractGpuCaseStats`
  - `ExtractAllGpuCaseStats`
  - `GpuCaseStatsCollector`
- [exact_gpu_constraints.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/exact_gpu_constraints.py)
  - `build_exact_constraint_nodes`
- [expr_nodes.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/expr_nodes.py)
  - `CaseSplitNode.interval`
  - `CaseSplitNode.feasible_case_values`
  - `CaseSplitNode.interval_with_feasible_cases`
- [constraint_set.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/constraint_set.py)
  - `check_all_exact`
  - `_evaluate_exact_upper_bounds`

## Validation And Artifacts

- Rebuilt `libtvm.so` with `ninja -C build-release -j4`.
- Artifacts:
  - `/tmp/projected_gpu_full_validation/optimizer/post_patch_timing_sk3.json`
  - `/tmp/projected_gpu_full_validation/optimizer/post_patch_exact_build_rows.json`
  - `/tmp/projected_gpu_full_validation/optimizer/post_patch_randomize_internal_sk2.json`

## Concrete Outcome

- `check_all_exact()` output matched the legacy composition of individual exact checks on sampled params for `sketch_index=3`.
- Batch FFI matched legacy single-case FFI on normalized outputs for sampled cases from `sketch_index=2`.
- Exact-only randomization improved materially:
  - `sketch_index=3`: about `695-708 ms` before to about `313 ms` after
  - `sketch_index=2`: about `526 ms` before to about `228-234 ms` after
- `ScheduleGenerator` init improved only marginally:
  - `sketch_index=2` exact build still about `20.47 s`
  - `sketch_index=3` exact build still about `34.95 s`

## Remaining Uncertainty

- The dominant init-time cost still appears to be the repeated post-vectorize lowering/pass pipeline per selector case, not Python/C++ call overhead or repeated IR walks alone.
- The next major speedup likely requires reducing the number of distinct cases processed or avoiding eager exact case-table construction on paths that do not need it.

## Next Recommended Owner

- Recommended owner: `validator`
- Recommended next step: run a narrow exact-vs-concrete and projected-generation validation shard on the patched build, then decide whether to pursue lazy exact-node construction or case-count reduction.
