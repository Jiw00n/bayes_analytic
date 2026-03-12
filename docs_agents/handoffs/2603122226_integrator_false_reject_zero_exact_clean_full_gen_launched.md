# 2603122226 Integrator False Reject Zero Exact Clean Full Gen Launched

## What changed

- Kept the concrete-first projected fallback path in [constraint_set.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/constraint_set.py) so projected false rejects on fully assigned params are suppressed by concrete or exact confirmation.
- Accepted the divisor-candidate propagation optimization in [domain_propagator.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/domain_propagator.py) to reduce `propagate_domain()` cost on `sp_*` variables.
- Repaired and updated [profile_schedule_generator_timing.py](/root/work/tvm-ansor/gallery/constrained_gen/profile_schedule_generator_timing.py) so it measures the current concrete-first generation path instead of the old exact-only path.
- Added a fused projected-metadata helper in [exact_gpu_constraints.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/exact_gpu_constraints.py) for pre-vectorize metadata collection. This change was semantics-clean in narrow checks, but measured gain was negligible.

## Files checked

- [constraint_set.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/constraint_set.py)
- [domain_propagator.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/domain_propagator.py)
- [exact_gpu_constraints.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/exact_gpu_constraints.py)
- [profile_schedule_generator_timing.py](/root/work/tvm-ansor/gallery/constrained_gen/profile_schedule_generator_timing.py)
- [validate_exact_gpu_constraints.py](/root/work/tvm-ansor/gallery/constrained_gen/validate_exact_gpu_constraints.py)
- [validate_projected_gpu_generation.py](/root/work/tvm-ansor/gallery/constrained_gen/validate_projected_gpu_generation.py)

## Exact validation outcome

- Full exact-vs-concrete validation completed clean after sharding the run into `8 x 114` sketches.
- Aggregate result:
  - `checked=912`
  - `false_reject=0`
  - `false_accept=0`
  - `final_checker_mismatch=0`
- Aggregate artifacts:
  - [aggregate_summary.json](/tmp/projected_gpu_full_validation/validator/concrete_first_all_projected_exact_sharded_20260312/aggregate_summary.json)
  - [aggregate_mismatch.jsonl](/tmp/projected_gpu_full_validation/validator/concrete_first_all_projected_exact_sharded_20260312/aggregate_mismatch.jsonl)
- Per-shard artifacts root:
  - `/tmp/projected_gpu_full_validation/validator/concrete_first_all_projected_exact_sharded_20260312/`

## Generation validation outcome so far

- Narrow and wider generation checks stayed clean on the concrete-first path:
  - `/tmp/projected_gpu_full_validation/validator/shared_fallback_gen_smoke_20260312/summary.json`
  - `/tmp/projected_gpu_full_validation/validator/concrete_first_generation_slice_130_20260312/summary.json`
  - `/tmp/projected_gpu_full_validation/validator/concrete_first_generation_wide64_20260312/summary.json`
  - `/tmp/projected_gpu_full_validation/validator/concrete_first_generation_wide128_191_20260312/summary.json`
- Observed outcomes from completed wider shards:
  - `0~63`: `3200/3200` success, `concrete_invalid=0`, `rand_fail=0`
  - `128~191`: running during this note, but had no invalid or randomize failure in progress logs through sketch `187`

## Profiling outcome

- Representative hard sketch measurements after the concrete-first fix and divisor propagation:
  - `sketch 2` init-only: about `268ms`
  - `sketch 2` `randomize_params(max_retries=1)`: about `33ms ~ 40ms` depending on cache state
  - `sketch 2` internal breakdown with concrete-final cache cleared:
    - `propagate_domain ~26.7ms`
    - `get_concrete_final_result ~8.5ms`
    - `filter_by_constraints ~4.4ms`
- Artifacts:
  - `/tmp/projected_gpu_full_validation/optimizer/post_concrete_first_profile_sk2_cacheclear_20260312.json`
  - `/tmp/projected_gpu_full_validation/optimizer/divisor_prop_profile_sk2_20260312.json`
  - `/tmp/projected_gpu_full_validation/optimizer/divisor_prop_profile_sk133_20260312.json`

## Full generation sweep

- Full `2000 attempts/sketch` generation validation has been launched as `16 x 57`-sketch detached shards.
- Root:
  - `/tmp/projected_gpu_full_validation/validator/concrete_first_full_generation_2000_20260312/`
- Launch manifest:
  - [launch.json](/tmp/projected_gpu_full_validation/validator/concrete_first_full_generation_2000_20260312/launch.json)
- PIDs at launch:
  - `92132..92147`
- Early progress snapshot:
  - `shard_00`: `[2/57] sketch=1 status=ok success=2000 invalid=0 rand_fail=0`
  - `shard_02`: `[3/57] sketch=116 status=ok success=2000 invalid=0 rand_fail=0`
  - `shard_03`: `[1/57] sketch=171 status=ok success=2000 invalid=0 rand_fail=0`

## Remaining uncertainty

- The full `2000 attempts/sketch` generation sweep is still running, so no global generation invalid-rate summary exists yet.
- The fused projected-metadata change in [exact_gpu_constraints.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/exact_gpu_constraints.py) did not show a meaningful measured win and could be reverted later if we want to keep the hot path simpler.

## Next recommended owner

- `validator`
  - poll the detached generation shards under `/tmp/projected_gpu_full_validation/validator/concrete_first_full_generation_2000_20260312/`
  - merge per-shard `summary.json` and `invalid.jsonl` once complete
- `reviewer`
  - review the final merged generation results and confirm whether any sketch still shows `concrete_invalid` or `no_randomize_success`
