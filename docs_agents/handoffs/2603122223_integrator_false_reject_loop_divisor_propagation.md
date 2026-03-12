# 2603122223 Integrator False Reject Loop Divisor Propagation

## What changed

- Kept the concrete-first projected fallback in [constraint_set.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/constraint_set.py) for fully assigned params so projected false rejects stay suppressed by concrete or exact confirmation.
- Repaired [profile_schedule_generator_timing.py](/root/work/tvm-ansor/gallery/constrained_gen/profile_schedule_generator_timing.py) against the current `ConstraintSet` / `DomainPropagator` / concrete-final API so it measures the real concrete-first generation path instead of an outdated exact-only path.
- Optimized [domain_propagator.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/domain_propagator.py) upper-bound propagation:
  - added `_candidate_values_for_domain()`
  - for `sp_*` vars, `propagate_domain()` now bisects over actual divisor candidates of the remaining split extent instead of the full integer interval
  - kept existing fast paths for split-structure and pure-product upper bounds

## Files checked

- [constraint_set.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/constraint_set.py)
- [domain_propagator.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/domain_propagator.py)
- [profile_schedule_generator_timing.py](/root/work/tvm-ansor/gallery/constrained_gen/profile_schedule_generator_timing.py)
- [param_sampler.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/param_sampler.py)
- [exact_gpu_constraints.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/exact_gpu_constraints.py)

## Raw artifacts

- Exact narrow repros after divisor propagation:
  - `/tmp/projected_gpu_full_validation/validator/divisor_prop_exact_sk2_20260312/summary.json`
  - `/tmp/projected_gpu_full_validation/validator/divisor_prop_exact_sk18_20260312/summary.json`
- Generation narrow/broader shards after divisor propagation:
  - `/tmp/projected_gpu_full_validation/validator/divisor_prop_gen_sk2_3_20260312/summary.json`
  - `/tmp/projected_gpu_full_validation/validator/divisor_prop_gen_sk117_20260312/summary.json`
  - `/tmp/projected_gpu_full_validation/validator/divisor_prop_gen_sk133_20260312/summary.json`
  - `/tmp/projected_gpu_full_validation/validator/divisor_prop_gen_shard130_20260312/summary.json`
- Profiling outputs:
  - `/tmp/projected_gpu_full_validation/optimizer/divisor_prop_profile_sk2_20260312.json`
  - `/tmp/projected_gpu_full_validation/optimizer/divisor_prop_profile_sk133_20260312.json`
  - `/tmp/projected_gpu_full_validation/optimizer/post_false_reject_fix_profile_sk2_20260312.json`

## Concrete outcome

- Reproduced false rejects remain fixed on the known exact mismatch sketches:
  - `sketch 2`: false reject `0`
  - `sketch 18`: false reject `0`
- Generation smoke stayed clean after the propagation optimization:
  - `sketch 2~3`: `100/100` success, `invalid=0`
  - `sketch 117`: `50/50` success, `invalid=0`
  - `sketch 133`: `50/50` success, `invalid=0`
  - `sketch 130~137`: `400/400` success, `invalid=0`
- Measured performance improved on representative hard sketches without changing these narrow outcomes:
  - `sketch 2` `randomize_params(max_retries=1)`: about `42.2ms -> 33.4ms`
  - `sketch 2` `propagate_domain` inside one randomize run: about `29.2ms -> 21.0ms`
  - `sketch 133` ad hoc representative generation mean over `50` attempts: about `101.6ms -> 84.0ms`

## Remaining uncertainty

- Full exact sweep on the current divisor-propagation patch is still running:
  - PID `91236`
  - command output target: `/tmp/projected_gpu_full_validation/validator/divisor_prop_exact_full_20260312/summary.json`
- This note is `single-session validation only` for the latest propagation optimization. A separate validator/reviewer pass is still needed for sign-off.

## Next recommended owner

- `validator`
  - finish the full exact sweep and record whether global false reject / false accept / final mismatch stay at zero on the patched propagation path
- `reviewer`
  - review the full exact artifacts plus the new generation shard artifacts before the final full `2000 attempts/sketch` generation sweep
