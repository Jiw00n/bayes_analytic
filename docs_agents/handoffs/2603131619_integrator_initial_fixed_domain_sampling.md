## Summary

- Added initial fixed-domain assignment in [param_sampler.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/param_sampler.py) so singleton domains are eagerly assigned before the main sampling loop when the assignment is order-safe.
- Restored `pure_product_max_vthread` handling and cleaned partial constraint printing.
- Validation was `single-session validation only` because multi-agent spawn was unavailable at the time of execution.

## Files Changed

- [param_sampler.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/param_sampler.py)
- [constraint_set.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/constraint_set.py)
- [var_order_planner.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/var_order_planner.py)
- [schedule_generator.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/schedule_generator.py)
- [order_test.ipynb](/root/work/tvm-ansor/gallery/constrained_gen/order_test.ipynb)

## Implementation

- Added `ParamSampler._try_assign_initial_fixed_vars(...)`.
- The helper:
  - scans the requested `var_order`
  - eagerly assigns a variable when its domain is singleton and all preceding split vars in the same split group are already fixed
  - updates `sym_map`, `domains`, `result`, and `group_remaining`
  - runs `propagate_domain(...)` after each such assignment
  - repeats to a fixpoint
- `_randomize_params_with_order(...)` now samples only the remaining non-fixed vars after this initialization pass.

## Related Fixes

- Fixed [constraint_set.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/constraint_set.py) `_collect_vthread_clamped_sp_names()` regex so vthread clamp factors are detected again.
- Updated [var_order_planner.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/var_order_planner.py) to build pure-product phase vars from normalized product-form factors instead of raw tree vars.
- Updated [schedule_generator.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/schedule_generator.py) partial constraint evaluation to print `eval: (partial)` instead of `EVAL_FAIL(...)`.

## Validation

- Generation smoke:
  - artifact: `/tmp/projected_gpu_full_validation/validator/initial_fixed_domain_smoke_260313/summary.json`
  - result: `4 sketches`, `20/20 randomize_success`, `concrete_invalid=0`
- Raw-state smoke:
  - artifact: `/tmp/projected_gpu_full_validation/validator/raw_state_initial_fixed_smoke_260313/summary.json`
  - result: `8/8 ok`, `concrete_invalid=0`, `exceptions=0`
- Local check on sketch `600`:
  - `grid_0__pure_product_max_vthread` restored to `['sp_2_0', 'sp_3_0', 'sp_4_0']`

## Remaining Uncertainty

- Validation is narrow only.
- The eager singleton assignment is intentionally conservative: it only fixes split vars when earlier vars in the same split group are already fixed.

## Next Recommended Owner

- `validator` for a wider shard if this initialization behavior is going to be kept and relied on broadly.
