## Summary

- Adjusted `execution_max_threads_pure_product` ordering so its variables are emitted in `step_idx` order instead of expression-factor order.

## Files Changed

- `gallery/constrained_gen/modules/var_order_planner.py`

## Concrete Change

- `_collect_scoped_product_phase_vars(...)` now accepts `order_by_step_idx`.
- `execution_max_threads_pure_product` calls it with `order_by_step_idx=True`.
- Sorting key is `(step_idx, length_idx)`.

## Verification

- Representative check on the current `order_test.ipynb` target sketch now yields:
  - `grid_0__execution_max_threads_pure_product ['sp_1_1', 'sp_2_1', 'sp_3_1', 'sp_4_1']`

## Remaining Uncertainty

- This tweak only changes intra-phase ordering for `max_threads` pure-product variables.
- No broader validation was run because semantics are unchanged beyond ordering.

## Next Recommended Owner

- `integrator`

## Validation Note

- `single-session validation only`
