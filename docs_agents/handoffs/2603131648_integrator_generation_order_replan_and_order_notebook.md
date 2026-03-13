## Summary

- Replaced the planner phase families with the new execution / memory / instruction ordering requested for grid-local generation order.
- Updated `order_test.ipynb` so the final cell walks each phase in order and prints the constraint state after that prefix.
- Ran a narrow smoke on one pure-product case and one non-product case; both generated concrete-valid schedules.

## Files Checked / Changed

- `gallery/constrained_gen/modules/var_order_planner.py`
- `gallery/constrained_gen/modules/schedule_generator.py`
- `gallery/constrained_gen/order_test.ipynb`

## Concrete Changes

- Execution level now uses:
  - `execution_max_threads_pure_product`
  - `execution_max_vthread_pure_product`
  - `execution_block_split_structure`
  when a pure-product execution constraint exists.
- Execution level now uses:
  - `execution_non_product_direct_arm`
  - `execution_non_product_gate_vars`
  when the scope has no execution pure-product.
- Memory level now uses:
  - `memory_split_structure`
  with `step_idx` ascending and inner `length_idx` first.
- Instruction level now uses:
  - `instruction_scaled_product_upper_bound`
  - `instruction_non_product_min`
- Vectorize non-product vars are now collected without falling back to legacy ordering; a representative leftover (`sp_26_0`) is now assigned to `instruction_non_product_min`.
- `order_test.ipynb` now:
  - prints fixed singleton domains before generation
  - prints constraints before generation
  - iterates phase-by-phase, applies prefix generation up to that phase, and prints params / fixed values / domains / constraints after each phase

## Artifacts

- `/tmp/projected_gpu_full_validation/validator/order_replan_smoke_260313/summary.json`
- `/tmp/projected_gpu_full_validation/validator/order_replan_smoke_260313/details.jsonl`

## Outcome

- Representative pure-product case (`sketch_idx=160` in `order_test.ipynb`) now shows the requested phase order with no leftover legacy vars.
- A sampled non-product case (`vm_mod_fused_nn_adaptive_avg_pool3d`) also shows the requested execution `direct-arm -> gate-vars` ordering.
- Narrow smoke result: `rows=6`, `concrete_invalid=0`.

## Remaining Uncertainty

- This change was validated only on a narrow smoke, not a broader generation sweep.

## Next Recommended Owner

- `validator` for a broader shard if we want to lock in the new ordering semantics across more sketches.

## Validation Note

- `single-session validation only`
