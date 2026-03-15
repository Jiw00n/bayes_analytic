## Summary

Checked the current constrained generation path for `batch_matmul`-related false rejects.

## What Was Checked

Used the current active workflow:

- `SketchPolicy(...).generate_concrete_sketches()`
- `ScheduleGenerator.from_task_state(...)`
- `gen.randomize_params()`
- `gen.check_all_pruning(...)`
- `gen.check_all_exact(...)`
- `gen.check_all_final(...)`

Also ran a deeper probe that sampled params with full-validation disabled and then compared:

- `check_all_pruning(params)`
- `check_all_exact(params)`
- `check_all_final(params)`

to look for `final == []` with symbolic/exact violations still present.

## Batch-Matmul Tasks Probed

- `149` `vm_mod_fused_nn_batch_matmul_1`
- `293` `vm_mod_fused_nn_batch_matmul_2`
- `383` `vm_mod_fused_nn_batch_matmul`
- `492` `vm_mod_fused_nn_batch_matmul`
- `584` `vm_mod_fused_nn_batch_matmul`
- `585` `vm_mod_fused_nn_batch_matmul_3`
- `763` `vm_mod_fused_nn_batch_matmul_2`
- `819` `vm_mod_fused_nn_batch_matmul_4`

## Concrete Outcomes

- Single `gen.randomize_params()` checks on representative batch-matmul tasks `149, 293, 383, 492, 584, 585` all returned params with:
  - `pruning = []`
  - `exact = []`
  - `final = []`
- Deeper probe over tasks `149, 293, 383, 492, 584, 585, 763, 819`:
  - `40` sampled assignments per task
  - `1` concrete sketch per task
  - no reproducer found where `check_all_final(params)` passed but `check_all_pruning(params)` or `check_all_exact(params)` failed

## Remaining Uncertainty

- This probe does **not** prove that no projected-domain false reject remains inside prefix filtering or candidate elimination before a full assignment is materialized.
- It only shows that for the current batch-matmul shard, I did not find a reproducer at the full-assignment check level.

## Next Recommended Owner

- `validator` if you want a broader shard focused on prefix/domain false rejects rather than full-assignment false rejects.
