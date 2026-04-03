## What was run or changed

- Preserved loop `min` in symbolic state instead of collapsing everything to `(0, extent)`.
- Added `max_threads` runtime-index special handling:
  - if a block's concrete `thread extent` contains `blockIdx_*` or `threadIdx_*`,
    that block's `thread` side is treated as `projected_thread`
  - `vthread` remains direct symbolic
  - `max_threads_per_block` is now `thread * vthread`, with the `thread` side projected only for those special blocks
- Verified narrow generator execution for an already-generated task and confirmed skip behavior still works.

## Exact files and functions changed

- `modules/sym_types.py`
  - `SymIter`
    - added `min_value`
    - repr/clone updated to show and preserve `(min, extent)`

- `modules/symbolic_state.py`
  - symbolic-state storage now keeps `min_value`
  - printing now shows `(min, extent)`
  - trivial-loop suppression now keeps loops with `extent == 1` if `min != 0`
  - added state holders used by later recovery:
    - `_min_source_state`
    - `_ca_saved_mins`

- `modules/symbolic_state_bridge.py`
  - `build_symbolic_state`
    - stores original concrete state into `sym._min_source_state`

- `modules/transform_applier.py`
  - introduced original-state min recovery helpers:
    - `_to_state_object`
    - `_replay_partial_state`
    - `_infer_bound_partial_from_state`
    - `_infer_bound_from_state`
    - `_get_safe_saved_min`
    - `_symexpr_from_tir`
    - `_recover_iter_min`
  - `_clone_real_stage` now copies `min_value`
  - `_restore_stage_extents_if_needed` now restores:
    - extent from structural replay
    - min from original concrete replay
  - `_infer_bound_final` refreshes final `min_value` from original full state
  - `_apply_cache_read`, `_apply_cache_write`, `_apply_compute_at`, `_apply_compute_root`
    - now preserve non-zero symbolic loop mins
  - `_clamp_positive_split_extent`
    - no longer collapses `min(sp_i_j, 1)` to `1` just because clamp bound is `1`

- `modules/constraint_set.py`
  - binding item collection now stores:
    - `sym_min`
    - `min_tree`
  - preprocess resets caches for projected-thread integration:
    - `_min_source_bounded`
    - `_runtime_index_thread_block_scopes`
    - `_effective_thread_constraint_items`
    - `_projected_thread_items_by_scope`
  - added helpers:
    - `_get_min_source_bounded`
    - `_has_runtime_index_vars_in_primexpr`
    - `_collect_runtime_index_thread_block_scopes`
    - `_build_projected_thread_items_by_scope`
    - `_build_effective_thread_constraint_items`
  - `_build_max_threads_constraints`
    - now uses effective thread items
  - `_build_max_threads_per_block_constraints`
    - now uses effective thread items plus direct `vthread`
  - improved projected block-scope matching:
    - projected launch ordinals are now matched only against runtime-index scopes,
      not all direct thread scopes

- `modules/gpu_projection_constraints.py`
  - `_collect_projected_gpu_metadata`
    - now also collects `thread_launches`
  - `build_projected_gpu_context`
    - now stores `thread_launches`
  - added public helper:
    - `project_runtime_upper_expr`
  - added:
    - `build_projected_thread_launch_items`

- `modules/schedule_generator.py`
  - `__init__`
    - added caches used by the new projected-thread path:
      - `_min_source_bounded`
      - `_runtime_index_thread_block_scopes`
      - `_effective_thread_constraint_items`
      - `_projected_thread_items_by_scope`

## Concrete outcome

- `SymbolicState` no longer loses non-zero loop mins by construction.
- `max_threads` family now has this intended structure:
  - `max_threads`
    - per-axis thread item
    - source is either `direct_thread` or `projected_thread`
  - `max_threads_per_block`
    - `thread * vthread`
    - thread side becomes projected only when the block is classified as runtime-index thread block
  - `max_vthread`
    - unchanged direct symbolic path

- The runtime-index block classification currently uses the original concrete bound:
  - `real_it.range.extent`
  - if that expression contains `blockIdx_*` or `threadIdx_*`, the block scope is marked as special

- Projected thread launch extents are collected from pre-vectorize TIR and then upper-projected over runtime domains before being turned into constraint items.

## Validation run

- Passed:
  - `python3 -m py_compile modules/sym_types.py modules/symbolic_state.py modules/symbolic_state_bridge.py modules/transform_applier.py modules/constraint_set.py modules/gpu_projection_constraints.py modules/schedule_generator.py`
  - `python3 generate_programs.py --task-index 400 --records-per-task 1 --workers 1`
    - result: `selected=1 ok=1 skipped=1 exhausted=0 failed=0`

- Spot-check:
  - task `570`
    - `RUNTIME_SCOPES []`
    - all `max_threads` items still reported `source='direct_thread'`

## Important current limitations

- I did not find a real sampled task yet where `_collect_runtime_index_thread_block_scopes(...)` returned a non-empty scope during the quick checks I ran.
  - So the new projected-thread path compiles and is wired in,
    but I have not yet exercised it on a confirmed runtime-index thread block from the current task set.

- Planner is not updated yet.
  - Current planner still derives pure-product phases directly from `item['tree']`.
  - This is in `modules/var_order_planner.py`, especially:
    - `_build_var_order_phase_entries`
    - `_collect_scoped_product_phase_vars`
  - That means projected-thread blocks will not get the intended generation order semantics automatically.

## Why planner still needs work

- Right now planner reads product structure from:
  - `g.constraint_set._extract_product_form_meta(item['tree'])`
- This worked for direct items because actual mixed thread/vthread trees could still normalize into a usable product form.
- For projected-thread blocks, `max_threads_per_block.tree` becomes:
  - `projected_thread_tree * direct_vthread_tree`
- That whole tree may no longer have a usable pure-product decomposition.
- As a result:
  - `thread_pure` can disappear for projected-thread blocks
  - `vthread_pure_from_block` can also disappear because planner looks at the whole aggregate tree rather than the vthread-side generation view

## Recommended next step

- Update planner to separate:
  - checking/pruning tree
  - generation-time pure-product view

- The clean path is:
  1. add planner-only product metadata to execution items in `modules/constraint_set.py`
     - e.g. `planner_product_meta`
     - and for `max_threads_per_block`, split planner views for thread/vthread sides
  2. update `modules/var_order_planner.py`
     - make `thread_pure` consume planner meta from `max_threads`
     - make `vthread_pure` consume planner meta from:
       - `max_vthread`
       - vthread-side planner meta of `max_threads_per_block`
  3. keep projected-thread blocks out of pure-thread generation unless a safe planner-side product view is explicitly provided

## Artifact path

- `docs_agents/handoffs/260317202816_projected_thread_and_symbolic_min_handoff.md`
