## What was changed

- Removed the temporary `max_threads` projected-thread special-case that had been added for runtime-index thread blocks.
- Removed the temporary symbolic-state dependence on the original concrete state for loop `min` recovery.

## Files changed

- `modules/constraint_set.py`
  - removed projected-thread integration for `max_threads`
  - removed projected-thread integration for `max_threads_per_block`
  - restored both families to direct symbolic binding collection
  - deleted the temporary helpers and caches related to:
    - runtime-index thread block detection
    - projected thread item materialization
    - original concrete bound caching for that path

- `modules/gpu_projection_constraints.py`
  - removed projected thread-launch collection from projected GPU metadata
  - removed `project_runtime_upper_expr(...)`
  - removed `build_projected_thread_launch_items(...)`
  - kept projected support for `vectorize` and `shared_memory`

- `modules/schedule_generator.py`
  - removed temporary caches introduced only for the projected-thread path

- `modules/symbolic_state_bridge.py`
  - `build_symbolic_state()` no longer stores the original concrete state on the symbolic state

- `modules/symbolic_state.py`
  - removed `_min_source_state`
  - clone path no longer carries original concrete-state min source

- `modules/transform_applier.py`
  - removed original concrete-state min replay and recovery
  - `_clone_real_stage(...)` now initializes loop mins to `0`
  - `_restore_stage_extents_if_needed(...)` no longer reads `range.min` from original concrete state
  - `_infer_bound_final(...)` no longer overwrites `min_value` from original concrete state
  - `CacheReadStep` / `CacheWriteStep` insertion no longer clone mins from original concrete replay
  - `_recover_iter_min(...)` now uses only current symbolic state / saved symbolic mins

## Why this was changed

### 1. `max_threads` projection was not justified by the observed case

The motivating example had runtime-index loops like:

```text
for p (...)
for ci (...)
```

but the actual execution constraints for the same task were still reconstructed purely from split variables:

- `max_threads`
- `max_threads_per_block`
- `max_vthread`

all had `runtime_vars=[]` when inspecting the actual constraint trees.

That means the mere presence of runtime-index loops in symbolic state does **not** imply that `max_threads`-family constraints need projection.

The right criterion is:

- projection is only needed if the **constraint target itself** still contains runtime vars
- not merely because some loop min/index map contains runtime vars

For the inspected task (`570`), that criterion was not met.

### 2. Concrete-state-backed symbolic min recovery was not a valid symbolic representation

The previous temporary fix made symbolic-state loop mins depend on the original concrete state through:

- original-state replay
- `infer_bound_from_state(original_state)`
- direct copying of `range.min`

This was not canonical symbolic reconstruction.

If concrete assignments change, those mins also change, which means the symbolic state was not representing sketch structure alone.

So the concrete-state dependency was removed.

## Validation

- `python3 -m py_compile modules/constraint_set.py modules/gpu_projection_constraints.py modules/schedule_generator.py`
  - passed after removing projected-thread path

- `python3 generate_programs.py --task-index 400 --records-per-task 1 --workers 1`
  - passed
  - summary:
    - `selected=1 ok=1 skipped=1 exhausted=0 failed=0`

- `python3 -m py_compile modules/sym_types.py modules/symbolic_state.py modules/symbolic_state_bridge.py modules/transform_applier.py`
  - passed after removing concrete-state min dependence

- `build_symbolic_state()` still runs successfully on task `570`

## Important current state

- `SymbolicState` is again canonical with respect to sketch structure:
  - it no longer depends on the original concrete state for loop mins

- However, this also means runtime-index loop mins such as:

```text
for p (blockIdx_x // ... + threadIdx_x ..., 1)
for ci (blockIdx_x % ... + threadIdx_x ..., 1)
```

are **not** currently reconstructed.

The current symbolic state falls back to:

```text
for p (0,1)
for ci (0,1)
```

for those cache/shared/runtime-index loops unless their mins are derived directly by symbolic step logic.

This is expected after removing concrete-state copying.

## Recommended next step

If those runtime-index loop mins must appear symbolically, the next work should be:

- **not** to reintroduce concrete-state copying
- but to symbolically reconstruct loop mins inside `TransformApplier`

The most likely place is:

- `CacheReadStep` / `CacheWriteStep` inserted stages
- possibly via consumer-side stencil/index analysis already present in:
  - `modules/transform_applier.py`
    - `_analyze_cache_read_stencil`
    - `_analyze_index_expr`

In other words:

- current state is symbolically sound but less expressive
- the next step is symbolic `min` reconstruction, not concrete-state restoration

## Artifact path

- `docs_agents/handoffs/260317205455_remove_projected_thread_and_concrete_min_dependency.md`
