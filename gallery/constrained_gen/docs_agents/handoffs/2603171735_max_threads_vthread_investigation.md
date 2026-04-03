## What was run or changed

- Reverted my prior edits in:
  - `modules/constraint_set.py`
  - `modules/deprecated/gpu_case_constraints.py`
  - `modules/deprecated/gpu_projection_diagnostics.py`
  - `modules/gpu_projection_constraints.py`
  - `modules/schedule_generator.py`
  - `../../src/auto_scheduler/exact_gpu_constraints.cc`
- Verified the revert with `git status --short` and `git diff -- <files>`.
- Read current code paths for `max_threads`, `max_vthread`, and `threads per block`.

## Exact files and functions checked

- `modules/constraint_set.py`
  - `preprocess`
  - `_build_max_threads_constraints`
  - `_build_max_vthread_constraints`
  - `_check_max_threads`
  - `_check_max_vthread`
  - `_build_thread_per_block_constraint_item`
  - `_collect_thread_binding_axes`
  - `_collect_vthread_binding_axes`
  - `_ensure_projected_gpu_constraints`
  - `check_all_pruning`
  - `check_all_exact`
  - `_evaluate_exact_upper_bounds`
  - `_ensure_exact_gpu_constraints`
- `modules/deprecated/gpu_case_constraints.py`
  - `build_exact_constraint_nodes`
- `modules/gpu_projection_constraints.py`
  - `build_projected_constraint_nodes`
- `modules/schedule_generator.py`
  - `check_all_hybrid`
  - `_ensure_exact_gpu_constraints`
  - `_ensure_projected_gpu_constraints`
  - inspector `get_raw_exact_constraints_str`
- `../../src/auto_scheduler/exact_gpu_constraints.cc`
  - `GpuCaseStatsCollector`
  - `BuildGpuCaseStats`

## Concrete outcome

- Current symbolic/pruning path:
  - `max_threads` is built from direct binding axes.
  - It includes per-axis `threadIdx.*` items and a block-scoped aggregate `threads per block` item formed from `thread` and `vthread` factors together.
  - `max_vthread` is built separately from direct `vthread` binding axes only, with per-axis upper bounds.
- Current exact path:
  - `max_threads` has a standalone exact case node for aggregate `threads per block`.
  - `max_vthread` has a standalone exact case node for `max_vthread`.
  - C++ exact extraction computes both `vthread_upper` and `max_threads_upper`, while `max_threads_upper` already multiplies `vthread_extent_` into `ThreadsPerBlock()`.
- Current asymmetry:
  - `max_threads` = per-axis thread checks + aggregate `threads per block`
  - `max_vthread` = per-axis vthread checks only
  - exact/projected layers additionally carry a standalone `max_vthread` node

## Important inconsistencies observed

- In `modules/constraint_set.py`, the import of `build_exact_constraint_nodes` is commented out while `_ensure_exact_gpu_constraints()` still calls it.
- In the same file, `_ensure_projected_gpu_constraints()` has a `max_vthread` branch that calls `build_projected_constraint_nodes(g._exact_gpu, ...)`, but the `_ensure_exact_gpu_constraints()` call right above it is commented out.
- So the direct symbolic `max_vthread` path is present and readable, but the standalone projected/exact `max_vthread` plumbing is internally inconsistent in the current file state.

## Artifact paths

- No external validator artifacts.
- Durable note: `docs_agents/handoffs/2603171735_max_threads_vthread_investigation.md`

## Remaining uncertainty

- No behavior run was executed after the revert; this note is based on current source inspection only.
- The exact path appears partially disconnected in Python, so source structure and runnable behavior may differ if some callers never hit that path.

## Next recommended owner

- Integrator for deciding whether `max_threads` and `max_vthread` should be unified around:
  - direct symbolic binding only, or
  - direct symbolic + exact aggregate handling for both
