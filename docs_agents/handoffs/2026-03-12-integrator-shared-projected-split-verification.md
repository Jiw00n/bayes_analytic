# Shared Projected Split Verification

- Agent: `integrator`
- Date: `2026-03-12`
- Status: `completed`

## What Was Checked

- Verified the current shared-memory projected path in:
  - [constraint_set.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/constraint_set.py)
  - [exact_gpu_constraints.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/exact_gpu_constraints.py)
- Verified that `build_shared_memory_constraints()` now uses the lightweight projected builder and does not force `_ensure_exact_gpu_constraints()` during init.
- Re-measured hard-sketch init on `sketch_index=2`.
- Verified lazy exact behavior for `enabled_constraints=('shared_memory',)`.
- Compared the symbolic shared-memory sum against concrete lowered shared allocation on representative hard sketches.

## Files And Functions Checked

- [constraint_set.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/constraint_set.py)
  - `build_shared_memory_constraints`
  - `_ensure_projected_gpu_constraints`
- [exact_gpu_constraints.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/exact_gpu_constraints.py)
  - `build_projected_gpu_context`
  - `build_projected_shared_memory_constraint_node`
- [transform_applier.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/transform_applier.py)
  - shared fused extent capture around `_apply_fuse` and `_apply_split`
- [symbolic_state.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/symbolic_state.py)
  - `get_shared_memory_extents`

## Commands And Artifacts

- Local timing/behavior checks were run inline from the repo root with:
  - `source /root/work/venv/bin/activate`
  - `export TVM_HOME=/root/work/tvm-ansor`
  - `export PYTHONPATH=$TVM_HOME/python`
  - `export TVM_LIBRARY_PATH=$TVM_HOME/build-release`
- No standalone artifact file was written for these ad hoc checks.

## Concrete Outcome

- Current hard-sketch init timings on `sketch_index=2`:
  - `shared_only`: about `0.244 s`
  - `vectorize_only`: about `0.224 s`
  - `all`: about `0.248 s`
  - `max_threads + innermost_split + split_structure`: about `0.0009 s`
- `shared_only` no longer builds exact GPU constraints during init:
  - before projected build: `_exact_gpu is None`
  - after `build_shared_memory_constraints()`: `_exact_gpu is None`
  - after `check_shared_memory_exact()`: `_exact_gpu is not None`
- On hard sketches `2` and `3`, the symbolic shared-memory total matched the concrete lowered shared allocation bytes on sampled params in the checks run during this session.

## Remaining Uncertainty

- I did not run a broad validation sweep in this session.
- Validator smoke regression is still pending as a separate role-owned follow-up.

## Next Recommended Owner

- Recommended owner: `validator`
- Recommended next step: run a narrow projected-generation smoke shard for the shared-heavy sketches and record raw artifacts.
