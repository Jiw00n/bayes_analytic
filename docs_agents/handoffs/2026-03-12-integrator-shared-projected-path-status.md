# Shared Projected Path Status

- Agent: `integrator`
- Date: `2026-03-12`
- Status: `completed`

## What Was Checked

- Verified the current `shared_memory` projected path in:
  - [gallery/constrained_gen/modules/constraint_set.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/constraint_set.py)
  - [gallery/constrained_gen/modules/exact_gpu_constraints.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/exact_gpu_constraints.py)
- Measured hard-sketch init time on `sketch_index=2` for:
  - `shared_only`
  - `vectorize_only`
  - `all`
- Ran a concrete comparison between symbolic shared-memory totals and lowered shared `Allocate` totals on the hard batch-matmul sketches.

## Concrete Outcome

- The current workspace already routes `shared_memory` projected construction through `build_projected_shared_memory_constraint_node(...)` instead of exact case-table enumeration.
- On `sketch_index=2`, init timings are now roughly:
  - `shared_only`: `0.244 s`
  - `vectorize_only`: `0.224 s`
  - `all`: `0.248 s`
- `shared_only` and `all` no longer build exact GPU constraints during init.
- For sampled params on `sketch_index=2` and `sketch_index=3`, the symbolic shared-memory total derived from shared-stage extents matched the concrete lowered shared `Allocate` byte total on every sample checked.

## Artifacts

- Validator smoke note: [2026-03-12-validator-shared-split-generation-smoke.md](/root/work/tvm-ansor/docs_agents/handoffs/2026-03-12-validator-shared-split-generation-smoke.md)
- Validator raw summary: `/tmp/projected_gpu_full_validation/validator/shared_split_smoke_20260312/summary.json`

## Remaining Uncertainty

- The narrow smoke shard does not establish full-rate generation behavior.
- The next likely performance target is no longer `shared_memory` init; if more bottleneck work is needed, re-profile `randomize_params()` and any remaining exact-build paths under wider workloads.

## Next Recommended Owner

- Recommended owner: `reviewer`
- Recommended next step: decide whether the narrow smoke artifact is sufficient or whether a slightly wider shard is needed before treating the shared projected path as accepted.
