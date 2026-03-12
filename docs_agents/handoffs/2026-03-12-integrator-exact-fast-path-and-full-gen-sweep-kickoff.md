# Exact Fast Path And Full Generation Sweep Kickoff

- Agent: `integrator`
- Date: `2026-03-12`
- Status: `in_progress`

## What Changed

- Added a concrete-evaluable fast path for exact upper-bound evaluation in:
  - [gallery/constrained_gen/modules/constraint_set.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/constraint_set.py)
  - [gallery/constrained_gen/modules/exact_gpu_constraints.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/exact_gpu_constraints.py)
- The new path keeps the existing interval-based fallback for partial assignments, but for fully assigned params it evaluates feasible exact cases concretely and takes the per-kind max directly.

## Local Checks

- Hard-sketch timing check after the fast path:
  - `sketch_index=2` randomization about `0.04 s`
  - `sketch_index=3` randomization about `0.06 s`
- Narrow generation smoke passed:
  - [summary.json](/tmp/projected_gpu_full_validation/validator/local_exact_fast_path_gen_smoke_20260312/summary.json)
  - [invalid.jsonl](/tmp/projected_gpu_full_validation/validator/local_exact_fast_path_gen_smoke_20260312/invalid.jsonl)
- Narrow exact-vs-concrete shard reproduced one pre-existing `shared_memory` projected false reject on `sketch_index=2`:
  - [summary.json](/tmp/projected_gpu_full_validation/validator/local_exact_fast_path_exact_smoke_20260312/summary.json)
  - [mismatch.jsonl](/tmp/projected_gpu_full_validation/validator/local_exact_fast_path_exact_smoke_20260312/mismatch.jsonl)

## Full Sweep

- Kicked off full generation validation in 6 shards under:
  - `/tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/`
- Current shard processes:
  - shard `0`: pid `65399`
  - shard `1`: pid `65400`
  - shard `2`: pid `65401`
  - shard `3`: pid `65402`
  - shard `4`: pid `65403`
  - shard `5`: pid `65404`
- Per-shard outputs:
  - `summary.json`
  - `invalid.jsonl`
  - `run.log`

## Remaining Uncertainty

- The full 912-sketch x 2000-attempt sweep is still running.
- A small probe on `sketch_index=152` suggests the full sweep may take many hours even with 6-way sharding.
- Reviewer sign-off is still pending.

## Next Recommended Owner

- Recommended owner: `reviewer`
- Recommended next step:
  - once the shard summaries land, aggregate counts and identify any invalid-rate-heavy or randomize-fail-heavy sketches
  - decide whether the pre-existing `shared_memory` projected false reject needs specialist follow-up separately from the exact fast path
