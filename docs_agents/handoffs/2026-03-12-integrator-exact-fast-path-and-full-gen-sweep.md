# Exact Fast Path And Full Generation Sweep

- Agent: `integrator`
- Date: `2026-03-12`
- Status: `in_progress`

## What Changed

- Updated [exact_gpu_constraints.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/exact_gpu_constraints.py)
  - `build_exact_constraint_nodes`
  - added `case_expr_vars` metadata per exact kind
- Updated [constraint_set.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/constraint_set.py)
  - `_evaluate_exact_upper_bounds`
  - added `_can_evaluate_exact_cases_concretely`
  - added `_evaluate_exact_upper_bounds_concretely`
- Concrete effect:
  - when exact case expressions depend only on concrete assigned params, exact checking now scans feasible selector cases once and uses `evaluate()` instead of `interval()` for the case expressions
  - the old interval-based fallback remains in place for non-concrete cases

## Local Checks Run

- Hard-sketch equivalence/timing script on sketches `2` and `3`
  - confirmed the new path returns concrete maxima where the old interval path often returned `None`
  - observed `randomize_params(max_retries=2)` down to roughly:
    - sketch `2`: `~0.042 s`
    - sketch `3`: `~0.061 s`
- Narrow generation smoke:
  - `python gallery/constrained_gen/validate_projected_gpu_generation.py --all-sketches --start 2 --limit 2 --attempts-per-sketch 20 --max-retries 8 --print-every 1 --summary-path /tmp/projected_gpu_full_validation/validator/local_exact_fast_path_gen_smoke_20260312/summary.json --invalid-path /tmp/projected_gpu_full_validation/validator/local_exact_fast_path_gen_smoke_20260312/invalid.jsonl`
  - outcome:
    - `sketch=2`: `success=20 invalid=0 rand_fail=0`
    - `sketch=3`: `success=20 invalid=0 rand_fail=0`
- Narrow exact-vs-concrete shard:
  - `python gallery/constrained_gen/validate_exact_gpu_constraints.py --start 2 --limit 2 --summary-path /tmp/projected_gpu_full_validation/validator/local_exact_fast_path_exact_smoke_20260312/summary.json --mismatch-path /tmp/projected_gpu_full_validation/validator/local_exact_fast_path_exact_smoke_20260312/mismatch.jsonl --print-every`
  - outcome:
    - reproduced one pre-existing projected `shared_memory` false reject on `sketch_index=2`
    - mismatch was `runtime_projection_upper_bound_insufficient`
    - this did not appear to be introduced by the new exact fast path because the exact side remained `final_ok=true`, `concrete_ok=true`

## Full Sweep Launched

- Full generation validation was launched as six background shards under:
  - `/tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/`
- Active shard commands:
  - shard `0`: `--start 0 --limit 152`
  - shard `1`: `--start 152 --limit 152`
  - shard `2`: `--start 304 --limit 152`
  - shard `3`: `--start 456 --limit 152`
  - shard `4`: `--start 608 --limit 152`
  - shard `5`: `--start 760 --limit 152`
- Common validation flags:
  - `--all-sketches`
  - `--attempts-per-sketch 2000`
  - `--max-retries 64`
  - `--print-every 5`
- Active background python PIDs at note time:
  - `65399`
  - `65400`
  - `65401`
  - `65402`
  - `65403`
  - `65404`

## Artifact Paths

- Narrow generation smoke:
  - `/tmp/projected_gpu_full_validation/validator/local_exact_fast_path_gen_smoke_20260312/summary.json`
  - `/tmp/projected_gpu_full_validation/validator/local_exact_fast_path_gen_smoke_20260312/invalid.jsonl`
- Narrow exact-vs-concrete:
  - `/tmp/projected_gpu_full_validation/validator/local_exact_fast_path_exact_smoke_20260312/summary.json`
  - `/tmp/projected_gpu_full_validation/validator/local_exact_fast_path_exact_smoke_20260312/mismatch.jsonl`
- Full sweep shard roots:
  - `/tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_0/`
  - `/tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_1/`
  - `/tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_2/`
  - `/tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_3/`
  - `/tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_4/`
  - `/tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_5/`

## Remaining Uncertainty

- The full `912 sketches x 2000 attempts` sweep is still running and had not produced `summary.json` files yet at note time.
- A quick single-sketch probe in the same area suggests the full sweep is a multi-hour to multi-day batch even with six-way parallelism.
- The projected `shared_memory` false reject on `sketch_index=2` is still open and may affect overall randomize-fail rates in some regions.

## Next Recommended Owner

- Recommended owner: `validator`
- Recommended next step:
  - monitor the six background shard processes to completion
  - preserve each shard `summary.json` and `invalid.jsonl`
  - then hand the full artifact set to `reviewer` for aggregation and issue triage
