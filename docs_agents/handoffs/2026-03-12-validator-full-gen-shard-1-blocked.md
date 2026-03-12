# Full Generation Validation Shard 1 Blocked

- Agent: `validator`
- Date: `2026-03-12`
- Status: `blocked`
- Run Type: `generation_validation`

## Scope

- Goal:
  - run full generation validation for shard 1 with `start=152`, `limit=152`, `attempts_per_sketch=2000`
- Shard or indices:
  - requested shard range `152..303`
- Reason for this run:
  - post-patch full-sweep validation for the exact fast-path work

## Files Checked

- `gallery/constrained_gen/validate_projected_gpu_generation.py`: generation validation entry point
- `gallery/constrained_gen/modules/projected_gpu_validation.py`: shared validation helpers used by the driver
- `gallery/constrained_gen/modules/schedule_generator.py`: observed indirectly because the driver keeps a per-sketch generator and concrete-final cache alive across attempts

## Commands Run

```bash
source /root/work/venv/bin/activate
export TVM_HOME=/root/work/tvm-ansor
export PYTHONPATH=$TVM_HOME/python
export TVM_LIBRARY_PATH=$TVM_HOME/build-release

python gallery/constrained_gen/validate_projected_gpu_generation.py \
  --all-sketches --start 152 --limit 152 \
  --attempts-per-sketch 2000 --max-retries 64 \
  --summary-path /tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_1/summary.json \
  --invalid-path /tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_1/invalid.jsonl

python gallery/constrained_gen/validate_projected_gpu_generation.py \
  --all-sketches --start 152 --limit 1 \
  --attempts-per-sketch 20 --max-retries 64 --print-every 1 \
  --summary-path /tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_1_probe_small/summary.json \
  --invalid-path /tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_1_probe_small/invalid.jsonl

python gallery/constrained_gen/validate_projected_gpu_generation.py \
  --all-sketches --start 152 --limit 1 \
  --attempts-per-sketch 200 --max-retries 64 --print-every 1 \
  --summary-path /tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_1_probe_200/summary.json \
  --invalid-path /tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_1_probe_200/invalid.jsonl

python gallery/constrained_gen/validate_projected_gpu_generation.py \
  --all-sketches --start 152 --limit 1 \
  --attempts-per-sketch 1000 --max-retries 64 --print-every 1 \
  --summary-path /tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_1_probe_1000/summary.json \
  --invalid-path /tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_1_probe_1000/invalid.jsonl
```

## Artifacts

- Raw output dir:
  - `/tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/`
- Stable probe summaries:
  - `/tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_1_probe_small/summary.json`
  - `/tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_1_probe_200/summary.json`
- Stable probe invalid files:
  - `/tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_1_probe_small/invalid.jsonl`
  - `/tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_1_probe_200/invalid.jsonl`
- Requested shard artifacts:
  - not produced

## Result Summary

- Sketches processed:
  - full shard: not completed
  - probe sketch `152`: completed at `20` attempts and `200` attempts
- Main counts:
  - probe `20 attempts`: `randomize_success=20`, `concrete_invalid=0`, `randomize_fail=0`
  - probe `200 attempts`: `randomize_success=200`, `concrete_invalid=0`, `randomize_fail=0`
- Reproducer confirmed:
  - `partial`

## Key Findings

- The requested direct shard command (`152 sketches x 2000 attempts`) did not produce any summary or invalid artifact before the process died.
- A single-sketch probe at `attempts_per_sketch=1000` also died before writing artifacts.
- A single-sketch probe at `attempts_per_sketch=200` is stable and completes successfully.
- A batched orchestration attempt using `152 sketches x 200 attempts` as one subprocess also died with `SIGTERM` after reaching only `[5/152]`.
- This means the blocker is not the shard index itself; it is the long-lived per-process validation budget.

## Uncertainty

- Missing evidence:
  - exact root cause of the process termination
- Known limitations of this run:
  - shard 1 was not completed
  - shard 5 was not started because shard 1 never reached a viable execution strategy
- Likely hypothesis from observed behavior only:
  - the current validation strategy does not sustain very large per-process attempt counts
  - a plausible contributor is growth in per-sketch in-memory state such as concrete-result caching, but this note does not establish root cause

## Next Owner

- Recommended owner: `reviewer`
- Recommended next step:
  - review whether this validator evidence is sufficient to escalate to `integrator` or `specialist` for a validation-execution strategy change before attempting shard 1 and shard 5 again
