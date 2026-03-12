# Shared Split Generation Smoke

- Agent: `validator`
- Date: `2026-03-12`
- Status: `completed`
- Run Type: `generation_validation`

## Scope

- Goal: confirm that the current projected shared-memory path still passes a narrow concrete-validation smoke run on shared-heavy sketches
- Shard or indices: `all-sketches start=2 limit=2` which maps to sketch indices `2` and `3`
- Reason for this run: post-change regression smoke after the projected shared-memory split; `single-session validation only`

## Files Checked

- [gallery/constrained_gen/validate_projected_gpu_generation.py](/root/work/tvm-ansor/gallery/constrained_gen/validate_projected_gpu_generation.py): `run`
- [gallery/constrained_gen/modules/projected_gpu_validation.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/projected_gpu_validation.py): `build_schedule_generator`, concrete-diagnostic helpers used by the driver

## Commands Run

```bash
source /root/work/venv/bin/activate
export TVM_HOME=/root/work/tvm-ansor
export PYTHONPATH=$TVM_HOME/python
export TVM_LIBRARY_PATH=$TVM_HOME/build-release
python gallery/constrained_gen/validate_projected_gpu_generation.py --all-sketches --start 2 --limit 2 --attempts-per-sketch 1 --max-retries 1 --summary-path /tmp/projected_gpu_full_validation/validator/shared_split_smoke_20260312/summary.json --invalid-path /tmp/projected_gpu_full_validation/validator/shared_split_smoke_20260312/invalid.jsonl
```

## Artifacts

- Raw output dir: `/tmp/projected_gpu_full_validation/validator/shared_split_smoke_20260312/`
- Summary file: `/tmp/projected_gpu_full_validation/validator/shared_split_smoke_20260312/summary.json`
- Detailed file: `/tmp/projected_gpu_full_validation/validator/shared_split_smoke_20260312/invalid.jsonl`

## Result Summary

- Sketches processed: `2`
- Main counts: `randomize_success=2`, `randomize_fail=0`, `concrete_pass=2`, `concrete_invalid=0`
- Reproducer confirmed: `no`

## Key Findings

- Sketch `2` (`vm_mod_fused_nn_batch_matmul_3`) succeeded on its single sampled attempt and passed concrete GPU verification.
- Sketch `3` (`vm_mod_fused_nn_batch_matmul_1`) succeeded on its single sampled attempt and passed concrete GPU verification.

## Uncertainty

- Missing evidence: no exact-vs-concrete mismatch shard was run here, only generation validation.
- Known limitations of this run: this is a tiny smoke shard with one attempt per sketch, so it does not establish full-rate behavior or broad false-reject coverage.

## Next Owner

- Recommended owner: `reviewer`
- Recommended next step: inspect the smoke artifact and decide whether the evidence is sufficient, or request a slightly wider shard before accepting the shared-memory split as low risk.
