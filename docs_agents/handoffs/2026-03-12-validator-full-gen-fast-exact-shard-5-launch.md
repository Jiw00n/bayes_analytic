# Full Generation Validation Shard 5 Launch

- Agent: `validator`
- Date: `2026-03-12`
- Status: `needs_followup`
- Run Type: `generation_validation`

## Scope

- Goal: run the full generation validation shard for sketch indices `760..911` after the exact fast-path change
- Shard or indices: `--all-sketches --start 760 --limit 152`
- Reason for this run: part of the requested full-sweep generation validation at `2000` attempts per sketch

## Files Checked

- `/root/work/tvm-ansor/gallery/constrained_gen/validate_projected_gpu_generation.py`: script entry point
- `/root/work/tvm-ansor/gallery/constrained_gen/modules/projected_gpu_validation.py`: validation helper path used by the script

## Commands Run

```bash
source /root/work/venv/bin/activate
export TVM_HOME=/root/work/tvm-ansor
export PYTHONPATH=$TVM_HOME/python
export TVM_LIBRARY_PATH=$TVM_HOME/build-release
setsid -f bash -lc 'source /root/work/venv/bin/activate && export TVM_HOME=/root/work/tvm-ansor && export PYTHONPATH=$TVM_HOME/python && export TVM_LIBRARY_PATH=$TVM_HOME/build-release && echo $$ > /tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_5/pid.txt && exec python -u /root/work/tvm-ansor/gallery/constrained_gen/validate_projected_gpu_generation.py --all-sketches --start 760 --limit 152 --attempts-per-sketch 2000 --max-retries 64 --print-every 1 --summary-path /tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_5/summary.json --invalid-path /tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_5/invalid.jsonl > /tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_5/run.log 2>&1'
ps -p $(cat /tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_5/pid.txt) -o pid=,stat=,etime=,cmd=
```

## Artifacts

- Raw output dir: `/tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_5/`
- Summary file: `/tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_5/summary.json`
- Detailed file: `/tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_5/invalid.jsonl`
- Progress log: `/tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_5/run.log`
- PID file: `/tmp/projected_gpu_full_validation/validator/full_gen_fast_exact_20260312/shard_5/pid.txt`

## Result Summary

- Sketches processed: not yet available
- Main counts: not yet available
- Reproducer confirmed: `partial`

## Key Findings

- Running the shard through attached `exec_command` sessions was unreliable because long periods with no per-sketch output caused the session to terminate before the validation completed.
- A detached `setsid -f bash -lc ...` launch succeeded, and the validator process was confirmed running via `ps` with PID recorded in `pid.txt`.

## Uncertainty

- The shard was still running when this note was written, so no `summary.json` or invalid-count totals were available yet.
- `run.log` had not emitted the first per-sketch progress line at the time of this note, so the first sketch in the shard may be slow.

## Next Owner

- Recommended owner: `validator`
- Recommended next step: poll `pid.txt`, `run.log`, and `summary.json` until completion, then leave a completion note with final counts and standout invalid/randomize-fail-heavy sketches.
