# 2603131239 Full-Task Repeated Raw-State Sweep

- Agent: `validator`
- Date: `2026-03-13 12:39`
- Status: `completed`
- Run Type: `generation_validation`

## Scope

- Goal: validate the structural representative state refactor against repeated raw concrete states across all tasks from `gallery/dataset/network_info`
- Shard or indices: full task set, `849` tasks
- Reason for this run: confirm that repeated `SketchPolicy.generate_concrete_sketches()` calls remain clean when building `SymbolicState` through the structural representative state path

## Files Checked

- `gallery/constrained_gen/modules/param_manager.py`: `build_symbolic_state`
- `gallery/constrained_gen/modules/schedule_generator.py`: `ScheduleGenerator(..., task=task, base_state=state)`, `randomize_params`
- `gallery/constrained_gen/modules/tvm_verify.py`: `params_to_state_from_state`, `lower_with_gpu_passes`, `verify_gpu_module_errors`
- `gallery/constrained_gen/modules/structural_sketch.py`: structural representative state path

## Commands Run

```bash
source /root/work/venv/bin/activate
export TVM_HOME=/root/work/tvm-ansor
export PYTHONPATH=$TVM_HOME/python
export TVM_LIBRARY_PATH=$TVM_HOME/build-release
cd /root/work/tvm-ansor/gallery/constrained_gen
python - <<'PY'
# full-task repeated raw-state validation driver
# for each task:
#   create SketchPolicy(task, RandomModel())
#   call generate_concrete_sketches() 3 times
#   for each raw state:
#     build_symbolic_state(task, state)
#     ScheduleGenerator(sym, task=task, base_state=state)
#     randomize_params(rng=..., max_retries=1)
#     params_to_state_from_state(...)
#     lower_with_gpu_passes(...)
#     verify_gpu_module_errors(...)
# write summary.json and details.jsonl under /tmp/projected_gpu_full_validation/validator/raw_state_repeated_full_2603131206/
# PY
```

## Artifacts

- Raw output dir: `/tmp/projected_gpu_full_validation/validator/raw_state_repeated_full_2603131206/`
- Summary file: `/tmp/projected_gpu_full_validation/validator/raw_state_repeated_full_2603131206/summary.json`
- Detailed file: `/tmp/projected_gpu_full_validation/validator/raw_state_repeated_full_2603131206/details.jsonl`
- Run log: `/tmp/projected_gpu_full_validation/validator/raw_state_repeated_full_2603131206/run.log`

## Result Summary

- Sketches processed: `849` tasks, `2547` repeated `generate_concrete_sketches()` calls, `2736` raw states validated
- Main counts:
  - `randomize_success=2736`
  - `concrete_valid=2736`
  - `concrete_invalid=0`
  - `exceptions=0`
  - `elapsed_sec=1874.48`
- Reproducer confirmed: `yes`

## Key Findings

- No repeated raw-state failures were found in the full-task sweep. Every validated raw state produced params, reconstructed cleanly via `params_to_state_from_state(...)`, and passed concrete lowering plus `verify_gpu_module_errors(...)`.
- Structural diversity across repeated raw states was observed in `45` tasks, with `unique_sketch_fingerprint_count > 1` and matching `unique_step_signature_count > 1`. This appears to be expected sketch diversity from repeated `generate_concrete_sketches()` calls, not a refactor regression.
- Family coverage in this run:
  - `conv2d=626`
  - `dense=46`
  - `batch_matmul=42`
  - `pool=54`
  - `misc=81`

## Uncertainty

- Missing evidence: this run used `repeat_count_per_task=3` and `attempts_per_state=1`, so coverage is broad but intentionally shallow per raw state.
- Known limitations of this run: no multi-process cold-start matrix and no deeper per-state retry depth beyond a single `randomize_params(max_retries=1)` attempt.

## Next Owner

- Recommended owner: `reviewer`
- Recommended next step: review the full-task raw-state sweep artifacts and decide whether this evidence is sufficient to accept the structural representative state refactor for repeated raw-state generation paths.
