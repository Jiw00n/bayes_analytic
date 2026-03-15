# 2603132310 Exact GPU Parallel Extract

- Agent: `integrator`
- Date: `2026-03-13 23:10`
- Status: `accepted`
- Decision Topic: `no-cache exact GPU init latency reduction`

## Inputs Considered

- Reviewer note:
- Specialist note:
- Relevant artifacts:
  - [2603132305_optimizer_exact_no_cache_hotspot_investigation.md](/root/work/tvm-ansor/docs_agents/handoffs/2603132305_optimizer_exact_no_cache_hotspot_investigation.md)
  - `/tmp/projected_gpu_full_validation/optimizer/exact_hotspot_20260313/task383_reconfirm.json`
  - `/tmp/projected_gpu_full_validation/optimizer/exact_hotspot_20260313/task383_deep_profile.json`

## Files Checked

- `src/auto_scheduler/exact_gpu_constraints.cc`: `ExtractAllGpuCaseStats`, `LowerSymbolicPostVectorizeWithPipeline`
- `gallery/constrained_gen/modules/exact_gpu_constraints.py`: `build_exact_constraint_nodes`
- `gallery/constrained_gen/modules/constraint_set.py`: `_ensure_exact_gpu_constraints`, `_evaluate_exact_upper_bounds`
- `gallery/constrained_gen/validate.py`: narrow validate harness path

## Decision

- Chosen direction:
  - Parallelize per-case exact GPU stat extraction inside `ExtractAllGpuCaseStats(...)` using `tvm::support::parallel_for`.
  - Keep ordering deterministic by writing each case result into an indexed temporary buffer, then rebuilding the returned `Array`.
- Rejected alternatives:
  - caching extractor output
  - broad symbolic exact rewrite
  - Python-side selector tightening as the primary fix
- Why:
  - The measured hotspot was linear per-case lowering cost in `_EXTRACT_ALL_GPU_CASE_STATS(...)`.
  - This patch preserves the existing exact extraction semantics and only removes serial execution of independent cases.

## Impact

- Files changed:
  - `src/auto_scheduler/exact_gpu_constraints.cc`
- Validation run in main session:
  - `python -m py_compile gallery/constrained_gen/modules/exact_gpu_constraints.py gallery/constrained_gen/modules/constraint_set.py gallery/constrained_gen/validate.py`
  - `python gallery/constrained_gen/validate.py --task-index 29`
  - `python gallery/constrained_gen/validate.py --task-index 383`
- Measured outcome:
  - task `29`: `~2.36s`, pass
  - task `383`: `~8.99s`, pass
  - isolated `_ensure_exact_gpu_constraints()` on task `383`: `~6.11s`
  - prior baseline in this session for task `383` validate total: `~21.1s`
  - prior baseline in this session for task `383` exact init: `~20.3s`
- Remaining uncertainty:
  - separate validator/reviewer sign-off has not completed yet
  - this is narrow validation only

## Next Owner

- Recommended owner: `validator`
- Recommended next step:
  - rerun the same narrow shard as a role-pure validator pass and, if clean, have reviewer judge whether the evidence is sufficient before any broader rollout
