# projected/exact owner split

## What changed

- Split projected GPU pruning helpers out of exact owner code.
- Added Python projected owner:
  - `gallery/constrained_gen/modules/gpu_projection_constraints.py`
- Added C++ projected registration owner:
  - `src/auto_scheduler/projected_gpu_constraints.cc`
- Kept exact case-table materialization and post-vectorize exact lowering in:
  - `gallery/constrained_gen/modules/gpu_case_constraints.py`
  - `src/auto_scheduler/exact_gpu_constraints.cc`

## Files changed

- `gallery/constrained_gen/modules/gpu_projection_constraints.py`
  - owns:
    - `build_projected_gpu_context(...)`
    - `build_projected_vectorize_constraint_node(...)`
    - `build_projected_shared_memory_constraint_node(...)`
    - `build_projected_constraint_nodes(...)`
- `gallery/constrained_gen/modules/gpu_case_constraints.py`
  - now exact-focused
  - no longer owns projected helper implementations
  - still imports projected context internally for `build_exact_constraint_nodes(...)`
- `gallery/constrained_gen/modules/constraint_set.py`
  - active projected imports now come from `gpu_projection_constraints.py`
  - exact imports stay on `gpu_case_constraints.py`
- `src/auto_scheduler/projected_gpu_constraints.cc`
  - registers:
    - `constrained_gen.lower_symbolic_pre_vectorize`
    - `constrained_gen.list_vectorized_loop_extents`
- `src/auto_scheduler/exact_gpu_constraints.cc`
  - keeps only exact registrations:
    - `constrained_gen.lower_symbolic_post_vectorize`
    - `constrained_gen.extract_gpu_case_stats`
    - `constrained_gen.extract_all_gpu_case_stats`
  - removed leftover pre-vectorize lowering implementation that had become unused after the split
- active docs updated:
  - `AGENTS.md`
  - `docs_agents/CODEX_WORKING_CONTEXT.md`
  - `docs_agents/HANDOFF_WORKFLOW.md`

## Why

- Active generation no longer uses exact symbolic checks on its normal runtime path.
- Projected vectorize/shared-memory pruning is still active generation-critical.
- The old `gpu_case_constraints.py` / `exact_gpu_constraints.cc` owners mixed projected and exact responsibilities, which made it harder to reason about what active generation actually depends on.

## Verification

Environment:

- `source /root/work/venv/bin/activate`
- `export TVM_HOME=/root/work/tvm-ansor`
- `export PYTHONPATH=$TVM_HOME/python`
- `export TVM_LIBRARY_PATH=$TVM_HOME/build-release`

Ran:

- `python -m py_compile gallery/constrained_gen/modules/gpu_projection_constraints.py gallery/constrained_gen/modules/gpu_case_constraints.py gallery/constrained_gen/modules/constraint_set.py`
- `cmake --build build-release -j4`
- `python gallery/constrained_gen/validate.py --task-index 0`
- `python gallery/constrained_gen/generate_programs.py --task-index 0`

Observed:

- build succeeded
- `validate.py --task-index 0` succeeded
- `generate_programs.py --task-index 0` succeeded and saved a record

## Remaining uncertainty

- `validate.py` still exercises `check_all_exact(...)` for diagnostics/reporting, so exact code remains live for validation and diagnostics even though active generation no longer depends on it at runtime.
- `gpu_projection_diagnostics.py` still intentionally reaches into exact helpers.
- Broader validator/reviewer follow-up is still appropriate because this touched:
  - `constraint_set.py`
  - `gpu_case_constraints.py`
  - `src/auto_scheduler/projected_gpu_constraints.cc`
  - `src/auto_scheduler/exact_gpu_constraints.cc`

## Next recommended owner

- `validator` for a small projected-generation shard plus one exact/diagnostic smoke
- then `reviewer` if this split needs formal sign-off
