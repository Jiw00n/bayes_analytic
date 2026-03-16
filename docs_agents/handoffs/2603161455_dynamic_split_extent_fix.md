## Dynamic split extent fix

- Owner: integrator
- Files changed:
  - `modules/symbolic_state.py`
  - `modules/transform_applier.py`
  - `modules/schedule_generator.py`
  - `modules/param_sampler.py`
  - `modules/domain_propagator.py`
  - `modules/concrete_gpu_verify.py`

### What changed

- Recorded per-`SplitStep` symbolic current extent during symbolic replay as `SymbolicState._split_step_extents`.
- Switched active sampling/materialization paths to evaluate split extents lazily from current `sym_map` instead of always using representative `g._sp_extents`.
- Patched concrete state reconstruction to rewrite raw record `SP` step extents from evaluated dynamic extents before reloading the TVM state.

### Verification

- `python -m py_compile modules/symbolic_state.py modules/transform_applier.py modules/schedule_generator.py modules/param_sampler.py modules/domain_propagator.py modules/concrete_gpu_verify.py`
- Narrow reproducer on workload `1097323f3970e5c881ad3a0028ca79cb` with `ScheduleGenerator.from_task_state(...)` and repeated `randomize_params()`:
  - sampled `step52` extents: `2048, 4096, 8192`
  - sampled `step57` extents: `32, 392, 448, 1568, 3136, 6272`

### Outcome

- Generated schedules are no longer forced to keep vectorize-local `SplitStep.extent` at the representative concrete sketch value.
- The visible effect is in emitted/reconstructed concrete states; constraints were intentionally left unchanged.

### Remaining uncertainty

- Reconstructing a different ground-truth record from the same sketch fingerprint via only `sp_/ur_` values still does not reproduce that record's exact raw `SplitStep.extent` fields.
- This patch fixes the “generated extents are frozen” bug, not full equivalence with every TVM ground-truth record in the same sketch class.

### Recommended next owner

- validator, if a wider generated-shard comparison against `/root/work/tvm-ansor/gallery/dataset/to_measure_gen_programs` is needed.
