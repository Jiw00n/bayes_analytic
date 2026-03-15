# active-generation unique schedule search and final dedupe

## What changed

- Added concrete-state fingerprinting for active generation in:
  - `gallery/constrained_gen/modules/concrete_gpu_verify.py`
- Added stateful unique-schedule search on top of the existing constrained sampler in:
  - `gallery/constrained_gen/modules/param_sampler.py`
  - `gallery/constrained_gen/modules/schedule_generator.py`
- Reworked active generation to collect task-wide unique concrete schedules and run a final in-memory dedupe before save in:
  - `gallery/constrained_gen/generate_programs.py`

## Implementation summary

- Concrete dedupe key:
  - build a stable hash from the serialized measure-record step payload for the concrete `State`
- Online generation behavior:
  - keep per-generator exhausted-prefix state
  - search over split params plus unroll params
  - prune only when a subtree is exhausted
  - reject leaf assignments that reproduce an already-seen concrete-state fingerprint
- Active generation behavior:
  - keep `seen_fingerprints` task-wide across sketches
  - build each `ScheduleGenerator` once per sketch
  - request successive unique schedules from the generator until the target count is reached or the sketch is exhausted
  - final-dedupe the collected `MeasureInput`/`MeasureResult` pairs before `save_records_batch(...)`

## Files / functions checked

- `gallery/constrained_gen/modules/concrete_gpu_verify.py`
  - `build_state_record_steps_payload(...)`
  - `concrete_state_fingerprint(...)`
- `gallery/constrained_gen/modules/param_sampler.py`
  - `next_unique_schedule(...)`
  - `_search_next_unique_from_prefix(...)`
  - `_ensure_unique_search_state(...)`
  - `_order_candidates_for_prefix(...)`
- `gallery/constrained_gen/modules/schedule_generator.py`
  - `next_unique_schedule(...)`
  - `get_unique_search_stats(...)`
- `gallery/constrained_gen/generate_programs.py`
  - `build_measure_record(...)`
  - `dedupe_measure_records(...)`
  - `process_task(...)`

## Local validation run

Environment:

- `source /root/work/venv/bin/activate`
- `export TVM_HOME=/root/work/tvm-ansor`
- `export PYTHONPATH=$TVM_HOME/python`
- `export TVM_LIBRARY_PATH=$TVM_HOME/build-release`

Commands:

- `python -m py_compile gallery/constrained_gen/modules/concrete_gpu_verify.py gallery/constrained_gen/modules/param_sampler.py gallery/constrained_gen/modules/schedule_generator.py gallery/constrained_gen/generate_programs.py`
- in-memory uniqueness smoke on task `0`:
  - collected `5`
  - unique fingerprints `5`
  - sampler stats `{'emitted_unique': 5, 'duplicates_skipped': 0, 'exhausted_prefixes': 5}`
- `python gallery/constrained_gen/generate_programs.py --task-index 0 --records-per-task 5`
  - saved unique `5/5`
  - `duplicates_skipped=0`
  - `search_exhausted=0`
- `python gallery/constrained_gen/generate_programs.py --task-index 29 --records-per-task 5`
  - saved unique `5/5`
  - `duplicates_skipped=0`
  - `search_exhausted=0`
- in-memory multi-sketch smoke on task `29`
  - first two sketches each yielded `3` unique schedules
  - combined task-level unique fingerprint count `6`
- in-memory final-dedupe smoke on task `29`
  - collected `5`
  - deduped `5`
  - dropped `0`

## Artifact note

- `generate_programs.py` writes to the canonical dataset output path and appends records there.
- Because of that append behavior, saved-file total record counts are not a clean validation signal on an already-populated output file.
- The reliable narrow validation signal for this patch is the in-memory uniqueness / dedupe smoke above.

## Concrete outcome

- The active generation path now has:
  - task-wide online concrete-state dedupe
  - subtree exhaustion-based branch pruning
  - final in-memory save-time dedupe
- Narrow local smoke did not show duplicate emissions or regressions on the checked tasks.

## Remaining uncertainty

- The narrow smoke did not hit a task that actually produced duplicate concrete states, so it validates the new control flow and no-regression path more strongly than it validates duplicate-skipping counts under a known-collision workload.
- Broader validator/reviewer follow-up is still appropriate because this touched:
  - `param_sampler.py`
  - `schedule_generator.py`
  - `concrete_gpu_verify.py`
  - `generate_programs.py`

## Next recommended owner

- `validator` for a narrow raw artifact run on the patched active generation path
- then `reviewer` to decide whether the evidence is sufficient
