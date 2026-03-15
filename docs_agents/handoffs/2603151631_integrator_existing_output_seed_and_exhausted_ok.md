# existing-output fingerprint seeding and exhausted-ok termination

## What changed

- Updated `gallery/constrained_gen/generate_programs.py` so active generation:
  - seeds task-wide `seen_fingerprints` from the existing canonical output file before generating
  - treats zero-new-candidate exhaustion as a normal successful termination instead of a failure

## Files changed

- `gallery/constrained_gen/generate_programs.py`

## Implementation summary

- Added:
  - `load_existing_output_fingerprints(task, output_path)`
- `process_task(...)` now:
  - reads existing saved records from `get_to_measure_gen_filename(task)`
  - computes their concrete-state fingerprints
  - initializes task-wide `seen_fingerprints` with those existing fingerprints
  - returns `ok=True`, `stage="exhausted"` when no new unique schedules can be produced

## Validation

Environment:

- `source /root/work/venv/bin/activate`
- `export TVM_HOME=/root/work/tvm-ansor`
- `export PYTHONPATH=$TVM_HOME/python`
- `export TVM_LIBRARY_PATH=$TVM_HOME/build-release`

Commands:

- `python -m py_compile gallery/constrained_gen/generate_programs.py`
- in-memory seed check on task `0`
  - existing output fingerprint seed count: `245`
  - additional unique schedules after seeding: `0`
- `python gallery/constrained_gen/generate_programs.py --task-index 0 --records-per-task 1`
  - printed `seeded_existing=245`
  - sketch exhausted without generating a new record
  - terminated with `EXHAUSTED ... new_unique=0`
  - outer summary reported `successes=1 failures=0`

## Concrete outcome

- Existing saved records are now part of the dedupe seed for active generation.
- If a task has no new unique schedules left, `generate_programs.py` now exits that task successfully as exhausted instead of failing.

## Remaining uncertainty

- Validation was narrow and used an already-populated output file for task `0`.
- A broader shard would still be useful if we want confidence across more workloads with mixed empty/non-empty output files.

## Next recommended owner

- none required for the requested patch
