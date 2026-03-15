## Summary

- Implemented the first pass of `gallery/constrained_gen/measure_programs.py`.
- Kept the public CLI minimal:
  - one positional `input_path`
  - optional `--workers`
- Reused the repo's existing TVM measurement path instead of inventing a second one:
  - `auto_scheduler.RecordReader(...).read_lines()`
  - rebuild `MeasureInput(task, inp.state)` with the canonical task from `load_and_register_tasks()`
  - `auto_scheduler.measure.ProgramMeasurer(...)`
  - `auto_scheduler.RecordToFile(...)`

## Files Changed

- `gallery/constrained_gen/measure_programs.py`

## Accepted Input Forms

- one generated record file path
- one directory path

For directory input:

- files are discovered recursively
- only `.json` files are considered
- traversal is deterministic:
  - sorted directory names
  - sorted file names

## Output Path Behavior

- measured records are written through `get_measure_record_filename(task, task.target)`
- this keeps the repo's existing measured-gen log path contract:
  - `/root/work/tvm-ansor/gallery/dataset/measured_gen_programs/{target.model}/{clean_name((task.workload_key, target.kind))}.json`
- writes go through `RecordToFile(...)`, so repeated runs append to the same measurement log

## Validation And Failure Contract

- a source file must contain records for exactly one workload key
- mixed-workload files are rejected at:
  - `stage = "validate_input_records"`
- structured failures use stable fields:
  - `input_path`
  - `stage`
  - `error`

## Worker Model

- `--workers` applies file-level parallelism only
- when `--workers > 1`, the parent process launches child `measure_programs.py <file>` subprocesses and aggregates their results
- this keeps measurement isolated per file and avoids building a second in-process measurement workflow

## Verification

- `python -m py_compile gallery/constrained_gen/measure_programs.py`
- `python gallery/constrained_gen/measure_programs.py 'gallery/dataset/to_measure_gen_programs/([1aa729c96f4afc0cf6bf84dff07364c6,[1,18,9,1,512],[1,1,1,1,512]],cuda).json'`

Observed smoke output:

```text
Get 6 programs to measure:
......*E*E*E*E*E*E[measure] start /root/work/tvm-ansor/gallery/dataset/to_measure_gen_programs/([1aa729c96f4afc0cf6bf84dff07364c6,[1,18,9,1,512],[1,1,1,1,512]],cuda).json

Time elapsed for measurement: 2.22 s
[measure] OK inputs=6 errors=6 source=/root/work/tvm-ansor/gallery/dataset/to_measure_gen_programs/([1aa729c96f4afc0cf6bf84dff07364c6,[1,18,9,1,512],[1,1,1,1,512]],cuda).json saved=/root/work/tvm-ansor/gallery/dataset/measured_gen_programs/unknown/([1aa729c96f4afc0cf6bf84dff07364c6,[1,18,9,1,512],[1,1,1,1,512]],cuda).json
measure_programs_summary input_files=1 successes=1 failures=0 measured=6 measure_errors=6
```

## Notes

- The narrow smoke did not hit an entrypoint blocker.
- The measured results for this sample all carried TVM measurement errors (`measure_errors=6`), but the measurement pipeline itself completed and wrote the output log.
- The output path used `target.model = "unknown"` for this task in the current environment, which is why the measured log landed under `measured_gen_programs/unknown/`.

## Next Recommended Owner

- `integrator` if the next pass needs broader directory smoke coverage, worker-path validation, or summary/report enrichment around measurement error classes.
