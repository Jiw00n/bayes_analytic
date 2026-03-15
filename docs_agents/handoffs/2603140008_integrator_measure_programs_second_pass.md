## Summary

- Updated `gallery/constrained_gen/measure_programs.py` for the second pass.
- Kept the input contract unchanged:
  - one file or one directory
  - optional `--workers`
- Kept the underlying measurement path unchanged:
  - `RecordReader(...)`
  - `MeasureInput(task, inp.state)`
  - `ProgramMeasurer(...)`
  - `RecordToFile(...)`

## Files Changed

- `gallery/constrained_gen/measure_programs.py`

## Reporting Changes

- Per-file success output now distinguishes:
  - pipeline/file processing success
  - usable measurement outcome
- Each successful file result now carries:
  - `pipeline_ok`
  - `usable_measurement`
  - `measured`
  - `measure_errors`
  - `measure_error_histogram`

Final summary now reports:

- `pipeline_successes`
- `pipeline_failures`
- `usable_measurements`
- `files_with_measure_errors`
- `measured`
- `measure_errors`
- histogram by `MeasureErrorNo` name when available

## Measure Error Semantics

- `measure_errors` do **not** count as pipeline failures by themselves.
- Exit status remains tied to pipeline/file processing failure only:
  - nonzero exit if any file failed at discovery, load, validation, worker, or measurement-exception stage
  - zero exit if all files completed through the TVM measurement pipeline, even if some or all measured results contain TVM measure errors
- Usability is therefore reported separately through:
  - `usable_measurements`
  - `files_with_measure_errors`
  - `measure_error_histogram`

## Verification

- `python -m py_compile gallery/constrained_gen/measure_programs.py`
- reran the first-pass single-file smoke:
  - `python gallery/constrained_gen/measure_programs.py 'gallery/dataset/to_measure_gen_programs/([1aa729c96f4afc0cf6bf84dff07364c6,[1,18,9,1,512],[1,1,1,1,512]],cuda).json'`
- ran a small directory/worker smoke:
  - `python gallery/constrained_gen/measure_programs.py gallery/dataset/to_measure_gen_programs --workers 2`

Observed single-file smoke output:

```text
Get 6 programs to measure:
......*E*E*E*E*E*E[measure] start /root/work/tvm-ansor/gallery/dataset/to_measure_gen_programs/([1aa729c96f4afc0cf6bf84dff07364c6,[1,18,9,1,512],[1,1,1,1,512]],cuda).json

Time elapsed for measurement: 2.25 s
[measure] OK pipeline=1 usable=0 inputs=6 errors=6 source=/root/work/tvm-ansor/gallery/dataset/to_measure_gen_programs/([1aa729c96f4afc0cf6bf84dff07364c6,[1,18,9,1,512],[1,1,1,1,512]],cuda).json saved=/root/work/tvm-ansor/gallery/dataset/measured_gen_programs/unknown/([1aa729c96f4afc0cf6bf84dff07364c6,[1,18,9,1,512],[1,1,1,1,512]],cuda).json
measure_programs_summary input_files=1 pipeline_successes=1 pipeline_failures=0 usable_measurements=0 files_with_measure_errors=1 measured=6 measure_errors=6
measure_error_histogram
  RUNTIME_DEVICE=6
```

Observed directory worker smoke output:

```text
Get 2 programs to measure:
..*E*E[measure] start /root/work/tvm-ansor/gallery/dataset/to_measure_gen_programs/([25252ef28760d56401943904a46661f3,[1,16,16,480],[1,1,1,480]],cuda).json
Time elapsed for measurement: 2.02 s
[measure] OK pipeline=1 usable=0 inputs=2 errors=2 source=/root/work/tvm-ansor/gallery/dataset/to_measure_gen_programs/([25252ef28760d56401943904a46661f3,[1,16,16,480],[1,1,1,480]],cuda).json saved=/root/work/tvm-ansor/gallery/dataset/measured_gen_programs/unknown/([25252ef28760d56401943904a46661f3,[1,16,16,480],[1,1,1,480]],cuda).json
Get 6 programs to measure:
......*E*E*E*E*E*E[measure] start /root/work/tvm-ansor/gallery/dataset/to_measure_gen_programs/([1aa729c96f4afc0cf6bf84dff07364c6,[1,18,9,1,512],[1,1,1,1,512]],cuda).json
Time elapsed for measurement: 2.21 s
[measure] OK pipeline=1 usable=0 inputs=6 errors=6 source=/root/work/tvm-ansor/gallery/dataset/to_measure_gen_programs/([1aa729c96f4afc0cf6bf84dff07364c6,[1,18,9,1,512],[1,1,1,1,512]],cuda).json saved=/root/work/tvm-ansor/gallery/dataset/measured_gen_programs/unknown/([1aa729c96f4afc0cf6bf84dff07364c6,[1,18,9,1,512],[1,1,1,1,512]],cuda).json
measure_programs_summary input_files=2 pipeline_successes=2 pipeline_failures=0 usable_measurements=0 files_with_measure_errors=2 measured=8 measure_errors=8
measure_error_histogram
  RUNTIME_DEVICE=8
```

## Outcome

- No new entrypoint blocker was hit.
- The reporting distinction is now explicit:
  - pipeline succeeded for all tested files
  - none of the tested files produced usable measurements in this environment
  - the concrete error class observed in both smokes was `RUNTIME_DEVICE`

## Next Recommended Owner

- `integrator` if the next pass should add more research-facing aggregation or persist richer summary artifacts without changing the TVM measurement path.
