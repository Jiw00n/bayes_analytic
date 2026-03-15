# 2603151731 Task 86 4000-Unique Active-Path Profile

- Agent: `optimizer`
- Date: `2026-03-15 17:31`
- Status: `completed`
- Optimization Topic: `task 86 active-generation 4000-unique profiling`

## Scope

- Goal: profile the current active-generation path for task index `86` on a `4000`-unique-schedule run without using `generate_programs.py` as the top-level entrypoint.
- Target metric: wall-clock breakdown and cumulative-time hotspot ranking for the internal concrete-sketch -> `ScheduleGenerator.from_task_state()` -> `next_unique_schedule()` -> `build_measure_record()` path.
- Why this hotspot was selected: recent local timings on tasks `84/85/86` showed task `86` as the slowest among that set.

## Files Checked

- `gallery/constrained_gen/generate_programs.py`: `generate_concrete_sketches()`, `build_measure_record()`
- `gallery/constrained_gen/modules/schedule_generator.py`: `ScheduleGenerator.from_task_state()`, `ScheduleGenerator.next_unique_schedule()`
- `gallery/constrained_gen/modules/param_sampler.py`: `ParamSampler.next_unique_schedule()`, `_random_unique_rollout()`
- `gallery/constrained_gen/modules/domain_propagator.py`: `propagate_domain()`, `_tighten_upper_domain_from_candidates()`, `filter_by_constraints()`
- `gallery/constrained_gen/modules/expr_nodes.py`: `ExprNode.evaluate()`, `interval()`
- `gallery/constrained_gen/modules/task_paths.py`: `load_and_register_tasks()`

## Measurement Setup

- Commands or scripts run:
  - activated repo env with `source /root/work/venv/bin/activate`
  - exported `TVM_HOME=/root/work/tvm-ansor`
  - exported `PYTHONPATH=$TVM_HOME/python`
  - exported `TVM_LIBRARY_PATH=$TVM_HOME/build-release`
  - ran an inline Python timing script that:
    - loaded tasks via `load_and_register_tasks()`
    - selected `tasks[86]`
    - generated concrete sketches with `generate_concrete_sketches(task)`
    - iterated the active path with task-wide `seen` fingerprints until `4000` uniques were emitted
    - called `build_measure_record(task, payload["state"])` for each emitted unique
  - ran a second inline Python script with identical control flow under `cProfile.Profile()`
- Input shard or workload:
  - task index `86`
  - task desc `vm_mod_fused_nn_conv2d_add_add_clip_divide_multiply_4`
  - sketch count observed: `1`
- Baseline artifact path:
  - `/tmp/projected_gpu_full_validation/optimizer/task86_4000_unique_20260315_171611/`

## Bottleneck Finding

- Confirmed hotspot: `next_unique_schedule()` dominates the 4000-unique run; inside it, the heaviest stack remains `ParamSampler._random_unique_rollout()` -> `DomainPropagator.propagate_domain()` / `_tighten_upper_domain_from_candidates()` -> `ExprNode.evaluate()`.
- Evidence:
  - wall-clock timing run:
    - `generate_concrete_sketches`: `0.007050 s`
    - `ScheduleGenerator.from_task_state` total: `1.075973 s`
    - `next_unique_schedule` total over `4000` uniques: `338.335779 s`
    - `next_unique_schedule` average per unique: `0.084584 s`
    - `build_measure_record` total: `0.150702 s`
    - `build_measure_record` average per unique: `0.000038 s`
    - end-to-end total: `340.869264 s`
  - component share of wall-clock total:
    - `next_unique_schedule`: about `99.26%`
    - `from_task_state`: about `0.32%`
    - `build_measure_record`: about `0.04%`
    - `generate_concrete_sketches`: about `0.00%`
  - cProfile top cumulative-time summary for the 4000-unique run:
    - `schedule_generator.py:532(next_unique_schedule)` `531.106 s`
    - `param_sampler.py:488(next_unique_schedule)` `531.100 s`
    - `param_sampler.py:345(_random_unique_rollout)` `530.896 s`
    - `domain_propagator.py:312(propagate_domain)` `393.195 s`
    - `domain_propagator.py:350(_tighten_upper_domain_from_candidates)` `380.039 s`
    - `expr_nodes.py:326(evaluate)` `366.104 s`
    - `param_sampler.py:303(_validate_sample)` / `schedule_generator.py:162(check_all_hybrid)` / `concrete_gpu_verify.py:39(lower_with_gpu_passes)` stay visible but materially below domain propagation

## Change Summary

- What changed: no code changes; profiling only.
- What was intentionally not changed: no generator semantics, search order, or logging behavior was modified.

## Results

- Before:
  - 20-sample smoke for task `86` from recent local timings:
    - `generate_concrete_sketches`: `0.0068 s`
    - `from_task_state`: `0.8866 s`
    - `next_unique_schedule` average: `0.0875 s`
    - `build_measure_record` average: `0.000038 s`
- After:
  - 4000-unique run completed without early exhaustion: `generated_unique=4000`, `search_exhausted_before_target=false`
  - `next_unique_schedule` average stayed essentially unchanged at scale: `0.084584 s` per unique
  - no duplicates were skipped during the 4000-unique run: `duplicates_skipped=0`
  - bottleneck ranking did not materially change inside the sampling path: domain propagation and expression evaluation still dominate
  - one coarse-grain ranking change is important:
    - at 20 samples, `from_task_state` was a noticeable second component next to sampling
    - at 4000 uniques, `from_task_state` becomes amortized noise while the `next_unique_schedule` loop overwhelmingly dominates the full run
- Artifact paths:
  - `/tmp/projected_gpu_full_validation/optimizer/task86_4000_unique_20260315_171611/task86_4000_unique_timing.json`
  - `/tmp/projected_gpu_full_validation/optimizer/task86_4000_unique_20260315_171611/task86_4000_unique_cprofile.prof`
  - `/tmp/projected_gpu_full_validation/optimizer/task86_4000_unique_20260315_171611/task86_4000_unique_cprofile_top.txt`

## Correctness Risk

- What might regress: none from this task; this was measurement only.
- Required follow-up validation: if a hotspot patch is proposed in `param_sampler.py` or `domain_propagator.py`, route implementation ownership through `integrator` and require narrow validator coverage afterward because these paths affect active generation semantics.

## Next Owner

- Recommended owner: `integrator`
- Recommended next step: use the attached timing and cProfile artifacts to decide whether the next optimization attempt should target `DomainPropagator.propagate_domain()` / `_tighten_upper_domain_from_candidates()` or search-state work inside `ParamSampler._random_unique_rollout()`.
